# 系统方法拆解(Method)

> MagicSim 是系统论文,这里按"组件 → 运行时拓扑 → 三驱动器 → 异步规划 → episode 生命周期"逐层拆开。

---

## 0. 总体一图(ASCII)

```
                          ┌───────────────────────────────────────────────┐
                          │  YAML-first 规范:内容/布局/行为/暴露 解耦       │
                          └───────────────────────┬───────────────────────┘
                                                  │ 构建
                                                  ▼
   ┌───────────────────────────  Env-Core:一个批处理同步运行时  ───────────────────────────┐
   │  IsaacRLEnv host  →  SimulationContext · PhysX · RTX render · 批处理 sim_step()          │
   │  BaseEnv          →  TerrainManager · LayoutManager · SceneManager · Seeded DR           │
   │                      reset_idx · manager-local replay · P1/P2/P3 生命周期                │
   │  SyncBaseEnv      →  CameraManager · CaptureManager · 原生标注器                          │
   │  SyncRobotEnv     →  PlannerManager.step → RobotManager.step → IsaacRLEnv.step           │
   └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │  wraps SyncRobotEnv as self.scene
                                                ▼
                  ┌──────────────  MDP 边界:TaskBaseEnv(Gym 兼容)  ──────────────┐
                  │  step(action, env_ids) → obs · reward · term/trunc · info/success  │
                  └───────┬───────────────────────┬──────────────────────────┬────────┘
                          │                       │                          │
              ┌───────────▼──────────┐ ┌──────────▼───────────┐ ┌────────────▼─────────────┐
              │ Driver 1: RL 训练     │ │ Driver 2: AutoCollect │ │ Driver 3: Inference/Replay│
              │ policy 直接 step      │ │ Command→Skill→Planner │ │ planned:policy/replay     │
              │ (current)            │ │ →Robot→Record (current)│ │ 用 RobotManager 动作空间  │
              └──────────────────────┘ └───────────────────────┘ └──────────────────────────┘
```

---

## 1. 组件拆解(Manager-based 运行时)

MagicSim 的可扩展性单元是**批处理仿真器进程**。一个 Isaac Sim 实例托管许多子环境;**每个 manager 拥有共享 stage 上 per-env 的状态切片**:

| Manager | 负责 |
|---------|------|
| SceneManager / LayoutManager / TerrainManager | 世界内容、布局、地形 |
| RobotManager | 机器人动作空间、闭环控制、step |
| PlannerManager / GlobalPlannerManager | 规划请求调度、cuRobo IK/MotionGen、导航(Nav/DWB) |
| AtomicSkillManager | 技能状态机:submit/pull 规划请求 |
| CameraManager / CaptureManager / TiledCaptureManager | 相机流、RGB-D 捕获 |
| 原生标注器(native annotators) | 末端 waypoint、物体 3D 包围盒、affordance 目标 |
| AutoCollectManager | 指令采样、任务推进 |
| RecordManager | per-env 观测/动作/采集流缓冲,成功才写盘 |
| (Avatar / Nav / Language managers) | avatar 动画、导航、L1/L2/L3 语言标注 |

> Environment wrapper 通过统一的 **reset & step 生命周期**把这些 manager 串起来。

---

## 2. 运行时拓扑与两条边界(Figure 2)

**边界 1:Env-Core ↔ 任务层。** Env-Core 的 managers + 同步 wrapper 构建并步进物理世界;在机器人环境中通过 `SyncRobotEnv` 暴露。层级:
```
IsaacRLEnv host  →  BaseEnv  →  SyncBaseEnv  →  SyncCollectEnv  →  SyncRobotEnv
```

**边界 2:MDP 边界。** `TaskBaseEnv` 包裹 `SyncRobotEnv`(作为 `self.scene`),定义:
- 观测空间(observation space)
- 动作空间(action space)
- 奖励(reward)
- 终止(termination)
- 成功信号(success)

> **这条边界是不同研究模式的会合点**:边界之下复用同一个世界/机器人/传感器/规划器/确定性;边界之上挂 RL 策略、脚本采集栈、或策略/回放 driver。
>
> 拓扑是**刻意非对称**的:RL 训练是直接 step `TaskBaseEnv` 的外部循环;脚本采集是在同一任务之上叠 wrapper(`AsyncRobotEnv` 加技能/规划,`AutoCollectEnv` 加指令采样+录制);推理/回放是同一 MDP 上的外部 driver。

---

## 3. 一个 MDP,三个驱动器(§3.4,Table 3)

| 驱动器 | 状态 | 动作路径 |
|--------|------|----------|
| **RL 训练** | current | policy 直接 step `TaskBaseEnv`;动作由 task + RobotManager 动作空间解释;返回批量 obs/reward/term/success |
| **脚本 AutoCollect** | current | `AutoCollectEnv` 经 `AsyncRobotEnv` 包裹同一任务;路径 `Command→Skill→Planner→Robot→Record`;per-env 状态机驱动脚本演示 |
| **Inference / Replay** | **planned** | `InferenceRunner` 用 policy 或回放动作流 step 同一 `TaskBaseEnv`;走 RobotManager 动作空间,**不**经过 Collect 层的 AtomicSkill / GlobalPlanner |

> **两轴正交**:driver(怎么 step:RL/AutoCollect/Inference)与 exposure(怎么用:benchmark rollout / dataset trajectory / agent interaction)互不绑定。driver 变,批处理运行时不变。

---

## 4. Agent 级异步(§3.5,§6.7,Figure 3)

核心机制:**物理时间同步,语义时间异步**。

```
同步物理时间:  t ── t+1 ── t+2 ── t+3 ── t+4   (所有活跃 env 跨越同一 sim_step() 屏障)
                │      │      │      │      │
Env0  采样指令 → AtomicSkill阶段 → 提交IK/MotionGen future → 轮询(保持姿态/安全动作) → 消费结果执行
Env1  ...      (与 Env0 处于不同语义阶段)
Env2  GlobalPlanner动作 → RobotManager step → 成功判定 → 语言L1/L2/L3 → RecordManager保存 → reset
Env3  ...
        └─ per-env 语义状态在 env_id 上异步 ─┘

Off-loop 异步规划计算服务(非阻塞 future;微批求解):
   请求队列/Future  →  微批窗口  →  批处理 cuRobo IK/MotionGen  →  future done/per-env 结果
```

**submit-and-poll 契约**:技能或 global planner 提交请求 → 收到 future → batch 继续步进 → 结果就绪时消费。两个关键属性:
1. 仿真循环**永不被慢解阻塞**;
2. per-agent 请求被聚合成**更大的批量求解**,而非许多孤立调用(提高 GPU 利用率)。

这也是"无魔法动作"的机制:高层指令实例化 task command → 推进 AtomicSkill 状态机 → 提交规划请求 → 执行机器人动作 → 观测 → 更新任务状态。

---

## 5. Episode 生命周期:交互如何变成数据(§3.6,Figure 4)

```
Specifications (场景/任务/本体/相机/DR种子/录制配置)
        ▼
Manager reset:可复现初始状态
   Layout → Scene → Terrain → Avatar/Nav → Camera/Capture → Robot → Planner
        ▼
  ┌──────────── 步进循环:一个批处理物理 tick + per-env 语义状态 ────────────┐
  │  Command (AutoCollectManager 采样/推进)                                  │
  │  Skill phase (AtomicSkillManager submit/pull)                          │
  │  Planner action (GlobalPlannerManager:cuRobo / Nav / DWB)              │
  │  Execution (PlannerManager → RobotManager → sim_step())                │
  │  Observe & evaluate (get_obs / reward / termination / task state)      │
  │  Buffer (RecordManager:obs/action/collect 流)                          │
  │  Annotate (CaptureManager + 原生标注器 + Language L1/L2/L3)             │
  └────────────────────────────┬───────────────────────────────────────────┘
                               ▼
                    成功且结束? ──no──► running:保留缓冲,下一 tick
                               │
                              yes
                               ▼
              Save trajectory:action/ · collect/ · env/info/ · env/camera/ · language/
                               │   (失败或截断 → flush/re-roll,清空缓冲,保留最新观测,reset_idx/下一任务)
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                       ▼
  Benchmark rollout      Dataset trajectory       Agent-facing interaction (planned)
  obs/reward/done/       对齐的动作/采集 trace/    命令、观测、后果随时间展开
  success                相机/标注/语言
```

> **关键:在 MagicSim 里,交互在成功并被记录之前不是数据。** 成功信号扮演双重角色——既是 MDP 的评测信号,又是数据采集的写入门控(write gate)。

---

## 6. 仿真引擎:统一多物理骨干(§4)

**参数化构造**。每个仿真实体不是固定外观/动力学的 USD prop,而是经代码导入/生成后,赋予仿真族、绑定材质、解析位姿尺度、附加求解器状态、暴露观测 hook。抽象写法:

```
x_e = ( a_e, f(e), T_e, s_e, θ_vis_e, θ_phys_e, θ_solver_e, h_e )
```
- `e = (n,c,i)`:环境 id、逻辑类别、实例 id;`c` = 含义(mug/tray/shirt…),`f(e)∈F` = 怎么仿真。
- `a_e` 资产源,`T_e` 世界变换,`s_e` 尺度,`θ_vis/phys/solver` 视觉/物理/求解器参数,`h_e` 观测标注 hook。

**仿真族 K=15**(`K=|F|` 写成变量,可扩展):

| 方法类 | 族 |
|--------|----|
| 刚体与铰接 | Rigid, Articulation, RigidInstancer |
| 有限元(FEM) | Deformable, FEMCloth |
| 粒子/位置基 | Garment, Fluid, Sand, Rice, Inflatable |
| 约束链 | Rope |
| 力场 | Magnet |
| 体积/发射效果 | Fire, Flow |
| 动画驱动 | Avatar |

**耦合多物理**两层含义:① **co-execution**(共存于同一 stage/episode);② **physical coupling**(跨求解器边界真实相互作用,如 Rigid×Deformable、Rigid×Fluid)。

---

## 7. 伪代码:三驱动器复用一个 MDP + 异步规划

```python
# ---- 一次定义:任务 MDP ----
task = TaskBaseEnv(scene=SyncRobotEnv(...))   # obs/action/reward/term/success

# ---- Driver 1: RL 训练(直接 step) ----
obs = task.reset()
for _ in range(T):
    action = policy(obs)                       # 外部 RL 循环
    obs, reward, term, trunc, info = task.step(action, env_ids)

# ---- Driver 2: AutoCollect(Command→Skill→Planner→Robot→Record) ----
collect = AutoCollectEnv(AsyncRobotEnv(task))  # 叠 AtomicSkill + GlobalPlanner + 采样/录制
for _ in range(T):
    collect.sample_or_advance_command()        # 采样/推进指令
    skill = collect.atomic_skill.step()        # 技能状态机
    fut   = skill.submit_planner_request()     # 提交 → 拿 future(非阻塞)
    # ... batch 继续步进其它 env ...
    if fut.done(): action = fut.result()       # 就绪才消费
    else:          action = skill.hold_or_safe_action()
    obs, r, term, trunc, info = collect.step(action, env_ids)
    collect.record.buffer(obs, action, annotations, language)
    if info["success"] and info["done"]:
        collect.record.save_trajectory(env_id) # 仅成功落盘
    else:
        collect.record.flush_if_failed(env_id)

# ---- Driver 3: Inference/Replay(planned,走 RobotManager,跳过 Skill/Planner 层) ----
runner = InferenceRunner(task)                 # policy 或 replay 动作流
```

---

## 8. 实现要点与复现风险(Pitfalls)

- **确定性是契约**:reset 顺序、不相交种子流(disjoint seeded streams)、状态快照、`reset_to` 必须严格;任何全局随机源(`torch`/`numpy` 全局 RNG)都会破坏 per-env 可复现性。
- **异步规划的"挂起态"**:future 未就绪时环境必须有合法的"保持姿态 / 安全动作",否则机器人会在等待中失稳——这是 planner-in-the-loop 的核心工程难点。
- **微批窗口调参**:微批窗口太小 → GPU 利用率低;太大 → 语义延迟高、机器人长时间挂起。是吞吐与实时性的权衡旋钮。
- **共享 stage 的隔离**:一个 env 的坏状态可能污染共享 Isaac Sim stage;manager per-env 切片 + 选择性 reset(`env_ids`)是缓解手段。
- **成功门控的偏差**:只保留成功轨迹 → 数据无失败/恢复样本;若下游需要 robust policy,需额外设计负样本采集。
- **P1/P2/P3 对象生命周期**:动态对象的创建/销毁需与 reset/replay 一致,否则回放对不上初始条件。
- **driver 状态不一**:Inference/Replay 与 Agent/VLM serving 仍是 planned;复现 Agent 闭环时不要假设这些已开箱即用。
