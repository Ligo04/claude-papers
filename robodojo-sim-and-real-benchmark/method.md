# RoboDojo 方法论拆解

> 论文：*RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies*
> arXiv 2607.04434v3（2026-07-08，cs.RO）
> 本文是 **system / benchmark 论文**，不是提出新策略算法的论文——它的“方法”是评测系统架构：如何用统一接口连接仿真诊断与真机验证，并给出可复现、防刷榜、可持续扩展的协议。以下拆解以 `FACTS.md` 为论文事实源；源码细节另见 `implementation-map.md`。

---

## 1. 系统总览

RoboDojo 由三大组件构成，彼此的关系可以概括为一句话：

> **策略只需要对接 XPolicyLab 一次，就能同时跑仿真评测和真机评测，最终汇入同一份公开榜单。**

```
                        ┌─────────────────────────────┐
                        │        被评测策略 (Policy)     │
                        │  Hy-Embodied / π0.5 / X-VLA /  │
                        │  GR00T-N1.7 / ... (30 个已接入) │
                        └───────────────┬───────────────┘
                                        │  统一策略接口
                                        │  (obs update / act predict /
                                        │   policy reset / batched query)
                                        ▼
                        ┌─────────────────────────────┐
                        │         XPolicyLab           │
                        │   "集成一次，处处评测"          │
                        │  数据转换工具 / 训练模板 /       │
                        │  部署流程 / 评测脚本 标准化       │
                        └───────┬───────────────┬───────┘
                                │               │
                 ┌──────────────┘               └──────────────┐
                 ▼                                              ▼
   ┌───────────────────────────┐                 ┌───────────────────────────┐
   │   仿真评测 (Isaac Sim/Lab)   │                 │  RoboDojo-RealEval (真机)   │
   │  42 任务 / 5 能力维度         │                 │  18 任务 / 3 本体            │
   │  异构并行 (Heterogeneous     │                 │  标准化硬件 + 触摸屏软件      │
   │  Parallelism) + cuRobo v2   │                 │  + transparent overlay 复位  │
   │  + VR 遥操作 采集数据         │                 │  + 云端远程评测              │
   └──────────────┬────────────┘                 └──────────────┬────────────┘
                  │  score / success rate                       │  score / success rate
                  │  (3 seeds mean±std)                          │  (10 trials, 3 评委 double-blind)
                  └───────────────────┬──────────────────────────┘
                                      ▼
                        ┌─────────────────────────────┐
                        │      公开榜单 (Leaderboard)     │
                        │  AI MMLab Club 维护             │
                        │  公开 layouts + hidden          │
                        │  verification layouts 防作弊    │
                        └─────────────────────────────┘
```

三块的分工非常清晰，各自解决动机中提到的一个痛点：

| 组件 | 解决的问题 | 关键设计 |
|---|---|---|
| 仿真平台 | benchmark 任务太简单/太窄，测不出「泛化/记忆/精度/长程/开放语义」 | 42 任务、5 维度、异构并行提速同时保多样性 |
| RoboDojo-RealEval | 真机评测贵、慢、难复现 | 标准化硬件夹具 + 布局回放 + 云端远程 |
| XPolicyLab | 每接入一个新策略都要重新写胶水代码 | 统一策略接口，一次集成两边通用 |

下面逐块拆解。

---

## 2. 仿真平台拆解

### 2.1 MagicSim 基础设施 + 配置优先设计

v3 明确说明 RoboDojo **直接建立在 MagicSim 基础设施上**：复用其 manager-based runtime、环境启动/物理渲染配置、确定性 seed 管理和物理对象抽象，再连接 Isaac Lab 的 vectorized RL interface。RoboDojo 的新增工作集中在机器人 benchmark 的任务构造、场景随机化、评测协议和数据/策略接入，而不是从头重写 USD/PhysX 仿真底座。

在这层基础上，核心设计是**任务规约（task spec）与仿真执行（simulation runtime）解耦**。每个任务由模块化 **YAML** 描述：

```yaml
# 概念示意，非论文原文 YAML
task: insert_tubes
assets:
  - id: tube_rack
    source: articulated_lib / ClothesNet / Meshy_AI_recon
  - id: tube
    count_range: [1, N]
layout:
  init_distribution: ...       # 物体初始位姿分布
  clutter_range: [0, 25]       # 杂物随机化上限（对比 RoboTwin2.0 的 10）
randomization:
  lighting: [...]
  background_texture: [...]
success_condition: ...
```

好处：同一套仿真 runtime 可以承载 42 个差异极大的任务，新增任务只需写 YAML，不需要改仿真引擎代码；也让"资产库 + 物理引擎 + 任务定义"三者可以独立迭代。

### 2.2 确定性 seed 采样的 reset 流程

```
reset(seed):
    rng = deterministic_RNG(seed)
    object_pose      = sample(rng, init_distribution)
    joint_state      = sample(rng, joint_init_range)
    clutter_layout   = sample(rng, clutter_range)      # ≤25 个杂物
    lighting         = sample(rng, lighting_range)
    background_tex   = sample(rng, background_texture_range)
    load_scene(object_pose, joint_state, clutter_layout, lighting, background_tex)
```

同一个 seed 让**场景采样过程可重复**，不同 seed 提供场景多样性。这里不能写成“整个 rollout 位级完全相同”：论文的跨 GPU 稳定性实验仍观察到轻微渲染/底层执行非确定性（最大 SR 标准差 1.1 个百分点、score 标准差 1.07）。统一场景构造接口支持刚体、铰接体和可变形体。

### 2.3 物理接地数字资产库（灵感 ManiTwin）

资产来源分三路：

1. **刚体**：在线开源资产库 + 用 **Meshy AI** 对真实日用物做三维重建，重建后统一标注类别 / 语言描述 / 放置区域 / 成功检查逻辑 / 操作 affordance（例如"可抓取区域""可插入孔位"）。
2. **铰接体**：开源库 + 3D 扫描件，统一标准化运动学结构（关节轴、限位）与碰撞几何。
3. **可变形体**（布料等）：来自 **ClothesNet**，同样标准化材料参数。

这一层的目的是让"看起来真实"的资产真正具备"physically-grounded"（有正确碰撞体、有正确关节限位、有正确材料参数），否则策略在仿真里学到的抓取/操作策略无法反映真实物理难度。

### 2.4 异构并行 Heterogeneous Parallelism 【核心系统创新】

这是论文中有直接吞吐 A/B 的核心系统贡献。先对比两种并行范式：

**(a) 同构克隆并行（标准 vectorized 仿真做法）**

```
Env 0: [Scene Template A] pose=p0, seed=s0
Env 1: [Scene Template A] pose=p1, seed=s1
Env 2: [Scene Template A] pose=p2, seed=s2
Env 3: [Scene Template A] pose=p3, seed=s3
...
```
所有并行环境共享**同一个场景模板**，只有物体位姿/随机种子等状态变化，场景多样性被压缩到“同一模板的变体”。

**(b) 异构并行（RoboDojo 方案）**

```
Env 0: [任务 insert_tubes,  N_clutter=3,  资产集合 A] ← 独立采样的场景配置
Env 1: [任务 classify_objects, N_clutter=18, 资产集合 B]
Env 2: [任务 match_and_pick,  articulated door, 资产集合 C]
Env 3: [任务 fold_cloth,     deformable asset,  资产集合 D]
        │
        ▼ 所有环境仍共享同一个 vectorized 接口 step()
   ┌─────────────────────────────────────────┐
   │     shared vectorized stepping interface │
   │     每个 env 保留独立场景与任务状态          │
   └─────────────────────────────────────────┘
```

关键点：多个环境**仍然共享 vectorized stepping interface**，但**每个环境独立持有场景配置、任务状态、随机种子、观测流和成功条件**。论文附录给出的实现描述是：关闭 cloned environments 的严格 physics replication，让每个环境从 seed-controlled 配置实例化自己的 scene modules；大规模评测时再按 episode seeds 分到多 GPU 的独立 simulator processes。论文没有宣称自定义了新的 physics kernel，也没有给出资产内存布局算法。

论文后官方源码与这一描述一致：`InteractiveSceneCfg(replicate_physics=False)` 是关键配置，`TaskEnv` 再把 per-env seed 传播给 Scene/Robot/Camera manager。完整路径见 `implementation-map.md`。

为什么论文把它概括为“兼顾效率与多样性”：传统同构配置需要用多组任务/场景进程覆盖多样性，RoboDojo 则让并行环境在同一向量化评测批次中承载不同任务与场景。论文的系统测量是：zero-action **77.4 vs 40.0 interactions/s（1.94×）**；接入 π0.5 后 **64.0 vs 39.2 interactions/s（1.63×）**。论文没有把加速逐项分解成加载、物理、渲染和调度成本，因此教学 demo 中的“模板重载开销”只是解释权衡的简化模型，不是论文验证过的唯一因果机制。

硬件配置：**8×RTX 4090**，每卡 1 进程、每进程 10 个并行环境；π0.5 在同一局域网内的 **A800** 服务器推理。v3 还补充：RoboDojo 渲染 `640×480`，用于横向参考的 RoboTwin 2.0 渲染 3 路 `320×240` 且每卡 2 个进程，因此两者的 77.4/44.6 不能当作等负载的纯并行算法 A/B。

### 2.5 数据采集双模式

```
采集模式选择:
  if 任务可由低层技能组合表达 (grasp/place/handover/insert/open/close/stack/push_up):
      使用 cuRobo v2 运动规划器自动合成轨迹   # 高吞吐、低标注成本
  else:  # 难以自动分解/需要人类直觉的任务
      使用 VR 遥操作采集                       # 覆盖自动合成的长尾
```

最终训练集规模：**35 个目录（34 训练任务 + 1 个 DLC 辅助目录）× 100 条轨迹 = 3,500 条轨迹，共 1,859,602 帧，20.66 小时双臂数据，25 Hz 记录**。注意 **Open 维度不提供训练数据**——这是有意设计，Open 专门测试“训练时未见的任务规约、但组成技能在别的上下文中出现过”的重组与迁移能力。额外 100 条 DLC 轨迹是 train-only 的数据级增强，与评测任务独立。v3 没有展开 `DLC` 的英文全称，因此不替作者补造缩写。

---

## 3. RoboDojo-RealEval 拆解

真机平台要解决"贵、慢、难复现"，方案是把"物理空间的一切自由度"用标准化夹具锁死，只留下策略本身的自由度。

### 3.1 标准化硬件

| 项目 | 规格 |
|---|---|
| 占地 | 1.5 m × 1.5 m |
| 桌面 | 1.2 m × 1.2 m |
| Head camera | 1× Gemini 335L |
| Wrist camera | 2× Gemini 305 |
| 光源 | LED 线光源，24 小时连续（消除昼夜光照漂移） |
| 结构件 | 模块化结构件 + 标准 3D 打印夹具（固定臂/相机/工作空间的相对位姿） |
| 支持本体 | ARX X5, Piper, Piper X（3 本体） |

标准化的意义：不同实验室复现同一套 RoboDojo-RealEval 站点时，相机内外参、光照条件、工作空间几何都是"同一份图纸"生产出来的，从而使跨站点评测结果具备可比性。

### 3.2 触摸屏软件流程

```
触摸屏 Web 界面:
  [场景复位] → [策略执行] → [急停] → [视频采集] → [云端打分]
```

### 3.3 transparent overlay 布局复位机制

这是解决"人工摆放物体无法做到像仿真 reset 一样精确"的关键 trick：

```
复位一个 trial 的物理布局:
  1. 从预采集的目标布局图中取出参考图像 (target_layout_image)
  2. 触摸屏上把 target_layout_image 以半透明形式叠加到实时相机流上
     screen.render(overlay(target_layout_image, live_camera_feed, alpha=0.5))
  3. 操作员手动移动物体，直到实时物体轮廓与半透明参考轮廓重合
  4. 确认复位 → 记录 trial 起始状态 → 开始执行策略
```

本质上是用"人眼 + 视觉对齐"替代仿真里的"确定性 seed 采样"，把不可能做到数值级精确复位的真实世界问题，降级为"视觉对齐到可接受误差"的可操作问题。这也是后续第 4 节"hidden layouts 一致性"依赖的基础设施。

### 3.4 云端远程评测

站点支持本地部署，也支持云端远程评测——策略方不需要亲自到场，可以提交策略到云端排队，在标准化站点上远程执行、远程回收视频与分数。

真机效率数据：**3.4 小时完成 180 次物理 trials；平均每任务 10 trials 需 11.2 分钟**。

---

## 4. XPolicyLab 拆解

目标：让"评测方"不需要为每个策略单独写一套仿真接口 + 一套真机接口，而是策略方只需实现**一套**统一接口，两边都能跑。

### 4.1 统一策略接口

```
interface Policy:
    def reset(self) -> None:
        """策略内部状态复位（如清空历史观测缓存，对应 Memory 维度任务尤其关键）"""

    def update_obs(self, obs: Observation) -> None:
        """接收一帧观测（RGB×3 + robot states + language instruction）"""

    def get_action(self) -> list[Action]:
        """基于当前内部状态预测一个 action chunk"""

    def update_obs_batch(self, obs_batch: list[Observation]) -> None: ...

    def get_action_batch(self, env_idx_list=None) -> list[list[Action]]:
        """批量 action chunk，按仍在运行的环境索引对齐。"""
```

### 4.2 "集成一次，处处评测"的落地方式

```
策略开发者:
   实现一次 Policy 接口
        │
        ├── 数据转换工具  → 把策略原生数据格式转换成 RoboDojo 标准格式
        ├── 训练模板      → 复用 XPolicyLab 提供的训练 pipeline 骨架
        ├── 部署流程      → 同一份部署代码可指向仿真 or 真机 endpoint
        └── 评测脚本      → 标准化调用 update_obs[_batch] / get_action[_batch]
        │
        ▼
   在仿真里快速跑 42 任务迭代调参
        │  (最小改动，只切换 endpoint 配置)
        ▼
   部署到 RoboDojo-RealEval 远程真机评测
```

策略内部架构（VLA、world-action model、经典模仿学习等）由各实现保留，XPolicyLab 标准化外部接口与工作流。论文冻结版本集成 **30 个策略**（Hy-Embodied-0.5-VLA、π0.5、X-VLA、GR00T-N1.7 等），说明接口覆盖了多类现有实现，但不保证所有未来架构都无需额外适配。

---

## 5. 评测协议算法流

### 5.1 仿真评测伪代码

```python
def evaluate_sim(policy, seeds=3, trials_per_task=50):
    dimension_scores = {}   # 5 个能力维度各自的均值
    for dim in ["Generalization", "Memory", "Long-Horizon", "Precision", "Open"]:
        task_scores = []
        for task in tasks_of(dim):
            per_seed_scores = []
            for seed in range(seeds):                      # 3 个随机种子
                set_policy_seed(policy, seed)
                episode_results = []
                for episode in range(trials_per_task):      # 每任务 50 episodes
                    if dim == "Generalization":
                        # 50 = 25 standard + 25 random(未见场景变化)
                        cfg = standard_cfg() if episode < 25 else random_cfg()
                    else:
                        cfg = default_cfg(task)
                    env = reset(task, cfg, seed=deterministic_seed(task, episode, seed))
                    policy.reset()
                    result = rollout(policy, env)            # success(0/1) + partial progress score
                    episode_results.append(result)
                per_seed_scores.append(aggregate(episode_results))
            task_scores.append(mean_over_seeds(per_seed_scores))   # mean±std 报告
        dimension_scores[dim] = mean(task_scores)

    total_score = mean(dimension_scores.values())   # 5 维度均值，而非全任务平均
                                                      # → 防止任务数多的维度主导总分
    return dimension_scores, total_score
```

要点核对：全部 42 任务 × 50 episodes = **2,100 episodes**；每策略 3 seeds × 50 trials/task = **每任务 150 trials**；总分是**5 个维度的均值**（不是 42 个任务的简单平均），这样任务数最多的 Generalization（12 任务）不会稀释掉任务数少的 Memory（6 任务）的权重。

### 5.2 真机评测伪代码

```python
def evaluate_real(policy, robots=("ARX X5", "Piper", "Piper X"), trials_per_task=10):
    results = {}
    for robot in robots:
        for task in tasks_for_robot(robot):          # 每本体 6 任务，共 18 任务
            task_scores = []
            for trial in range(trials_per_task):      # 每任务 10 trials
                target_layout = load_precaptured_layout(task, trial)
                operator_reset_via_transparent_overlay(target_layout)   # 3.3 节的复位机制
                policy.reset()
                video, action_log = execute_policy_on_hardware(policy, robot)
                # 3 位评委 double-blind 独立打分（含最终成功 + 中间子步骤进度）
                scores = [judge_i.score(video, action_log) for judge_i in judges]
                trial_score = mean(scores)
                task_scores.append(trial_score)
                publish(video, trial_score)            # 公开视频+分数，附申诉机制
            results[(robot, task)] = mean(task_scores)
    return results   # Overall Avg = 跨 robot × task 聚合
```

### 5.3 防作弊协议

```
官方认证榜准入条件（AND 关系，全部满足才算 "认证"）:
  1. 必须走官方在线评测系统（不允许自行报数）
  2. 仿真结果必须报 3 seeds 的 mean±std；真机结果必须覆盖全部 3 本体
  3. 必须通过 hidden verification layouts 的一致性检查
  4. 必须公开 checkpoint / 训练部署代码 / 配置 / 复现说明（经由 XPolicyLab 标准化格式）
  5. 必须发布评测视频
不满足 → 单独放入 closed-source / unverified track，不视为官方 verified leaderboard 条目
```

`hidden verification layouts` 的作用：公开 layouts 是主评测集，允许策略方在其上调参；但额外在一批**不公开**的 layouts 上跑一遍，如果两者性能差异异常（例如公开集上手调过拟合、刷榜），就能被检测出来——这是"benchmark 治理"层面的工程化手段，而不是算法层面的创新，但对榜单公信力至关重要。榜单本身由公益组织 **AI MMLab Club** 维护。

---

## 6. 复现风险与实现陷阱

| 风险点 | 具体表现 | 影响 |
|---|---|---|
| **Isaac Sim / Isaac Lab 版本依赖** | 仿真平台建立在 NVIDIA Isaac Sim + Isaac Lab 之上，二者迭代较快，YAML 配置驱动的 task spec 与特定版本的 asset schema / physics API 绑定 | 版本升级可能导致已发布任务 YAML 不兼容，复现旧榜单结果需要锁定平台版本 |
| **异构并行的 GPU 显存/负载均衡** | 每个环境资产复杂度不同；论文说明按 seeds 做多 GPU sharding，但没有给出动态负载均衡算法 | “重”场景可能决定显存峰值或拖慢一组环境；场景复杂度感知调度是合理工程方向，但属于推断，不是论文已验证方案 |
| **Meshy AI 资产的物理保真度** | 用 Meshy AI 从图像/视频重建的真实物体网格，几何细节和真实碰撞体（尤其薄壁、镂空、可变形部分）之间可能有偏差 | 抓取 affordance 标注依赖重建质量，若碰撞体粗糙会导致仿真中"能抓住"但真实世界"抓不住"（或反之），影响 sim 侧 Precision 维度结果的可信度 |
| **真机硬件成本与复现门槛** | 需要 3 套本体（ARX X5 / Piper / Piper X）+ 标准化夹具 + 双相机规格（Gemini 335L / 305）+ 24 小时 LED 光源等专用硬件 | 独立实验室要严格复现 RoboDojo-RealEval 站点需要非平凡的硬件采购和 3D 打印夹具制作，门槛高于纯仿真 benchmark |
| **hidden layout 一致性维护** | hidden verification layouts 需要长期保密且跨站点/跨时间保持物理布局可复现（依赖 transparent overlay 复位精度） | 若复位存在系统性视觉对齐误差，会引入"hidden 检查本身不稳定"的噪声，削弱防作弊机制的判别力；且 hidden 集一旦泄漏需要重新构建，维护成本持续存在 |
| **π0.5 推理与仿真吞吐的耦合** | 论文效率数据显示接入真实策略推理后加速比从 1.94× 降到 1.63×，推理服务位于同一局域网的 A800 上 | 复现时不能混用 zero-action 和真实策略口径；网络、推理服务与 8×RTX 4090 仿真共同决定端到端吞吐 |

---

## 小结

RoboDojo 的方法论价值不在于新模型，而在于同时处理三个问题：**仿真任务多样性与效率**、**真机评测可复现性**、**策略生态集成成本**。异构并行提供了明确的系统测量（1.94×/1.63×）和场景多样性机制；RoboDojo-RealEval、XPolicyLab 与榜单治理则负责把结果变成可比较、可复查的评测协议。复现难点集中在 Isaac Sim/MagicSim 版本与资产、异构场景的资源调度、三套真机平台的搭建维护，以及 hidden layouts 的长期保密和一致性验证。
