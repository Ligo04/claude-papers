# RoboDojo 方法论拆解

> 论文：*RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies*
> arXiv 2607.04434v1（2026-07-05，cs.RO）
> 本文是 **system / benchmark 论文**，不是提出新算法的论文——它的"方法"就是它的**技术架构**：如何用统一的接口把「仿真评测」和「真机评测」缝合成一个可复现、防作弊、可持续扩展的闭环。以下拆解严格依据事实底稿 `FACTS.md`，不引入未在底稿中出现的数字。

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

### 2.1 配置优先设计（Configuration-First，灵感来自 MagicSim）

核心思想：**任务规约（task spec）与仿真执行（simulation runtime）解耦**。每个任务由一份模块化 **YAML** 描述：

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

同一个 seed → 完全相同的场景（可复现），不同 seed → 场景多样（防止策略死记硬背）。统一接口同时支持刚体（rigid）、铰接（articulated，如抽屉/门）、可变形（deformable，如布料）三类资产，reset 逻辑对三者一致。

### 2.3 物理接地数字资产库（灵感 ManiTwin）

资产来源分三路：

1. **刚体**：在线开源资产库 + 用 **Meshy AI** 对真实日用物做三维重建，重建后统一标注类别 / 语言描述 / 放置区域 / 成功检查逻辑 / 操作 affordance（例如"可抓取区域""可插入孔位"）。
2. **铰接体**：开源库 + 3D 扫描件，统一标准化运动学结构（关节轴、限位）与碰撞几何。
3. **可变形体**（布料等）：来自 **ClothesNet**，同样标准化材料参数。

这一层的目的是让"看起来真实"的资产真正具备"physically-grounded"（有正确碰撞体、有正确关节限位、有正确材料参数），否则策略在仿真里学到的抓取/操作策略无法反映真实物理难度。

### 2.4 异构并行 Heterogeneous Parallelism 【核心系统创新】

这是论文中"最硬的系统贡献"。先对比两种并行范式：

**(a) 同构克隆并行（标准 vectorized 仿真做法）**

```
Env 0: [Scene Template A] pose=p0, seed=s0
Env 1: [Scene Template A] pose=p1, seed=s1
Env 2: [Scene Template A] pose=p2, seed=s2
Env 3: [Scene Template A] pose=p3, seed=s3
...
```
所有并行环境共享**同一个场景模板**，只有物体位姿/随机种子不同——GPU 上同一份 mesh/碰撞体/资产图可以直接广播，效率最高，但场景多样性被严重压缩到"同一模板的变体"。

**(b) 异构并行（RoboDojo 方案）**

```
Env 0: [任务 insert_tubes,  N_clutter=3,  资产集合 A] ← 独立采样的场景配置
Env 1: [任务 classify_objects, N_clutter=18, 资产集合 B]
Env 2: [任务 match_and_pick,  articulated door, 资产集合 C]
Env 3: [任务 fold_cloth,     deformable asset,  资产集合 D]
        │
        ▼ 所有环境仍共享同一个 vectorized 接口 step()
   ┌─────────────────────────────────────────┐
   │   单次 batched physics step (GPU kernel)   │
   │   每个 env 各自的资产/碰撞图/关节结构独立      │
   └─────────────────────────────────────────┘
```

关键点：多个环境**仍然共享一个 vectorized 接口一起步进**（即仍然是批量 GPU 仿真，而不是退化成串行仿真），但**每个环境独立采样场景配置**——不同物体类别、不同几何、不同杂物数量、不同铰接结构、不同任务布局都可以同时存在于同一批次里。工程上需要额外做资产打包/内存布局管理来让异构的碰撞体/关节树在同一个 batched kernel 里高效步进，并支持**多 GPU sharding**（把不同环境分片到不同 GPU 上）来横向扩展吞吐。

为什么能"既提速又保多样性"：
- 同构并行虽然快，但要覆盖 42 任务 × 5 维度的多样性，需要**顺序切换场景模板重新编译/加载**，这个"reload"开销才是同构方案的隐藏成本；
- 异构并行把"多样性"直接编码进单次 batched step 里，避免了频繁 reload，因此在"零动作 rollout"测试中：**异构 77.4 interactions/s vs 非异构 40.0 → 1.94× 加速**；在**接入 π0.5 真实推理**的场景下（推理本身有延迟，测的是端到端流水线）：**异构 64.0 interactions/s，加速比降为 1.63×**（因为推理耗时占比上升，仿真侧的加速收益被摊薄）。

硬件配置：**8×RTX 4090**，每卡 1 进程、每进程 10 个并行环境；π0.5 的推理放在单独的 **A800** 上执行（避免推理和仿真抢卡）。

### 2.5 数据采集双模式

```
采集模式选择:
  if 任务可由低层技能组合表达 (grasp/place/handover/insert/open/close/stack/push_up):
      使用 cuRobo v2 运动规划器自动合成轨迹   # 高吞吐、低标注成本
  else:  # 难以自动分解/需要人类直觉的任务
      使用 VR 遥操作采集                       # 覆盖自动合成的长尾
```

最终训练集规模：**35 个任务目录（34 训练任务 + 1 个 DLC 辅助目录）× 100 条轨迹/任务 = 3,500 条轨迹，共 1,859,602 帧，20.66 小时双臂数据，25 Hz 记录**。注意 **Open 维度不提供训练数据**——这是有意设计，Open 维度专门测试"训练时未见的任务规约、但组成技能在别的上下文中出现过"的重组与迁移能力，而不是让策略靠模仿学会。额外的 **100 条 DLC（Diversity/Data-Level-Clutter类）轨迹**是 train-only 的数据级增强，与评测任务本身独立、不会泄漏解法。

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
    def reset(self, task_config) -> None:
        """策略内部状态复位（如清空历史观测缓存，对应 Memory 维度任务尤其关键）"""

    def update_observation(self, obs: Observation) -> None:
        """接收一帧观测（RGB×3 + robot states + language instruction）"""

    def predict_action(self) -> Action:
        """基于当前内部状态预测下一步动作"""

    def batched_query(self, obs_batch: List[Observation]) -> List[Action]:
        """批量查询接口——支撑仿真侧异构并行的高吞吐 rollout"""
```

### 4.2 "集成一次，处处评测"的落地方式

```
策略开发者:
   实现一次 Policy 接口
        │
        ├── 数据转换工具  → 把策略原生数据格式转换成 RoboDojo 标准格式
        ├── 训练模板      → 复用 XPolicyLab 提供的训练 pipeline 骨架
        ├── 部署流程      → 同一份部署代码可指向仿真 or 真机 endpoint
        └── 评测脚本      → 标准化调用 batched_query / predict_action
        │
        ▼
   在仿真里快速跑 42 任务迭代调参
        │  (最小改动，只切换 endpoint 配置)
        ▼
   部署到 RoboDojo-RealEval 远程真机评测
```

策略内部架构（VLA/RL/规则式规划器等）完全不受约束——XPolicyLab 只标准化"接口边界"，不侵入策略实现本身。目前已集成 **30 个策略**（Hy-Embodied-0.5-VLA、π0.5、X-VLA、GR00T-N1.7 等），是这套接口设计已被验证可行的直接证据。

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
                    policy.reset(task.config)
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
                policy.reset(task.config)
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
不满足 → 单列展示，标注为"非认证 (non-certified)"，不进入官方主榜排序
```

`hidden verification layouts` 的作用：公开 layouts 是主评测集，允许策略方在其上调参；但额外在一批**不公开**的 layouts 上跑一遍，如果两者性能差异异常（例如公开集上手调过拟合、刷榜），就能被检测出来——这是"benchmark 治理"层面的工程化手段，而不是算法层面的创新，但对榜单公信力至关重要。榜单本身由公益组织 **AI MMLab Club** 维护。

---

## 6. 复现风险与实现陷阱

| 风险点 | 具体表现 | 影响 |
|---|---|---|
| **Isaac Sim / Isaac Lab 版本依赖** | 仿真平台建立在 NVIDIA Isaac Sim + Isaac Lab 之上，二者迭代较快，YAML 配置驱动的 task spec 与特定版本的 asset schema / physics API 绑定 | 版本升级可能导致已发布任务 YAML 不兼容，复现旧榜单结果需要锁定平台版本 |
| **异构并行的 GPU 显存/负载均衡** | 每个环境资产不同（不同数量物体、不同铰接/可变形复杂度），batched kernel 内各环境计算量不均衡；多 GPU sharding 需要动态分配避免某些 GPU 因分到"重"场景而拖慢整批 | 显存峰值由最重的场景决定，负载不均会侵蚀 1.94×/1.63× 的加速收益，实际部署需要做场景复杂度感知的调度 |
| **Meshy AI 资产的物理保真度** | 用 Meshy AI 从图像/视频重建的真实物体网格，几何细节和真实碰撞体（尤其薄壁、镂空、可变形部分）之间可能有偏差 | 抓取 affordance 标注依赖重建质量，若碰撞体粗糙会导致仿真中"能抓住"但真实世界"抓不住"（或反之），影响 sim 侧 Precision 维度结果的可信度 |
| **真机硬件成本与复现门槛** | 需要 3 套本体（ARX X5 / Piper / Piper X）+ 标准化夹具 + 双相机规格（Gemini 335L / 305）+ 24 小时 LED 光源等专用硬件 | 独立实验室要严格复现 RoboDojo-RealEval 站点需要非平凡的硬件采购和 3D 打印夹具制作，门槛高于纯仿真 benchmark |
| **hidden layout 一致性维护** | hidden verification layouts 需要长期保密且跨站点/跨时间保持物理布局可复现（依赖 transparent overlay 复位精度） | 若复位存在系统性视觉对齐误差，会引入"hidden 检查本身不稳定"的噪声，削弱防作弊机制的判别力；且 hidden 集一旦泄漏需要重新构建，维护成本持续存在 |
| **π0.5 推理与仿真吞吐的耦合** | 论文效率数据显示接入真实策略推理后加速比从 1.94× 降到 1.63×，推理延迟（A800 上）会摊薄仿真侧异构并行的收益 | 复现效率数字时需要注意"零动作 rollout"和"真实策略 rollout"两种测量口径不可混用，且推理硬件（A800）与仿真硬件（RTX 4090）分离的部署方式也是复现条件之一 |

---

## 小结

RoboDojo 的方法论价值不在于一个新模型，而在于把三个此前互相割裂的问题——**仿真任务多样性 vs 效率**、**真机评测可复现性**、**策略生态集成成本**——分别用一个系统级方案解决，并用统一接口把三者缝合进同一条评测流水线。其中**异构并行**是唯一具有明确算法/工程贡献的部分（1.94×/1.63× 加速 + 场景多样性保留），其余两块（RoboDojo-RealEval、XPolicyLab）更偏工程标准化与治理设计。复现该工作的主要难度也相应集中在异构并行的调度实现、真机硬件的标准化制作，以及 hidden layout 机制的长期维护上。
