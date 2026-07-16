# 论文总结 · ABC: Scalable Behavior Cloning with Open Data, Training, and Evaluation

## 背景脉络（Background）

行为克隆（Behavior Cloning, BC）已经是机器人操作学习的主流范式，但通往一个可靠 BC 系统的技术栈其实很深：硬件搭建 → 大规模遥操作数据采集 → 支持异构数据流的高效训练基础设施 → 有效的学习配方 → DAgger 式在线微调 → 昂贵的真机评测。

问题在于，当前 SOTA 系统大多在工业实验室的**私有数据**上开发，论文公开的训练配方细节往往不足以复现，社区因此**无法判断哪些设计决策真正驱动了性能提升**——前沿进展变得不透明。与此同时，现有的开源数据集要么规模不足、要么依赖昂贵的定制硬件、要么缺乏配套的训练基础设施与评测套件，研究者难以在同一条起跑线上开展公平比较。

ABC 的目标就是打破这一局面：提供一套**可复现、可扩展的完整操作栈**（数据 + 模型 + 仿真 + 评测），让社区能够共同学习和验证 BC 领域的"ABC"（基本功）。

## 问题陈述（Problem）

- BC 技术栈跨度大、环节多，任何一环的私有化都会让外部研究者无法复现关键结论。
- 现有开源遥操作数据集（DROID、BridgeData-V2、RH20T、Open X-Embodiment 等）在规模、硬件成本、任务多样性（尤其是精细灵巧任务）上难以同时满足。
- 架构设计（DiT vs VLA、connector 选择、视觉编码器选择）缺乏在统一真机 benchmark 下的系统性 ablation，社区对"哪些设计决策真正重要"缺乏共识。
- 真机评测昂贵，如何用低成本的离线指标或仿真作为可靠代理来加速迭代，尚无定论。

## 主要贡献（Contributions）

1. **ABC-130K**：迄今最大的开源双臂遥操作数据集——**3,553 小时 / 134,806 episodes / 195 任务**，采集自成本仅 **8,000 美元**的双臂 YAM station，涵盖 pick-and-place、fine pick-and-place、insertion/ejection、folding、sorting、tool use、tying/untying 七大 primitive 类别，并含 1,552 小时子集的 subgoal 标注。
2. **ABC-Models**：DiT（ABC-DiT）与 VLA（ABC-VLA）两族策略，通过统一的真机 ablation 系统研究架构设计（视觉编码器、conditioning 接口）与算力缩放（batch size × 训练步数）对性能的影响。
3. **ABC-Sim**：MuJoCo 仿真环境 + Blender path-tracing 高保真重渲染管线，配套 10 个仿真任务与 400+ 小时 VR 仿真遥操作数据，用于研究并验证 sim-to-real 相关性。
4. **ABC-Eval**：大规模真机评测套件（>100 小时真机 rollout + 评分 rubric），随论文一并公开评测分数，为社区提供统一评测基准。
5. **全开源可复现**：硬件设计、采集/训练/部署代码、仿真栈全部公开，任何研究者都可以从零复现整条 BC 流水线。

## 方法概览（Approach in brief）

> 详细拆解见 [`method.md`](./method.md)。

- **统一输入/输出接口**：ABC-DiT 与 ABC-VLA 均以 3 张 224×224 图像（顶部相机 + 两个夹爪相机）+ 14D 本体感受关节状态为输入，输出 30 步 chunk 的绝对关节位置目标，执行频率 30 Hz，异步执行带 action prefix 以防 mode switching。
- **ABC-DiT 三变体 ablation**：CLIP-AdaLN（baseline）、CLIP-Cross-Attention、DINO-Cross-Attention，全部使用 32 层 / 24 头 / hidden 1536 的统一容量，隔离视觉表示与 conditioning 接口两个变量。
- **ABC-VLA 三种 connector**：Cross Attention、Cross Attention + FAST、AdaLN（改自 LBM-2），固定 Gemma 3 4B 骨干与 42M 参数 DiT action head，只变 VLM 特征注入 DiT 的接口方式；并引入"多次采样"方差缩减 trick，摊销单次 VLM forward 到多组 (噪声, 时间步)。
- **算力缩放实验**：在 1,536 / 4,608 / 9,216 三档 global batch 与 100K / 200K 训练步下比较 DiT 与 VLA 的扩展曲线。
- **离线指标有效性分析**：比较训练 loss、验证 loss、验证 action error 与真机 strict/progress 的相关性。
- **ABC-Sim sim-real 相关性实验**：12 个 checkpoint（ABC-DiT/ABC-VLA + 内部数据/仿真数据混合）在 3 个任务各 50 trials 下同时评测仿真与真机表现。
- **DAgger 恢复行为配方**：针对折箱子这类长程精细任务，先用严格 SOP 采集微调数据，再采集"卡住时人接管、记录整段 rollout"的 DAgger 干预数据，按 80:10:10（上一轮数据 : 当前轮干预 : 当前轮其余）比例持续微调。

## 关键结果（Key Results）

- **数据规模**：ABC-130K 达到 **3,553 小时 / 134,806 episodes / 195 任务**，采集平台为 **8,000 美元**的双臂 YAM station，规模与成本组合优于同类开源数据集（DROID 350h 单臂 Franka、AgiBot-World ~3,000h 但硬件 $30,000、MolmoAct2-BimanualYAM 720h/28 任务）。
- **架构 ablation（Table 1，真机评测，200k steps，内部 7,000h 语料，3 任务 Bottles/Dishrack/Mugs 各 50 trials）**：最优 DiT 变体 **DiT DINOv3-xattn** 达到 Mean progress **67.5%**（Mean strict 32.9%）；最优 VLA 变体 **VLA Pooled AdaLN** 达到 Mean progress **61.4%**（Mean strict 32.8%），显著优于 VLA FAST+X-Attn（32.6%）与 VLA X-Attn（11.7%）。
- **sim-real 相关性**：strict success 的 Pearson 相关系数 **r=0.85**（p=4.2e-4, n=12），task progress 的 Pearson 相关系数 **r=0.91**（p=5.0e-5, n=12），证明仿真是真机表现的有意义预测器。
- **离线指标有效性**：验证 action error（固定 10 diffusion steps）与真机 strict success 相关性最强，**r=-0.89**（p=3.8e-6），训练 loss 次之 **r=-0.84**；而验证 loss 与真机表现几乎不相关（**r=-0.04**, p=0.886），不能作为模型选择依据。

## 定量指标表格

### 数据规模对比

| 数据集 | 规模 | 硬件/本体 | 成本 |
|---|---|---|---|
| **ABC-130K**（本文） | 3,553h / 134,806 episodes / 195 任务 | 双臂 YAM station | **$8,000** |
| DROID | 350h | 单臂 Franka | — |
| BridgeData-V2 | ~100h | 单臂 WidowX | — |
| RH20T | 110k 轨迹 / 41 技能 | 5 个单臂平台 | — |
| AgiBot-World | ~3,000h（偏长程宏操作） | 双臂 AgiBot G1 | $30,000 |
| MolmoAct2-BimanualYAM | 720h / 28 任务 | 双臂 YAM | — |

### Table 1：架构 ablation（真机评测，200k steps，内部 7,000h 语料）

| 模型 | Mean strict | Mean progress | Latency (ms/chunk) |
|---|---:|---:|---:|
| VLA Pooled AdaLN | 32.8% | 61.4% | 17.24 |
| VLA FAST+X-Attn | 3.6% | 32.6% | 19.24 |
| VLA X-Attn | 0.0% | 11.7% | 19.24 |
| **DiT DINOv3-xattn**（最优） | 32.9% | **67.5%** | 37.4 |
| DiT CLIP-AdaLN | 13.4% | 47.3% | 27.5 |
| DiT CLIP-xattn | 24.5% | 58.8% | 37.5 |

### Sim-real 相关性系数

| 指标 | Pearson r | p 值 | n |
|---|---:|---:|---:|
| Strict success | 0.85 | 4.2e-4 | 12 |
| Task progress | 0.91 | 5.0e-5 | 12 |

### 离线指标 vs 真机 strict success 相关性

| 离线指标 | Pearson r | 说明 |
|---|---:|---|
| 验证 action error（固定步数） | **-0.89**（p=3.8e-6, ρ=-0.93） | 相关最强，可作代理指标 |
| 训练 loss | -0.84 | 次强 |
| 验证 loss | -0.04（p=0.886） | 不显著，不能用于模型选择 |

## 一句话总结

**ABC 用 8,000 美元的双臂平台采出规模最大的开源遥操作数据集 ABC-130K，通过统一真机 ablation 证明 connector/视觉表示设计而非骨干规模决定 DiT/VLA 性能，并用 r=0.85~0.91 的 sim-real 强相关把昂贵的真机评测变成社区可复现的迭代闭环，最终以 DAgger 恢复行为配方把灵巧长程任务真正落到真机上。**
