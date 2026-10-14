# S4R 论文摘要与实证结果

**论文**：*S4R: Scaling for Rigid-Body Interpenetration Resolution*，Zhiyang Dou、Ang Zhao 等，arXiv:2609.20524v1，2026 年 9 月 17 日。本文属于高级难度的几何优化与系统实证研究；数学核心是局部接触线性化、凸二次规划与数值延拓，实践重点是 mesh 查询、状态验证和失败处理。

本文的数字来自 [论文 v1](https://arxiv.org/pdf/2609.20524)，页码按 PDF 的 1–29 页计；代码信息来自 [Dynamics-Modeling](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving)。这里是文献阅读与代码接口核对，**没有在本机重跑完整论文基准**。

## 问题：有效的物体，不一定构成有效的场景

单个刚体 mesh 可以完全有效，但程序化放置、生成式场景布局或随机初始化会让多个物体互相穿透。深度重叠会破坏 IPC 一类方法对初始无交叉状态的要求，也可能让基于冲量或罚力的引擎产生很大的修正运动。S4R 要消除互穿，同时尽量保留原布局；最终物体必须恢复原尺寸。它不模拟重力、摩擦或惯性，是进入物理仿真之前的**静态布局修复阶段**。[论文 §1、§3、§4.7，pp.1–5、14](https://arxiv.org/pdf/2609.20524#page=14)

直接在深穿透状态建立接触线性化容易失败：最近特征可能不唯一，接触方向会突变；即使距离函数在局部分支上光滑，一次大位移也会放大被忽略的二阶项。S4R 因而改变求解的起点与过程。

## 核心方法

1. 每个物体围绕固定参考中心缩小至 `s_min = 0.01`，处理过近参考中心，并检查小尺度状态。
2. 沿共同尺度参数 (s\in(0,1]) 恢复尺寸。每个尺度增量用最小范数凸接触 QP 求平移修正；默认固定方向。
3. 用包围球推导固定姿态膨胀的保守接触事件尺度；用冻结 witness/normal 的 gap 预测跨过部分检测步骤。
4. 在 (s=1) 执行精确评估与有预算的 tail refinement，修正遗漏或过时的接触。

这里每次最小化的是增量 (\sum_i\|\Delta p_i\|^2)，不是最终位姿相对初始位姿的全局最小偏差。低累计位移是实验观察，不能从逐步最小范数直接推出。[论文 §3.2–3.5，pp.5–9](https://arxiv.org/pdf/2609.20524#page=5)

## 评测协议决定了数字的含义

| 项目 | 论文协议 |
|---|---|
| 主数据 | Kubric 的 40 个 watertight 家居 mesh；HY3D-Bench 过滤后的 13 个 decimated / 12 个 full-resolution 模板；Thingi10K 的 430 个 mesh |
| 场景规模 | 主表 (N=40) 至 5000；图 5 扩展至 30,000 |
| 随机性 | 默认 3 seeds：42、123、456；高凹度实验 5 seeds，packing 实验 8 seeds |
| 密度控制 | 主实验 spawn box 按 ((N/40)^{1/3}) 扩张，使接触密度近似不变 |
| 归一化 | 每个 mesh 最长包围盒边归一化至 0.1 world units；RMSD、穿透深度和 margin 都在这些单位下 |
| Pen. | 统一 evaluator 中负分数 body pair 数量，阈值严格为 (\tau=0) |
| RMSD | 参考中心相对初始位置的 RMS 位移；含初始扰开和 tail 修正 |
| 时间 | `setup + solve`，包含本方法的 BVH/oracle 构建、GPU runtime/JIT 等；排除场景生成、mesh 加载和共享最终评估 |
| 硬件 | CPU：Xeon E5-2680 v4 单 socket、14 物理核；GPU：单张 RTX 2080 Ti，另配同样 host 核数 |
| 比较边界 | 运行时间排名在同一硬件 tier 内进行；部分方法使用 OBB、convex hull 或自己的检测后端，但最终姿态统一按 mesh 评分 |

论文的 FCL score 是“正 boundary distance / 负最大局部 contact depth”的拼接量，**不是全局光滑 signed distance，也不是全局最小平移穿透深度**。在线评估没有 containment 分支，watertight 保留结果另做离线 containment audit。空 contact 集按 touching、score 0 处理。RMSD 对还会旋转的 AVBD 和 ISIR 可能低估表面总运动。[论文 §4.1、App.A.7–A.9，pp.9–10、22–24](https://arxiv.org/pdf/2609.20524#page=24)

## 主结果：速度优势明显，但不是位移最小

下表是论文 Table 2 的 3-seed means，单位为秒；除 ISIR 的 (N=5000) 外，所有单元都报告 Pen.=0。

| (N) | S4R CPU RMSD / 时间 | QP/LCP RMSD / 时间 | PD-PGS RMSD / 时间 | S4R-Warp RMSD / 时间 | ISIR RMSD / 时间 |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.036 / 0.2 | 0.038 / 0.5 | 0.028 / 1.1 | 0.035 / 1.5 | 0.175 / 14 |
| 100 | 0.035 / 0.4 | 0.039 / 1.7 | 0.027 / 3.4 | 0.035 / 1.6 | 0.175 / 14 |
| 200 | 0.035 / 1.0 | 0.039 / 3.6 | 0.026 / 9.2 | 0.034 / 1.7 | 0.241 / 51 |
| 500 | 0.034 / 3.0 | 0.040 / 11.7 | 0.024 / 36.5 | 0.033 / 2.1 | 0.298 / 135 |
| 1000 | 0.035 / 7.7 | 0.040 / 29.8 | 0.025 / 69.2 | 0.034 / 2.9 | 0.356 / 427 |
| 2000 | 0.037 / 17.5 | 0.042 / 79.8 | 0.025 / 150 | 0.036 / 4.6 | 0.394 / 1305 |
| 5000 | 0.037 / 68.1 | 0.042 / 220 | 0.025 / 404 | 0.036 / 9.7 | 0.445 / 1775，平均 39 对残留 |

S4R CPU 在此 125 倍场景规模跨度内的 RMSD 维持 0.034–0.037；S4R-Warp 为 0.033–0.036。(N=5000) 时，S4R CPU 比 QP/LCP 快约 3.2 倍、比 PD-PGS 快约 5.9 倍，但 PD-PGS 的位移更低。论文不声称更低的渐近复杂度：候选对生成仍对所有 (N(N-1)/2) 对执行向量化扫描，最坏为 (\Theta(N^2))。[论文 §4.3、Table 2，pp.11–12](https://arxiv.org/pdf/2609.20524#page=12)

跨数据集 (N=2000) 时，S4R 在 HY3D-Bench 上为 Pen.=0、RMSD 0.021、22.2 s；在 Thingi10K 上为 0、0.026、21.4 s。高凹度 HY3D 测试中，S4R 与 AVBD-OBB 均达 Pen.=0，RMSD 分别为 0.020 与 2.530；时间为 5.8 s 与 3.6 s。后者的 126 倍位移差同时受 OBB proxy 与 mesh 几何差异影响，不应全部归因于优化算法。[论文 Table 3、Table A8，pp.12、26](https://arxiv.org/pdf/2609.20524#page=26)

## 消融揭示真正起作用的部分

| 消融 | 论文观察 | 解读 |
|---|---|---|
| 单次 Direct-QP | (N=40)：0.12 s、RMSD 0.028，但 13.7 对残留；50 次重线性化 0.41 s、Pen.=0；S4R 0.16 s、Pen.=0 | 单步更快不等于完成修复 |
| 关闭 tail | Kubric (N=1000)：85.3 对残留，开 tail 后 0；HY3D-Full (N=2000)：212.3→0；Thingi10K (N=2000)：129.7→0 | tail 是最终可行性的重要组成，不是可随意省略的后处理 |
| cache (M=1\to3) | 0.234→0.157 s，RMSD 0.033→0.036 | 约 1.5 倍速度伴随小幅位移代价 |
| coarse stride + SOI | 固定 `ds=.05`：20 步、0.15 s；SOI：17.7 步、0.16 s | 小场景中主要收益来自 coarse stride，不能把全部提速归给 SOI |
| 稀疏 QP | (N=40)：平均变量 120→56，时间同为 0.16 s | 减少维度，但此时检测才是瓶颈 |
| 6-DOF | Kubric (N=40)：RMSD 0.036→0.031，0.16→2.4 s | 约 13% 位移改善，约 15 倍时间；旋转导致重新查询 |

大型运行中，检测与 tail 占总耗时 73–97%，主循环 QP 不超过 1%；把 OSQP 换成更快 QP 后端未必改变整体时间。上述数字来自 Table 4、A11–A18，pp.12、27–29。[论文消融与时间分解](https://arxiv.org/pdf/2609.20524#page=28)

## 应用证据与失败边界

- **IPC 初始化**：紧密 Kubric 场景原有 236–333 对互穿，深度至 0.073；S4R 后 Pen.=0，最小 gap 为 (6\times10^{-5}) 至 (3\times10^{-4})。论文只检查 barrier 可有限求值，**没有运行 IPC solve**。内部 margin `d_hat=.02` 也不意味着最终每对物体都有 .02 间隙，因为 tail 在零穿透时终止。
- **引擎启动**：50 个 YCB 实例、关闭重力的一秒 rollout，原场景在 MuJoCo / PyBullet / Isaac Gym (PhysX) 的峰值速度为 7.8 / 12.5 / 230.6 m/s；经 S4R 后约为 0。该实验两侧都使用相同 convex hull 碰撞几何。
- **极密集自由空间**：球形 spawn 的 (N=500,1000,2000) 在默认 tail 下平均仍有 1.3、32、192 对残留；改为 200-iteration tail 后清除，时间为 20、64、215 s，RMSD 至多再增加 5.1%。
- **受限空间**：有容器墙约束时，packing fraction 约 .14 的 3 个 seeds 仍成功；约 .18 时全部失败，残留 12–35 对。这是特定实验的经验失败点，不是普遍 packing 上界。
- **其他局限**：近重合参考中心、封闭 cavity 中嵌套物体、接触特征切换、病态 active set、需要大旋转或多支持面的场景。缩放路径不一定能找到端点已存在的可行解。

这些边界见论文 §4.7、§5、App.A.13–A.15，pp.14、16–17、26–27。[论文应用与局限](https://arxiv.org/pdf/2609.20524#page=16)

## 与作者博客和公开代码一起读

[作者博客 *To Scale or Not to Scale*](https://frank-zy-dou.github.io/blog/to-scale-or-not-to-scale/) 将 S4R 与 Coverage Axis/++ 放在一起：后者通过 dilation 改变候选球的覆盖关系和抽象容差；S4R 用缩放改变中间问题，再回到指定尺寸。统一缩放整个 scene 保持交叠关系；S4R 必须相对各物体的参考中心改变大小，才会产生 clearance。最终恢复原尺寸并不会消除 continuation 的历史偏向。

**论文 v1 与代码必须分开记录。** `91cbaf2` 的 README 报告 CPU seed 42、(N=40) 为 RMSD 0.0400、0.35 s，(N=1000) 为 0.0365、15.7 s；GPU 对应为 0.0372、6.7 s 与 0.0346、17.0 s。这些不是论文的 3-seed mean。README 说明 broad phase 已保留 `d_hat` band 内的所有 pair，GPU 增加 full-scale exact FCL 验证与必要修正；当前结果接口也明确区分 continuation、native convergence 和最终 evaluator 状态。这里不以 README 数字替代实测，更不把它与论文表混算速度比。[公开代码 README，固定 commit](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving/README.md)

阅读时持续区分三件事：**几何零穿透、请求的空间关系满足、进入仿真后的稳定与任务成功**。博客的 SimReady Gate 例子正说明，几何谓词全部通过后，MuJoCo settle 仍可能破坏圆形水果的 upright 要求。
