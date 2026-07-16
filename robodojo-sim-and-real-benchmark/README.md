# RoboDojo: A Unified Sim-and-Real Benchmark for Generalist Robot Manipulation Policies

## 论文是什么

RoboDojo 是首个**统一 sim-and-real** 的通用机器人操作策略评测基准。它在仿真侧提供 **42 个任务、覆盖 5 大能力维度**（泛化 Generalization、记忆 Memory、长程 Long-Horizon、精度 Precision、开放语义 Open）的高密度诊断集，在真机侧提供 **18 个任务、覆盖 3 种双臂本体**（ARX X5、Piper、Piper X）的可复现远程评测平台 **RoboDojo-RealEval**，并配套统一策略集成框架 **XPolicyLab** 把二者打通——同一策略实现可以先在仿真快速迭代，再最小改动部署到远程真机评测。目前已接入 **30 个策略**并建立公开榜单，由公益组织 AI MMLab Club 维护，含 hidden-layout 防作弊机制。

## 难度等级

**Intermediate**（中级）—— 本文是系统设计 + 大规模实证型 benchmark 论文，不涉及艰深的新算法推导，但要理解其价值需要熟悉：机器人操作策略（VLA/模仿学习）的评测范式、Isaac Sim/Isaac Lab 仿真基础、以及"评测基准设计"本身的方法论权衡（任务覆盖度、防刷榜、sim-real gap）。

## 学习材料导航

本文件夹包含以下学习材料：

| 文件/目录 | 内容 |
|---|---|
| `README.md` | 本文件，总览与导航 |
| `summary.md` | 背景、问题陈述、核心贡献与关键结果 |
| `insights.md` | 深层洞察与批判性思考 |
| `method.md` | 方法细节：仿真平台、异构并行、RoboDojo-RealEval、XPolicyLab |
| `mental-model.md` | 心智模型与类比，帮助快速建立直觉 |
| `qa.md` | 常见问题与解答（Q&A 形式） |
| `reflection.md` | 学习反思与延伸思考 |
| `code/` | 相关代码片段或复现笔记 |
| `index.html` | 可视化学习页面 |
| `images/` | 论文图表与配图 |

## 关键要点（Key Takeaways）

- RoboDojo 用"仿真侧能力维度诊断 + 真机侧部署可靠性验证"的**二维评测哲学**，取代过去 sim 和 real 割裂的评测方式，两者互补而非要求一一对齐迁移。
- 仿真基准设计聚焦此前基准忽视的挑战：更密集的场景随机化（最多 **25 个杂物** vs RoboTwin 2.0 的 10 个）、部分可观测的记忆任务、需维持进度的长程任务、以及考验重组与迁移能力而非模仿的开放语义任务。
- **异构并行仿真（Heterogeneous Parallelism）**是本论文最硬的系统贡献：多环境共享一个 vectorized 接口步进，但每个环境独立采样场景配置，零动作 rollout 场景下带来 **1.94× 加速**（77.4 vs 40.0 interactions/s）。
- 顶尖策略（Hy-Embodied-0.5-VLA）在仿真总榜的 average score 仅 **13.07**（success rate 8.80%），而 Human Expert 参考达到 **80.42/76.03%**，揭示通用操作策略与人类水平之间的巨大差距。
- **精度（Precision）**和**开放语义（Open）**是两大瓶颈维度：Open 维度几乎全面崩溃（榜首 π0.5 仅 1.98/1.67%），说明当前策略难以做开放语义 grounding 和技能重组。
- 真机榜单中 π0.5 以 22.9/12.8% 领先，但在 Piper X 本体上普遍表现极差（多数策略接近 0），说明策略对不同运动学/工作空间配置的鲁棒性仍严重不足。
- 基准配套 hidden verification layouts、double-blind 真机打分、申诉机制等工程化治理手段，为"评测诚信"提供了可复用的方法论范式。

## 预计学习时间

约 **2.5–3.5 小时**（含通读论文、消化榜单数据、理解异构并行系统设计）。

## 文件夹结构概览

```
robodojo-sim-and-real-benchmark/
├── README.md          # 本文件：总览与导航
├── summary.md         # 背景/问题/贡献/关键结果
├── insights.md        # 深层洞察
├── method.md          # 方法细节
├── mental-model.md    # 心智模型
├── qa.md              # Q&A
├── reflection.md      # 学习反思
├── code/              # 代码片段/复现笔记
├── index.html         # 可视化学习页面
├── images/            # 论文图表
├── meta.json          # 元数据
└── paper.pdf          # 论文原文
```

## 论文链接

- arXiv: [2607.04434v1](https://arxiv.org/abs/2607.04434) (2026-07-05, cs.RO)
- 项目网站: [robodojo-benchmark.com](https://robodojo-benchmark.com)
- 关联页面: XPolicyLab.github.io
