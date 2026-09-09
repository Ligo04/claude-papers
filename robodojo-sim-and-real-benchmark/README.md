# RoboDojo: A Unified Sim-and-Real Benchmark for Generalist Robot Manipulation Policies

## 论文是什么

RoboDojo 是一个**统一 sim-and-real** 的通用机器人操作策略评测基准。它在仿真侧提供 **42 个任务、覆盖 5 大能力维度**（泛化 Generalization、记忆 Memory、长程 Long-Horizon、精度 Precision、开放语义 Open）的高密度诊断集，在真机侧提供 **18 个任务、覆盖 3 种双臂本体**（ARX X5、Piper、Piper X）的可复现远程评测平台 **RoboDojo-RealEval**，并配套统一策略集成框架 **XPolicyLab** 把二者打通——同一策略实现可以先在仿真快速迭代，再最小改动部署到远程真机评测。论文冻结版本接入 **30 个策略**并建立公开榜单，由非营利组织 AI MMLab Club 与全球学术机构伙伴共同治理，含 hidden-layout 防作弊机制。

> 当前资料对应 **arXiv 2607.04434v3（2026-07-08）**。目录里原有的 v1 材料已升级；版本差异、源码快照和证据边界分别记录在 `FACTS.md`、`implementation-map.md` 与 `source-status.md`。

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
| `FACTS.md` | v3 可核查事实、表格数字与证据边界 |
| `implementation-map.md` | 论文概念到 RoboDojo/XPolicyLab 官方源码的映射 |
| `source-status.md` | PDF、arXiv、仓库快照、许可证和验证状态 |
| `code/` | 相关代码片段或复现笔记 |
| `index.html` | 可视化学习页面 |
| `images/` | 论文图表与配图 |

## 关键要点（Key Takeaways）

- RoboDojo 用"仿真侧能力维度诊断 + 真机侧部署可靠性验证"的**二维评测哲学**，取代过去 sim 和 real 割裂的评测方式，两者互补而非要求一一对齐迁移。
- 仿真基准设计聚焦此前基准忽视的挑战：更密集的场景随机化（最多 **25 个杂物** vs RoboTwin 2.0 的 10 个）、部分可观测的记忆任务、需维持进度的长程任务、以及考验重组与迁移能力而非模仿的开放语义任务。
- **异构并行仿真（Heterogeneous Parallelism）**是论文中有直接吞吐 A/B 的核心系统贡献：多环境共享一个 vectorized 接口步进，但每个环境独立采样场景配置，zero-action rollout 下为 **1.94×**（77.4 vs 40.0 interactions/s）。
- v3 明确说明仿真栈建立在 **MagicSim 的 manager-based runtime 和物理对象抽象**之上，再连接 Isaac Lab 向量化接口；官方源码中的关键抓手是 `replicate_physics=False`、per-env seed 与 Scene/Robot/Camera manager。
- 顶尖策略（Hy-Embodied-0.5-VLA）在仿真总榜的 average score 仅 **13.07**（success rate 8.80%），而 Human Expert 参考达到 **80.42/76.03%**，揭示通用操作策略与人类水平之间的巨大差距。
- **精度（Precision）**和**开放语义（Open）**是两大瓶颈维度：Open 维度几乎全面崩溃（榜首 π0.5 仅 1.98/1.67%），说明当前策略难以做开放语义 grounding 和技能重组。
- 真机榜单中 π0.5 以 22.9/12.8% 领先，但在 Piper X 本体上普遍表现极差（多数策略接近 0），说明策略对不同运动学/工作空间配置的鲁棒性仍严重不足。
- 基准配套 hidden verification layouts、double-blind 真机打分、申诉机制等工程化治理手段，为"评测诚信"提供了可复用的方法论范式。

## 预计学习时间

约 **3–4 小时**（含通读论文、消化榜单数据、理解异构并行系统设计和源码映射）。

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
├── FACTS.md           # v3 事实底稿
├── implementation-map.md # 论文到官方源码的映射
├── source-status.md   # 来源与验证边界
├── code/              # 代码片段/复现笔记
├── index.html         # 可视化学习页面
├── images/            # 论文图表
├── meta.json          # 元数据
├── paper.pdf          # v3 论文原文
└── archive/v1/        # 原 v1 PDF 与图片归档
```

## 论文链接

- arXiv: [2607.04434v3](https://arxiv.org/abs/2607.04434v3)（v1 2026-07-05；v3 2026-07-08，cs.RO）
- 项目网站: [robodojo-benchmark.com](https://robodojo-benchmark.com)
- 官方 RoboDojo: [RoboDojo-Benchmark/RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
- 官方 XPolicyLab: [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)
