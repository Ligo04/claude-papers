# 参数曲面连续碰撞检测中的浮点鲁棒性

> Xuwen Chen, Junyu Wang, Cheng Yu, Xingyu Ni, Meng Zhang, Bin Wang, Mengyu Chu, Baoquan Chen
> ACM Transactions on Graphics (SIGGRAPH 2026), Vol. 45, No. 4, Article 141 · DOI [10.1145/3811310](https://doi.org/10.1145/3811310)

## 这篇论文讲什么？

**连续碰撞检测（Continuous Collision Detection, CCD）** 要在一个时间步内找出两个运动物体**最早**接触的
时刻和位置。它是物理仿真与建模中防止物体相互穿透的核心组件。对**三角网格**而言，可证明安全的 CCD 已经成熟；
但对**光滑参数曲面**（Bézier / NURBS 曲面片）却一直没有浮点鲁棒的方法。本文给出两个"首次"：(1) 一个对参数
曲面**可证明无漏报（false negative）** 的 CCD 算法——基于 TDIBM（时间相关包含法），通过一种全新的
**"系数误差 vs. 算术误差"分解**实现；(2) 一个含 **24,246 个有理算术精确真值案例**的基准数据集，其构造方式
是**反演 CCD 流程**：从规定的碰撞结果出发，反向求解出能产生该碰撞的曲面配置。

## 难度

**高级 / 高度理论化。** 核心贡献是一连串针对分支密集型几何算法的浮点误差界推导。需要熟悉 IEEE-754 舍入、
区间/包含法、Bézier 曲面、以及凸包（包络）构造。

## 如何浏览这些材料

| 文件 | 用途 | 阅读顺序 |
|------|------|---------|
| `summary.md` | 背景、问题、贡献、关键数据 | 最先 |
| `insights.md` | **"为什么有效"——从这里开始建立理解** | 核心 |
| `method.md` | 算法 + 误差界数学，逐步拆解 | 深入 |
| `mental-model.md` | 如何在脑中给这项工作归类；前置知识 | 定位 |
| `qa.md` | 15 道自测题（基础 → 进阶） | 检验自己 |
| `reflection.md` | 扩展方向、脆弱假设、失效模式 | 之后 |
| `code/` | 核心思想的可运行 Demo | 上手 |
| `index.html` | 交互式探索器（包络、误差偏移、结果） | 把玩 |
| `images/` | 抽取的图与页面预览 | 参考 |

## 关键要点

1. **难点不是误差的大小，而是误差破坏了"拓扑"。** 一次被翻转的浮点比较不仅仅是把一个数值轻微挪动——它能破坏
   分段线性包络的**组合结构**（保留/丢弃了错误的线段），产生几何上无效的结果。经典的"累积误差界 + 放宽阈值"
   推理在这里**失效**。
2. **分而治之。** 把包络总误差拆成 `ε_coeff`（系数不精确、算术精确）+ `ε_arith`（系数精确、算术不精确），
   各自单独定界，从而绕开难以处理的联合分析。
3. **始终朝安全方向保守。** 每个谓词 `θ < 0` 改为 `θ < −ε_θ`；把两条包络互相推开（`ε_constr`、`δ`）；
   对结果做截断。所有偏置都倾向于**误报（false positive，无害的多余碰撞）**，而非**漏报（false negative，
   = 穿透）**。
4. **反演问题以获得真值。** 正向计算 CCD 真值需要符号求解器（Mathematica 对非孤立接触每个案例 >1 天）。
   于是反过来：**规定**答案，再用精确有理算术反向求解——并用"二进分数（dyadic）性质"技巧保证最终的 double
   无损可表示。
5. **结果：** TDIBM-E 在全部六类接触场景下**消除了所有漏报**，相对启发式仅 ~40%、相对原始 TDIBM ≤120% 的
   开销——而启发式（TDIBM-H）只是在一个手调常数下"恰好"有效，并无任何保证。

## 预计学习时间

- 速览（summary + insights）：**45–60 分钟**
- 工作级理解（+ method + code）：**3–4 小时**
- 深度掌握（+ 复现误差推导、运行 Demo）：**1–2 天**

## 目录结构

```
floating-point-robust-parametric-ccd/
├── README.md            ← 你在这里
├── summary.md
├── insights.md          ← 最重要的概念文件
├── method.md
├── mental-model.md
├── qa.md
├── reflection.md
├── meta.json
├── paper.pdf
├── index.html           ← 交互式探索器
├── code/
│   ├── envelope_robustness_demo.py
│   ├── inverse_construction_demo.py
│   └── floating_point_error_bound_demo.py
└── images/
```
