# MAS-PNCG：高效多层级预条件非线性共轭梯度法用于增量势能接触

## 论文概述

本文提出 MAS-PNCG，一种面向 IPC（Incremental Potential Contact）仿真的高效非线性共轭梯度求解器。核心思想是将多层级加性 Schwarz（MAS）预条件器引入非线性 CG 框架，通过 Sparse-Input Woodbury 增量更新策略避免每次迭代重建预条件器，同时用 2D 子空间最小化替代启发式搜索方向，并引入逐子域保守 CCD。相比 GIPC 和 StiffGIPC，分别实现最高 5.66× 和 2.07× 加速。

## 难度级别

**高级** — 需要数值优化、域分解预条件、IPC barrier 能量、Woodbury 矩阵公式等背景知识。

## 学习路线

1. `summary.md` — 快速掌握贡献和结果
2. `insights.md` — 理解核心思想和设计取舍
3. `method.md` — 算法细节、伪代码、实现陷阱
4. `mental-model.md` — 定位论文在研究图谱中的位置
5. `qa.md` — 15 道分级问题自测理解
6. `reflection.md` — 开放问题与扩展方向
7. `code/` — 可运行代码演示
8. `index.html` — 交互式论文探索器

## 核心要点

- **PNCG 避免 Hessian 组装**但传统预条件器（Jacobi）在刚性/接触密集场景收敛极慢
- **MAS 预条件器**分层捕获全局耦合，但在非线性 CG 中每迭代重建代价过高
- **Sparse-Input Woodbury 更新**利用接触 barrier Hessian 的稀疏低秩结构增量维护预条件器
- **2D 子空间最小化**用近似 Hessian 信息替代启发式 β 公式，直接求解最优搜索方向
- **逐子域保守 CCD**避免全局步长限制，不同区域按局部需求推进

## 预计学习时间

- 快速浏览：30 分钟（summary + insights）
- 深入理解：2-3 小时（含 method + qa）
- 完整学习：4-5 小时（含代码实现 + 交互探索）

## 文件结构

```
mas-pncg-multilevel-preconditioned-ncg/
├── README.md          ← 本文件
├── paper.pdf          ← 原始论文
├── meta.json          ← 元数据
├── summary.md         ← 背景、贡献、结果
├── insights.md        ← 核心洞察与取舍
├── method.md          ← 算法详解与伪代码
├── mental-model.md    ← 心智模型与知识定位
├── qa.md              ← 15 道分级问答
├── reflection.md      ← 反思与扩展方向
├── code/              ← 代码演示
├── images/            ← 论文图片
└── index.html         ← 交互式探索器
```
