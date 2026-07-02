# AGIPC: Adaptive In-Solve Algebraic Coarsening for GPU IPC

**作者:** Xuan Wang, Zhaofeng Luo, Minchen Li, Taku Komura, Kemeng Huang  
**发表:** SIGGRAPH 2026  
**代码:** [GitHub](https://github.com/KemengHuang/Adaptive-GIPC)

## 论文简介

AGIPC 提出了一种 GPU 友好的代数自适应粗化方法，在隐式时间积分的 Newton 求解过程中动态减少自由度（DoF），无需显式修改网格拓扑。核心思想是将自适应性表达为选择性的边坍缩过程——通过 Green 应变增量准则标记边为"可坍缩"或"受保护"，然后用并行哈希聚合将精细节点合并为粗"超级节点"。粗化后的线性系统通过 Galerkin 投影代数组装，求解后再延拓回精细网格。整个流程与 IPC 的 barrier 能量无缝集成，端到端利用 GPU 并行性，在各种场景下实现了相比 StiffGIPC 高达 3 倍的加速。

## 难度等级

**进阶（Advanced）** — 需要有限元、隐式积分、IPC 接触力学和 GPU 并行计算的背景知识。

## 学习路线

| 文件 | 内容 | 建议时间 |
|------|------|----------|
| [summary.md](summary.md) | 背景、问题、贡献、结果 | 10 分钟 |
| [insights.md](insights.md) | 核心洞察与概念转变 | 15 分钟 |
| [method.md](method.md) | 算法流程与技术细节 | 25 分钟 |
| [mental-model.md](mental-model.md) | 心智模型与知识图谱 | 10 分钟 |
| [qa.md](qa.md) | 15 道自测题 | 20 分钟 |
| [reflection.md](reflection.md) | 扩展方向与局限性 | 10 分钟 |
| [code/](code/) | 代码演示 | 15 分钟 |
| [index.html](index.html) | 交互式探索器 | 自由探索 |

**预计总学习时间：约 1.5-2 小时**

## 关键收获

1. **代数粗化 vs 显式重网格化**：保持拓扑不变，仅在线性系统层面做 DoF 缩减，保证 IPC barrier 的 C² 连续性
2. **Green 应变增量准则**：每个 Newton 迭代动态决定哪些边需要保护，自然追踪弹性波传播
3. **仿射嵌入**：对大聚合体用 12-DoF 仿射变换代替 3-DoF 平移，保持旋转运动
4. **后粗化修正**：少量 CG 迭代恢复高频细节，类似多重网格的 smoother

## 目录结构

```
agipc-algebraic-coarsening-gpu-ipc/
├── README.md              ← 你在这里
├── paper.pdf              ← 原始论文
├── meta.json              ← 元数据
├── summary.md             ← 摘要与贡献
├── insights.md            ← 核心洞察
├── method.md              ← 方法详解
├── mental-model.md        ← 心智模型
├── qa.md                  ← 问答自测
├── reflection.md          ← 反思与扩展
├── index.html             ← 交互式探索器
├── code/                  ← 代码演示
│   └── algebraic_coarsening_demo.py
└── images/                ← 论文图片
```
