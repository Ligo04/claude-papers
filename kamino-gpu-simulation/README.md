# Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies

## 概述

Kamino 是由 Disney Research (苏黎世) 和 NVIDIA 联合开发的 GPU 物理求解器，专门解决带有**运动学闭环 (kinematic loops)** 的复杂机器人系统的大规模并行仿真问题。传统 GPU 仿真器假设树状运动学拓扑，而 Kamino 采用极大坐标 (maximal-coordinate) 表述，将闭环约束与普通关节约束统一处理，并通过 Proximal-ADMM 算法在 GPU 上高效求解。该系统首次在单张 GPU 上以 4096 个并行环境训练了具有 6 个运动学闭环的双足机器人 DR Legs。

## 难度评估

- **难度等级**: 高级 (Advanced)
- **论文性质**: 系统设计 + 算法工程
- **方法复杂度**: 多阶段 — 极大坐标建模 → 约束优化 → GPU 并行化

## 学习材料导航

| 文件 | 内容 | 优先级 |
|------|------|--------|
| `summary.md` | 背景、问题、贡献、关键结果 | ⭐⭐⭐ |
| `insights.md` | 核心思想、设计权衡、与现有工作对比 | ⭐⭐⭐ |
| `method.md` | 算法详解、伪代码、架构分解 | ⭐⭐⭐ |
| `mental-model.md` | 知识定位、前置知识、研究图谱 | ⭐⭐ |
| `qa.md` | 15 道问答题 (基础/中级/高级) | ⭐⭐ |
| `reflection.md` | 扩展思考、开放问题、实践局限 | ⭐ |
| `code/` | 代码演示 | ⭐⭐ |
| `index.html` | 交互式可视化探索器 | ⭐⭐ |

## 关键收获

1. **极大坐标表述**使闭环约束与普通关节约束结构相同，无需特殊处理
2. **Proximal-ADMM** 将前向动力学转化为对偶锥约束优化，自然统一双边约束、关节限位和摩擦接触
3. **异构世界**设计允许在单 GPU 上批量仿真结构不同的机器人
4. 稀疏求解器在约束维度 > 300 时超越稠密求解器
5. DR Legs 是首个在 GPU 仿真器中以原生闭环处理训练 RL 策略的复杂机构

## 预计学习时间

- 快速浏览: 30 分钟 (README + summary + insights)
- 深度学习: 2-3 小时 (所有材料 + 代码)
- 完整掌握: 需要刚体动力学和凸优化背景

## 文件夹结构

```
kamino-gpu-simulation/
├── README.md              # 本文件
├── paper.pdf              # 原始论文
├── meta.json              # 元数据
├── summary.md             # 论文摘要
├── insights.md            # 深度洞察
├── method.md              # 方法详解
├── mental-model.md        # 心智模型
├── qa.md                  # 问答练习
├── reflection.md          # 反思与扩展
├── index.html             # 交互式探索器
├── code/                  # 代码演示
│   └── padmm_demo.py      # Proximal-ADMM 求解器演示
└── images/                # 提取的图片
```
