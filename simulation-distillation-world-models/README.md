# Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation

## 概述

本文提出 SimDist 框架，利用物理仿真器大规模预训练世界模型（包含编码器、动力学模型、奖励模型和价值函数），然后在真实世界部署时冻结除动力学模型外的所有组件，仅用少量真实数据微调动力学模型。这将真实世界适应问题简化为监督式系统辨识，同时保留了仿真中学到的长时域规划信号。

## 难度等级

**高级** — 涉及 MBRL、POMDP、采样规划（MPPI）、sim-to-real transfer 等多个领域的交叉知识。

## 关键收获

1. 世界模型的模块化结构天然适合 sim-to-real：奖励/价值模型只需正确排序状态，无需精确校准
2. 仅微调动力学模型避免了灾难性遗忘和长时域信用分配问题
3. 多样化仿真数据（专家 + 次优 + 扰动）是有效规划的前提
4. 15-30 分钟真实数据即可实现显著性能提升

## 预计学习时间

- 快速浏览：30 分钟（summary.md + insights.md）
- 深入理解：2-3 小时（含 method.md + code demos）
- 完整掌握：4-5 小时（含 qa.md 自测 + reflection.md）

## 文件结构

```
├── README.md              # 本文件
├── meta.json              # 元数据
├── paper.pdf              # 原始论文
├── summary.md             # 背景、贡献、结果总结
├── insights.md            # 核心洞察与深度分析
├── method.md              # 技术细节与算法流程
├── mental-model.md        # 研究定位与思维框架
├── qa.md                  # 15 道自测题
├── reflection.md          # 扩展思考与开放问题
├── code/                  # 代码演示
│   ├── world_model_planning_demo.py
│   └── dynamics_adaptation_demo.py
├── images/                # 论文图片
└── index.html             # 交互式探索器
```
