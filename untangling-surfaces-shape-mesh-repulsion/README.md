# Untangling Surfaces via Shape and Mesh Repulsion

## 概述

本文提出一个基于能量的框架，通过耦合形状级（shape-level）高斯自接触能量和网格级（mesh-level）Minkowski 惩罚来消除曲面网格的自相交。核心洞察是：成功的解缠需要同时处理全局形状交互（远距离纠缠）和局部网格交互（三角形级退化）。该方法不改变网格连通性，适用于非流形、非可定向、多物体场景等广泛输入类型，在所有基准测试中实现了 100% 的自相交消除率。

## 难度等级

**高级** — 涉及微分几何、能量优化、Minkowski 和、自适应带宽选择等深层数学概念。

## 关键收获

1. 嵌入性（embeddedness）是全局几何性质，不能仅通过局部碰撞处理来保证
2. 高斯自接触能量在自相交配置上保持有限（不像经典排斥能量会发散），同时提供全局解缠梯度
3. 自适应带宽选择让能量自动聚焦于最相关的空间尺度
4. Minkowski 差的半空间表示提供了光滑可微的穿透深度度量
5. 两级能量互补：形状级驱动大尺度解缠，网格级解决局部退化

## 预计学习时间

- 快速浏览：30 分钟（summary.md + insights.md）
- 深入理解：3-4 小时（含 method.md + 数学推导）
- 完整掌握：5-6 小时（含 qa.md + code demos + reflection.md）

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
│   ├── gaussian_selfcontact_2d.py
│   └── minkowski_penalty_demo.py
├── images/                # 论文图片
└── index.html             # 交互式探索器
```
