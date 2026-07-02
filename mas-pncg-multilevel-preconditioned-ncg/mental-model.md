# 心智模型

## 问题分类

**优化求解器加速** — 面向 IPC 隐式时间积分中非线性优化问题的快速求解。

核心命题：能否在保持 MAS 预条件器质量的同时，避免 Newton 法的 Hessian 组装代价？

## 前置知识图谱

| 知识领域 | 需要程度 | 说明 |
|---------|---------|------|
| IPC / barrier 能量 | ★★★★★ | 理解接触如何被公式化为优化问题 |
| 共轭梯度法 (CG/PCG) | ★★★★★ | 线性和非线性 CG 的区别 |
| 域分解 / MAS | ★★★★☆ | 多层级加性 Schwarz 的结构和作用 |
| Woodbury 矩阵恒等式 | ★★★★☆ | 低秩更新矩阵逆的公式 |
| Newton 法 | ★★★☆☆ | 理解为什么 MAS-PNCG 要避免 Newton |
| GPU 并行 | ★★☆☆☆ | 理解性能对比的背景 |
| CCD (连续碰撞检测) | ★★☆☆☆ | CCCD 改进的背景 |

## 知识定位图

```
IPC 求解器加速
├── Newton 法加速（保持 Newton 框架）
│   ├── 线性求解加速
│   │   ├── AGIPC — 自适应代数粗化减少 DoF
│   │   ├── GPU PCG — 并行化 CG 内循环
│   │   └── 子空间方法 — 投影到低维空间
│   ├── Hessian 组装加速
│   │   ├── GIPC — GPU 并行 Hessian 组装
│   │   └── StiffGIPC — 更高效的 GPU 实现
│   └── 近似方法
│       ├── 投影动力学 — 局部/全局分裂
│       ├── 中轴线缩减 — 降低自由度
│       └── 模板下降 — 避免精确 Hessian
│
├── 非 Newton 法加速（★ MAS-PNCG 在此）
│   ├── PNCG — 完全避免 Hessian 组装
│   │   ├── Jacobi-PNCG — 简单但收敛慢
│   │   └── ★ MAS-PNCG — 多层级预条件 + Woodbury 增量
│   └── L-BFGS — 拟 Newton 近似
│
└── 混合方法
    └── 自适应策略 — 根据场景切换
```

## 类比理解

### 类比 1：建筑翻修 vs 重建

**Newton + MAS（GIPC/StiffGIPC）**就像每次修改建筑时都推倒重建：
- 先画完整蓝图（Hessian 组装）
- 按蓝图建造（线性求解）
- 每次修改都重新画蓝图、重新建造

**MAS-PNCG** 就像增量翻修：
- 保持主体结构不变（粗层级 MAS 冻结）
- 只对变化的房间重新布局（Level-0 Woodbury 更新）
- 偶尔做一次全面检查（全局重启）
- 大部分时间只需要局部调整（增量更新代价近零）

### 类比 2：GPS 导航

**Newton 法**：每走一步都重新计算从起点到终点的完整最优路径（精确但慢）

**Jacobi-PNCG**：只看眼前路面的坡度往下走（快但容易在复杂地形迷路）

**MAS-PNCG**：保持一张全局地图（MAS 粗层级），实时更新附近的路况信息（Woodbury 更新），在两个候选方向中选最优（2D 子空间），不同区域的车辆独立调速（CCCD）

### 类比 3：天气预报更新

- **Full Rebuild** = 每小时重新运行完整的全球气候模型
- **Freeze** = 用今早的预报应付一整天
- **Woodbury Update** = 用新的卫星数据增量修正局部预报，保持大尺度模式不变

## 这篇论文的学术贡献定位

1. **方法论贡献**：首次将 MAS 预条件器成功应用于非线性 CG 框架（之前只能用于线性 PCG 内循环）
2. **技术贡献**：Sparse-Input Woodbury 增量更新策略，利用接触 barrier 的物理结构
3. **工程贡献**：逐子域 CCCD 和改进下界估计
4. **认知贡献**：揭示了"接触 Hessian 变化的低秩结构"这一物理洞察的计算价值

## 研究谱系

```
Wu et al. 2022 (GPU MAS for elastodynamics)
     │
     ├──→ StiffGIPC (Huang et al. 2025): MAS 用于 Newton-PCG 内循环
     │         │
     │         └──→ ★ MAS-PNCG: MAS 用于非线性 CG（Woodbury 增量更新）
     │
Shen et al. 2024 (PNCG for IPC, Jacobi)
     │
     └──→ ★ MAS-PNCG: 解决 Jacobi 预条件器不足的问题
```
