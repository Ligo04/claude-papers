# 论文摘要

## 背景

增量势能接触（IPC）通过 barrier 能量将接触动力学重新表述为无约束优化问题，保证无穿透仿真。然而 Newton 法每次迭代需要组装完整 Hessian 矩阵并求解大规模线性系统，计算代价极高。GIPC 和 StiffGIPC 通过 GPU 并行加速了 Newton-PCG 内循环，但 Hessian 组装和线性求解仍是瓶颈。

预条件非线性共轭梯度法（PNCG）完全跳过 Hessian 组装，仅需梯度评估即可迭代下降。但现有 IPC-PNCG 实现依赖简单 Jacobi 预条件器，无法捕获刚性材料和密集接触带来的复杂耦合，导致收敛极慢。

## 问题陈述

如何在非线性 CG 框架中使用强大的多层级预条件器（MAS），而不陷入每次迭代重建预条件器的高昂代价？

核心矛盾：
- MAS 需要基于系统 Hessian 的子域分解 → 但 PNCG 刻意避免 Hessian 组装
- 非线性 CG 中有效曲率每迭代都在变化（接触激活/释放）→ 不能像 Newton-PCG 那样固定 MAS

## 主要贡献

1. **Sparse-Input Woodbury Level-0 更新**：利用接触 barrier Hessian 的稀疏低秩结构（每个接触贡献一个 rank-1 项），通过 Woodbury 公式增量更新细粒度子域逆矩阵，代价几乎为零。粗层级冻结不变，仅周期性重建。

2. **最优 2D 子空间最小化**：将 PNCG 搜索方向限制在由预条件梯度和前一方向张成的 2D 子空间内，利用近似 Hessian（通过 Woodbury 更新获得）解析求解最优组合系数 (μ, ν)，替代 Fletcher-Reeves / Polak-Ribière 等启发式 β 公式。

3. **逐子域保守 CCD（CCCD）**：为每个子域独立计算安全步长 αd，允许远离接触的区域以最优速度推进，而仅对涉及最硬接触的子域进行阻尼。同时引入改进的下界估计，比标准 ACCD 更紧。

## 关键结果

### 性能对比（Table 2）

| 场景 | 网格规模 | vs GIPC | vs StiffGIPC |
|------|---------|---------|-------------|
| Octopus Stack | 10K, 39K | 3.71× | 1.97× |
| Single Bunny | 20K, 80K | 3.64× | 1.54× |
| Two Bunnies | 40K, 160K | **5.66×** | **2.07×** |
| Armadillo | 8K, 22K | 2.12× | 1.03× |
| Dragon High | 78K, 274K | 3.25× | 1.16× |

### 关键指标

- 搜索方向计算时间：从 GIPC 的 0.1-7.9s 降至 0.01-0.39s
- 预条件器更新策略：Woodbury 更新收敛率与 Full Rebuild 几乎一致，计算时间低两个数量级
- CCCD vs ACCD：CCD 迭代次数从 30,000+ 降至个位数，端到端加速约 10%
- 2D 子空间 vs 启发式 β：一致地更快更稳定收敛（Figure 5）
- 材料鲁棒性：跨 500× 刚度范围（2.15×10⁴ Pa 到 10⁷ Pa）稳定收敛，Jacobi-PNCG 频繁发散

### 支持的能量模型

- ARAP（As-Rigid-As-Possible）
- Stable Neo-Hookean (SNH)
- Baraff-Witkin 布料
