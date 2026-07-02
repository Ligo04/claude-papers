# AGIPC 方法详解

## 算法总览

```
┌─────────────────────────────────────────────────────┐
│                  Newton 迭代循环                      │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ 细网格    │──▶│ 边标记   │──▶│ 并行聚合         │ │
│  │ 梯度/Hessian│  │ (Green   │   │ (Warp-level     │ │
│  │ 计算      │   │  strain) │   │  hash mapping)   │ │
│  └──────────┘   └──────────┘   └────────┬─────────┘ │
│                                          │           │
│  ┌──────────┐   ┌──────────┐   ┌────────▼─────────┐ │
│  │ 细网格    │◀──│ 延拓     │◀──│ 代数组装         │ │
│  │ 后精炼    │   │ (Prolong)│   │ Hc = UᵀHfU      │ │
│  │ (≤10 CG) │   │          │   │ gc = Uᵀgf       │ │
│  └──────────┘   └──────────┘   └────────┬─────────┘ │
│       │                                  │           │
│       ▼                         ┌────────▼─────────┐ │
│  线搜索 + 收敛检查              │ 粗网格 PCG 求解   │ │
│                                 └──────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## 组件分解

### 1. IPC 能量与隐式时间积分

标准 IPC 框架：每个时间步求解

$$x^{n+1} = \arg\min_x \frac{1}{2h^2}\|x - \hat{x}\|_M^2 + \Psi(x) + \sum_k b(d_k(x))$$

- 第一项：惯性项（质量加权）
- Ψ(x)：弹性势能（FEM 或 mass-spring）
- b(d)：IPC barrier 能量，防止穿透

Newton 迭代中，每步需要组装并求解 Hf·Δx = -gf，这是计算瓶颈。

### 2. 边标记（Edge Tagging）

**核心创新之一：** 用 Green 应变增量判断边是否可坍缩。

对每条边 e = (i, j)：

```
ΔG_e = G_i^{(k)} - G_i^{(k-1)}    （当前与上一 Newton 迭代的 Green 应变差）
‖ΔG_e‖_F < τ  →  标记为 collapsible
```

- τ = 5×10⁻⁵（默认阈值）
- 首次 Newton 迭代：所有边标记为 protected（无上一步参考）
- 接触区域附近的边始终 protected

**为什么用 Green 应变而非位移？**
Green 应变是旋转不变的，刚体旋转不会导致误标记。位移大的区域如果只是刚体运动，应变增量仍为零。

### 3. 并行聚合（Parallel Aggregation）

将可坍缩边的端点合并为 "超级节点"（supernode）。

**Warp-level hash mapping 方案：**

```
1. 每条 collapsible 边的两端点映射到同一 hash bucket
2. 使用 warp-level 原子操作并行合并
3. Connected component → 一个 supernode
4. Protected 边的端点保持独立
```

GPU 友好：无需显式拓扑修改，只需维护 fine→coarse 映射表。

### 4. 代数组装（Algebraic Assembly）

**Galerkin 投影：**

```
粗网格 Hessian:  Hc = Uᵀ · Hf · U
粗网格梯度:      gc = Uᵀ · gf
```

U 是 prolongation 矩阵（稀疏，每行只有一个非零块）。

实际实现中不显式构造 U，而是通过 GPU reduction kernel 直接在 fine-scale 数据上操作：

```python
# 概念伪代码
for each fine node i:
    coarse_id = mapping[i]
    gc[coarse_id] += gf[i]           # 梯度归约
    for each neighbor j of i:
        Hc[mapping[i], mapping[j]] += Hf[i, j]  # Hessian 归约
```

使用 BCOO（Block COO）格式存储粗网格 Hessian，利用对称性减半存储。

### 5. 仿射嵌入（Affine Embedding）

当一个 supernode 聚合了超过 32 个 fine 节点时，3-DoF 平移自由度不足以表达局部变形。

**解决方案：** 切换到 12-DoF 仿射变换

```
x_fine = A · x̄ + t

其中 A ∈ R³ˣ³ (旋转+变形), t ∈ R³ (平移)
→ 每个 supernode 12 个自由度
```

判定规则：
- ≤32 个 fine 节点 → 3-DoF（纯平移）
- >32 个 fine 节点 → 12-DoF（仿射）

12-DoF 能保留旋转和剪切信息，避免大 supernode 的过度约束。

### 6. 后粗化精炼（Post-Coarsening Refinement）

粗网格解延拓回细网格后，执行少量（≤10 次）CG 迭代进行精炼：

```
1. 粗网格求解: Hc · Δxc = -gc
2. 延拓: Δxf = U · Δxc
3. 精炼: 在细网格上以 Δxf 为初始值，运行 ≤10 次 CG
4. 线搜索: 确保能量下降
```

精炼步骤修正了粗化引入的近似误差，代价很低（仅几次 CG 迭代）。

### 7. 线搜索与收敛

与标准 IPC 一致：
- CCD（连续碰撞检测）确定最大步长
- 过滤线搜索确保 barrier 约束
- 能量下降 Armijo 条件

## 完整伪代码

```
Algorithm: AGIPC Newton Solve
Input: 当前位置 x, 惯性目标 x̂, 时间步 h
Output: 更新后位置 x*

1. for k = 1, 2, ... do
2.    计算细网格梯度 gf 和 Hessian Hf
3.    if k == 1 then
4.       所有边标记为 protected
5.    else
6.       for each edge e = (i,j) in parallel:
7.          ΔG = G_i^(k) - G_i^(k-1)
8.          if ‖ΔG‖_F < τ and e 不在接触区域:
9.             tag[e] = collapsible
10.         else:
11.            tag[e] = protected
12.   end if
13.   
14.   // 并行聚合
15.   mapping = WarpHashAggregate(edges, tags)
16.   
17.   // 代数组装粗网格系统
18.   gc = ReduceGradient(gf, mapping)
19.   Hc = ReduceHessian(Hf, mapping)  // BCOO 格式
20.   
21.   // 识别大 supernode，切换仿射嵌入
22.   for each supernode s:
23.      if size(s) > 32: use 12-DoF affine
24.      else: use 3-DoF translation
25.   
26.   // 粗网格求解
27.   Δxc = PCG(Hc, -gc)
28.   
29.   // 延拓
30.   Δxf = Prolongate(Δxc, mapping)
31.   
32.   // 后精炼
33.   Δxf = RefineCG(Hf, gf, Δxf, max_iter=10)
34.   
35.   // 线搜索
36.   α = FilteredLineSearch(x, Δxf)
37.   x = x + α · Δxf
38.   
39.   if ‖gf‖ < ε: break  // 收敛
40. end for
```

## 实现要点与陷阱

| 要点 | 说明 |
|------|------|
| Green 应变缓存 | 需保存上一 Newton 迭代的 Green 应变，增加显存开销 |
| Hash 冲突处理 | Warp-level hash 需处理冲突，影响聚合质量 |
| BCOO 对称性 | 只存上三角，组装时需注意对称贡献 |
| 接触边保护 | barrier 能量涉及的边必须 protected，否则穿透 |
| 仿射嵌入切换 | 32 节点阈值是经验值，太小浪费 DoF，太大过度约束 |
| 首次迭代 | 第一次 Newton 迭代无法粗化（无历史应变），退化为标准求解 |
| 精炼迭代数 | ≤10 次 CG 是上限；论文发现通常 3-5 次就够 |

## 超参数敏感性

| 参数 | 默认值 | 影响 |
|------|--------|------|
| τ (应变阈值) | 5×10⁻⁵ | 越大→粗化越激进→越快但越不精确 |
| 仿射切换阈值 | 32 节点 | 越大→更多 3-DoF→更快但大 supernode 精度下降 |
| 精炼 CG 上限 | 10 | 越多→越精确但耗时增加 |
| BCOO block size | 3×3 | 固定，与 3D 坐标维度匹配 |
