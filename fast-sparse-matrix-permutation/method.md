# Method · 算法逐步拆解

## 总览

输入：稀疏对称半正定矩阵 `A ∈ ℝ^{n×n}`、网格 `M`、嵌套剖分深度 `nd_level`
输出：填充缩减重排向量 `perm` + 对应的消去树 `etree`

> 网格只用于生成 patch 和商图。**若已有 patch 划分，则网格输入不必要。**

### Algorithm 1（顶层流程）

```
1: patches ← GetPatches(M)                       # Step 1: 在 GPU 上算 patch
2: G, gmap ← BuildGraphAndGroups(A, M, patches)  # Step 2: 建图 + 提升 patch 到组映射
3: etree   ← BuildEtree(G, gmap, nd_level)       # Step 3: patch 引导的嵌套剖分，同步建 etree
4: perm    ← ComputePerm(etree, G)               # Step 4: 遍历树拼接局部排序成全局重排
```

只有 Step 1 用 GPU，其余在 CPU 上。

---

## 组件拆解

### Step 1 — Patch 计算（GPU）

- 把网格顶点划分成连通的 **patch**（每个 patch 是网格上一块连通区域，默认目标 256 个顶点）。
- 实现：GPU 上用 **Lloyd 算法**（k-means 式聚类，RXMesh 的并行 patching）。
- 很多图形应用本来就有 patch 式划分（remeshing、聚类、区域分解），可直接复用，零额外成本。

**为什么重要**：patch 把 n 个顶点的大图压成 `|patches| ≈ n/256` 个节点的小图。分隔集计算（NP-complete）的搜索空间随之骤减。

### Step 2 — 把 patch 提升到矩阵图（构建 gmap）

- 构建 `A` 的图 `G = (V_G, E_G)`（去掉对角线），其中 `(i,j) ∈ E_G ⟺ A_ij ≠ 0`。
- 构建**组映射** `gmap : V_G → {1,…,|patches|}`，把每个图顶点分配到它所属的 patch。
  - **标量情形**（如标量 Laplacian）：每个网格顶点 ↔ 一个图顶点，直接映射。
  - **块情形**（如 2×2 Hessian 块）：每个网格顶点 ↔ 一小块图顶点，整块映到同一 patch。覆盖标量 Laplacian 和块结构 Hessian 两种系统。

### Step 3 — Patch 引导的嵌套剖分（同步建 etree）

对图 `G` 和组映射 `gmap`，递归做 `nd_level` 层嵌套剖分，**边递归边建 etree**：

- 每层：算分隔集 `s`，把当前子图切成左/右子图 `g_ℓ` 和 `g_r`，把分隔集存到当前树节点，递归两个孩子。
- 递归到深度 0 时，叶子存剩余顶点。
- etree 表示为**存在 1D 数组里的满二叉树**：节点索引 `idx`，孩子是 `2·idx+1` 和 `2·idx+2`。
- 实践中 nd_level 取 **9–10** 即可提供足够并行性。

#### GetSeparator（Algorithm 2，核心子程序）

```
输入: g, gmap   →   输出: s, g_ℓ, g_r
1: q       ← QuotientGraph(g, gmap)             # 在 patch 组上建商图
2: part_q  ← BiPartition(q)                     # 用 METIS 在商图上二划分（带平衡约束）
3: super_s ← SuperSeparator(g, gmap, part_q)    # 把划分提升回 V(g)，得到分隔集的"超集"（边界顶点）
4: s       ← RefineSeparator(g, super_s, gmap, part_q)  # 细化以缩小分隔集、维持平衡
5: (g_ℓ, g_r) ← LeftRight(g, s, gmap, part_q)   # 诱导出左右子图供下一层递归
```

**五步直觉**（对应 Figure 5）：在小商图上快速二划分 → 提升回原图得到一条"粗"的分界带（超集）→ 用 METIS 节点细化把这条带收窄 → 切出左右子图。

#### 关键实现优化：商图增量更新

> 痛点：每个树节点都要更新商图来产生有效二划分，但**从头重建商图要扫描整个矩阵**——大矩阵上极贵。

解法：**只建一次商图**，从根到叶递归时只做增量更新：
- 调整**节点权重**：减去已标记为分隔集的部分。
- 调整**边权重**：移除分隔集节点关联的边。

这就是"构建一次、跨递归复用"，是相对 METIS（每层重新粗化）的核心提速来源之一。

#### 分隔集计算的瓶颈

- 第 1–2 步（商图二划分）**快**，因为商图比 `A` 小得多。
- 第 3 步（把商图划分提升回 `V_G` 得到超集）是**整个计算的瓶颈**。
- 第 4 步用 METIS 的节点细化策略收窄分隔集。

### Step 4 — 局部排序 + 拼接（ComputePerm，Algorithm 3）

```
1: for idx in [0,…,|etree|-1]:
2:     g_idx ← InducedSubgraph(g, etree[idx].nodes)
3:     etree[idx].perm_local ← LocalOrder(g_idx)    # 默认用 AMD
4: perm ← []
5: for idx in schedule(etree):                       # 按选定调度遍历树
6:     perm.Append(etree[idx].perm_local)
```

两阶段：
1. **局部排序**：对每个树节点诱导的子图，用 **AMD**（默认）算局部排序并缓存。
2. **拼接**：按用户选定的 **schedule**（树遍历顺序）把局部排序串起来成全局 `perm`。

---

## Schedule（调度）—— 一个被低估的自由度

- **有效调度的约束**：父节点不能排在孩子之前（尊重消去依赖）。
- **不同有效调度不改变 fill-in**，但因局部性/并行性不同而影响性能。

| 调度 | 适用 | 原因 |
|------|------|------|
| **Post-order（后序）** | CPU（如 CHOLMOD 默认） | 提升时间局部性：兄弟子树处理完后立刻被父节点消费，缓存复用好 |
| **Level-order / wavefront（层序）** | GPU（如 cuDSS） | 从叶子逐层处理，暴露更多并行性 |

**注意陷阱**：
- CHOLMOD 不暴露用户调度，内部自建执行序——给无效调度不会崩，但分隔集先于孩子组装会降性能。
- cuDSS **要求**与其执行模型一致的调度（level-order/wavefront）——不兼容的调度会**运行时崩溃**。

---

## 块结构 Hessian 的处理（§6.5）

- Symmetric Dirichlet 等二阶系统：每个 Laplacian 非零对应一个稠密 2×2 Hessian 块 → 行列数 2×、非零数 ~4×。
- 技巧：把每个 2×2 块**合并成单个图节点**建 `G`，在压缩图上做重排，最后把 `perm` 和 `etree` **展开回完整 Hessian**。重排成本与标量 Laplacian 相当。

---

## 实现陷阱与复现风险

1. **patch 连通性**：必须保证每个 patch 诱导连通子图。不连通会让 GetSeparator 受限，分隔集更大更不平衡，质量明显下降。
2. **调度兼容性**：cuDSS 必须喂 level-order 调度，否则运行时失败。
3. **patch size 是敏感旋钮**：小网格对它敏感（粗 patch 过度约束）；默认 256 是大网格上的折中。复现时若网格偏小应调小。
4. **商图增量更新的正确性**：擦除分隔集节点的权重/边时必须严格一致，否则后续递归的二划分无效。
5. **盈亏平衡意识**：固定矩阵 + 上千次求解时本方法会变慢，复现对比实验要选对工作负载（单次/少量求解）。
6. **nd_level 取值**：太浅并行性不足，太深叶子太碎；9–10 是经验值。

## 超参数敏感度速查

| 超参 | 默认 | 影响 |
|------|------|------|
| patch size | 256 | ↑ 更快但 fill-in ↑；小网格敏感 |
| nd_level | 9–10 | 控制并行粒度 |
| 局部排序 | AMD | vs METIS：快 ~12×，fill-in 略高 |
| schedule | 按后端 | CPU post-order / GPU level-order |
| BiPartition | METIS | 在小商图上，便宜 |

## 复杂度直觉

稀疏 Cholesky 复杂度 `O(Σ_i d(i)²)`，`d(i)` 是因子 `L` 第 `i` 列的非零数（Lipton et al. 1979）。块化（标量→2×2）使有效列密度约翻倍，故 Hessian 分解远贵于 Laplacian（1M 网格：72ms vs 300ms），此时重排占比下降、加速也相应变小。
