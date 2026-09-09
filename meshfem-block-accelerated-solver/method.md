# 方法拆解

## 1. 数学对象

令节点变量为 `x_i ∈ R^d`，元素 `e` 只访问节点集合 `I_e`：

```text
E(x) = Σ_e E_e(x_{I_e})
g = ∂E/∂x
H = ∂²E/∂x²
```

对每个元素，局部 Hessian 的 `(a,b)` 子块大小是 `d×d`。只有两个节点同时出现在某个元素中时，全局块 `H_ij` 才可能非零。

## 2. 算法流

```text
元素拓扑
   │
   ├─► block sparsity pattern（可缓存）
   │
   ├─► 每个元素计算局部 g_e / H_e
   │        └─► TBB thread-local CSC，最后合并
   │
   ├─► 对非凸 Hessian 做投影/策略控制（按 Newton 外层配置）
   │
   ├─► symbolic ordering + multifrontal analysis
   │        └─► 模式未变时复用；动态模式由 LRU 管理
   │
   ├─► BlockCatamari numeric Cholesky
   │
   └─► solve Δx，再由 line search / 负曲率策略更新
```

## 3. 伪代码

```text
pattern = cache.lookup(element_connectivity, active_contact_set)
if pattern is missing:
    pattern = build_block_pattern(elements)
    cache.insert(pattern)

local_matrices = thread_local_empty(pattern)
parallel_for element e:
    ge, He = energy_gradient_hessian(e, x)
    local_matrices[thread].add(e, ge, He)
g, H = merge(local_matrices)

H = project_or_shift_if_needed(H)
symbolic = symbolic_factorization(pattern)  # reuse when possible
L = block_cholesky(symbolic, H)
dx = solve(L, -g)
x = globalization_step(x, dx)
```

## 4. 官方实现中的关键组件

### 块 CSC

`BlockCSCHessian` 用块列/块行描述稀疏模式，同时支持标量值数组；其设计假定对角块存在，并可通过 `ColumnScanner` 在单调列访问中降低查找成本。对角块保留完整小矩阵时，分解阶段可以更直接地调用块 kernel。

### 并行装配

官方 `ParallelAssembly.hh` 为线程维护局部矩阵，再做合并。这样避免多个元素同时写入同一个全局块的锁竞争；代价是局部暂存和合并带宽。

### 模式缓存

`SparsityLRU` 将静态模式和偶尔变化的动态条目分开管理。仓库当前实现的默认值是缓存预算约为静态条目数的 1%，软过期年龄 25，硬过期年龄 250；这些是实现默认，不是本文正文的实验参数。

### BlockCatamari

官方 README 将其定位为实数 supernodal sparse Cholesky `LLᵀ`，支持 BCSC 以利用块结构，并用 TBB 实现并行。若系统仍明显非正定，外层必须先处理负曲率或 Hessian 投影；不能把它当成任意对称不定系统的 LDLᵀ 黑盒。

## 5. 实现陷阱

1. **块模式和标量模式不一致**：忘记插入对角块会让后续 factorization 失败。
2. **局部变量排序不稳定**：元素贡献若没有统一块顺序，排序-合并会产生隐蔽的重复或漏加。
3. **只测 factorization**：应分别记录 pattern、assembly、symbolic、numeric、solve，才能知道优化是否真的覆盖总时间。
4. **接触模式过于动态**：缓存预算太小导致频繁重建，太大则浪费内存；要按活动集变化率调参。
5. **SPD 假设不成立**：非凸 Newton 的 Hessian 可能有负特征值，需要投影、shift、信赖域或负曲率方向。
6. **线程局部矩阵过大**：线程数增加不一定更快，需观察内存带宽和合并成本。

## 6. 复现风险

- 需要 MeshFEMCore、MeshFEMSparse、BlockCatamari、SuiteSparse/CHOLMOD 和 CMake 工具链；Python demo 还需要编译绑定。
- 不同 ordering、BLAS/MKL、线程数和接触活动集会改变结果，不能仅凭单机一次运行推断普适加速。
- 当前没有 ACM 正文，所以本目录没有复制论文图表，也没有声称复现论文的精确指标。
