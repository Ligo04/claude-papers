# 分级问答

## 基础 (Q1-Q5)

### Q1: MAS-PNCG 解决的核心问题是什么？

<details>
<summary>答案</summary>

IPC 仿真中，Newton 法需要每次迭代组装完整 Hessian 并求解大规模线性系统，代价极高。PNCG 避免 Hessian 组装但需要好的预条件器。简单的 Jacobi 预条件器在刚性/接触密集场景收敛极慢，而强大的 MAS 预条件器每次迭代重建代价过高。MAS-PNCG 通过 Woodbury 增量更新解决了这一矛盾，让 MAS 能高效应用于非线性 CG 框架。

</details>

---

### Q2: 为什么接触 barrier Hessian 适合用 Woodbury 公式更新？

<details>
<summary>答案</summary>

接触 barrier Hessian 有三个关键性质：(1) 稀疏性——每个接触只涉及少量顶点；(2) 低秩性——SPD 投影后近似为 rank-1 项 k·nnᵀ；(3) 局部变化——相邻迭代间大部分接触稳定。因此 Hessian 变化可以表示为少量 rank-1 更新的叠加，恰好是 Woodbury 矩阵恒等式的理想输入。

</details>

---

### Q3: MAS-PNCG 相比 GIPC 和 StiffGIPC 的加速倍数范围是多少？

<details>
<summary>答案</summary>

在 5 个基准场景中，MAS-PNCG 相比 GIPC 实现 2.12×-5.66× 加速，相比 StiffGIPC 实现 1.03×-2.07× 加速。最大加速出现在 Two Bunnies 场景（5.66×/2.07×），接触模式快速变化时优势最大。最小加速在 Armadillo 场景（2.12×/1.03×），接触较简单。

</details>

---

### Q4: 什么是全局重启，何时触发？

<details>
<summary>答案</summary>

全局重启是完全重建 MAS 层次结构（包括粗层级）的操作。通过监控正交性损失指标 r_k = |g_{k+1}ᵀ z_k| / (g_{k+1}ᵀ z_{k+1}) 来判断。当 r_k 超过阈值 δ=0.3 时触发，表示 Woodbury 近似误差累积过大，预条件器质量显著退化。实验中平均每 3-10 次 PNCG 迭代触发一次。

</details>

---

### Q5: MAS-PNCG 的性能优势主要来自哪个计算阶段？

<details>
<summary>答案</summary>

主要来自搜索方向计算阶段。Table 2 显示 GIPC 和 StiffGIPC 的搜索方向计算（含 Hessian 组装和线性求解）需要 0.1-7.9s，而 MAS-PNCG 仅需 0.01-0.39s。这是因为 MAS-PNCG 完全避免了 Hessian 组装，只需梯度评估和 Woodbury 增量更新的预条件器应用。

</details>

---

## 中级 (Q6-Q10)

### Q6: Sparse-Input Woodbury 更新如何分类处理不同接触状态？

<details>
<summary>答案</summary>

四种情况：
1. **新接触** (c ∉ C_base)：注入完整 rank-1 项 u = √k_curr · n_curr
2. **法线旋转** (n_curr · n_base < cos25°)：数值不稳定，按新接触处理，保留旧项作为"幽灵墙"增加鲁棒性
3. **刚度增加** (k_curr > k_base)：只添加差分 u = √(k_curr - k_base) · n_curr
4. **刚度减少** (k_curr ≤ k_base)：忽略，保持较硬近似避免 SPD 破坏

每个子域还用 Top-K (K=8) 截断按显著性 ΔS 排序保留最重要的更新。

</details>

---

### Q7: 2D 子空间最小化相比传统 β 公式有什么优势？

<details>
<summary>答案</summary>

传统 PNCG 使用启发式 β 公式（如 Fletcher-Reeves、Polak-Ribière）来组合梯度和前一方向，缺乏曲率信息。2D 子空间最小化利用 Woodbury 近似 Hessian 构建局部二次模型，在 {-μz_{k+1} + νp_k} 子空间内解析求解最优系数。这消除了启发式选择，系数 μ 自然起到步长作用（考虑二阶曲率），允许用 α=1.0 作为 CCD 初始步长。Figure 5 显示 2D 子空间法一致地比所有 β 公式收敛更快更稳定。

</details>

---

### Q8: CCCD（逐子域保守 CCD）如何避免"数值锁定"问题？

<details>
<summary>答案</summary>

标准 CCD 计算全局最小 TOI，一个最硬接触会限制所有区域的步长。CCCD 为每个 MAS 子域独立计算安全步长 α_d，只约束涉及该接触的子域。远离接触的区域可以全速推进。最终位置更新为 x_{k+1} = x_k + Σ_d S_dᵀ α_d S_d p_{k+1}，每个子域用自己的步长。此外，CCCD 用改进的下界（基于相对位移而非绝对位移）提供更紧的估计，将 CCD 迭代从 30,000+ 减少到个位数。

</details>

---

### Q9: MAS-PNCG 为什么在 Armadillo 场景加速不明显（仅 1.03× vs StiffGIPC）？

<details>
<summary>答案</summary>

Armadillo 场景（8K 顶点, 22K 四面体）规模较小且接触模式相对简单。在这种情况下：(1) Newton 法的二次收敛优势更大，每步减少的误差更多；(2) Hessian 组装代价因规模小而不那么突出；(3) PNCG 的每迭代代价优势被需要更多迭代数的劣势抵消。MAS-PNCG 的优势在大规模、接触密集场景中更显著。

</details>

---

### Q10: Woodbury 更新的 Capacitance System 是什么？

<details>
<summary>答案</summary>

Capacitance System 是将 Woodbury 公式高效实施为四步求解过程：
1. **Base Solve**: z_base = B_d · g_d（用缓存的基础逆乘梯度）
2. **Update Projection**: W_d = B_d · U_d，r = U_dᵀ · z_base（投影更新矩阵）
3. **Capacitance Solve**: (I_K + U_dᵀ · W_d) · λ = r（求解 K×K 小系统）
4. **Correction**: z_d = z_base - W_d · λ（校正得到最终预条件方向）

因为 Top-K 截断限制了 K=8，Capacitance 系统只是 8×8 小矩阵，求解代价极低。

</details>

---

## 高级 (Q11-Q15)

### Q11: MAS-PNCG 与 AGIPC（自适应代数粗化）解决的是同一层面的问题吗？

<details>
<summary>答案</summary>

不是，它们解决不同层面的瓶颈：

- **AGIPC** 加速 Newton 法**内循环**的线性求解——通过代数粗化减少 PCG 求解的自由度，仍在 Newton 框架内
- **MAS-PNCG** 完全**跳过 Newton 外循环**——用非线性 CG 替代 Newton 法，避免 Hessian 组装

理论上两者可以互补：AGIPC 的粗化策略可能用于 MAS-PNCG 的粗层级重建加速，但目前没有这种组合。两者也有竞争关系——如果 AGIPC 足够快，Newton 法的优势可能回升。

</details>

---

### Q12: 为什么刚度减少时选择忽略而不是减去差分？

<details>
<summary>答案</summary>

这涉及 SPD（对称正定）保证。减去刚度意味着从 Hessian 近似中移除正定贡献，可能导致预条件器不再正定（矩阵出现负特征值），从而产生无效的搜索方向甚至发散。

保持较硬的近似（即高估刚度）虽然不够精确，但：(1) 保证 SPD 不被破坏；(2) 物理上等同于"保守估计"——认为接触比实际更硬，搜索步长更小但方向正确；(3) 误差会被后续迭代修正或全局重启清除。

这是 over-approximate 优于 under-approximate 的典型案例。

</details>

---

### Q13: Powell 重启策略中 r_k 的物理含义是什么？为什么阈值选 0.3？

<details>
<summary>答案</summary>

r_k = |g_{k+1}ᵀ z_k| / (g_{k+1}ᵀ z_{k+1}) 衡量相邻预条件梯度的正交性。在精确 CG 中，相邻梯度应严格正交（r_k = 0）。当 r_k 增大，说明：(1) 预条件器质量退化（Woodbury 近似误差累积）导致搜索方向偏移；(2) 共轭性被破坏，CG 退化为梯度下降。

在预条件空间中评估（而非原始空间）更准确，因为它过滤了病态矩阵的影响。

阈值 0.3 是经验值。太小（如 0.1）导致频繁重启，失去 Woodbury 更新的效率优势；太大（如 0.5）允许过多误差累积，收敛退化。0.3 平衡了维护成本和预条件器质量。

</details>

---

### Q14: 如果将 MAS-PNCG 应用于没有接触的纯弹性仿真，还有优势吗？

<details>
<summary>答案</summary>

优势会大幅减弱：

1. **Woodbury 更新失去用武之地**：没有接触意味着 barrier Hessian 为零，没有低秩更新可做，MAS 只能靠全局重启维护
2. **弹性 Hessian 变化模式不同**：弹性 Hessian 通常平滑连续变化，不具备稀疏低秩的接触 Hessian 特征，增量更新策略不适用
3. **Newton 法优势更大**：无接触时 Hessian 组装代价相对更低，Newton 的二次收敛优势更明显

但 PNCG 本身（不含 Woodbury 技巧）在某些大变形场景中仍可能因避免 Hessian 组装而有优势，尤其是网格极大时。

</details>

---

### Q15: 论文中的"幽灵墙"策略（法线旋转时保留旧接触）是否可能引入虚假力？

<details>
<summary>答案</summary>

是的，但影响可控。当法线旋转超过 ε_rot 时，旧接触项被保留（"幽灵墙"），同时添加新方向的接触项。这意味着预条件器"记住"了一个不再存在的接触方向，等同于在该方向施加了额外刚度。

影响可控的原因：
1. 预条件器只影响搜索方向，不影响能量评估——最终收敛到正确解
2. 多余的刚度使搜索步长更保守，增加迭代数但不导致错误
3. 全局重启会清除所有历史残留
4. 法线旋转超过 25° 的接触相对少见（大多数接触方向稳定）

本质上，这是用"过度保守"换"数值稳定"的取舍，在工程实践中很常见。

</details>

---
