# 方法拆解：从 EF 相交到方向化 penalty

## 1. IP 背景

标准增量势能步骤写成

\[
x^*=\arg\min_x \frac{1}{2h^2}\|x-\tilde x\|_M^2+\Psi(x),
\]

但 interior-point barrier 需要初态无穿透。论文把每条 contour 转为临时约束 \(C_k(x)>0\)，并在这个 minimization 里加入可解缠的 response energy。

## 2. 三类线性 ray test

三个测试共享

\[
[r\mid a\mid b][\delta,\alpha,\beta]^T=d.
\]

- **VF**：source vertex \(p\) 沿 \(r\) 到 destination triangle 的 barycentric 点。
- **EE**：source edge 扫成 parallelogram strip，与 destination edge 相交。
- **FV**：source face 扫成 prism，等价于 destination vertex 沿 \(-r\) 反向打到 source face；代码复用 reverse-VF traversal。

令 \(n_h=a\times b\)，则

\[
\delta_h(r)=\frac{\det[d\mid a\mid b]}{\det[r\mid a\mid b]}=\frac{\nu_h}{n_h^Tr}.
\]

\(n_h\) 是 characteristic normal：VF/FV 是面法向，EE 是两条 edge 方向叉积。

## 3. 双侧 cluster culling

对每个 raw hit 建立 source primitive 与 destination primitive 的乘积单元。只有两侧都满足对应 adjacency/incidence 才连边。得到 component \(\Gamma_j\) 后，保留触碰 contour 两侧 one-ring seed 的 component，并以 EF anchor coverage 选物理 band； detached component 视为几何碎片。

这一步解释了 `type-free`：不必先判断 Closed/Eight/BLI，只需要 contour anchors + primitive adjacency。

## 4. 10 个初始方向

论文正文的 10 个方向：

\[
D_{scatter}=\{\pm q_0,\pm q_1,\pm q_2\},
\]

加上两侧中心差方向 \(\pm(c_1-c_0)\)，以及从 contour centroid 指向两侧 mesh centroid 的 2 个方向，共 10 个。散布矩阵为

\[
S_w=\sum_{v\in A} A_v(x_v-c_C)(x_v-c_C)^T.
\]

最小特征值对应的 axis 常常捕捉最薄、最容易分离的方向，但只作 seed；每个 seed 必须重新 ray-cast 验证。

> 当前仓库固定 SHA 的实现日志记录了方向集合的工程调整：默认 `EmbedMeshDirection` 为最小方差方向的正负；扩展模式可加入全局质心径向、LDA、cone/低差异样本，而旧的 `±(c1-c0)` 不再是无条件方向。读代码时把“论文 10 个 seed”和“当前可配置 combo 集合”分开。

## 5. 方向目标与 IRLS

对固定 hit set \(S_k\)，冻结 \(\nu_h\)，用

\[
\delta_h(r)=\nu_h/(n_h^Tr)
\]

构造带安全间隙 \(\epsilon_{safe}\) 的 IRLS surrogate：

\[
F_k(r)=\sum_{h\in S_k}\mu_h\omega_h^{(k)}(\delta_h(r)+\epsilon_{safe})^2.
\]

\(p_k\) 从 2 线性升到 \(p_{max}\)，权重聚焦深穿透；\(\mu_h\) 是由 barycentric weights 和 nodal mass 推导的 effective mass。导数为

\[
\frac{\partial\delta_h}{\partial r}=-\frac{\delta_h n_h}{n_h^Tr}.
\]

把 gradient/Hessian 投影到 \(T_rS^2\) 的二维 basis；Hessian 正定时取 Newton step，否则走 Cauchy/steepest-descent。归一化 retract 回球面，每次 trial 都重新 ray-cast + cull；只有完整 objective 下降才接受。

## 6. Response assembly

选出 \(r^*\) 和 \(S^*\) 后，所有 hit 共享 contour depth \(D_C(r^*)\)：

\[
C_h(x_h)=n_h^T\left(\sum_jw_{h,j}x_{h,j}\right)-D_C(r^*),\qquad
E_h=\frac12\kappa \bar A_h C_h^2.
\]

固定 barycentric weights 意味着 \(C_h\) affine，故

\[
\nabla E_h=\kappa\bar A_h C_hJ_h^T,\qquad
\nabla^2E_h=\kappa\bar A_hJ_h^TJ_h.
\]

这就是为什么它可以在现有 Newton/PCG 循环里与 elastic/contact 能量一起装配。

## 7. adaptive k-ring 与 CCR

从两侧 seeds 做 BFS：vertex 要在 k-ring 内，edge 的两个端点都在，face 的三个端点都在；frontier 尚未脱离当前 ring 时令 \(k\leftarrow\min(2k,k_{max})\)。论文在 878 contours 上报告 raw hits 减少 152.5×（几何均值），median 减少 99.44%，且 valid-hit parity 为 100%。

论文 CCR 用当前 Euclidean chord distance：

\[
\phi_s(v)=\frac{c_s(v)-c_{1-s}(v)}{c_s(v)+c_{1-s}(v)+\epsilon}\le\tau,
\quad \tau=0.25.
\]

当前仓库额外支持 rest-geodesic/current-space 混合与 AND 逻辑，目的是减少折叠空气捷径；这是实现扩展，不应冒充论文已经完成的消融。

## 8. 冲突抑制

多条 contour 的 response band 可能重叠。论文把 contour 作为节点、重叠作为边、penetration area 作为权重，目标是最大权独立集；实际用按 area 降序的 greedy heuristic，保留一个主导 separation field，抑制相邻冲突 contour。当前代码日志将其称为 directed greedy response suppression，并不重新合并 contour geometry。

## 9. 伪代码

```text
for contour C:
    build seeds A0, A1 and candidate V/E/F domains
    for k in adaptive_k_ring_levels:
        for direction r in initial_combos:
            raw = raycast_VF_EE_reverseFV(C, r, candidates[k])
            support = dual_side_cluster_cull(raw, A0, A1)
            score = max_depth(support)^2 * area_sum(support)
        if support_frontier_converged:
            break
    r0 = best_valid_combo(score)
    r* = projected_newton_or_cauchy(r0, re-evaluate_full_discrete_pipeline)
    response[C] = fixed_barycentric_penalty_pairs(r*, support[r*])
resolve_cross_contour_conflicts(response)
solve_implicit_newton_pcg(elastic + contact + response)
```

## 复现风险

- PDF 表格依赖作者硬件、精度和场景参数；不能拿 demo 输出替代论文结果。
- 仓库的 `ext/luisaCompute` 和 `external/instant-mesh-intersection-repair` 是 submodule，未初始化时无法编译完整 solver。
- Jang 的完整 `Dataset_distributed` 不随仓库分发，需要按 README 向原作者获取。
- GPU/CPU PRP、普通 physics backend 和 `use_gpu_untangling` 是独立开关；只写 `--backend cuda` 不代表 PRP contour evaluator 已经在 GPU 上运行。
- 单一方向无法表达连续扭转；CCR 和 candidate domain 没有全局 locality theorem。
