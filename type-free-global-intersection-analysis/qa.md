# 自测题（15 题）

### 1. 论文要修复的输入状态是什么？

<details>
<summary>答案</summary>

已经存在 edge–face 穿透的薄壳、布料或可变形体状态；这类状态不能直接交给依赖无穿透初态的 interior-point barrier。

</details>

---

### 2. GIA 和 ICM 的主要短板分别是什么？

<details>
<summary>答案</summary>

GIA 依赖 contour 类型分类和对应 flood-fill 规则，容易受检测噪声、开边界和非流形结构影响；ICM 只最小化局部 contour 几何，深层/嵌套穿透时会卡在局部极小。

</details>

---

### 3. 三种 ray evidence 是什么？

<details>
<summary>答案</summary>

VF、EE、FV。FV 在实现上可看作 destination vertex 沿反方向打到 source face 的 reverse-VF 查询。

</details>

---

### 4. \(D_\mathcal C\) 和 \(J(r)\) 各表示什么？

<details>
<summary>答案</summary>

\(D_\mathcal C\) 是一个 contour 的最大 hit travel distance；\(J\) 是它的平方乘以 response support 的面积/质量总和。

</details>

---

### 5. 为什么论文说 type-free？

<details>
<summary>答案</summary>

同一条“候选方向 raycast → 双侧 hit graph → contour-anchored component”路径覆盖开/闭 contour、自相交和跨物体穿透，不需要预先选择 Closed、Eight、BLI 等拓扑分支。

</details>

---

### 6. 为什么只在 source 侧做 adjacency 不够？

<details>
<summary>答案</summary>

不同 destination primitive 的 hit 可能在 source mesh 上相邻，却属于不同响应带。product-complex 规则要求 source 和 destination 两侧同时满足 adjacency/incidence，避免把远处层串成一个 component。

</details>

---

### 7. Cramer 形式对方向搜索有什么帮助？

<details>
<summary>答案</summary>

固定节点位置时 \(\nu_h=n_h^Td\) 不随方向变化，\(\delta_h(r)=\nu_h/(n_h^Tr)\) 只需更新一个分母；这让固定 hit basin 内的梯度/Hessian 可以解析计算。

</details>

---

### 8. 为什么要先离散采样再 Newton？

<details>
<summary>答案</summary>

hit identities 和 cluster membership 会随方向跳变，整体目标非凸、非光滑；离散 seeds 用来覆盖多个 basin，Newton/IRLS 只在命中集合固定的局部区域内提案。

</details>

---

### 9. adaptive k-ring 的停止条件是什么？

<details>
<summary>答案</summary>

保留 support 的 frontier 已经脱离当前 ring，或达到了完整 contour anchor coverage；否则扩大两侧 k-ring。当前代码还把 frontier convergence 作为 response 有效性条件。

</details>

---

### 10. 论文的 side restriction 有什么风险？

<details>
<summary>答案</summary>

切向 ray 可能沿表面走很远并触碰远端边界；side mask 抑制这种 over-extension，但论文明确它是 heuristic，没有 correctness 或 global-locality 保证。

</details>

---

### 11. 为什么 IRLS surrogate 不等价于直接最小化论文的 \(J\)？

<details>
<summary>答案</summary>

IRLS 使用固定 hit 集合和加权平方和作为局部二次 proposal；真实 trial 仍要重新 raycast/cull，并用原始 minimax 风格 objective 接受。因此 surrogate 只提供方向提案，不能替代完整离散评价。

</details>

---

### 12. 约束为什么能进入 Newton/PCG？

<details>
<summary>答案</summary>

hit 的 barycentric weights 在一次响应中固定，所以 \(C_h\) 对节点位置是 affine；其 Hessian 为零，平方 penalty 只贡献显式梯度和常数 Gauss–Newton block。

</details>

---

### 13. 论文表 6 的 100% success 是否证明普适性？

<details>
<summary>答案</summary>

不能。它只说明在给定 59-case 数据和参数下 Ours 清掉了 EF；论文和 README 都承认复杂 self-intersection、刚体 hollow/concave 区域可能失败，且没有全局收敛定理。

</details>

---

### 14. 当前代码为什么不能只照抄论文公式复现？

<details>
<summary>答案</summary>

当前 comparator 加了 coverage² normalization，CCR 支持 rest/current 多种模式，response depth 有 3 mm offset/cap，direction optimization 默认 0 次，且 `--backend cuda` 不等于 GPU PRP evaluator。必须记录实际 config 和 commit。

</details>

---

### 15. 若要证明方法改进了“形状”，应增加哪些指标？

<details>
<summary>答案</summary>

至少同时报告 EF、triangle inversion、stretch/bending energy、volume ratio、clearance 和 solver work；只看 EF=0 会把过度拉伸或错误拓扑变化误判为成功。

</details>
