# 自测题

## 基础（1–5）

### 1. S4R 要解决的输入和输出是什么？

<details>
<summary>答案</summary>

输入是已经给定位姿、但三角网格之间可能严重互穿的刚体集合；输出是尽量少改变参考布局、恢复到 full size、通过统一 mesh-level evaluator 的无互穿位姿。它是静态初始化修复，不是带重力和摩擦的动力学模拟。

</details>

---

### 2. 为什么先把物体缩小？

<details>
<summary>答案</summary>

固定各物体的 reference center 后缩小边界，会增大物体之间的 clearance，使初始几何更容易分离。论文默认 `s_min=0.01`，再沿 `s` 逐步恢复到 1；整场景同比缩放不会改变交叠关系。

</details>

---

### 3. `Pen=0` 在论文中具体表示什么？

<details>
<summary>答案</summary>

统一最终 evaluator 对 body pair 的 piecewise FCL score 没有负值，即没有报告 penetrating pair。论文还对 watertight 结果做 containment audit；它不是说布局满足动力学稳定、摩擦、upright 或任务关系。

</details>

---

### 4. 默认每步 QP 优化什么变量？

<details>
<summary>答案</summary>

默认只优化每个 body 的平移增量 `Δp_i`，目标是 `1/2 Σ||Δp_i||²`。6-DOF 版本另外加入小角旋转变量和杠杆臂项，但不是默认路径。

</details>

---

### 5. 论文使用了哪些主要数据集？

<details>
<summary>答案</summary>

Kubric、HY3D-Bench 和 Thingi10K。主比较使用 Kubric，跨数据集实验覆盖 HY3D-Bench 与 Thingi10K；主表做到 `N=5000`，图 5 进一步展示到 `N=30000` 的大场景。

</details>

---

## 中级（6–10）

### 6. 直接在深穿透状态线性化为什么不可靠？

<details>
<summary>答案</summary>

最近特征可能靠近 medial axis 或 feature-switch 区域，接触方向会突变；即使在平滑分支上，大修正也会放大 Taylor 二阶余项。一次线性化既可能推错方向，也可能留下 residual，所以 S4R 用多个浅接触子问题替代一次深修正。

</details>

---

### 7. `E_ij` 和 frozen-witness gap predictor 各自做什么？

<details>
<summary>答案</summary>

`E_ij=e_i(n_ij)+e_j(-n_ij)` 是两物体沿接触方向的单侧支持宽度，用来保守估计 `ds` inflation 会闭合多少投影间隙。预测式 `ℓ≈d̃+nᵀ(Δp_j−Δp_i)−dsE` 把当前 gap、平移和尺度闭合写进一条线性约束；它不是下一次真实 mesh score 的恒等式。

</details>

---

### 8. SOI event jump、cache 和 tail refinement 的分工是什么？

<details>
<summary>答案</summary>

SOI 用包围球推断固定姿态下不会马上接触的尺度区间，减少检测；cache 在 full detection 之间传播冻结 witness 的 gap 预测，默认每 `M=3` 步刷新；tail 在 `s=1` 重新检测并对 penetrating 和 near-contact 邻居一起解 correction QP，清理主循环留下的 residual。

</details>

---

### 9. 为什么 tail 要包含已经分离但靠近的 pair？

<details>
<summary>答案</summary>

只修当前穿透 pair 可能把某个 body 推进邻居，产生 domino effect。把 `0≤d̃<d_hat` 的 near-contact 也放进 full-active-set QP，可以预留局部间隙，减少一次修复制造另一次穿透。

</details>

---

### 10. 论文的速度优势主要来自哪里？

<details>
<summary>答案</summary>

主要来自减少昂贵的 mesh proximity 查询和利用接触稀疏性、event skip、cache 与 warm start；Appendix 的时间分解显示 detection 加 tail 约占总耗时的 73–97%，QP 通常不超过 1%。因此单纯替换 QP backend 不是首要优化方向。

</details>

---

## 高级（11–15）

### 11. 为什么逐步 minimum-norm 不等于全局最小最终 RMSD？

<details>
<summary>答案</summary>

每个 QP 只在当前 active set、当前 witness 和当前尺度下最小化本步增量；早期增量会改变后续最近特征、接触图和可行通道。增量的累计和没有全局 objective 的 joint optimality 条件，路径历史可能留下 bias。论文报告 RMSD 随 N 近似平坦是实验现象，不是由局部 QP 推出的定理。

</details>

---

### 12. Eq.(2) 的包围球条件证明了什么、没有证明什么？

<details>
<summary>答案</summary>

它给出一个充分的起始尺度条件：若参考中心距离减去 margin 大于两个缩放包围半径之和，包围球不会相交。它不证明非凸 mesh 在之后每一个中间尺度都有可行路径，也不证明 full-size 能到达；近中心会被确定性扰开并计入 RMSD。

</details>

---

### 13. 如何解释论文“成功”与当前代码 `status=converged` 的边界？

<details>
<summary>答案</summary>

论文主表的成功是统一 evaluator 在预算内 `Pen=0`。当前 README 把条件写得更显式：必须 continuation 到 `s=1`、pose 有限、full-size shared evaluator 无 penetrating pair，且墙/关节约束满足；GPU 还要求 sampled oracle 后的 exact FCL recheck/correction 通过。中间尺度干净或 solver 自己返回不等于 converged。

</details>

---

### 14. 为什么博客说 Coverage Axis dilation 和 S4R scaling 不能混为一谈？

<details>
<summary>答案</summary>

Coverage Axis 在形状抽象阶段用 `r'_i=r_i+δ` 改变候选球的覆盖关系，选中后保留球恢复原半径，表面不被移动；S4R 固定 reference centers，实际缩放 body 以改变 clearance，并把尺度当作求解路径。前者的 δ 是表示容差，后者的 s 是几何修复参数；两者的单位、单调性和验收目标不同。

</details>

---

### 15. 设计一个能暴露 S4R 失败边界的实验。

<details>
<summary>答案</summary>

固定同一 mesh pool 和 seed，逐渐提高容器 packing fraction，同时加入 nested cavity、近重合 centers、薄长非凸件和不同 cache refresh `M`。记录每个尺度的 QP status、active-pair 数、full-size Pen/maxPen、tail stop reason、RMSD 与阶段耗时。将 `continuation_complete`、`native_converged`、最终 exact evaluator 和后续 MuJoCo settle 分开报告，避免把几何路径堵塞误判为物理失败。

</details>

---
