# 自测题：5 基础 + 5 中级 + 5 高级

## 基础题

### 1. UniVTAC 试图解决的两个基础设施瓶颈是什么？

<details>
<summary>答案</summary>

第一是真实世界触觉数据昂贵、传感器不统一、难以大规模覆盖丰富接触状态；第二是缺少统一且物理有效的视触觉操作 benchmark，导致策略和表示难以公平比较。UniVTAC 用仿真数据生成、encoder 预训练和八任务评测把两者放进同一流程。

</details>

---

### 2. 论文的三个主要贡献分别是什么？

<details>
<summary>答案</summary>

一是支持三类视触觉传感器、可控接触数据和自动操作 API 的 UniVTAC Platform；二是用形状、接触、位姿多路径监督预训练的 UniVTAC Encoder；三是包含八项任务并补充真机验证的 UniVTAC Benchmark/实验体系。

</details>

---

### 3. 论文列出的三类传感器和五个原子操作 API 是什么？

<details>
<summary>答案</summary>

传感器是 GelSight Mini、ViTai GF225 和 Xense WS；API 是 Grasp、Move、Place、Probe、Rotate。论文把光学/机械差异映射到相机内参、gelpad 网格和渲染方式，把轨迹规划与触觉反应式接触控制组合起来。

</details>

---

### 4. UniVTAC Encoder 的 Shape、Contact、Pose 三条路径各预测什么？

<details>
<summary>答案</summary>

Shape 重建带 marker 与无 marker 的 RGB 图；Contact 预测 gelpad 深度图和 marker 的 2D 位置；Pose 回归 7 维相对位姿（3D 平移 + 4D 四元数）。它们分别对应全局形状、局部接触形变和外部度量空间定位。

</details>

---

### 5. UniVTAC Benchmark 有多少项任务，覆盖哪些感知维度？

<details>
<summary>答案</summary>

共有八项：Lift Bottle、Pull-out Key、Lift Can、Put Bottle in Shelf、Insert Hole、Insert HDMI、Insert Tube、Grasp Classify。它们覆盖 pose reasoning、shape perception 和 contact-rich interaction 三个维度。

</details>

---

## 中级题

### 6. 为什么数据生成要故意加入随机失败和接触后的纠正？

<details>
<summary>答案</summary>

一次成功且无碰撞的 expert 轨迹通常没有有意义的接触变化，触觉策略学不到错位、局部接触和修正动作。随机失败让轨迹包含接触不确定性，expert 再根据仿真中的接触反馈纠正，使数据更接近真实操作；穿透和明显滑移仍会被 invalid 规则排除。

</details>

---

### 7. 抓取阶段的自适应速度方程如何工作？

<details>
<summary>答案</summary>

当 `d_min=d_max`（无接触）时使用 `v_fast` 快速靠近；当 `d_min<d_max`（已经接触）时使用 `min(|d_min−δ_th|, v_slow)`，限制闭合速度并朝目标压入阈值靠近。它是生成数据时的安全控制器，用于避免非物理穿透，不是下游策略的学习损失。

</details>

---

### 8. 论文的多任务损失是什么，权重如何设置？

<details>
<summary>答案</summary>

`L_shape = MSE(Î_marked,I_marked)+MSE(Î_pure,I_pure)`，`L_contact = MSE(D̂,D)+MSE(M̂,M)`，`L_pose=MSE(p̂,p)`，总损失为 `λ_s L_shape + λ_c L_contact + λ_p L_pose`。实验设置 `λ_s=1.0`、`λ_c=0.5`、`λ_p=0.5`。

</details>

---

### 9. 为什么训练时有多个 decoder，部署时却只保留 encoder？

<details>
<summary>答案</summary>

多个 decoder 给共享 latent 施加互补的物理归因约束，使它同时保留形状、接触和位姿信息。部署策略只需要这个 tactile-centric latent；丢弃 RGB/depth/marker/pose decoder 可以避免额外推理开销，也不需要在真实传感器上获得仿真特权标签。

</details>

---

### 10. 仿真主实验的比较对象、训练量和平均结果是什么？

<details>
<summary>答案</summary>

比较 ACT（无触觉）、VITaL 和 ACT + UniVTAC Encoder；每项任务使用 50 条自动采集完整轨迹训练，并做 100 次测试 rollout。八任务平均成功率分别是 30.9%、40.5% 和 48.0%，因此相对 vision-only 的绝对提升是 17.1 个百分点。

</details>

---

## 高级题

### 11. 双视图 RGB 重建如何帮助模型把物体形状与传感器伪影分开？

<details>
<summary>答案</summary>

带 marker 图像包含物体外观、gelpad 形变、marker 纹理和光照；无 marker 的 counterfactual 图只保留更接近接触本身的外观。共享 latent 必须同时生成两种视图，因此不能只记住 marker 的固定纹理来降低像素误差，必须提取能跨视图解释的物体几何。它仍是软性的归因偏置，不是严格的可辨识性证明。

</details>

---

### 12. 表 II 为什么能支持“Pose 路径有独立价值”，但不能证明它在所有任务都最重要？

<details>
<summary>答案</summary>

在同一 ACT 设置下，Contact+Shape 平均 43.9%，Full（再加入 Pose）达到 48.0%，且 Lift Bottle 等需要空间关系的任务改善明显，这支持 Pose 在该实验中的额外贡献。但这是固定八任务、固定数据和固定策略的消融；不同机器人、动作空间或任务可能需要不同的路径权重，不能外推为普遍排序。

</details>

---

### 13. 数据规模结果应如何解释，为什么不能简单说“数据越多每个任务越好”？

<details>
<summary>答案</summary>

八任务平均值从 34.2%（0）升到 48.0%（200K），说明总体上更大、更丰富的仿真预训练有益。但逐任务表格存在波动，例如部分任务在 100K 相比 50K 回落；数据分布、任务难度和有限 rollout 方差都会影响单任务曲线。因此结论应是平均扩展趋势，而不是每个任务严格单调的因果定律。

</details>

---

### 14. 真机实验为什么提供了 sim-to-real 证据，又有哪些重要混杂因素？

<details>
<summary>答案</summary>

Encoder 只用仿真合成触觉数据预训练，部署到 Marvin + 双 ViTai GF225 时不再微调；三项任务平均从 vision-only 的 43.3% 到 vision + UniVTAC 的 68.3%，说明表征可能捕捉了跨域共享的接触/位姿线索。可是每项只有 150 条示范和 20 次 rollout，成功由人工判定，策略按任务分别训练，真机只覆盖一种机器人和一种传感器，所以不能据此证明跨硬件、跨物体或长期部署的普适迁移。

</details>

---

### 15. 如果要独立复现论文，最小实验计划和必须声明的缺口是什么？

<details>
<summary>答案</summary>

先在公开仓库中核对 HDF5 字段和 encoder 头，使用少量合成样本复现三类 MSE 目标与路径消融，再逐步进入 Isaac Sim/TacEx 的八任务策略评测，最后才做真机实验。必须声明：完整数据采集、随机种子、环境和硬件都很重；仓库快照的采集/评测目前只注明支持 GelSight Mini；在线数据集 viewer 有 schema CastError；README 与 LICENSE 的许可证说法不一致。因此只跑标准库概念 demo 或单个 encoder 训练，不能写成“复现了论文成功率”。

</details>
