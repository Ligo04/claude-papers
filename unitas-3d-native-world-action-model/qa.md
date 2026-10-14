# 自测题

## 基础题

### 1. UNITAS 说的 3D-native 是什么？

<details>
<summary>答案</summary>

它指 observation token、action flow 和 scene flow 在同一个 metric 3D frame 中对齐。重点是共享物理坐标接口，不是单纯把 RGB 输入换成 RGB-D。

</details>

---

### 2. UNITAS 的三个主要输出变量是什么？

<details>
<summary>答案</summary>

A 是 action flow，u 是可执行 end-effector action chunk，S 是 action-conditioned scene flow。policy mode 只需要 u；simulator mode 还生成 A 和 S。

</details>

---

### 3. 机器人每只手臂和人手分别使用多少个点？

<details>
<summary>答案</summary>

每只机器人手臂从 URDF gripper geometry 采样 20 个点；人手使用 21 个 MANO-derived keypoints。

</details>

---

### 4. UNITAS 在 LIBERO 的平均成功率是多少？

<details>
<summary>答案</summary>

表 1 报告的 LIBERO average 是 99.8%。这是多个 LIBERO suites 的平均，不是单个任务成功率。

</details>

---

### 5. 两个真实机器人任务的成功率是多少？

<details>
<summary>答案</summary>

Organize Books 为 18/20 = 90%，Pack the Ball 为 16/20 = 80%。这是每个任务 20 次试验的结果。

</details>

---

## 中级题

### 6. 为什么 UNITAS 需要 world-aligned positional embedding？

<details>
<summary>答案</summary>

pixel displacement 受相机视角和深度影响，不能稳定表示物理位移。将视觉 token 绑定到 world-frame position 后，模型能把观察、点轨迹和场景变化放进同一 metric reference。

</details>

---

### 7. 没有 depth 时 UNITAS 怎么做空间 grounding？

<details>
<summary>答案</summary>

沿 camera ray 采样 64 个候选深度，范围为 0.1–2.0 m，用视觉 feature 计算 routing 权重，再聚合 world-frame positional features。LIBERO-Plus 上 routed PE 无 depth 为 84.43%，ground-truth depth PE 为 84.68%。

</details>

---

### 8. physical-time tokenizer 解决什么问题？

<details>
<summary>答案</summary>

不同数据源的 frame rate 和 trajectory horizon 不同。UNITAS 用真实 elapsed seconds，而不是 frame index，将每个 point trajectory 编成一个 token，使相同物理时刻在不同数据源中有一致含义。

</details>

---

### 9. scene tokens 为什么不能看 noisy action flow？

<details>
<summary>答案</summary>

训练时 scene flow 应该学习由 clean action flow 产生的环境后果。如果 scene tokens 直接读取 noisy action flow，模型可能利用噪声捷径，破坏 action-conditioned scene prediction 的因果结构。

</details>

---

### 10. 为什么 policy mode 比 world-model mode 快？

<details>
<summary>答案</summary>

policy mode 只 denoise executable action chunk，跳过 future action-flow 和 scene-flow branches。RTX 5090 D 上 action-only 是 55.77 ms/chunk，而 actions + both flows 为 305.49–561.60 ms/chunk，取决于 scene query 数。

</details>

---

## 高级题

### 11. UNITAS 与 PointWorld 的根本区别是什么？

<details>
<summary>答案</summary>

PointWorld 以 externally supplied robot motion 为条件预测 scene flow；UNITAS 同时学习从 instruction 和 observation 生成 executable action，并预测该 action 造成的 scene flow。因此 UNITAS 既是 policy，也是 action-conditioned world model。

</details>

---

### 12. 为什么 action-flow history 能提升 robustness？

<details>
<summary>答案</summary>

当前图像可能无法表达刚刚发生的接触、失败或回退。历史 action flow 提供短时运动上下文，使模型知道当前状态是如何到达的。消融中去掉 history tokens 后，LIBERO-Plus overall 从 85.74% 降到 75.92%，robot initial-state perturbation 尤其明显。

</details>

---

### 13. 99.8% LIBERO 是否证明 UNITAS 已解决真实机器人操作？

<details>
<summary>答案</summary>

不能。LIBERO 是仿真 benchmark；真实实验只覆盖 Organize Books 和 Pack the Ball，各 20 trials。UNITAS 的真实 scene-flow transfer 主要是 qualitative 展示，论文结论应限定在已评测任务和协议。

</details>

---

### 14. 混合数据预训练带来的主要收益是什么？

<details>
<summary>答案</summary>

总体 LIBERO-Plus success 提升 2.6 points；更大的增益出现在 robot initial-state perturbation（84.8% 到 94.2%）和 language perturbation（77.6% 到 85.0%）。这说明跨数据源迁移主要帮助 distribution robustness，而非只提升同分布拟合。

</details>

---

### 15. 如果要扩展 UNITAS，最值得先验证什么？

<details>
<summary>答案</summary>

先做 latency-aware query selection：只保留与当前 contact、uncertainty 和可移动物体相关的 scene queries，验证能否在接近 action-only latency 的成本下保持 action–scene consistency。必须同时报告 success、Moving ADE/FDE、Acc@1cm、query count 和 wall-clock。
</details>

---

