# 15 个自测问题

## 基础

### 1. cluster、collar 和自由胶带分别做什么？

<details>
<summary>答案</summary>

cluster 让卷内多数材料共享刚体运动；collar 保留释放前沿的弯曲和解除黏附能力；自由胶带保留薄壳自由度，可与新表面接触和黏附。放带时材料从 cluster 经 collar 进入自由区。

</details>

---

### 2. 为什么不能永久刚化整个胶带卷？

<details>
<summary>答案</summary>

放带需要下一段材料弯离卷面并脱离原支撑。永久刚化会禁止这种局部相对变形，所以需要逐步恢复薄壳自由度。

</details>

---

### 3. 四种配置中的两个字母轴表示什么？

<details>
<summary>答案</summary>

Coh/Bond 是胶带–胶带和胶带–hub 的界面选择，F/R 是全薄壳或 cluster+collar 的表示选择。Peel 的桌面界面在四种模型中均为已标定的 cohesive 接口。

</details>

---

### 4. virtual tetrahedron 增加质量或顶点吗？

<details>
<summary>答案</summary>

没有。它复用一个点与一个三角形的四个既有顶点，记录创建时形状，增加一个能量项。其参考体积用于能量权重，不代表加入真实胶层体积或独立运动坐标。

</details>

---

### 5. 论文是否已经证明实时封箱？

<details>
<summary>答案</summary>

没有。文中一次 Coh-R 封箱仿真在 RTX 5090 上以 304.1 秒墙钟推进 128.2 秒仿真，约 0.422× 实时，展示的是连续处理能力。

</details>

---

## 进阶

### 6. 为什么收紧同一个 Coh-F 的容差会改变摆角？

<details>
<summary>答案</summary>

容差决定非线性求解停止时剩余的数值误差。高刚度卷内界面可能使整体运动在实用容差下恢复不足。32 圈时收紧 `1e-2→1e-3 m/s`，角度范围 `9.7°→38.1°`，支持存在数值导致的运动抑制。

</details>

---

### 7. 为什么 bond 断裂力要除去 `Δt²`？

<details>
<summary>答案</summary>

增量势能中的弹性势乘了 `Δt²`。力阈值使用 N，应对物理势能 `E_b/Δt²` 求导。直接对增量势能求导再与固定 N 比较，会把阈值错误地依赖于步长。

</details>

---

### 8. 为什么压缩力很大也不能直接触发这个 release criterion？

<details>
<summary>答案</summary>

论文的 primary force criterion 要求 `d>d_ref`，再要求力模超过阈值。它用于张开断裂；压缩不能仅凭力模触发。滑动时若距离不增加，这条特定判据也不一定断开。

</details>

---

### 9. 为什么前沿候选必须相对旧自由区计算？

<details>
<summary>答案</summary>

若边遍历边把新释放顶点加入邻接判断，一次更新可能递归穿过整个 collar。旧自由区快照使所有候选同步判断，下一层只能留到下一次更新。真实三角网格可在宽度方向同时释放多个邻居，不应误解为每步全局只释放一个顶点。

</details>

---

### 10. 为什么总成功率一致仍可能预测错误？

<details>
<summary>答案</summary>

总率只看成功集合大小，不看成员。Wrap 两圈中 Coh-R 与真实均成功 4/55，但共同成功只有 1 次，其余各有 3 次只在一个域成功。应查看 TP/TN/FP/FN、BA 和正类召回率。

</details>

---

## 高级

### 11. 完整 reduced Hessian 为什么多一个项？

<details>
<summary>答案</summary>

`x(q)` 的旋转参数化非线性。对 `Jᵀg_x` 再求导得到 `JᵀH_xJ`，还得到 `Σ_j(g_x)_j ∇²_q[x]_j`。忽略后者是一种近似，需分析其稳定性和精度，不能当成原式。

</details>

---

### 12. `||FFᵀ−I||²` 单独能排除反射吗？

<details>
<summary>答案</summary>

不能。所有正交矩阵都让它为零，包含 `det(F)=-1` 的反射。这是对势能的数学分析，不等于作者求解器实际会穿越退化状态。其他能量、初态、CCD 和更新限制仍然重要。

</details>

---

### 13. `N=10` 的 collar 是否跨网格保持同一物理模型？

<details>
<summary>答案</summary>

不能直接保证。图距离以边数计，物理宽度随网格尺度和图连接改变。可扫描分辨率并固定材料长度/曲率范围，检验前沿、张力、轨迹与成本。论文没有给出充分的跨网格结论。

</details>

---

### 14. 怎样评价 Coh-R 的 Wrap 两圈 agreement 89.1%？

<details>
<summary>答案</summary>

TP=1、TN=48、FP=3、FN=3。agreement=49/55=89.1%，正类召回 1/4=25%，负类召回 48/51≈94.1%，BA≈59.6%。全失败基线的 agreement 为 51/55≈92.7%，BA 为 50%。Coh-R 有一点正类辨识，但不能称为可靠的两圈成功预测。

</details>

---

### 15. 若把该方法移到 IPC 仿真器，最危险的状态转移错误是什么？

<details>
<summary>答案</summary>

重复或丢失元素质量，恢复顶点速度为零，重设 rest geometry，丢失卷绕时界面历史，遗漏 bond 源对的 pair re-enable，以及把刚体弧线 CCD 当成直线顶点 CCD。它们分别影响惯性、能量、黏附、碰撞和运动连续性，必须独立验证。

</details>

---

数字来源：[论文 v2](https://arxiv.org/html/2609.28766v2) 和 [作者统计 JSON](https://genesis-embodied-ai.github.io/tapesim-website/assets/data/evidence.json)。反射、跨网格与工程问题的回答含本包推导。
