# 自测题：MPM Lite（5 基础 / 5 进阶 / 5 高级）

---

## 基础（Basic）

### 问题 1：MPM Lite 想要回答的核心问题是什么？它的答案是？
<details>
<summary>答案</summary>

核心问题是：**"MPM 的粒子必须是求积点吗？"（Must MPM particles be quadrature?）** 答案是**否**。论文论证：在求解时把粒子用作积分点只是逼近弱形式应力积分的一种选择；如果让粒子忠实地把状态重采样到固定的、网格对齐的胞心求积点上，就能把力装配和时间积分整体搬到网格上，从而让求解复杂度不再与每胞粒子数（PPC）相关。

</details>

---

### 问题 2：传统 MPM 的"两个结构性负担"是什么？
<details>
<summary>答案</summary>

1. **宽支撑的 B-spline 传输核**：用来缓解线性基的穿胞不稳定，但会复杂化并行、让隐式求解器变贵（宽模板）、并使边界条件含糊（B-spline 在节点不具备 Kronecker-delta 性质）。
2. **粒子求积的成本**：力通过粒子求积积分，成本直接正比于 PPC；隐式时每次梯度/Hessian-向量积都触发对全部粒子的 G2P2G 循环，造成性能悬崖。

</details>

---

### 问题 3：MPM Lite 一个时间步的三个阶段分别是什么？
<details>
<summary>答案</summary>

1. **Unload（卸载）**：粒子 → 胞心 → 网格节点，把质量、动量、体积、Kirchhoff 应力卸载到网格（Algorithm 2）。
2. **Integrate（积分）**：在网格节点上求解 $v^{n+1}$，显式或隐式，**纯 FEM，全程不访问粒子**（Algorithm 3）。
3. **Load（装载）**：网格节点 → 胞心 → 粒子，把更新后的速度和速度梯度装载回粒子，再平流位置、更新 $F_p$（Algorithm 4）。

</details>

---

### 问题 4：在 MPM Lite 中，粒子在求解阶段扮演什么角色？它存储哪些量？
<details>
<summary>答案</summary>

粒子退化为**仅携带运动学状态与材料历史的"信使（courier）"**，在求解（Integrate）阶段**完全不参与**。它存储位置 $x_p$、速度 $v_p$、变形梯度 $F_p$、速度梯度信息矩阵 $G_p$，以及材料历史。注意：应力**不**存储在粒子上（已追踪 $F_p$，$\tau_p$ 可就地计算），求解期间也不重新装载应力到粒子。

</details>

---

### 问题 5：MPM Lite 相对显式 MPM 和隐式 MPM 各取得了多少加速？硬件是什么？
<details>
<summary>答案</summary>

- 相对**显式** B-spline MPM：**1.88×**（Jelly Falling，MPM 404.1 s vs MPM Lite 215.4 s，CK-MPM 247.7 s）。
- 相对**隐式** MPM：**最高 15.9×**（扭转 faceless toy，PPC=24 时 MPM Lite 77.1 s vs 隐式 MPM 1225.4 s）。

硬件：**NVIDIA RTX Pro 6000 GPU + Intel Core i9-9980XE CPU**。

</details>

---

## 进阶（Intermediate）

### 问题 6：为什么不能把变形梯度 $F$ 直接平均到胞心？应该重采样什么？
<details>
<summary>答案</summary>

因为 $F$ **既不是强度量也不是广延量**：它把旋转和拉伸纠缠在一起（$F=RS$）。把带不同旋转的粒子的 $F$ 平均，得到矩阵极分解出的拉伸不对应任何真实应变状态，旋转会污染拉伸，导致非物理的力和求解器不稳定。

正确做法是重采样**广延的 Kirchhoff 应力含量 $V\tau$**（公式 6–7）：$V_c=\sum_p w_{cp}V_p$，$\tau_c=\sum_p w_{cp}(V_p\tau_p)/V_c$。它满足守恒 $\sum_c V_c\tau_c=\sum_p V_p\tau_p$，正好是弱形式力项 Riemann 和需要保持的量。

</details>

---

### 问题 7：隐式步需要胞心的变形梯度，但 MPM Lite 不存储胞心 $F$。它如何获得 $F_{\text{base}}$？
<details>
<summary>答案</summary>

通过**旋转无关的拉伸重建**。对各向同性超弹性材料，能量与刚度只通过 $F$ 的拉伸 $S$（奇异值）起作用，旋转 $R$ 无关。所以从已重采样的应力 $\tau_c$ 反向重建拉伸：设 $F_{\text{base}}^c:=S_c$ 使得 $P(S_c)S_c^\top=\tau_c$（公式 17）。具体：$\tau_c$ 的特征向量给主方向 $U$，主拉伸 $\sigma_i$ 由谱关系 $\tau_i=\sigma_i\partial\hat\psi/\partial\sigma_i$ 反解，再组装 $F_{\text{base}}=U\,\mathrm{diag}(\sigma_i)\,U^\top$。丢弃旋转的速度误差仅 $O(\Delta t^2)$/步。

</details>

---

### 问题 8：MPM Lite 的传输为何可以无写竞争地并行？
<details>
<summary>答案</summary>

因为胞心→节点（以及节点→胞心）传输用的是**常数权重** $w_{ic}=2^{-d}$（$d=2$ 或 3），且写成**纯 gather**形式——每个目标位置（节点或胞心）主动收集来自邻居的贡献，而不是每个源向多个目标散射（scatter）。gather 写入彼此独立，没有多个线程写同一地址的竞争，因此天然可并行（Fig. 2）。

</details>

---

### 问题 9：MPM Lite 的隐式增量势优化问题是什么形式？为什么这很重要？
<details>
<summary>答案</summary>

$$\min_v\ \sum_i\tfrac12 m_i\|v_i-v_i^n\|^2\ +\ \sum_c V_c^n\psi(F_c(v)),\quad F_c(v)=(I+\Delta t\,G_c(v))F_{\text{base}}^c.$$
第一项是惯性，第二项是离散弹性能（公式 16）。

重要性：这是一个**标准的六面体 FEM 增量势问题**，意味着任何现成的非线性求解器、预条件子、子空间积分方法都能**开箱即用**——PCG、multigrid-PCG、VBD 等，无需为 MPM 定制求解器。论文实测 VBD 比 Newton-PCG 再快 2×。

</details>

---

### 问题 10：Fig. 5（PPC 加速曲线）说明了什么核心结论？给出数据。
<details>
<summary>答案</summary>

它直观证明了**求解成本与 PPC 解耦**。扭转 faceless toy 实验中，PPC 从 8 增到 24：

| PPC | MPM Lite | 隐式 MPM | 加速 |
|-----|----------|----------|------|
| 8 | 63.5 s | 300.3 s | 4.7× |
| 12 | 68.1 s | 496.1 s | 7.3× |
| 16 | 70.5 s | 726.7 s | 10.3× |
| 20 | 72.8 s | 954.6 s | 13.1× |
| 24 | 77.1 s | 1225.4 s | 15.9× |

MPM Lite 几乎持平（63→77 s），隐式 MPM 近线性暴涨。由于抑制数值断裂常需 PPC≥20，MPM Lite 在"最需要高 PPC"的场景里提速最大。

</details>

---

## 高级（Advanced）

### 问题 11：为什么论文拒绝"速率形式的隐式应力"（Jaumann 更新）？
<details>
<summary>答案</summary>

因为这会破坏隐式优化求解器所需的结构。把 $\tau$ 用 Jaumann 客观率做隐式更新（公式 13），由于其中的**斜对称（skew-adjoint）项**的贡献，内力 Jacobian $\partial f^{\text{int}}/\partial v$ 即使弹性模量 $C$ 对称也**generically 非对称**，并且**不源于任何弹性势能**。基于优化的隐式（增量势最小化）要求 Hessian 对称且对应一个标量势函数，因此速率形式不适用。把 $\tau^{n+1}$ 放进 Jaumann 共旋的全隐式变体也有同样问题。这正是论文转向"旋转无关拉伸参考 + 势能形式"的动机。

</details>

---

### 问题 12：详细说明 StVK-Hencky 的 $\tau\to\sigma_i$ 闭式反演，以及它良定义的条件。
<details>
<summary>答案</summary>

主 Kirchhoff 应力 $\tau_i=2\mu\log\sigma_i+\lambda\sum_j\log\sigma_j$（公式 21）。令 $e_i=\log\sigma_i$，$s=\sum_j e_j=\log J$。

**步骤 1**：对 $i$ 求和：$\sum_i\tau_i=(2\mu+d\lambda)s$，故 $s=\dfrac{\mathrm{tr}\,\tau}{2\mu+d\lambda}$（公式 22）。

**步骤 2**：$e_i=\dfrac{\tau_i-\lambda s}{2\mu}$（公式 23）。

**步骤 3**：指数化 $\sigma_i=\exp\!\Big(\dfrac{\tau_i}{2\mu}\Big)\exp\!\Big(-\dfrac{\lambda}{2\mu}\dfrac{\mathrm{tr}\,\tau}{2\mu+d\lambda}\Big)$（公式 24）。

因为 $\tau_i$ 对 $\log\sigma_i$ 是**线性**的，映射 $e\mapsto\tau$ 一一对应，得到全局闭式反演。**良定义条件**：$\mu>0$ 且 $\lambda>-2\mu/d$（保证 $2\mu+d\lambda>0$），且 $J=\exp(s)>0$ 自动满足（因为是指数）。$\tau$ 的特征向量给出主方向，于是 $F_{\text{base}}=U\,\mathrm{diag}(\sigma_i)\,U^\top$。

</details>

---

### 问题 13：split Neo-Hookean 的 3D 拉伸重建为何需要解三次方程？分支如何选择？
<details>
<summary>答案</summary>

偏-体分裂后 Kirchhoff 应力 $\tau=\mu J^{-2/d}\mathrm{dev}(b)+\alpha(J)I$，$\alpha(J)=\frac{\kappa}{2}(J^2-1)$。**球面部分**（$\mathrm{tr}\,\tau/d=\alpha$）直接定 $J=\sqrt{1+\frac{2}{\kappa}\frac{\mathrm{tr}\,\tau}{d}}$（公式 31）。**偏量部分**给出偏移 $\delta_i=\frac{\tau_i-\alpha}{\mu J^{-2/d}}$（$\sum\delta_i=0$）。

设 $\beta_i=\sigma_i^2=m+\delta_i$，其中 $m=\bar\beta$ 是平均值。约束 $\prod_i\beta_i=J^2$ 给出关于 $m$ 的方程。2D 是 $m^2-\delta^2=J^2$（闭式 $m=\sqrt{J^2+\delta^2}$）。**3D 是三次方程** $m^3+mS_2+S_3-J^2=0$（公式 34，$S_2=\sum_{i<j}\delta_i\delta_j$，$S_3=\delta_1\delta_2\delta_3$）。

**分支选择**：必须选使所有 $\beta_i=m+\delta_i>0$ 的唯一实根（**正拉伸分支**），否则 $\sigma_i=\sqrt{\beta_i}$ 无意义。论文指出该三次方程在选正拉伸分支时保持稳健，用 Cardano 公式求解（附录 C）。

</details>

---

### 问题 14：MPM Lite 如何处理水？它与固体处理的关键区别是什么？
<details>
<summary>答案</summary>

无黏流体没有剪切强度，故**不存全变形梯度 $F_p$**，只追踪体积形变 $J_p=\det(F_p)$（标量）。为保持与应力传输框架一致，定义标量 Kirchhoff 压力：状态方程 $\psi(J)=\frac{\kappa}{2}(J-1)^2$，$\pi(J)=J\psi'(J)=\kappa J(J-1)$（用 Kirchhoff 而非 Cauchy 压力，使量随初始体积 $V_p$ 正确缩放）。

卸载广延压力矩 $\pi_c=\frac{1}{V_c}\sum_p w_{cp}V_p\pi_p$（公式 37），反演出基 Jacobian $J_{\text{base}}^c=\frac{1+\sqrt{1+4\pi_c/\kappa}}{2}$（公式 38）。隐式时 $J_c(v)=\det(I+\Delta t G_c(v))J_{\text{base}}^c$，最小化总势能（公式 39）。更新 $J_p^{n+1}=(1+\Delta t\sum_c w_{cp}\mathrm{tr}(G_c^{n+1}))J_p^n$。

**关键区别**：固体重采样张量 $V\tau$ 并重建拉伸张量 $S_c$；水只重采样**标量**压力含量并重建标量 $J_{\text{base}}$——是固体方案在 1 维（纯体积）情形的特例。

</details>

---

### 问题 15：把 MPM Lite 定位到"网格求积谱系"中，它相对 Sulsky & Gong (2016)、Al-Kafaji (2013)、staggered-grid MPM 做了哪些刻意选择？
<details>
<summary>答案</summary>

MPM Lite **承袭网格求积谱系**（把积分从移动粒子搬到固定网格点，与更新 Lagrangian FEM 接壤），但为图形学实践做了两个刻意选择：

1. **重采样到胞心、用紧致 $Q_1$（多线性）核**——而非 Al-Kafaji 那样在填满胞用 Gauss 求积、近界面回退到粒子求积；也不用 Sulsky & Gong 的 MLS 传输。
2. **力/Jacobian 全在网格装配**，使求解成本与 PPC 无关，同时粒子保留历史载体角色。

与 **staggered-grid MPM（Liang 2019）** 的对比：后者在交错网格上重排 FLIP/PIC 传输、以速率形式演化应力、针对显式；而 MPM Lite 针对 **APIC 风格核、基于变形梯度的超弹塑性、全隐式 FEM 模板**——这也是它必须发展旋转无关拉伸参考（而非速率形式应力）的根本原因。

</details>

---
