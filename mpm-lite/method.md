# 方法详解：MPM Lite

下标约定（全文一致）：
- $p$ = 粒子（particle，Lagrangian 状态载体）
- $c$ = 胞心求积点（cell-center quadrature，固定位置）
- $i$ = 网格节点（grid node，FEM 顶点）

---

## 1. 组件分解

MPM Lite = **粒子重采样模块** + **FEM 积分模块**。一步分三阶段（Algorithm 1）：

```
Unload  : 粒子 → 胞心 → 网格节点   （把状态卸载到网格）
Integrate: 网格节点上求解 v^{n+1}   （纯 FEM，显式或隐式，不碰粒子）
Load    : 网格节点 → 胞心 → 粒子    （把结果装载回粒子）
```

粒子全程只存 $(x_p,v_p,F_p,G_p)$ 与材料历史；$G_p$ 编码速度梯度信息（类 APIC 仿射模式）。

---

## 2. 数据流 ASCII 架构图（load / unload 通信层）

```
                        ┌──────────────────────────────────────────────┐
                        │            一个时间步 Δt                       │
                        └──────────────────────────────────────────────┘

   ┌─────────────┐   Unload (P→C)        ┌──────────────┐   Unload (C→I)      ┌────────────┐
   │  Particles  │  w_cp = N_c(x_p)      │ Cell Centers │  w_ic = 2^{-d}      │ Grid Nodes │
   │     p       │ ───────────────────▶  │      c       │ ─────────────────▶  │     i      │
   │ x,v,F,G,hist│  m_c, (mv)_c, G_c     │ m_c,v_c,G_c, │  m_i,(mv)_i (gather)│ m_i, v_i^n │
   │             │  V_c=Σ w_cp V_p       │ V_c, τ_c     │  纯 gather 无写竞争  │            │
   │             │  τ_c=Σ w_cp V_p τ_p   │              │                     │            │
   │             │        /V_c           │ (每材料 k:   │                     │            │
   │             │  (eqs 1-3, 6-7)       │  V_{c,k},    │   (eqs 8-9)         │            │
   └─────────────┘                       │  τ_{c,k})    │                     └─────┬──────┘
                                         └──────────────┘                           │
                                                                                    │ Integrate
                                                                                    │ 纯 FEM
   ╔═══════════════════════════════════════════════════════════════════════════╗   │ 不碰粒子!
   ║  关键: Integrate 阶段只在 c 与 i 之间往返, 粒子 p 完全不参与。            ║   │
   ║  显式: f_i = -Σ_c V_c τ_c ∇w_ic ; v_i^{n+1}=v_i^n+Δt f_i/m_i (eq 12)      ║   │ 显式 or
   ║  隐式: F_base=S_c (从 τ_c 闭式重建) ; min_v 惯性+Σ_c V_c ψ(F_c(v)) (eq16) ║   │ 隐式优化
   ╚═══════════════════════════════════════════════════════════════════════════╝   │
                                                                                    ▼
   ┌─────────────┐   Load (C→P)          ┌──────────────┐   Load (I→C)        ┌────────────┐
   │  Particles  │  v_p=Σ w_cp v_c       │ Cell Centers │  v_c=Σ w_ic v_i     │ Grid Nodes │
   │   (更新)     │ ◀───────────────────  │      c       │ ◀─────────────────  │     i      │
   │ x_p += Δt v_p│  G_p=Σ w_cp G_c       │ v_c^{n+1},   │  G_c=Σ v_i⊗∇w_ic    │ v_i^{n+1}  │
   │ F_p=(I+ΔtG)F │  (eqs 4-5)            │ G_c^{n+1}    │  (eqs 10-11)        │            │
   └─────────────┘                       └──────────────┘                     └────────────┘
```

**要点**：胞心 $c$ 是粒子与节点之间的**通信中转层**。Integrate 阶段（双框内）是纯网格运算，是隐式加速的来源——无论隐式迭代多少次，都不会触发粒子循环。

---

## 3. 运动学传输（公式 1–5）

### 卸载（P→C，类 APIC）
用多线性核 $w_{cp}=N_c(x_p)$ 把粒子运动学卸载到胞心：

$$m_c^n=\sum_p w_{cp}m_p \tag{1}$$
$$(mv)_c^n=\sum_p w_{cp}m_p\big(v_p^n+G_p^n(x_c-x_p^n)\big) \tag{2}$$
$$G_c^n=\sum_p w_{cp}m_p G_p^n\,/\,m_c^n \tag{3}$$

质量与动量传输与 APIC 一致。公式 3 中 $G_c$ 的**质量加权**使得局部速度场贴合二次 B-spline 重建到 $O(\Delta x^2)$（§3.4）。与 APIC/MLS-MPM 不同，MPM Lite **直接把速度梯度保留在求积点上**，而非每粒子从矩阵反演重建仿射速度。

### 装载（C→P，直接插值）
$$v_p^{n+1}=\sum_c w_{cp}v_c^{n+1} \tag{4}\qquad G_p^{n+1}=\sum_c w_{cp}G_c^{n+1} \tag{5}$$
然后平流 $x_p^{n+1}=x_p^n+\Delta t\,v_p^{n+1}$，更新变形梯度 $F_p^{n+1}=(I+\Delta t\,G_p^{n+1})F_p^n$。

---

## 4. 应力传输（公式 6–7）—— 全文核心

**绝对禁忌**：不要平均变形梯度 $F$（既非强度量也非广延量，混合旋转/拉伸会污染应力 → 非物理力 + 求解器不稳定）。

**正确做法**：重采样**广延的 Kirchhoff 应力含量 $V\tau$**：
$$V_c^n=\sum_p w_{cp}V_p \tag{6}\qquad \tau_c^n=\frac{\sum_p w_{cp}(V_p\tau_p^n)}{V_c^n} \tag{7}$$
满足守恒 $\sum_c V_c^n\tau_c^n=\sum_p V_p\tau_p^n$（弱形式力项的 Riemann 和被保持）。$\tau$ 的对称性可用来节省计算与存储。应力**不**装载回粒子也不存于粒子（已追踪 $F_p$，$\tau_p$ 可就地算）。

---

## 5. 变身有限元（公式 8–11）

卸载后我们有 $\{m_c^n,v_c^n,G_c^n,V_c^n,\tau_c^n\}$。**尽管没有变形梯度**，这已足以在格点上以更新 Lagrangian 六面体 FEM 视角做时间积分。采用质量集中（mass lumping），用 $w_{ic}=N_i(x_c)\equiv 2^{-d}$ 把数据从胞心 gather 到节点：

$$m_i^n=\sum_c w_{ic}m_c^n \tag{8}\qquad (mv)_i^n=\sum_c w_{ic}m_c^n\big(v_c^n+G_c^n(x_i-x_c)\big) \tag{9}$$
$$v_i^n=(mv)_i^n/m_i^n$$

常数权重 → 节点 gather，无写竞争（Fig. 2）。积分后把 $v_i^{n+1}$ 及其梯度采样回胞心：

$$v_c^{n+1}=\sum_i w_{ic}v_i^{n+1} \tag{10}\qquad G_c^{n+1}=\sum_i v_i^{n+1}\otimes\nabla w_{ic}(x_c) \tag{11}$$

其中 $\nabla w_{ic}=\nabla N_i(x_c)=(x_i-x_c)/(2^{d-2}\Delta x^2)$。再经公式 4–5 装载回粒子。

---

## 6. 增量势公式（公式 12–17, 39）

### 显式（公式 12）
$$f_i^n=-\sum_c V_c^n\tau_c^n\nabla w_{ic},\qquad v_i^{n+1}=v_i^n+\Delta t\,f_i^n/m_i^n$$

### 为什么不用速率形式隐式应力（§4.1，反面教材）
诱人但错误：保持应力速率形式，用 Jaumann 客观率让 $\tau$ 成为 $v$ 的隐函数（公式 13）。问题：由于**斜对称项**的贡献，Jacobian $\partial f^{\text{int}}/\partial v$ 即使 $C$ 对称也**generically 非对称**，不源于任何弹性势能 → **不适合基于优化的隐式求解器**。把 $\tau^{n+1}$ 放进 Jaumann 共旋也有同样问题。

### 旋转无关拉伸参考（公式 15–17）
更新 Lagrangian 的试探变形：
$$F_c(v)=(I+\Delta t\,G_c(v))F_{\text{base}}^c \tag{15}$$
后向 Euler 时间步求解标准六面体 FEM 优化（Gast 2015）：
$$\min_v\ \sum_i\tfrac12 m_i\|v_i-v_i^n\|^2\ +\ \sum_c V_c^n\psi(F_c(v)) \tag{16}$$
其中 $F_{\text{base}}^c$ 由旋转无关拉伸参考给出（各向同性 → 能量只依赖拉伸 $S$，旋转无关）：
$$F_{\text{base}}^c:=S_c\ \text{使得}\ P(S_c)S_c^\top=\tau_c^n,\quad P(\cdot)=\nabla\psi(\cdot) \tag{17}$$
附录 B：对各向同性 $\psi$ 该选择客观，丢弃旋转的速度误差仅 $O(\Delta t^2)$/步。

---

## 7. 拉伸重建（$\tau\to\sigma_i\to F_{\text{base}}$）

设 $F=U\Sigma V^\top$，$\Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_d)$，$\psi(F)=\hat\psi(\sigma_1,\dots,\sigma_d)$ 对称。谱形式下主 Kirchhoff 应力（公式 18）：
$$\tau_i=\sigma_i\frac{\partial\hat\psi}{\partial\sigma_i},\qquad \tau=U\,\mathrm{diag}(\tau_1,\dots,\tau_d)\,U^\top.$$
所以：对称 $\tau$ 的特征向量给出主方向 $U$；主拉伸 $\sigma_i$ 由 $d$ 个标量方程反解。组装 $F_{\text{base}}=U\,\mathrm{diag}(\sigma_i)\,U^\top$，由构造满足 $P(F_{\text{base}})F_{\text{base}}^\top=\tau$。

### 7.1 StVK with Hencky（公式 19–24）
能量 $\psi=\mu\sum_i(\log\sigma_i)^2+\frac{\lambda}{2}(\sum_i\log\sigma_i)^2$。主 Kirchhoff 应力（公式 21）：
$$\tau_i=2\mu\log\sigma_i+\lambda\sum_j\log\sigma_j.$$
令 $e_i=\log\sigma_i$，$s=\sum_j e_j=\log J$。对 $i$ 求和（公式 22）：$\sum_i\tau_i=(2\mu+d\lambda)s\Rightarrow s=\frac{\mathrm{tr}\,\tau}{2\mu+d\lambda}$。于是（公式 23–24）：
$$e_i=\frac{\tau_i-\lambda s}{2\mu},\qquad \sigma_i=\exp\!\Big(\frac{\tau_i}{2\mu}\Big)\exp\!\Big(-\frac{\lambda}{2\mu}\frac{\mathrm{tr}\,\tau}{2\mu+d\lambda}\Big).$$
**全局闭式反演**，对所有对称 $\tau$ 成立（$\mu>0$，$\lambda>-2\mu/d$），保证 $J=\exp(s)>0$。

### 7.2 Split Neo-Hookean（公式 25–35）
偏-体分裂：$\psi=\Psi_\mu(J^{-1/d}F)+\Psi_\kappa(J)$，$\Psi_\mu=\frac{\mu}{2}(\mathrm{tr}(F^\top F)-d)$，$\Psi_\kappa=\frac{\kappa}{2}(\frac{J^2-1}{2}-\log J)$（$\kappa=\frac23\mu+\lambda$ 在 3D，$\kappa=\mu+\lambda$ 在 2D）。Kirchhoff 应力（公式 28–29）：
$$\tau=\mu J^{-2/d}\mathrm{dev}(b)+\alpha(J)I,\qquad \alpha(J)=\frac{\kappa}{2}(J^2-1).$$
球面部分定 $J$（公式 31）：$J=\sqrt{1+\frac{2}{\kappa}\frac{\mathrm{tr}\,\tau}{d}}$。偏量部分定偏移 $\delta_i=\frac{\tau_i-\alpha}{\mu J^{-2/d}}$（$\sum_i\delta_i=0$）。令 $\beta_i=\sigma_i^2=m+\delta_i$，由 $\prod_i\beta_i=J^2$ 得 $m$ 的方程：
- **2D**（公式 33）：$m=\sqrt{J^2+\delta^2}$（$\delta=\delta_1=-\delta_2$），闭式。
- **3D**（公式 34–35）：$m^3+mS_2+S_3-J^2=0$（$S_2=\sum_{i<j}\delta_i\delta_j$，$S_3=\delta_1\delta_2\delta_3$），三次方程，选 $\beta_i=m+\delta_i>0$ 的唯一实根（Cardano，附录 C），再 $\sigma_i=\sqrt{\beta_i}$。

### 7.3 Water（§4.5，公式 36–40）
无黏流体不存全 $F_p$，只追踪 $J_p=\det(F_p)$。状态方程 $\psi(J)=\frac{\kappa}{2}(J-1)^2$，标量 Kirchhoff 压力 $\pi(J)=J\psi'(J)=\kappa J(J-1)$（用 Kirchhoff 而非 Cauchy，使量随 $V_p$ 正确缩放）。卸载广延压力矩 $\pi_c^n=\frac{1}{V_c^n}\sum_p w_{cp}V_p\pi_p^n$（公式 37），反演得基 Jacobian（公式 38）：
$$J_{\text{base}}^c=\frac{1+\sqrt{1+4\pi_c^n/\kappa}}{2}.$$
隐式：$J_c(v)=\det(I+\Delta t\,G_c(v))J_{\text{base}}^c$，$\min_v\sum_i\frac12 m_i\|v_i-v_i^n\|^2+\sum_c V_c^n\psi(J_c(v))$（公式 39）。更新 $J_p^{n+1}=(1+\Delta t\sum_c w_{cp}\mathrm{tr}(G_c^{n+1}))J_p^n$（公式 40）。

### 7.4 多材料混合（§4.4）
质量/动量可加、Eulerian 速度单值 → 所有材料共享 $v_c,v_i$；本构按材料分，故在同一胞心保留每材料 colocated 求积 $(V_{c,k},\tau_{c,k})$，各自重建拉伸、各自贡献能量和（公式 16）。**显式无需分开副本**（力 $\propto V\tau$ 可加）。

---

## 8. 算法伪代码（Algorithm 1–4，中文注释）

```
算法 1：MPM Lite 一个时间步
  输入: (x_p^n, v_p^n, F_p^n, G_p^n); 材料 {ψ_k}
  输出: (x_p^{n+1}, v_p^{n+1}, F_p^{n+1}, G_p^{n+1})
  1: (m_i, v_i^n, {V_{c,k}^n, τ_{c,k}^n}, v_c^n, G_c^n) ← Unload(x_p^n, v_p^n, F_p^n, G_p^n, {ψ_k})
  2: v_i^{n+1} ← Integrate((m_i, v_i^n), {V_{c,k}^n, τ_{c,k}^n}, {ψ_k})   # 纯 FEM，不碰粒子
  3: (x_p^{n+1}, v_p^{n+1}, F_p^{n+1}, G_p^{n+1}) ← Load(v_i^{n+1})
```

```
算法 2：Unload 卸载粒子状态
  1: 清空 i 和 c 上的所有数据
  2: for all 粒子 p:
  3:   τ_p^n ← Stress(F_p^n, mat_p)          # 用本构算 Kirchhoff 应力
  4:   k ← mat_p                              # 材料 ID；显式积分用 k=0
  5:   for all 受 p 影响的胞心 c:
  6:     m_c += w_cp m_p
  7:     v_c += w_cp m_p (v_p^n + G_p^n(x_c - x_p^n))   # 动量，含仿射外推
  8:     G_c += w_cp m_p G_p^n
  9:     V_{c,k} += w_cp V_p                  # 广延体积
 10:     τ_{c,k} += w_cp V_p τ_p^n            # 广延 Kirchhoff 应力含量 (关键!)
 11: for all m_c^n ≠ 0 的胞心 c:
 12:   v_c^n /= m_c^n ; G_c^n /= m_c^n        # 归一化
 13:   for all V_{c,k}^n ≠ 0 的材料 k (显式时 M={0}):
 14:     τ_{c,k}^n /= V_{c,k}^n               # 得到胞心 Kirchhoff 应力
 15: for all m_c^n ≠ 0 的胞心 c:               # 胞心 → 节点 gather
 16:   for all i ∈ Corners(c):  w_ic = 2^{-d}
 17:     m_i += w_ic m_c^n
 18:     v_i += w_ic m_c^n (v_c^n + G_c^n(x_i - x_c))
 19: for all m_i^n ≠ 0 的节点 i:  v_i^n /= m_i^n
```

```
算法 3：Integrate（显式或隐式）
  输入: (m_i^n, v_i^n); {(V_{c,k}^n, τ_{c,k}^n)}; {ψ_k}
  输出: v_i^{n+1}
  if 显式:
    f_i ← -Σ_c Σ_k V_{c,k}^n τ_{c,k}^n ∇w_ic        # k=0
    v_i^{n+1} ← v_i^n + Δt f_i / m_i^n
  else:  # 隐式
    for all 胞心 c:
      for all V_{c,k}^n ≠ 0 的材料 k:
        从 τ_{c,k}^n 按 §4.3 重建 S_{c,k}^n            # 旋转无关拉伸重建
    v_i^{n+1} ← argmin_v  Σ_i (1/2) m_i^n ||v_i - v_i^n||^2
                          + Σ_{c,k} V_{c,k}^n ψ_k((I + Δt G_c(v)) S_{c,k}^n)
```

```
算法 4：Load 信息回粒子
  输入: v_i^{n+1}
  for all m_c^n ≠ 0 的胞心 c:
    v_c^{n+1} ← Σ_{i∈Corners(c)} w_ic v_i^{n+1}
    G_c^{n+1} ← Σ_{i∈Corners(c)} v_i^{n+1} ⊗ ∇w_ic
  for all 粒子 p:
    v_p^{n+1} ← Σ_c w_cp v_c^{n+1}
    G_p^{n+1} ← Σ_c w_cp G_c^{n+1}
    x_p^{n+1} ← x_p^n + Δt v_p^{n+1}
    F_p^{n+1} ← (I + Δt G_p^{n+1}) F_p^n
```

---

## 9. 实现陷阱

1. **永远不要平均 $F$。** 用 $V\tau$ 重采样（公式 6–7）。这是正确性的根，不是优化项。
2. **利用 $\tau$ 的对称性。** 只存/算上三角，省内存省计算。
3. **3D 三次方程的分支选择。** Split Neo-Hookean 3D 解 $m$ 时必须选 $\beta_i=m+\delta_i>0$ 的实根（正拉伸分支），否则 $\sigma_i=\sqrt{\beta_i}$ 非法。用 Cardano 时显式判别并筛分支（附录 C）。
4. **质量集中（mass lumping）。** 节点质量用对角化，简化求解（公式 8）。
5. **StVK-Hencky 参数约束。** 需 $\mu>0$ 且 $\lambda>-2\mu/d$ 保证全局可反演且 $J>0$。
6. **gather 方向。** P→C、C→I、I→C、C→P 都尽量写成 gather（每个目标位置主动收集），常数权重 $w_{ic}=2^{-d}$ 无写竞争。
7. **隐式中 $F_{\text{base}}$ 在步首计算一次。** 整个非线性迭代过程 $S_c$ 固定，只有 $G_c(v)$ 随 $v$ 变。
8. **多材料：隐式要每材料独立 $(V_{c,k},\tau_{c,k})$；显式不必。**

---

## 10. 超参数敏感性

- **PPC**：决定抗数值断裂能力（实践 ≥20），但**对 MPM Lite 求解成本几乎无影响**（Fig. 5：63→77 s）。这是核心卖点。传输成本随 PPC 线性增长，但只在 Unload/Load 各一次。
- **$\Delta x$（网格间距）**：决定空间精度与求积分辨率；与 B-spline APIC 的差异是 $O(\Delta x^2)$。
- **$\Delta t$**：旋转-自由参考的误差是每步 $O(\Delta t^2)$；隐式可取较大 $\Delta t$（论文多用 $10^{-3}$ s），显式受 CFL 限制（如 Jelly $6\times10^{-5}$ s，Snow $10^{-4}$ s）。

---

## 11. 复现风险

- **拉伸重建数值稳定性**：StVK-Hencky 闭式稳健；split Neo-Hookean 3D 三次根的分支选择是最易出错处，需仔细处理近退化（$\delta_i\approx 0$）与负判别式情形。
- **应力守恒验证**：实现后应数值检查 $\sum_c V_c\tau_c=\sum_p V_p\tau_p$（见 code/stress_transfer_demo.py），偏差应到机器精度。
- **与 APIC 一致性**：速度/速度梯度差异应为 $O(\Delta x^2)$，可做收敛测试。
- **塑性 return mapping**：StVK-Hencky 的对数应变空间是 Drucker-Prager / Cam-Clay / von Mises return mapping 的载体；选 $\psi$ 主要为闭式 return mapping 而非匹配非线性弹性。
- **VBD/求解器接口**：因为是标准 FEM 增量势，复现时可直接复用现成求解器，但需保证 Hessian 装配在胞心、边界条件在节点施加。
