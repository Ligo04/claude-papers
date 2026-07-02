# 方法详解 (Method)

本文件逐组件拆解方法，含数学（式 2–12）、Algorithm 1 伪代码、无穿透 CCD 合并流程、$\rho$ 自适应、PD 负载均衡，以及复现陷阱与超参敏感性。

---

## 0. 组件总览

```
                          ┌─────────────── 控制器 (controller) ───────────────┐
                          │  编排帧级命令 · 聚合 worker 上报 · 全局收敛判定      │
                          └───────▲───────────────────────▲────────────────────┘
                                  │ 上报(Δq,r,s,TOI)        │ 广播信号 σ
              ┌───────────────────┴──────┐      ┌───────────┴──────────────────┐
              │  worker 1 (GPU)          │◄────►│  worker 2 (GPU)               │
              │  · 局部 Newton 解 (式5)  │ 邻居  │  · 局部 Newton 解 (式5)       │
              │  · 共识 z (式6) 去中心化 │ 通信  │  · 对偶 u (式7)               │
              │  · CCD on merge target   │      │  · 残差驱动 ρ (式12)          │
              └──────────────────────────┘      └───────────────────────────────┘
```

五大组件：(1) 分区/共享体模型；(2) 共识 ADMM 求解器（式 2–10）；(3) 无穿透 CCD 合并（§4.2）；(4) 质量感知 + 残差驱动 ρ（§4.3，式 11–12）；(5) PD 负载均衡（§5.2，式 13–14）。

---

## 1. 分区模型与共享体/重叠定义

- $N$ 个 worker，$\mathcal W = \{1,\dots,N\}$；worker $i$ 负责子区域 $\Omega_i$。
- 物体集 $\mathcal B = \{b_1,\dots,b_B\}$。worker $i$ 维护两类：**内部物体** $\mathcal B_i$ 和**与邻居 $j$ 共享的物体** $\mathcal B_{ij}$。
- 相邻 worker $i,j$（$j \in \mathcal N_i$）的界面记为 $\Gamma_{ij}$，其周围设**共享重叠区** $\Omega_{ij}$。
- 共享物体 = **界面变量**，其状态被复制到多个 worker 并通过共识协调。
- worker $i$ 的全部共享体：$\mathcal B_{i,\partial} := \bigcup_{j \in \mathcal N_i} \mathcal B_{ij}$；全局共享体：$\mathcal B_\partial := \bigcup_i \mathcal B_{i,\partial}$。
- **重叠宽度**：$w = \max(\alpha v_{\max} dt,\ w_{\min})$，$\alpha = 2$。第一项（$\alpha v_{\max} dt$）防止一个时间步内物体越过界面被漏检；第二项（$w_{\min}$ = 最小物体包围盒对角线）防止低速时共享区过窄。
- 邻居关系对称（$j \in \mathcal N_i \Leftrightarrow i \in \mathcal N_j$），$i \notin \mathcal N_i$。

### 两 worker 分区 ASCII 图（对应图 2）

```
        Ω₁ (worker 1, 橙色内部体)        Ω₂ (worker 2, 绿色内部体)
   ┌──────────────────────────┬──────────────────────────┐
   │  ●b1    ●b2               │                ●b6   ●b7 │
   │             ●b3           │          ●b8             │
   │                ╎◄── 共享重叠区 Ω₁₂ ──►╎              │
   │            ┌───┼──────────┼──────────┼───┐           │
   │            │   ╎  ◆bs1     ╎    ◆bs2  ╎   │  ← 共享体 │
   │            │   ╎ (蓝/橙双副本)        ╎   │   (双副本) │
   │            └───┼──────────┼──────────┼───┘           │
   │      ●b4       │          ║          │       ●b9     │
   │                │          ║          │              │
   └────────────────┴──────────╫──────────┴───────────────┘
                                ║
                       界面 Γ₁₂ (红色, 平面)
              ← 只能沿法向平移 →  (PD 控制器据此做负载均衡)

   图例: ● 内部体(单 worker 持有)   ◆ 共享体(被两 worker 复制)
         共享体 bs 在 worker1 上是 q_{1,bs}(蓝), 在 worker2 上是 q_{2,bs}(橙),
         二者通过共识变量 z_bs 协调一致。
```

---

## 2. 单步 ABD 优化（预备，式 1）

$$
q_{t+1} = \arg\min_q\ \tfrac{1}{2}\|q - \tilde q\|_M^2 + h^2 E_{\text{ARAP}}(q) + h^2 E_{\text{IPC}}(q),
\qquad \tilde q = q_t + h\dot q + h^2 M^{-1} f_{\text{ext}}.
$$

- $q$：广义坐标（仿射体 12-DoF/体堆叠）；$M$：仿射体质量矩阵；
- $E_{\text{ARAP}}$：弹性能（惩罚仿射矩阵偏离正交，体现近刚性）；
- $E_{\text{IPC}}$：碰撞障碍能；$\tilde q$：基于惯性 + 外力的预测位置；$h$：时间步长。

---

## 3. 共识 ADMM 求解器（式 2–10）

### 3.1 共识优化问题（式 2）

$$
\min_{\{q_i\},\{z_b\}}\ \sum_{i=1}^N F_i(q_i),\qquad \text{s.t. } q_{i,b} - z_b = 0,\ \forall i,\ \forall b \in \mathcal B_{i,\partial}.
$$

约束 $q_{i,b} = z_b$ 强制所有持有共享体 $b$ 的副本与公共共识变量 $z_b$ 一致。

### 3.2 局部目标与复制计数归一化（式 3）

$$
F_i(q_i) = \sum_{b \in \mathcal B_i \cup \mathcal B_{i,\partial}} \frac{1}{\kappa_b}\!\left[\tfrac{1}{2}\|q_{i,b}-\tilde q_{i,b}\|^2_{M_{i,b}} + h^2 E_{\text{ARAP}}(q_{i,b})\right] + h^2 \sum_{c \in \mathcal C_i} \frac{1}{\kappa_c} \phi_c(q_i).
$$

- $\kappa_b$ = 持有 $b$ 副本的 worker 数；$\kappa_c$ = 包含接触对 $c$ 的 worker 数；$\phi_c$ = 接触对 $c$ 的 IPC 障碍项；$\mathcal C_i$ = worker $i$ 上的接触对集。
- **关键**：朴素求和 $\sum_i F_i$ 会对复制体/跨分区接触**重复计数**；除以 $\kappa_b, \kappa_c$ 保证每个能量项对全局目标 $\sum_i F_i$ **恰好贡献一次**，与单 worker 表述一致。

### 3.3 缩放形式增广拉格朗日（式 4）

$$
\mathcal L = \sum_{i=1}^N \Bigg(F_i(q_i) + \sum_{b \in \mathcal B_{i,\partial}} \Big[\tfrac{\rho_{i,b}}{2}\|q_{i,b} - z_b + u_{i,b}\|^2 - \tfrac{\rho_{i,b}}{2}\|u_{i,b}\|^2\Big]\Bigg),
$$

$u_{i,b}$ 为缩放对偶变量，$\rho_{i,b}$ 为惩罚参数。

### 3.4 ADMM 三步交替

**(a) 局部求解（式 5）** —— 每个 worker 极小化 $F_i$，并用 $\rho$ 加权的二次项把每个共享体软锚定到当前共识 $z_b^k$：
$$
q_i^{k+1} = \arg\min_{q_i}\ F_i(q_i) + \sum_{b \in \mathcal B_{i,\partial}} \tfrac{\rho_{i,b}}{2}\|q_{i,b} - z_b^k + u_{i,b}^k\|^2.
$$
（实现上是局部 Newton 解，含 IPC 障碍 + CCD 线搜索，产出无穿透 $q_i^{k+1}$。）

**(b) 共识更新（式 6，去中心化）** —— 对每个共享体 $b$ 聚合各 worker 的 $\{q_{i,b}^{k+1}+u_{i,b}^k\}$：
$$
z_b^{k+1} = \arg\min_{z_b} \sum_{i \in \mathcal W_b} \tfrac{\rho_{i,b}}{2}\|z_b - (q_{i,b}^{k+1}+u_{i,b}^k)\|^2
= \frac{\sum_{i\in\mathcal W_b}\rho_{i,b}(q_{i,b}^{k+1}+u_{i,b}^k)}{\sum_{i\in\mathcal W_b}\rho_{i,b}},
$$
$\mathcal W_b := \{i \in \mathcal W \mid b \in \mathcal B_{i,\partial}\}$。**无需中央 worker**：每个 $i \in \mathcal W_b$ 与持有 $b$ 副本的邻居交换 $\{q_{j,b}^{k+1}, u_{j,b}^k, \rho_{j,b}\}$ 后本地算出**相同**的 $z_b^{k+1}$。

**(c) 对偶更新（式 7）** —— $u_{i,b}^{k+1} = u_{i,b}^k + q_{i,b}^{k+1} - z_b^{k+1}$。

### 3.5 停止判据（式 8–10）

三个度量（都用 $\ell_\infty$ 范数、按场景尺度 $l$ 与 $h$ 归一化）均 $< \theta$ 才停止：
$$
\tfrac{1}{hl}\|\Delta q_i^{k+1}\|_\infty < \theta,\quad \tfrac{1}{hl} r^{k+1}_\infty < \theta,\quad \tfrac{1}{hl} s^{k+1}_\infty < \theta.
$$
- **Newton 更新** $\|\Delta q_i^{k+1}\|_\infty$：局部解的位移变化；
- **原始残差**（式 9，共识失配）：$r_{i,b}^{k+1} := q_{i,b}^{k+1} - z_b^{k+1}$，$\ r^{k+1}_\infty := \max_{b\in\mathcal B_\partial}\max_{i\in\mathcal W_b}\|r_{i,b}^{k+1}\|_\infty$；
- **对偶残差**（式 10，合并共识状态的变化）：$s_b^{k+1} := z_b^{k+1} - z_b^k$，$\ s^{k+1}_\infty := \max_{b\in\mathcal B_\partial}\|s_b^{k+1}\|_\infty$。

收敛后，把所有局部副本 $q_{i,b}$ 替换为对应 $z_b$ 完成合并（须先通过 §4.2 的 CCD 检查）。

---

## 4. 无穿透 CCD 合并流程（§4.2，逐步）

> 动机：局部解 $q_i$ 无穿透，但共识 $z$（加权平均）可能互穿。仅在终止时认证可行性。

逐步过程（每次满足数值停止判据时执行）：

1. 局部求解得无穿透 $q_i^{k+1}$（式 5）；共识步得共享体估计 $z^{k+1}$（式 6）。
2. **构造候选合并状态** $\hat q_i^{k+1}$：把 $q_i^{k+1}$ 中**每个**共享体 $b \in \mathcal B_{i,\partial}$ 替换为 $z_b^{k+1}$（内部体保持不变）。
3. **沿轨迹做 CCD**：在 $q_i^{k+1} \to \hat q_i^{k+1}$ 的线性插值轨迹上跑连续碰撞检测，得到该 worker 的 $\text{TOI}_i$（time of impact，1 表示整条轨迹无穿透）。
4. **全体认证**：仅当**所有** worker 都报告 $\text{TOI}=1$ 时，才接受收敛并令 $q_{t+1} = z$；否则不接受，继续 ADMM 迭代。

```
   q_i^{k+1} (局部无穿透)  ──CCD──►  ĥq_i^{k+1} (用 z 替换共享体)
        worker1: TOI=1 ✓ ─┐
        worker2: TOI=1 ✓ ─┴─► 全部 TOI=1  ⇒  接受 q_{t+1}=z (无穿透)
        worker_j: TOI<1 ✗ ───► 任一失败    ⇒  拒绝, 继续迭代 (必要时缩小 dt)
```

### 角落情形与兜底（图 4）

人为构造（水平力达 586× 物体自重，把两副本推到障碍物两侧），合并直线路径被障碍挡住、CCD 反复失败。**自适应时间步**缓解：$dt=0.02\text{s}$（被挡）→ $dt=0.015\text{s}$（副本可绕过障碍逐步靠拢、最终合并）。其余所有实验未见 CCD 诱导的不收敛。

---

## 5. 质量感知 + 残差驱动 ρ 自适应（§4.3，式 11–12）

### 5.1 为什么不用 Hessian 全谱（式 11）

$$
H_{i,b} = M_{i,b} + H^{\text{ARAP}}_{i,b} + H^{\text{IPC}}_{i,b}.
$$
$\lambda_{\max}(H_{i,b})$ 或 $\text{mean}(\lambda)$ 会**严重高估**有效刚度：
- ARAP 最大特征值来自**三个缩放模态**，$E=10^{12}$ 时可达 $10^8$，主导 $\lambda_{\max}$；
- IPC 对数障碍在两体接近时特征值**爆炸**，且只惩罚改变距离 $d$ 的方向，曲率大但高度**瞬态**（接触对随距离在阈值 $\hat d$ 附近反复激活/失活）。

### 5.2 质量感知初始化

剔除 ARAP、IPC，只用惯性块 $M_{i,b}$。在 ABD 中 $\lambda_{\max}(M_{i,b}) \le m_b$，故用物体质量作刚度代理：
$$
\rho_{i,b} = \beta\, m_b,\qquad \beta = 1\ \text{(默认)}.
$$

### 5.3 残差驱动逐物体自适应（式 12，遵循 Boyd 2011）

$$
\rho_b^{k+1} = \begin{cases}
\tau\,\rho_b^k, & \|r_b^k\|_\infty > \mu\,\|s_b^k\|_\infty,\\[2pt]
\rho_b^k/\tau, & \|s_b^k\|_\infty > \mu\,\|r_b^k\|_\infty,\\[2pt]
\rho_b^k, & \text{otherwise}.
\end{cases}
$$
- 原始残差大于对偶残差 $\mu$ 倍 → 共识太弱 → 放大 $\rho$；反之缩小；
- $\tau = 2$，$\mu = 5$；
- **钳制**：$\rho_b \in [\sigma_{\min}\rho_0,\ \sigma_{\max}\rho_0]$，$\sigma_{\min}=0.001$，$\sigma_{\max}=1000$。
- 效果：各密度等级下 ADMM 迭代数减少 **22%–80%**。

> 实践提示（见 `code/consensus_admm_demo.py`）：乘性更新若**每次迭代**触发会让 $\rho$ 来回振荡；应**周期性**触发（demo 中每 5 次迭代调整一次），与 Boyd 的实践一致。

---

## 6. Algorithm 1：worker $i$ 上的共识 ADMM（可读伪代码）

```
输入: 邻居集 N_i, 共享体集 B_{i,∂}, 最大迭代 K

# --- 初始化帧 ---
1.  从邻居 j∈N_i 拉取 {M_b, id_b, q_b, q̇_b}，b∈B_{i,∂}
2.  构建局部数据与预测位置 q̃_i
3.  z_i^1 ← q̃_i ;  u_i^1 ← 0                       # 初始化 ADMM

# --- ADMM 主循环 ---
4.  for k = 1 .. K:
5.      if k > 1:
6.          拉取共享体量 {q_{j,b}^k, ρ_b^k, u_{j,b}^k}, j∈N_i, b∈B_{ij}
7.          用式(6)更新 z_i^k, 用式(7)更新 u_i^k        # 共识 + 对偶
8.          评估 ‖Δq_i^{k+1}‖_∞, r_i^k, s_i^k, 以及 TOI_i^k  # 停止度量
9.          向控制器上报 (‖Δq^{k+1}‖_∞, r, s, TOI)        # 控制器通信
10.         接收信号 σ
11.         if σ = end:
12.             break
13.         更新 ρ_b^{k+1}, b∈B_{i,∂}                    # 式(12) 惩罚自适应
14.     求解式(5)得 q_i^{k+1}                            # 局部 Newton 解
15. 赋予收敛后的共享体状态, 结束本帧
```

控制器侧：聚合所有 worker 上报，检查全局停止判据（含全员 TOI=1），广播信号 σ。

---

## 7. PD 负载均衡控制器（§5.2，式 13–14）

每个边界平面 $\Gamma_{ij}$ 参数化为点 $p_{i,j}$ + 单位法向 $n_{i,j}$（从 $i$ 指向 $j$）。两 worker 情形：设 $\tau_i^k$ = worker $i$ 在帧 $k$ 的平滑计算时间，$\eta^k = \tau_1^k/\tau_2^k$，失衡度
$$
T^k = \frac{\eta^k - 1}{\eta^k + 1} \in (-1, 1),
$$
$T^k=0$ 为完美均衡，符号指示哪个 worker 更慢。PD 更新
$$
\Delta p^k_{1,2} = K_P T^k + K_D(T^k - T^{k-1}),
$$
沿法向平移边界：$p_{1,2} := p_{1,2} + \Delta p^k_{1,2}\, n_{1,2}$，把负载在两 worker 间重新分配。

---

## 8. 实现陷阱与复现风险

### 8.1 必须注意的陷阱

- **复制计数归一化 $\kappa_b/\kappa_c$**：忘记除会**重复计数**复制体/跨分区接触，使全局目标与单 worker 不一致——这是分布式正确性的核心，不是可选优化。
- **去中心化 z 更新**：式 6 不需要中央节点；若误实现为"汇总到一个 worker 再广播"，会引入瓶颈且与论文架构不符。每个持有 $b$ 的 worker 应本地算出相同 $z_b$。
- **CCD 角落情形**：合并路径被挡导致反复 CCD 失败时，必须有自适应缩小 $dt$ 的兜底，否则可能不收敛。
- **$\rho$ 钳制**：残差驱动自适应必须钳制到 $[\sigma_{\min}\rho_0, \sigma_{\max}\rho_0]$，否则乘性更新会让 $\rho$ 漂移到数值不稳定区。
- **$\rho$ 自适应触发频率**：周期性触发而非每步触发，避免振荡。
- **停止判据归一化**：三个度量都要按 $\tfrac{1}{hl}$ 归一化（场景尺度 $l$、步长 $h$），否则 $\theta$ 在不同场景/步长下不可比。

### 8.2 复现风险

- **CCD 子程序**：论文复用 ABD 线性顶点轨迹的高效 CCD；若用通用 CCD，"沿合并轨迹做一次 CCD"的成本可能不再可忽略。
- **局部 Newton 解的细节**（IPC 障碍处理、线搜索、Hessian 投影）依赖 ABD 原文 [Lan 2022a]，论文未在正文展开。
- **PD 增益 $K_P, K_D$**、场景尺度 $l$、平滑时间窗等在正文未给具体值（部分在补充材料）。
- **$\lambda_{\max}(M_{i,b}) \le m_b$ 的推导**在补充材料，正文只给结论。
- **通信开销** ($t_{\text{sync}}$) 强依赖 1 Gb/s 阻塞网络，不同网络下加速比会变。

---

## 9. 超参敏感性速查

| 超参 | 默认 | 含义 | 敏感性 |
|---|---|---|---|
| $\beta$ | 1.0 | 质量感知系数 $\rho_0=\beta m_b$ | $\beta=1$ 给最低/近最低迭代；$\beta=0.1$ 也好；过小（如最轻密度 + $\beta=0.01$）共识太弱、收敛慢 |
| $\tau$ | 2 | $\rho$ 乘/除因子 | 控制自适应步幅，过大易振荡 |
| $\mu$ | 5 | 残差不平衡阈值 | 大 $\mu$ 减少对残差噪声的敏感（更少触发自适应） |
| $\theta$ | 场景相关（耦合例 $0.0008$） | 停止阈值 | 越小精度越高但迭代越多（长尾） |
| $\sigma_{\min},\sigma_{\max}$ | $0.001, 1000$ | $\rho$ 钳制下/上界（相对 $\rho_0$） | 防止数值不稳定 |
| $\alpha$ | 2 | 重叠宽度速度系数 | 大 → 更宽共享区，更少漏检但更多通信 |
| $h$ | 场景相关 ($0.003\sim0.0125$) | 时间步长 | CCD 反复失败时缩小以兜底 |
