# 方法拆解(Method)

> 三支柱:FLASH 仿真后端 + RGB-native real-to-sim + 先验引导 MPPI,组成 real-to-sim-to-real 闭环。

---

## 0. 总体一图(ASCII)

```
              ┌──────────────────────── OFFLINE(训练流)────────────────────────┐
              │ A. 合成数据生成:FLASH 模拟随机折叠 episode                       │
              │    → 渲染(相机姿态对齐真机)+ 顶点 GT  →  D_syn                  │
              │ B. 训练 RGB-native real-to-sim f_θ (纯合成数据)                  │
              │ C. 离线蒸馏一个基础策略(产出名义动作序列 Ū_t)                   │
              └──────────────────────────────┬──────────────────────────────────┘
                                             │ 部署
   ┌──────────────────────── ONLINE(每步闭环 real-to-sim-to-real)─────────────────────┐
   │  RGB 图 I_t                                                                          │
   │     │  (ii) DINOv2(冻结) → 视觉token V_t ┐                                          │
   │     ▼                                      ├─ concat → SelfAttn×N → 取前N → MLP解码  │
   │  N 个可学习 canonical tokens C ───────────┘                                         │
   │     │                                                                               │
   │     ▼   ŝ_t = f_θ(I_t):形变 3D 顶点(仿真兼容状态)                                │
   │  ┌──────────── (iii) 先验引导 MPPI ────────────┐                                    │
   │  │ 以 Ū_t 为锚:U^(k)=Ū_t+ε^(k), ε~N(0,Σ), k=1..K │                                  │
   │  │ (i) FLASH 并行 rollout:τ^(k)=F_{0:H}(ŝ_t,U^(k)) │                                │
   │  │ 代价 S(τ^(k)) → softmax 权重 w_k(温度λ)         │                                │
   │  │ 精修序列 U*_t = Σ_k w_k U^(k)                    │                                │
   │  └──────────────────────┬──────────────────────────┘                               │
   │     ▼  执行首动作 a*_t = U*_t[0]  →  新观测 I_{t+1}  →  循环(滚动时域)            │
   └─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 闭环执行公式(§3.1 System Overview)

每个控制步 t,机器人收到单张 RGB 观测 I_t:

1. **状态重建**:`ŝ_t = f_θ(I_t)`  —— real-to-sim 模块输出仿真兼容布料状态,用于同步仿真器。 (式1)
2. **动作精修**:给定离线策略的名义动作序列 Ū_t,控制器采样候选、在 FLASH 里并行 rollout 评估,执行精修序列首动作:
   `a*_t = MPPI(ŝ_t, Ū_t, F, c)`  —— F 是仿真 rollout 算子,c 是任务代价。 (式2)
3. 执行后得到新观测 I_{t+1},重复 —— **滚动时域 real-to-sim-to-real 循环**。

---

## 2. 支柱 (i):仿真后端 FLASH(§3.2)

仿真器在闭环里扮演**双重角色**:① 为 real-to-sim 训练提供合成监督 + 定义"仿真兼容状态空间";② 作为 MPPI 的推理时 rollout 引擎。因此其**物理保真度 / 数值稳定性 / rollout 效率**直接决定全管线性能。

**FLASH 关键特性**:
- 布料表示为**三角网格**,顶点是仿真自由度,连通性编码局部形变与表面几何。
- 通过**非光滑 Newton 迭代**求解 Signorini–Coulomb 摩擦接触条件,在大形变下保留 stick–slip(粘滑)行为。
- **local-global 积分器** + 该接触表述,把主要计算降为 **GPU 友好的稀疏矩阵运算**,支持稳定的重复 rollout。

状态 `x_t=(q_t, q̇_t)`(顶点位置与速度);给定作用于末端/抓取点的动作 a_t:
`x_{t+1} = F(x_t, a_t)`  —— 一步仿真,含操作、重力、接触、摩擦。 (式3)
在线时对候选动作序列**并行** rollout,送给 MPPI。

> **为何选 FLASH**(Table 1):K=64–512 并行下 SR 95–100%,而 Isaac Sim/Newton <60%、MSE/EMD 高约一个数量级;Newton 偶发数值不稳定;Genesis 无法稳定折叠被排除。Isaac Sim 每步快 1.5–1.9× 但最终状态保真差 —— **稳定与保真比每步速度更决定闭环成功率**。

---

## 3. 支柱 (ii):RGB-native real-to-sim 重建(§3.3)

目标:从单 RGB 图恢复 N 个材料顶点的形变 3D 位置 `ŝ_t={p̂_{t,i}}, p̂∈R³`(每个顶点对应 rest-shape 网格上一个固定物质点)。 (式4)

**① 仿真资产构造(离线,每个布做一次)**:从 ZED 2i RGB-D 帧重建固定拓扑 rest-shape 网格 —— SAM 2 分割 → 反投影 masked 深度成点云 → ball pivoting 三角化 → 各向同性 remeshing + 边界锁定平滑。材料顶点被 real-to-sim 与 rollout **共享**。

**② 合成监督**:在 FLASH 里用随机布料姿态/配置/形变模式模拟折叠 episode,相机姿态对齐真机标定,每帧配 GT 顶点 → M 对图像-状态对 `D_syn`。 (式5)

**③ 网络结构**:
- 冻结 **DINOv2** 提取 dense patch 特征,线性层 P 投影到维度 d:`V_t = P(Φ_DINO(I_t)) ∈ R^{M_v×d}`。 (式6)
- 维护 **N 个可学习 canonical tokens** `C={c_i} ∈ R^{N×d}`,每个对应一个网格顶点,训练中编码该物质点的持久身份与标准位置。
- 拼接后联合自注意力,取前 N 个 canonical 输出:`Z_t = SelfAttn([C; V_t])_{1:N}`。 (式7)
- 共享解码器:`p̂_{t,i} = D_θ(Z_{t,i})`。 (式8)

**④ 训练目标**:smooth-L1 损失 `L_r2s = (1/N)Σ ℓ_{smooth-L1}(p̂_i, p*_i)`。 (式9) 只训练 P、C、注意力层、D_θ,**DINOv2 冻结**。

**⑤ 鲁棒增广**(跨 sim-to-real 视觉漂移):输入级——渲染 RGB 的图像空间随机化;隐层级——给视觉 token 加噪与掩码,含**随机抓取点掩码**(在采样接触区周围),使模型从不完整/损坏视觉证据推断布料状态。

---

## 4. 支柱 (iii):先验引导 MPPI 精修(§3.4)

假设有一个基础策略,把当前布料状态映射为动作;部署时它在时域 H 上产出名义序列 Ū_t。**不直接执行**,而作为任务级先验被在线精修。

**① 先验引导采样**:在名义提议附近采 K 条候选:
`U^(k)_t = Ū_t + ε^(k), ε^(k)~N(0,Σ), k=1..K`  —— Σ 控制局部探索尺度。 (式10)

**② 仿真 rollout 评估**:`τ^(k)_t = F_{0:H}(ŝ_t, U^(k)_t)`,赋任务代价 `S(τ^(k))`(折叠目标 + 正则项:工作空间限制、桌面穿透、动作平滑)。 (式11)

**③ MPPI 加权**:代价转归一化权重(温度 λ):
`w_k = exp(−S(τ^(k))/λ) / Σ_j exp(−S(τ^(j))/λ)`。 (式12)
精修序列 `U*_t = Σ_k w_k U^(k)_t`。 (式13)

**④ 滚动时域执行**:执行 `a*_t = U*_t[0]`,收新观测,重复。基础策略被在线物理推理持续纠正,从局部执行误差与状态偏差中恢复。

---

## 5. 伪代码(端到端闭环)

```python
# 离线已就绪: f_theta (real-to-sim), base_policy, FLASH 仿真器 F, 资产网格(N 顶点)
def control_loop(robot, camera, F, f_theta, base_policy,
                 H=horizon, K=num_candidates, Sigma=expl_scale, lam=temperature):
    while not task_done:
        I_t   = camera.get_rgb()                 # 单张 RGB
        s_hat = f_theta(I_t)                      # (式1) DINOv2 + canonical tokens → 形变顶点
        F.set_state(s_hat)                        # 同步仿真孪生

        U_bar = base_policy.rollout(s_hat, H)     # 名义动作序列(先验锚点)

        # ---- 先验引导 MPPI ----
        candidates, costs = [], []
        for k in range(K):                        # 实际为 GPU 并行 batch
            eps   = sample_gaussian(0, Sigma, shape=U_bar.shape)
            U_k   = U_bar + eps                   # (式10) 锚定在 U_bar 附近
            tau_k = F.rollout(s_hat, U_k, H)      # (式11) 并行物理 rollout
            costs.append(task_cost(tau_k))        # 折叠目标 + 工作空间/穿透/平滑正则
            candidates.append(U_k)

        w = softmax([-c/lam for c in costs])      # (式12)
        U_star = sum(w_k * U_k for w_k, U_k in zip(w, candidates))  # (式13)

        robot.execute(U_star[0])                  # 滚动时域:只执行首动作
```

---

## 6. 实现要点与复现风险(Pitfalls)

- **仿真器选型是成败关键**:别只看每步速度。Isaac Sim 更快但最终状态保真差导致 SR 仅 ~50%。优先数值稳定 + 最终状态保真。
- **相机标定必须对齐**:合成渲染相机姿态要匹配真机标定,否则 real-to-sim 在真机上系统性偏移。
- **canonical token 数 = 顶点数 N**:网格粒度(remeshing 目标)决定 N,过密则 rollout 慢、过疏则丢操作相关形变。这是稀疏度的核心旋钮。
- **随机抓取点掩码**:训练时务必加,否则抓取点附近遮挡/接触区在真机上估计崩坏。
- **MPPI 温度 λ 与探索 Σ**:λ 太小→退化为 argmin(对代价噪声敏感),太大→平均化失去选择性;Σ 太大→候选飞出仿真有效域、rollout 不稳。
- **rollout 预算饱和**:可变形动力学并行性差,单 GPU 下 K×H 容易撞显存/时间墙,直接限制控制频率 —— 复现时先量 batch 上限。
- **per-garment 训练**:换新衣物需重做离线网格重建 + 重训 real-to-sim,不是即插即用。
- **基础策略质量下限**:先验引导是局部采样,若名义轨迹离正确解太远,可能无法跳出 —— base-policy 不能太差(Table 3 中"仅基础策略"仍有 1–3/10,说明先验本身弱时整体也弱)。
