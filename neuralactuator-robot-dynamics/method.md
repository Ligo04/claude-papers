# NeuralActuator 方法详解

> 依据: arXiv 2607.11734v1, METHOD 节 (原文行 288-855)。所有数字/公式均对照 `paper-facts.md` 与原文核实，未核实内容一律标注"未在原文中给出"。

## 1. 问题设定回顾

论文要解决的核心矛盾：低成本齿轮舵机（如 DYNAMIXEL XM430-W350）上，经典的线性电流-力矩模型

```
τ_act = K_t · i + b
```

在存在摩擦、迟滞、背隙、电流饱和、热漂移时不可靠（实测中甚至出现负斜率拟合）。而这类平台又没有力/力矩传感器，因此需要一个"无需力矩标签、无需力传感器"就能同时做（a）执行器动力学建模和（b）无感力觉估计的模型。NeuralActuator 用一个四头 Transformer + 可微仿真训练来同时解决这两件事。

式 (1) 定义了物理关节力矩的一般依赖关系：

```
τ_real_j(t) = f_τ(q_cmd_j(t), u_j(t), q_j(t), q̇_j(t), V_j(t), T_j(t), H_t)
```

其中 `u_j` 是平台相关信号：OpenManipulator-X 上是电机电流，SO-101 上是有符号负载寄存器，Franka 上是指令关节力矩。这个依赖关系是非线性、有历史依赖的——这正是用 Transformer 处理时序窗口的动机。

## 2. 输入特征构成

输入序列 `X_t ∈ R^{L×n_f}`，`L = H_hist + 1 = 9`：8 帧历史 `H_t = {x_{t-8}, ..., x_{t-1}}` + 当前帧 `x_t`。

每帧特征 `x_t` 由三族信号 + 跟踪误差构成（主实验低成本平台配置）：

| 族 | 具体量 |
|---|---|
| 指令目标 | `q_cmd`（关节）, `q_cmd_g`（夹爪单指棱柱坐标目标） |
| 本体感知 | `q, q̇`（关节位姿/速度）, `q_g`（夹爪坐标） |
| 执行器遥测 | `u`（电流/负载寄存器/指令力矩，平台相关）, `V`（电压）, `T`（温度） |
| 跟踪误差 | `e = q_cmd − q`, `e_g = q_cmd_g − q_g` |

所有特征在送入网络前做归一化。Franka 使用附录 B 中平台专属的特征向量（未在正文详述，此处不展开）。注意：仿真中的夹爪特征 `q_g` 是单指棱柱关节坐标，和 III-E 节中物理夹爪总开口 `a(α)` 是两个不同的量——前者是仿真标定量，后者是真实机构几何量，两者通过夹爪运动学映射关联但不等价。

## 3. Transformer Encoder

输入投影 + 可学习位置编码：`H_enc^(0) = X_t W_in + P`，`W_in ∈ R^{n_f×d}`，`P ∈ R^{L×d}`。

配置：4 层，`d = 192`，4 个注意力头，`d_ff = 384`。

每层是 **pre-norm 残差块 + query 依赖的逐元素门控注意力**（式 5-8）：

```
U^(l)   = LayerNorm(H_enc^(l-1))
A^(l)   = [MHA(U^(l)) ⊙ σ(U^(l) W_g^(l))] · W_o^(l)
Z^(l)   = H_enc^(l-1) + A^(l)
H_enc^(l) = Z^(l) + FFN_GELU^(l)(LayerNorm(Z^(l)))
```

关键点：`U^(l) W_g^(l)` 和 query 投影是**并行计算**的（都吃同一个 `U^(l)`），门控 `σ(U^(l) W_g^(l))` 逐元素乘在 MHA 的拼接输出（未过输出投影前）上，再统一过 `W_o^(l)`。这是一个輸入依赖的软门，让每个 token/通道能自适应地抑制或放大注意力输出的某些维度——比原始 Transformer 的固定线性输出投影多了一层非线性调制能力，参数开销很小（`W_g ∈ R^{d×d}`，和 `W_o` 同阶）。

末层 LayerNorm 后做**时序均值池化**（对 L=9 个 token 取平均）得到 `h_pool ∈ R^d`，再喂给四个 SiLU-MLP 输出头。

## 4. 四个输出头

1. **力矩代理 (torque surrogate)** `τ_pred = MLP_τ(h_pool) ∈ R^{n_m}`（式 9）。`n_m` 是建模的执行器通道数（OpenManipulator-X: `n_a=4` 臂关节 + 1 夹爪通道 = `n_m=5`）。对旋转关节是力矩量纲，对棱柱夹爪通道是力量纲，因此论文称其为"generalized-effort surrogate"，torque surrogate 只是简写。Franka 配置保留原生 affine PD 位置执行器，这个头此时提供的是"仿真控制代理"而非直接力矩量。
2. **原始外力** `f̂_raw ∈ R³`：从共享 Transformer 表征先算出一个 force-specific 隐特征，再用**独立线性读出**分别得到 `f̂_raw` 和门 `g`（两者共享该隐特征但读出层不同）。
3. **接触概率门** `g ∈ [0,1]`：最终力输出是两阶段的 `f̂ = g · f̂_raw`。训练时门标签 `g_gt = I[‖f_gt‖₂ > ε]`，`ε = 0.01 N`（这个 ε 仅用来区分"零/非零"标签，不是接触检测的物理阈值）。
4. **逐电机状态分数** `c = σ(MLP_cond(h_pool)) ∈ [0,1]^{n_m}`：架构上每个电机通道都输出一个分数，但 OpenManipulator-X 实验里**只监督/评估 Joint 3** 这一维，`c₃ → 1` 表示正常运行，`c₃ → 0` 表示机械受限（橡皮筋捆绑）。其余维度在该实验中不赋予概率解释。

## 5. 算法流程与架构图

```
                    ┌─────────────────────────────────────────┐
  X_t = (x_{t-8},…,x_t)  9 tokens × n_f                        │
        │                                                      │
        ▼                                                      │
  Win projection + positional embed  H_enc^(0)                 │
        │                                                      │
        ▼                                                      │
  ┌──────────────────────────┐                                 │
  │ ×4 encoder blocks:        │                                │
  │  U = LayerNorm(H)         │                                │
  │  A = [MHA(U)⊙σ(U·Wg)]·Wo  │  (pre-norm + gated attention)  │
  │  Z = H + A                │                                │
  │  H' = Z + FFN_GELU(LN(Z)) │                                │
  └──────────────────────────┘                                 │
        │                                                      │
        ▼  final LayerNorm                                     │
  temporal mean pooling over L=9 tokens → h_pool ∈ R^d          │
        │                                                      │
        ├────────────┬─────────────┬───────────────┐           │
        ▼            ▼             ▼               ▼           │
   MLP_τ         hidden_force   hidden_force    MLP_cond        │
   τ_pred          → f̂_raw      → g (linear)     → c            │
   (n_m)           (R³)          [0,1]           [0,1]^{n_m}    │
        │            │             │                            │
        │            └──────┬──────┘                            │
        │                   ▼                                   │
        │              f̂ = g · f̂_raw   (supervised output,      │
        │                               NOT fed back to sim)    │
        ▼                                                        │
  ¯τ_t = clip(τ_pred; τ_min, τ_max)                               │
        │                                                        │
        ▼                                                        │
  s_{t+1} = DiffSim(s_t, ¯τ_t)     (MuJoCo/MJX forward step)      │
        │                                                        │
        ▼                                                        │
  append x_t to history buffer H_t → next control cycle ─────────┘
```

**要点**：只有 `τ_pred`（经 clip 后的 `¯τ_t`）真正进入仿真状态更新；力、门、状态三个头的输出是纯监督输出，**不参与 `F_t` 的状态转移**，也不通过 `J_v(q)^T` 映射回力矩残差——即式 (4) 的 `τ_ID = τ_act + τ_ext` 分解只是"物理动机"，不是架构上强制的约束。

## 6. 物理解释（式 4）与其边界

```
τ_ID = M(q)q̈ + C(q,q̇)q̇ + g(q) = τ_act + τ_ext,   τ_ext = J_v(q)^T f_ext
```

`M, C, g(q)` 是仿真器内置的解析刚体项，`τ_act` 是真实关节侧执行器力矩。经典力估计方法（ID-Linear、ID-Friction、GMO 等）都依赖这个分解，用线性 `τ_act = K_t·i+b` 近似执行器项，再把残差 `τ_ID − τ_act` 映射成笛卡尔力。齿轮舵机上这个线性近似本身误差很大，会直接污染力估计。

NeuralActuator 不强制这个分解，而是让 `τ_pred`/`f̂` 各自独立学习。这带来一个必须理解的陷阱：

> **训练/评估用无负载模型 + MJX 接触关闭**：在 OpenManipulator-X 的接触/负载实验中，前向模型既不接收测量或预测的外力，也不接收负载体/连杆质量变化。因此 `¯τ` 学到的是"复现观测运动所需的仿真等效广义输入"，会把未建模的接触、负载、动力学失配**吸收进去**——它不是一个可辨识的物理执行器力矩，只有当外部负载缺失或已被前向模型表示、且其余动力学建模准确时，`¯τ` 才与真实关节力矩重合。

## 7. 训练：位姿监督的可微仿真 BPTT

由于低成本平台**没有直接的广义力矩标签**，力矩代理头通过位姿轨迹间接监督：预测的广义输入经仿真前向积分变成未来位姿，测得的真实位姿就是有效的代理监督目标。

**前向（式 10）**：
```
τ_pred_t = g_θ(X_t)
¯τ_t = clip(τ_pred_t; τ_min, τ_max)
s_{t+1} = F_t(s_t, ¯τ_t) = DiffSim(s_t, ¯τ_t)
```
`s_t = [q_t, q̇_t]` 是仿真关节状态。

**梯度（式 11-12）**：总损失 `L = Σ_k ℓ_k(s_k)`，
```
dL/dθ = Σ_{k=1}^{T} (∂ℓ_k/∂s_k)(ds_k/dθ)

ds_{t+1}/dθ = (∂F_t/∂s_t)(ds_t/dθ) + (∂F_t/∂¯τ_t)(d clip(g_θ(X_t))/dθ)
```
这就是标准的 BPTT：后一步的损失会沿着仿真器灵敏度和 clip 的次梯度一路传回更早的力矩预测，只要这两个灵敏度非零。论文特别指出（式 12 后一段）：真正的 Markov 状态是增广状态 `(s_t, H_t)`，式 12 为了可读性省略了历史缓冲这一项的显式梯度，但历史缓冲里的仿真状态分量同样通过链式法则贡献梯度。

### 损失函数全家（式 13-18）

设 `Huber_β` 是转折参数为 `β` 的 Huber（Smooth L1）损失。

```
L_joint = (1/|T|n_a) Σ_t Σ_j Huber_1(q_sim_{t,j} − q_real_{t,j})          (13)  关节位姿
L_grip  = (1/|T|)     Σ_t   Huber_1(κ_g(q_sim_{g,t} − q_real_{g,t}))       (14)  夹爪位姿，κ_g=1000（米→毫米）
L_force = (1/3|T_f|)  Σ_t Σ_k w_t v_{t,k} Huber_{β_f}(f̂_{t,k} − f_gt_{t,k}) (15)  力回归
L_gate  = (1/|T_f|)   Σ_t   BCE(g_t, g_gt_t)                                (16)  接触门
L_cond  = (1/|T_c|)   Σ_t   BCE(c_{t,3}, m_{t,3})                           (17)  Joint3 状态
L = λ_joint L_joint + λ_grip L_grip + λ_force L_force
  + λ_gate L_gate + λ_cond L_cond + λ_vel L_vel                            (18)
```

细节：
- `κ_g = 1000` 只对 OpenManipulator-X 的棱柱夹爪单指坐标残差生效，把米换算成毫米再算 Huber，防止夹爪这个小量纲的误差被淹没。
- `β_f`（力损失 Huber 转折参数）：**力传感器实验和最终已知重量微调阶段用 β_f=0.15**，其余情况 `β_f=1`。Franka 只监督竖直分量，用固定 `β=0.15`。
- `w_t` 是非零力样本的固定重加权系数（论文未给出具体数值，只说明是"fixed nonzero-force-sample reweighting"）。
- `v_{t,k}` 是力通道有效性掩码，默认全 1（缺失值哨兵也转成 0 参与训练），仅在开启"无效通道屏蔽"配置时置 0。
- `g_gt_t = I[‖f_gt‖₂ > 0.01N]`，`m_{t,3}=1` 表示 Joint3 正常，`=0` 表示机械受限。
- `L_vel`（速度损失）：部分 SO-101/Franka 训练用有限差分位姿速度做辅助 Smooth L1 rollout 损失，`λ_vel = 5`（SO-101）或 `5/10`（Franka 不同训练阶段），其余情况为 0。
- 任何数据不可用的任务损失，其权重直接置 0。
- 视觉监督实验里，可微渲染轮廓损失被加进式 18，沿"渲染器→仿真配置→可微动力学→NeuralActuator 参数"这条路径反传。

**优化器与算力**：AdamW，最多训练 100,000 epochs，但实践中最佳 checkpoint 通常在 30,000 epochs 内出现。默认工作站配置：Intel Core Ultra 7 265K（20 核 5.0GHz）+ 32GB 内存 + RTX 5080 16GB VRAM；附录 B 的空载 benchmark 用 RTX 4090，附录 C 的手眼标定用 A6000。训练/推理用 JAX + JIT，仿真基于 MuJoCo/MJX。

### 梯度视野诊断（Table II）

论文用 rollout 视野 `H ∈ {64,128,256,320,500}` 做梯度诊断：仿真器输入梯度范数在这些视野下都保持 `O(1e-2)` 量级；参数梯度余弦相似度（相对 H=128 的方向）始终 `≥ 0.96`；`‖∇_θ L‖` 的量级在 `H=256–320` 处出现**平台（plateau）**——即再增大视野，梯度量级不再显著增长。这个观测被用来指导课程学习的视野调度设计（先短视野后长视野，或直接选一个落在平台区的视野长度做主训练，兼顾梯度信噪比和 BPTT 计算成本）。

## 8. 伪代码

### 8.1 训练循环（含 BPTT 展开）

```python
# One training step: rollout H_train steps, accumulate loss, BPTT through DiffSim
def train_step(theta, batch_traj, H_train):
    s = init_state_from(batch_traj[0])          # s0 = [q0, qdot0] from real robot
    history = init_history_buffer(batch_traj)    # H_t = {x_{t-8},...,x_{t-1}}
    loss = 0.0
    for t in range(H_train):
        x_t = build_feature_frame(batch_traj, t, s)      # cmd/proprio/telemetry/tracking-error
        X_t = concat(history, x_t)                        # 9-token sequence
        h_pool = encoder_forward(theta, X_t)               # 4-layer gated-attention transformer
        tau_pred = mlp_tau(h_pool)                         # torque-surrogate head
        f_hat, gate = force_heads(h_pool)                  # f_raw, g (decoupled two-stage)
        c = mlp_cond(h_pool)                                # per-motor condition
        tau_bar = clip(tau_pred, tau_min, tau_max)          # bounded generalized input
        s = diffsim_step(s, tau_bar)                        # differentiable forward dynamics
        loss += huber1(s.q, batch_traj.q_real[t]) \
              + huber1(kappa_g * (s.q_grip - batch_traj.q_grip_real[t])) \
              + force_loss(gate * f_hat, batch_traj.f_gt[t]) \
              + gate_loss(gate, batch_traj.gate_gt[t]) \
              + cond_loss(c[3], batch_traj.cond_gt[t])
        history = shift_and_append(history, x_t)
    grads = backprop_through_time(loss, theta)   # eq.11-12: dL/dtheta via chain rule over all t
    theta = adamw_update(theta, grads)
    return theta, loss
```

### 8.2 在线推理（严格因果）

```python
# Online deployment: at each control cycle, only telemetry that has ALREADY arrived is used.
def online_control_cycle(theta, history_buffer):
    q, qdot, q_cmd = read_current_proprioception_and_command()
    i_t = read_live_motor_current()                     # measured NOW, not future
    x_t = build_feature_frame(q_cmd, q, qdot, i_t, V(), T(), e=q_cmd - q)
    X_t = concat(history_buffer, x_t)                    # only past + current, no future frames
    h_pool = encoder_forward(theta, X_t)
    tau_pred = mlp_tau(h_pool)
    f_hat_t, gate_t = force_heads(h_pool)
    c_t = mlp_cond(h_pool)
    f_ext_hat = gate_t * f_hat_t                          # feedback controller may consume this
    next_command = feedback_controller(f_ext_hat, ...)
    history_buffer = shift_and_append(history_buffer, x_t)   # x_{t+1} enters only after next packet
    return next_command, f_ext_hat, c_t
    # NOTE: cannot roll out candidate FUTURE commands counterfactually —
    # that would require a model of the corresponding future effort/telemetry signal,
    # which this pipeline does not have (see pitfall below).
```

## 9. 实现陷阱（论文明确强调或隐含的坑）

1. **clip 让位姿损失对越界原始幅值不敏感**：`¯τ = clip(τ_pred; τ_min, τ_max)` 只有裁剪后的值参与仿真状态更新，反传到 `τ_pred` 的梯度经过 clip 的次梯度——当 `τ_pred` 落在裁剪区间外时，`d clip(·)/dθ` 在那一段是 0（或近 0），意味着网络可以把原始输出打得任意大而不受位姿损失惩罚，只要裁剪后的值是对的。这解释了为什么"pre-clipping network output"和"applied surrogate ¯τ"要分开表述。
2. **力头不进状态更新**：`f̂ = g·f̂_raw` 是纯监督输出，绝不通过 `J_v(q)^T` 映射回力矩残差、也不进入 `F_t`。这意味着力估计质量完全依赖直接标签监督，和仿真动力学的准确性解耦——好处是力估计不会被仿真误差污染，坏处是力估计无法利用物理一致性约束来正则化。
3. **recorded-telemetry-conditioned rollout ≠ 反事实预测**：离线训练/评估用的是完整的、和真实轨迹时间对齐的 effort-telemetry 序列（包括未来时刻已经发生的电流等），所以多步 rollout 指标本质上是"给定真实遥测轨迹条件下"的重建误差，而不是对未执行候选指令的前瞻预测。只有在线部署才是严格因果的：每个控制周期只用已经到达的遥测（含当前实测电流），`x_{t+1}` 要等下一个遥测包到达才进历史缓冲。这也是论文局限 3 提到的——做 MPC 式反事实 rollout 需要额外一个 effort 响应模型，NeuralActuator 本身不提供。
4. **梯度视野课程**：`‖∇_θ L‖` 在 `H=256-320` 出现平台，这是训练稳定性和计算成本的权衡点——视野太短，长时程误差累积信号进不来梯度；视野太长，BPTT 计算/内存成本线性增长而梯度量级不再显著变化，收益递减。复现时如果自行选择视野，建议在这个区间内，而不是无脑拉满到 500。
5. **门控注意力的 `W_g` 和 query 投影是并行支路**：容易误实现成"先算 attention 再决定要不要过门"的串行流程，正确实现是 `U^(l)W_g^(l)` 独立于 MHA 内部的 Q/K/V 投影，只是共享同一个归一化输入 `U^(l)`，最后逐元素乘在 MHA 拼接输出上。
6. **不同平台的 `u` 语义不同**：OpenManipulator-X 是电流，SO-101 是有符号负载寄存器，Franka 是指令关节力矩——迁移代码到新平台时，这个字段的物理含义和量纲必须重新核实，不能假设是电流。

## 10. 超参敏感性与复现风险

- **仿真后端**：JAX + JIT，MuJoCo/MJX 做可微刚体动力学；论文也提到额外用 NVIDIA Warp（FeatherstoneIntegrator + `wp.Tape` 反向）做过验证，60Hz 环境率 × 5 substeps，armature 0.01——复现时后端选择本身就是一个自由度，两套后端的数值行为（接触求解器、积分器）可能带来结果差异，论文没有给出两者的定量对比误差。
- **训练规模**：AdamW 最多 100k epochs，但实际最佳 checkpoint 在 30k epochs 内出现——如果复现时训练不到 30k 就下结论"不收敛"，可能只是还没到平台区；反过来死磕到 100k 也未必比 30k 更好，容易过拟合到 recorded-telemetry-conditioned 的训练分布。
- **算力假设**：默认基准是 RTX 5080 16GB VRAM，论文没有报告在更小显存卡（如 8GB 级）上 batch size/视野长度是否需要缩减。
- **控制率**：主实验 60Hz 控制率，运行时数字（Table XII: 1.44M 参数、5.46M FLOPs、平均推理 0.25ms/P95 0.31ms、吞吐 4019Hz@b=1）都是在这个假设下测的；换到更高频控制回路（如 200Hz+ 力控）需要重新验证实时性余量。
- **β_f 和 λ 系列权重**：`β_f` 在不同训练阶段（力传感器 run / 已知重量微调 / 其余）取值不同（0.15 vs 1），`λ_vel` 在 SO-101(5) 和 Franka 不同阶段(5或10)也不同——这些是按平台/数据模态手工调的，论文未给出选择这些值的系统性准则，复现到新平台大概率需要重新网格搜索。
- **跨平台泛化的隐藏假设**：Franka 实验里指令位姿用了平滑代理 `q̃_cmd = 1.03·q_rec(t+5)`，这依赖**未来**的记录状态，所以 Franka 的力估计 benchmark 本质是离线 benchmark，不能直接当作在线可行性的证据——这是复现/迁移时最容易被忽略的一条限制。

## 11. 夹爪运动学（III-E 节，式 19）

物理机构：一个旋转电机通过对称双滑块-连杆机构驱动两个棱柱指爪。定义：`B, B'` 为两个曲柄销，`r=|OB|=|OB'|` 曲柄半径，`l=|AB|=|A'B'|` 连杆长度，`α` 曲柄半径与滑块轴之间的锐角，`δ` 滑块销到对应内侧指爪面的偏移。

对称开口：
```
a(α) = 2|AO| − 2δ = 2(√(l² − r²sin²α) + r·cosα) − 2δ    (式19)
```

这是物理机构的电机角-开口理想映射；仿真器用一个**标定过的单指棱柱坐标**表示夹爪（也就是第 2 节提到的 `q_g`，和 `a(α)` 是两个不同量）。

准静态对称抓取下，连杆是二力杆，承受轴向力 `T`。设 `β` 为连杆与滑块轴的夹角，每个指爪的闭合力：
```
F = T·cosβ
```
电机轴力矩平衡给出：
```
τ = 2Tr·sinγ,   γ = ∠OBA = ∠OB'A'
```
即夹爪的力传递同时依赖电机力矩和机构构型（`r, γ, β` 随开口角变化），不是简单的线性比例——这也是为什么夹爪通道单独用棱柱坐标建模、而不是直接把电机力矩当力用的原因。
