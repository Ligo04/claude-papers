# Method — SCORE 方法与算法细节

## 0. 三阶段总览（Real → Sim → Real）

```
┌─────────────── Real-to-Sim ───────────────┐  ┌──────── Sim ────────┐  ┌─── Sim-to-Real ───┐
│ 1. 真实演示 D_real 训 flow 基座 π_base(a|o) │  │ 3. 冻结 π_base       │  │ 5. 组合部署        │
│ 2. 从真实扫描构建仿真环境 + 域随机化        │→ │ 4. 学 steering π_steer│→ │   a=π_base(o,z)+a_r │
│                                             │  │   PPO + 稀疏奖励      │  │   免蒸馏/免微调     │
└─────────────────────────────────────────────┘  └─────────────────────┘  └───────────────────┘
```

核心设计选择：**在仿真里优化任务性能，同时把学到的策略约束到"真实数据训出的生成式策略的支撑集"内。不直接更新该策略，而是冻结 π_base、在其隐输入上学一个轻量 steering 策略 π_steer。**

---

## 1. 组件拆解（Component Breakdown）

### 1.1 基座策略：条件 flow matching 策略 π_base(a|o)

- **观测 o**：部署观测 = 点云 + 本体感觉 + 历史。
- **动作 a**：机器人动作（Franka 臂 + LEAP 手关节）。
- **Flow matching [Lipman 2023]**：学一个速度场 `v_θ(x_τ, τ, o)`，把先验 p₀（高斯）上的样本沿 τ∈[0,1] 输运到行为分布。
- **训练目标**（给定 `(o,a)~D_real, x₀~p₀, τ~U[0,1]`，插值 `x_τ = (1−τ)x₀ + τa`）：

  ```
  L_FM(θ) = E_{τ,(o,a),x₀} [ ‖ v_θ(x_τ, τ, o) − (a − x₀) ‖² ]        (Eq.4)
  ```

- **推理**：抽 `z ~ p₀`，从 `x₀ = z` 积分 `dx_τ/dτ = v_θ(x_τ, τ, o)` 到 `x₁`，得 `a = x₁ =: π_base(o, z)`。
  → 关键：**给定 o，π_base 是从隐噪声 z 到动作 a 的确定性映射**。

### 1.2 支撑集 = 模型诱导集合 A_base(o)

对固定观测 o，基座能生成的动作集合：

```
A_base(o) = { π_base(o, z) | z ∈ Z }
```

**只要只调 z，输出动作永远 ∈ A_base(o)**。这就是支撑集约束的实现机制——不需要判别器/保守 Q/额外正则，纯靠"只在噪声空间做 RL"这个结构保证。

### 1.3 Steering 策略 π_steer(z | o)

冻结 π_base 后，只学 π_steer。在策略时间步 k：

```
z_k ~ π_steer(·| o_k),    a_k = π_base(o_k, z_k)                       (纯 latent steering)
```

两种实例化：

| 方法 | 动作 | 约束类型 | 说明 |
|------|------|---------|------|
| **DSRL** [Wagenmaker 2025] | `a_k = π_base(o_k, z_k)` | **硬**支撑集约束 | 纯调噪声 z，动作严格 ∈ A_base(o) |
| **RFS**（=SCORE）[Su 2026] | `a_k = π_base(o_k, z_k) + a_r` | **软**支撑集约束 | 额外小残差 a_r 局部精修，给灵巧操作所需精度 |

- 论文里 `SCORE-DSRL` 用 DSRL，`SCORE` 用 RFS。
- SCORE 对支撑集约束方法是**模块化**的——DSRL/RFS 只是两个实例，未来可换其它（如 REFORM [33]）。

### 1.4 仿真优化目标

冻结 π_base，只优化 π_steer：

```
max_{π_steer}  J_sim(π_base ∘ π_steer)
  = E_{ z_k~π_steer(·|o_k),  a_k=π_base(o_k,z_k),  p_sim } [ Σ_{k=1}^T γ^{k−1} r(s_k, a_k) ]     (Eq.5)
```

- **稀疏奖励**：r 用与评估相同的成功条件定义，**所有任务同一套超参**，几乎不做 reward engineering。

### 1.5 PPO + 非对称 actor-critic

稀疏奖励在灵巧操作里样本效率低（成功接触序列稀有）。为充分利用并行仿真：
- 用 **PPO [Schulman 2017]** 优化 π_steer。
- **非对称 actor-critic**：
  - **Actor**：只接收**部署观测**（点云 + 本体感觉），输出 latent steering 分布 → 保证可部署。
  - **Critic**：训练时额外用**特权仿真状态**（privileged state）→ 更强的 value 信号、更快学习。
- 部署策略不受 critic 影响（critic 只在训练用）。

---

## 2. 算法流程（伪代码）

```python
# ============ 阶段 A: Real-to-Sim ============
# A1. 真实演示训 flow 基座
pi_base = train_flow_matching(D_real)          # Eq.4，训完冻结
# A2. 从真实扫描构建仿真环境
env_sim = build_from_scan(real_scene)
env_sim.enable_domain_randomization()          # 物体初始化 + 物理参数
env_sim.enable_artificial_failures()           # 接触后随机位移物体（保留恢复行为）

# ============ 阶段 B: Sim（支撑集约束 RL）============
pi_steer = init_steering_policy()              # 输出 (z, a_r) 分布
critic   = init_asymmetric_critic()            # 用特权状态

for iteration in range(N):                      # PPO 循环
    rollouts = []
    for env in parallel_envs(4096):            # 大规模并行
        o = env.deploy_obs()                   # 部署观测（点云+本体感觉+历史）
        z, a_r = pi_steer.sample(o)            # steering：采样隐噪声 + 残差
        a = pi_base(o, z) + a_r                # 前向 flow（冻结）+ 残差  -> RFS
        s_next, r = env.step(a)                # 稀疏奖励（成功条件）
        rollouts.append((o, env.priv_state(), z, a_r, r))
    # PPO 更新：actor 用 o，critic 用 priv_state
    ppo_update(pi_steer, critic, rollouts)     # pi_base 始终不动

# ============ 阶段 C: Sim-to-Real（直接部署）============
# 无蒸馏、无微调、不改 pi_base
def deployed_policy(o):
    z, a_r = pi_steer.sample(o)
    return pi_base(o, z) + a_r
```

**关键点**：整个过程 π_base 的权重**从未被梯度更新**——RL 的全部作用是"重新选择用哪个 z（以及加多小的 a_r）"，即在支撑集内重新分配概率。

---

## 3. 架构要点图（ASCII）

```
                部署观测 o = [点云, 本体感觉, 历史]
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
  ┌───────────────┐              ┌─────────────────┐
  │  π_steer(z|o) │  采样 z,a_r  │  Critic (训练用)  │◄── 特权仿真状态 s_priv
  │  (可训练)     │              │  非对称、只训练期  │
  └───────┬───────┘              └─────────────────┘
          │ z
          ▼
  ┌───────────────────┐
  │  π_base(o, z)     │  冻结的 flow matching
  │  隐噪声→动作 确定映射│  输出 ∈ A_base(o)  ← 支撑集约束在此自动成立
  └───────┬───────────┘
          │ + a_r  (RFS 软约束局部精修)
          ▼
        动作 a_k  ──►  仿真 p_sim / 真机 p_real
```

---

## 4. 逐步解释（Step-by-step）

1. **训基座**：真实数据 → flow matching 策略，学到多模态行为分布（含快/慢、精/糙各种模式）。
2. **建仿真**：真实场景扫描 → 仿真环境；域随机化增强迁移；人工失败注入维持恢复行为。
3. **冻结 + steering**：π_base 冻结，π_steer 在噪声空间学"该选哪个模式"。因只调 z，动作恒在支撑集内 → **可迁移性由结构保证**。
4. **PPO 训练**：非对称 critic 用特权状态加速稀疏奖励学习；4096 并行环境；每 40M 步评估一次。
5. **直接部署**：把 π_steer 与真机 π_base 组合上真机，无任何后处理。

---

## 5. 实现陷阱（Implementation Pitfalls）

- **只调 z 还是加残差？** 纯 DSRL 的硬约束在灵巧操作里精度不够（受限于基座离散动作集），SCORE 用 RFS 加小残差 a_r。**a_r 必须"小"**——太大就退化成 Residual-RL 甚至无约束，破坏支撑集约束的迁移保证。
- **恢复行为遗忘**：训练后期策略稳定首成功 → 不再访问失败态 → 遗忘恢复。**必须**在接触后按随机间隔位移物体（人工失败）来对抗，否则鲁棒性下降（100%→95%，定性上恢复明显变差）。
- **对称 critic 会掉样本效率**：若 critic 也只用点云+本体感觉，样本效率与最终仿真成功率显著下降（Fig.11）。特权状态是关键加速器。
- **感知偏移假设**：方法假设"充分视觉随机化下两域感知偏移最小"。若视觉泛化不足（如 distractor），steering 难以探索，仅部分工作空间成功。
- **稀疏奖励探索难**：灵巧操作成功接触序列稀有，没有并行仿真 + 非对称 critic 很难起步。

---

## 6. 超参敏感性（Hyperparameter Sensitivity）

- **一套超参跑所有任务**（论文卖点）——稀疏奖励用统一成功条件定义，降低了 reward tuning 负担。
- **残差幅度 a_r**：软/硬约束之间的关键旋钮，隐性敏感（论文归到 RFS/Appendix E）。
- **数据量 / 覆盖度**：非超参但强烈影响上限——Cube Pinch 100 条演示才达 100%；retry/play 数据显著扩大支撑集。
- **PPO 常规超参**：论文强调对比时保持一致；BC 基线（RialTo）的正则系数则极度敏感（10→崩溃、100→危险、100k→锁死、10k→折中）。

---

## 7. 复现风险（Reproduction Risks）

1. **仿真环境构建**：需要真实场景扫描 → 仿真资产的 pipeline，非平凡（附录 D）。
2. **DSRL / RFS 依赖**：需正确实现 flow 隐空间 steering（DSRL [11]）与残差 flow steering（RFS [12]）——这两篇是前置。
3. **LEAP Hand + Franka FR3 硬件**：真机复现门槛高；点云观测 + PointNet 编码需对齐。
4. **域随机化 + 人工失败**的具体范围未在正文完全给出（散落附录 D），是复现敏感点。
5. **代码**：项目页 https://weirdlabuw.github.io/score/ 承诺提供代码与视频（正文未列 GitHub 直链）。

---

## 8. 一句话记忆锚点

> **冻结 flow 基座，只在噪声上做 PPO；动作恒在"基座能生成的集合"内 → 提升与迁移同时拿下。**
