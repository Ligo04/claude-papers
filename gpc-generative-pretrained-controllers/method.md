# 方法细节 — GPC

## 总览：三阶段流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Skill Quantization  (§5)   —— 造"运动词表"，端到端 RL       │
│                                                                       │
│   参考动作 ŝ_{t:t+h}                                                   │
│        │                                                              │
│    ┌───▼───┐    z_t ∈ R^d    ┌─────┐   ẑ_t (离散)   ┌───────┐  a_t    │
│    │Encoder│ ─────────────▶ │ FSQ │ ─────────────▶ │Decoder│ ──────▶ │
│    └───────┘                 └─────┘                └───▲───┘  (PD目标)│
│                          ⌊(L/2)tanh(z)⌉               │ s_t (本体感知) │
│         └────────── 端到端 PPO + 跟踪奖励（STE 直通梯度）─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │ 冻结 FSQ 编码器/解码器
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: Generative Controller  (§6)  —— GPT 学 token 分布           │
│                                                                       │
│   FSQ 编码器产出 d 个 L 进制 token → 每 G 个打包 → d' 个大词表 token    │
│                                                                       │
│    s_t ──▶ ┌──────────────────────┐  自回归 p(z̃^j | s_t, z̃^{<j})      │
│           │ GPT Transformer Decoder│  teacher forcing + 交叉熵          │
│            └──────────────────────┘                                   │
│   推理：nucleus(top-p) 采样 token → 冻结 FSQ 解码器 → 动作             │
└─────────────────────────────────────────────────────────────────────┘
                              │ 冻结生成控制器
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Task Adaptation  (§7)  —— CoLA (PEFT) + SFT + RLFT          │
│                                                                       │
│   冻结主干 W₀ + CoLA 层(<1%参数, DoRA+FiLM条件于任务观测 c)            │
│   可选 SFT（示例动作，交叉熵偏置技能）→ RLFT（PPO 优化任务奖励）      │
│   探索时用无条件生成模型引导 nucleus 采样，保住自然行为                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Skill Quantization（§5）

### 组件拆解

- **跟踪框架**：沿用 DeepMimic [Peng 2018]。给定参考动作在每个时刻的目标状态 `ŝ_{t:T}`，训练策略 `π_θ(a_t | s_t, ŝ_{t:t+h})`。
  - 条件于**本体感知状态** `s_t` + 未来 `h` 帧参考目标。
  - 奖励 `r_t` 度量仿真状态 `s_{t+1}` 与参考动作的差异（跟踪奖励）。
- **编码器 E**：把目标状态序列 `ŝ_{t:t+h}` 映到连续潜向量 `z_t = E(ŝ_{t:t+h}) ∈ R^d`。
- **FSQ 量化**（式 4）：每维独立量化到 `L` 级：

  ```
  ẑ_t = ⌊ (L/2) · tanh(z_t) ⌉
  ```

  隐式码本大小 `L^d`（本文 `9^40`）。无显式码本 → 无 VQ-VAE 的坍缩/死码问题。
- **解码器 D**：接收当前状态 `s_t` 与离散码 `ẑ_t`，输出动作 `a_t ~ D(a_t | s_t, ẑ_t)`。动作 = 各关节 **PD 控制器的目标关节旋转**；PD 把目标转成关节力矩驱动仿真角色。

### 训练
- **算法**：PPO [Schulman 2017]，编码器-量化-解码器**端到端**联合优化。
- **梯度穿过量化**：STE（直通估计器）[Bengio 2013]——前向取整，反向梯度当作恒等直接传给 `tanh` 前的 `z`。

### 伪代码

```python
# Stage 1: FSQ 跟踪控制器端到端 RL 训练
for iteration in range(N_iters):
    # rollout：与物理仿真交互收集轨迹
    for t in range(rollout_len):
        z_cont = encoder(ref_states[t : t+h])        # (d,) 连续潜
        z_disc = round((L/2) * tanh(z_cont))          # FSQ 量化（前向）
        # STE：z_ste = z_cont + stop_grad(z_disc - z_cont)
        z_ste   = z_cont + (z_disc - z_cont).detach()
        action  = decoder(state[t], z_ste)            # PD 目标关节旋转
        target_torque = pd_controller(action, joint_state)
        next_state = sim.step(target_torque)          # Isaac Gym 一步
        reward  = tracking_reward(next_state, ref_states[t+1])
        buffer.add(state[t], action, reward, ...)
    # 用 PPO 更新 encoder+decoder（梯度经 STE 穿过量化）
    ppo_update(encoder, decoder, buffer)
```

---

## Stage 2: Generative Controller（§6）

### 从连续码到 token 序列
- 编码器每步产生 `d` 个 `L` 进制离散 token。直接建模 `d` 个 token 序列太长。
- **分组**：每 `G` 个连续 `L` 进制 token 打包成一个 `L^G` 词表的大 token，得到 `d' = d/G` 个分组 token：

  ```
  z̃_t = (z̃_t^0, ..., z̃_t^{d'-1}),   每个 z̃_t^j ∈ {0,...,L^G - 1}
  ```

### 自回归分解（式 5）

```
p_θ(z̃_t | s_t) = p_θ(z̃_t^0 | s_t) · Π_{j=1}^{d'-1} p_θ(z̃_t^j | s_t, z̃_t^{<j})
```

- 用 **GPT 式 Transformer decoder + 因果自注意力**：token `j` 只 attend 到之前的 `z̃_t^{<j}`，保证训练/推理一致。
- 每步预测下一码的类别分布（`L^G` 类）。

### 训练（式 6）
- **Teacher forcing + 交叉熵**：

  ```
  L_CE = - Σ_{j=0}^{d'-1} log p_θ(z̃_t^j | s_t, z̃_t^{<j})
  ```

  目标 token 由（冻结的）FSQ 编码器产出。

### 推理
- 每个时刻 `t`，自回归采样 `z̃_t^j ~ p(· | s_t, z̃_t^{<j})`。
- **nucleus (top-p) 采样** [Holtzman 2020]：只从累积概率前 `p` 的高概率子集采样，抑制低概率离群、同时保多样性。
- 采完整序列 `z̃_t` → **冻结 FSQ 解码器**解码成动作。

### 伪代码

```python
# Stage 2: 训练（teacher forcing）
for batch in dataset:
    s, ref = batch
    z_disc = fsq_encoder(ref)             # 冻结，产生 d 个 L 进制 token
    tokens = group(z_disc, G)             # d' 个 L^G 进制 token
    logits = transformer(s, tokens[:-1])  # 因果自注意力，预测下一 token
    loss   = cross_entropy(logits, tokens)
    loss.backward(); opt.step()

# Stage 2: 推理（自回归 + nucleus 采样）
def act(state):
    toks = []
    for j in range(d_prime):
        logits = transformer(state, toks)          # 只 attend 已生成
        toks.append(nucleus_sample(softmax(logits), top_p))
    z_disc = ungroup(toks, G)
    return fsq_decoder(state, z_disc)              # 冻结解码器 → 动作
```

---

## Stage 3: Task Adaptation — CoLA（§7）

### 7.1 Conditional Low-rank Adaptation（式 7）

在预训练权重 `W₀ ∈ R^{d_out × d_in}` 上加低秩、任务条件化的更新：

```
Wx = W₀x + m · [ B(diag(γ(c))·Ax + β(c)) ] / ‖ B(diag(γ(c))·Ax + β(c)) ‖_F
```

- `A ∈ R^{r×d_in}`, `B ∈ R^{d_out×r}`：可训练低秩矩阵，`r ≪ min(d_in, d_out)`。
- `m`：学习到的幅度向量（DoRA 的幅度/方向解耦）。
- `γ(c), β(c)`：轻量 MLP 从任务观测 `c` 生成，在低秩空间 `R^r` 做 **FiLM** 调制。
- 冻结 `W₀`，仅训 CoLA 参数 → **新增 <1% 参数**。

> 直觉：DoRA 提供稳定的低秩方向更新，FiLM 让同一套低秩子空间按任务上下文动态调制。

### 7.2 RLFT（RL 微调）
- **PPO** 优化任务奖励。
- 动作空间 = 长度 `d'` 的离散技能 token 序列，每 token 从 `L^G` 类分布采。
- 自回归预测 token，条件于当前状态 + 任务观测；冻结 FSQ 解码器解成动作。
- per-token log-prob 聚合成 PPO 目标，同一决策步内 token 共享 advantage。
- **探索正则**：用**无条件生成模型**引导 nucleus 采样，把探索约束在高概率（自然）区域，防止 RL 微调破坏自然行为。

### 7.3 SFT（监督微调，可选前置）
- 当有示例动作时，先 SFT：对离散码做交叉熵，提升「任务相关技能 token」的似然，把控制器偏向合适技能子集。
- 再从 SFT 初始化做 RLFT。
- 效果（Table 4）：SFT 降低预测 logits 的熵 → 更一致的风格化行为（如蹲走），任务性能不降。

---

## 关键超参与设置

| 项 | 值 |
|------|------|
| FSQ token 数 `d` | 40 |
| 每 token 量化级 `L` | 9 |
| 隐式码本 | `9^40` |
| 分组因子 `G` | 5（甜点，Table 3） |
| 优化器/算法 | PPO |
| 量化梯度 | STE 直通估计器 |
| 采样 | nucleus (top-p) |
| PEFT 参数占比 | <1% |
| 仿真 | Isaac Gym + ProtoMotions |
| 训练硬件 | 24× A100（Stage 1&2） |
| 推理硬件 | 单 RTX 4090 |
| 成功判据 | 全帧平均关节位置误差 < 0.5m |

---

## 实现坑点（Implementation Pitfalls）

1. **STE 必须正确**：`z_ste = z_cont + (z_disc - z_cont).detach()`，写反或忘 detach 会让量化不可导或梯度错误。
2. **tanh 饱和**：若编码器输出量级过大，`tanh` 饱和使多数维挤到边界级，隐式码本利用率下降。需注意编码器输出的初始化/归一化。
3. **分组因子权衡**：G 太大 → 词表 `L^G` 爆炸（G=8 → 4.3×10⁷，内存不可行）；G 太小 → 序列过长、自注意力昂贵。先从 G=5 试。
4. **端到端 vs 蒸馏**：别图省事先监督训编码器再冻结（FSQ-K），会显著掉点（Table 2）。可执行性必须进 RL 目标。
5. **RLFT 探索崩坏**：不加无条件生成模型引导的 nucleus 采样，RL 容易找到「高分但抽搐」的非自然策略。
6. **SFT 的多样性代价**：SFT 会降熵/多样性（Table 4），若下游需要多样行为需谨慎使用或调权重。
7. **teacher forcing / 因果掩码一致性**：训练用 teacher forcing，务必保证因果掩码与推理时逐 token 生成的依赖一致，否则 train/inference 失配。

---

## 复现风险（Reproduction Risks）

- **数据集**：Bones（680h，商业/特定数据集）不一定公开可得；AMASS（40h）公开可复现但规模小。
- **算力**：24× A100 训练，个人难复现完整规模；可先在 AMASS 小规模验证方法。
- **框架依赖**：ProtoMotions + Isaac Gym，环境/版本敏感。
- **代码**：论文给出 `github.com/NVLabs/`（未指向具体 repo），发布状态需确认。
