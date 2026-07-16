# 方法详解：RoboTTT 架构与训练配方

## 0. 总览

RoboTTT = **GR00T N1.7 主干 + TTT 层（时间维序列建模）+ 序列训练配方**。三个组件各解决一个问题：

```
问题                                  解法
────────────────────────────────    ─────────────────────────────────
如何以恒定推理成本编码长历史          TTT fast weights（梯度下降压缩进权重空间）
如何在保护预训练能力下加新模块        tanh 门控 + 只训新层的预训练策略
如何在固定显存下训练 8K 上下文        TBPTT + sequence action forcing
如何让模型"会用"上下文               损失掩码（视频模仿 / DAgger Distillation）
```

## 1. TTT 机制（预备知识）

fast weights W 参数化一个小网络 f_W: R^d → R^d（此处为两层 GeLU MLP）。对输入 token 序列 X，用可学习投影 θ_Q, θ_K, θ_V 得到 Q, K, V。每个时间步 t 做两步：

**Update（写入）**：让 f_W 把 K_t 映射到 V_t

```
W_t ← W_{t-1} − η ∇_W ℒ_FW( f_{W_{t-1}}(K_t), V_t ),    ℒ_FW = ‖v̂ − v‖²
```

**Apply（读出）**：用刚更新的权重回答查询

```
O_t = f_{W_t}(Q_t)
```

这个 update-then-apply 在**训练和推理时都执行**——推理时也在做梯度下降，故名 Test-Time Training。θ_Q/θ_K/θ_V、W₀、内层学习率 η 全部由外层任务损失学习（η 基础值 0.1，可学习）。

直觉：**update 把上下文编码进参数空间，apply 把它检索出来**。对比 full attention 在内存里保留所有 KV 并每步全量注意，TTT 把历史压缩成固定大小的 W。

## 2. 模型架构

### 2.1 组件与数据流

```
                    每个时间步 t
  图像 o_t ──► Eagle VLM ──► VL tokens Φ_t ─────────┐ (cross-attn，不进 TTT)
  语言 l  ──►                                        │
                                                     ▼
  [R_t (×16 register) | q_t (proprio) | Ã_t (noised actions)]
        │                                            │
        │      DiT block × 16（每块结构如下）          │
        │   ┌────────────────────────────────────┐   │
        └──►│ self-attn + cross-attn(Φ_t)         │◄──┘   ← 单时间步内
            │        │                            │
            │        ▼  沿时间维拼接 X=[R_1,q_1,Ã_1,...,R_T,q_T,Ã_T]
            │   TTT 层: update(Eq.1) → apply(Eq.2) │      ← 跨时间步
            │        │                            │
            │   O = tanh(α)⊙O_TTT + O_attn        │      ← 门控残差
            └────────────────────────────────────┘
                     │
                     ▼  k 步去噪（flow matching）
              action chunk A_t = [a_t, ..., a_{t+H-1}]
```

**分工原则**：attention 处理**单时间步内**的信息（register/proprio/action token 互相注意 + cross-attend 当步 VL tokens），TTT 层处理**跨时间步**的信息。

### 2.2 三个关键设计

- **Register tokens（N=16）**：每个时间步前置 16 个可学习 token，注意所有其他 token。VL tokens Φ 数量大，直接过 TTT 层太贵，于是让少量 register token"打包"VL 信息带过时间维。消融：加 register tokens 相对提升 18%，但对无 TTT 的 GR00T N1.7 毫无帮助——它们只有配合 TTT 的时间建模才有意义。
- **tanh 门控**：每个 DiT 层学一个 α ∈ R^d，初始化 0.001，输出 `O = tanh(α)⊙O_TTT + O_attn`。训练初期 TTT 贡献≈0，模型从纯预训练行为出发逐渐学会用上下文，不破坏预训练能力（借鉴 Flamingo）。
- **fast model 用 MLP 而非线性层**：TTT-Linear 变体比 MLP 差 27%。非线性容量是压缩稠密机器人流的关键。

### 2.3 参数量

GR00T N1.7 DiT 538M；16 个 TTT 层每层约 +10M → 总计 690M。RoPE 位置编码（θ=10000）。

## 3. 序列训练配方

### 3.1 外层目标（元学习）

数据集 D = {ξ⁽ⁱ⁾}，每条轨迹 ξ = {(l, o_t, q_t, A_t)}。训练序列 = 完整轨迹或连续子轨迹（≤最大上下文长度）。序列损失：

```
ℒ_fm(ξ; W₀) = (1/T) Σ_t ℓ_t( ξ_t ; W_{t-1} )
```

内层：TTT 沿序列更新 W（Eq.1）；外层：对每个时间步算 flow-matching 损失、平均后对**全部参数（含 W₀）**求梯度。W₀ 通过 gradient-of-gradients 元学习——fast weight 的初始化和更新动力学被"塑造"成适合机器人轨迹的样子。**这正是 RoboTTT 有上下文缩放律而 GDN 没有的机制根源**：更长的训练序列展开更多内层更新步，给外层更深的塑造空间。

### 3.2 Sequence Action Forcing（关键稳定性技巧）

flow-matching 目标：去噪 Ã_t = τ·A_t + (1−τ)·ε。序列训练时**每个 action chunk 独立采样噪声水平**：

```
τ_t = s·(1−u),   u ~ Beta(1.5, 1),   s = 0.999
ℒ = (1/T) Σ_t E_{τ_t, ε} ‖ v_θ(Φ_t, A_t^{τ_t}, q_t; W_{t-1}) − (A_t − ε) ‖²
```

若整条序列共享一个 τ（full-sequence diffusion 风格），序列会整体一致地"易"（低噪）或"难"（高噪），训练不稳定（呼应 Diffusion Forcing）。消融：去掉它后动作不精确到无法推进任务。

### 3.3 TBPTT（显存与上下文长度解耦）

```
序列:  [═══ segment 1 ═══│═══ segment 2 ═══│═══ segment 3 ═══]
梯度:   ◄──── 段内回传 ────┤◄──── 段内回传 ────┤◄──── 段内回传 ────┤  (边界 detach)
fast W: ────────────────── 连续传递，TTT 不间断 ──────────────────►
```

- 显存 ∝ segment 长度，而非总序列长 → 任意长上下文在固定显存下可训。
- W₀ 仍通过第一段（其更新直接源于 W₀）收到元学习梯度。

### 3.4 训练流程与超参数

| 阶段 | 数据 | 上下文 | 步数 | 硬件 | 可训参数 | LR 调度 |
|------|------|--------|------|------|----------|---------|
| 预训练 | 桌面双臂机器人 + 第一人称人类数据 (EgoScale)，偏长轨迹 | 逐步增至目标（如 8K） | 30K | 16× GB200 | **只训 TTT/GDN 新层**，其余冻结 | WSD，峰值 2e-5 |
| Post-train | 每个下游任务各自数据 | 1K | 20K | 8× GB200 | 全参数 | cosine，峰值 5e-5 |

其他：AdamW（weight decay 1e-5）；per-device batch size：上下文 ≤4K 时为 4（全局 64），>4K 时为 1（全局 16）。预训练轨迹长度分布：41.4% 在 2K-4K，17.1% >4K。

**部署**：YAM 双臂 + 4× RealSense D405（顶/底/左腕/右腕，480p），RTX 5090 工作站，30Hz 控制，每步 k 步去噪出 H 步 action chunk；rollout 从学到的 W₀ 出发，逐步更新并传递 fast weights。

## 4. 从上下文中学习（损失掩码机制）

核心机制：**对选定时间步 mask 掉 flow-matching 损失 → 该时间步只更新 fast weights、不作为模仿目标**（纯上下文）。两个实例：

### 4.1 人类视频一次性模仿

训练：拼接 [ξ_video | ξ_robot]（同一任务配置），视频段掩码损失（只写 fast weights），机器人段算动作损失（条件于已被视频更新过的 fast weights）。每个训练配置收 5-20 条人类视频。所有配置共用同一 prompt "assemble circuit"，逼迫模型只能从视频中辨认目标配置。
测试：给一条**未见配置**的人类视频 → 一次性模仿（6/10 成功 vs GDN 0/10）。

### 4.2 DAgger Distillation

```
DAgger 轨迹: [A^R 机器人动作(次优) | A^H 人类纠正 | A^R ... | A^H ...]

fast weight 更新:  全部时间步（含失败动作）─── "失败是纠正的问题陈述"
flow-matching 损失: 只算 A^H 人类纠正 ─────── "只模仿纠正"
```

非对称设计把"失败→纠正"映射蒸馏进 fast weights（Algorithm Distillation 的机器人实例化）。部署时模型自主执行纠正，其纠正又进入历史被 fast weights 吸收——形成闭环的 on-the-fly 改进。100 条 DAgger 轨迹：标准 DAgger 平均 +9%，DAgger Distillation 平均 +33%（RoboTTT +36%、GDN +29%）。

## 5. 复现要点与坑

1. **必须做 sequence action forcing**——共享噪声水平会直接训崩。这是序列化 flow-matching 策略最重要的坑。
2. **TBPTT 边界只 detach 梯度、不重置 fast weights**——重置会切断跨段上下文积累。
3. **门控 α 初始化要足够小**（0.001），否则未训练的 TTT 输出会污染预训练行为。
4. **预训练冻结主干、只训 TTT 层**；直接全参数长序列预训练成本高且可能损伤 VLM。
5. **内层 LR 可学习但要有合理基值**（0.1）；内层优化器就是普通 SGD（Muon 等留作 future work）。
6. **上下文短于任务时长会掉点**：1K（约半分钟）短于最短任务时，推理时 fast weights 更新步数和 RoPE 位置都超出训练分布。训练上下文应覆盖部署 episode 长度。
7. 评测用**固定初始物体摆放**（记录并复现）保证方法间可比。

## 6. 基线设置（供公平对比参考）

- **GR00T N1.7**：官方实现，单步上下文。
- **GR00T N1.7 Hist.**：官方实现扩展一帧历史。注意它在 Pup Go Car 上**低于**无历史版（39.5% vs 57%）——朴素拼历史引入虚假相关。
- **GDN**：TTT 层原位替换为 Flash Linear Attention 库的 Gated DeltaNet 层，层位置、门控、参数量全部对齐——严格隔离"更新规则"这一个变量。
- 非序列模型训练到匹配的计算预算；序列模型 post-train 用 1K 上下文。
