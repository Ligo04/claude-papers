# ABC: Scalable Behavior Cloning — 方法论拆解

> arXiv 2606.27375v1 · UC Berkeley / MIT / Amazon FAR / XDOF / CMU · https://abc.bot

---

## 1. 系统总览

ABC 不是单一模型，而是一整套**行为克隆（BC）操作栈**，由四大组件构成一个「数据 → 模型 → 三级评测代理 → 迭代」的闭环。核心设计哲学：真机评测（真实成功率）是唯一「gold standard」，但极其昂贵（每个 checkpoint 要跑数十次 trial），所以需要更便宜的**代理指标（proxy metrics）**在各个环节前置过滤设计决策，只有最有希望的候选才送去烧真机时间。

四大组件：

- **ABC-130K**：3,553 小时 / 134,806 episodes / 195 任务的双臂遥操作数据集（$8,000 YAM 平台采集）。
- **ABC-Models**：ABC-DiT（diffusion transformer）与 ABC-VLA（VLM + diffusion action head）两族策略，用真机 ablation 定架构。
- **ABC-Sim**：MuJoCo + Blender 仿真栈，400+ 小时 VR 仿真遥操作数据，验证 sim-real 相关性（r=0.85~0.91）。
- **ABC-Eval**：>100 小时真机 rollout + 评分 rubric 的大规模评测套件。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              ABC 闭环                                     │
│                                                                           │
│   ┌───────────────┐        ┌───────────────────────┐                    │
│   │  ABC-130K     │        │      ABC-Models        │                   │
│   │ 3,553h/134.8K │───────▶│  ABC-DiT (2B)          │                   │
│   │ episodes/195  │        │  ABC-VLA (Gemma3-4B    │                   │
│   │ tasks, YAM     │        │           +42M DiT head)│                  │
│   │ ($8,000)      │        └───────────┬───────────┘                    │
│   └───────────────┘                    │                                 │
│                                          ▼                                 │
│                     ┌────────────────────────────────────────┐          │
│                     │        三级评测代理（由廉价到昂贵）        │          │
│                     │                                          │          │
│                     │  L1 离线指标                             │          │
│                     │   train loss / val action error(固定步数) │          │
│                     │   ──r=-0.84~-0.89, 便宜但粗──▶ 过滤明显差的设计│      │
│                     │             │                            │          │
│                     │             ▼                            │          │
│                     │  L2 ABC-Sim (MuJoCo 240Hz / Blender)     │          │
│                     │   ──r=0.85(strict)/0.91(progress)──▶     │          │
│                     │   无硬件也能迭代设计决策                    │          │
│                     │             │                            │          │
│                     │             ▼                            │          │
│                     │  L3 真机 ABC-Eval                        │          │
│                     │   3任务×50 trials，唯一 ground truth      │          │
│                     └────────────────┬─────────────────────────┘         │
│                                       │                                    │
│                                       ▼                                    │
│                      迭代：架构选择 / 算力分配 / DAgger 数据补采             │
│                                       │                                    │
│                                       └──────────────▶ 回到 ABC-Models     │
└─────────────────────────────────────────────────────────────────────────┘
```

闭环的关键洞察：**代理指标越靠近数据采集端越便宜、但与真机相关性越弱**；ABC 的贡献之一就是量化了每一级代理与真机的相关系数，告诉后来者「在哪一级可以放心地做设计决策，在哪一级必须烧真机时间」。

---

## 2. ABC-130K 数据管线（第 2 节）

### 2.1 规模与硬件

| 指标 | 数值 |
|---|---|
| Episodes | 134,806 |
| 任务数 | 195 |
| 总时长 | 3,553 小时 |
| 采集平台 | 双臂 YAM station |
| 平台成本 | $8,000 |

对比同类数据集：DROID（350h 单臂 Franka）、BridgeData-V2（~100h WidowX）、RH20T（110k 轨迹/41 技能/5 单臂平台）、AgiBot-World（~3,000h 双臂，$30,000 AgiBot G1）、MolmoAct2-BimanualYAM（720h/28 任务，同为 YAM）。ABC-130K 的定位是「低成本双臂 + 大规模 + 覆盖精细灵巧任务」的组合，在这三个维度上没有直接对标者同时占优。

### 2.2 任务体系：7 个 primitive 类别

195 个任务按**主导接触模式与控制策略**归为 7 类：

```
Pick-and-Place ─────┐  ← 任务数最多
Fine Pick-and-Place ─┤  ← 任务数最多
Insertion/Ejection ──┤
Folding ─────────────┼─▶ 每类内变化物体 + 初始配置，构成任务内多样性
Sorting ─────────────┤
Tool Use ────────────┤
Tying/Untying ───────┘
```

难度谱系从简单的 pick-and-place、handover 一路延伸到高难灵巧行为：折纸飞机、折纸箱（后述 DAgger 案例）、从钱包里精细取出信用卡。

### 2.3 Subgoal 元数据

- 全量元数据：匿名遥操作者 ID + 采集时间戳（可用于分析 operator 差异，但论文未深入展开这块分析）。
- **1,552 小时子集**带 subgoal 标注：把连续子轨迹标注为子任务的文字描述，为语言条件模型（尤其 ABC-VLA）和分层策略研究提供更细粒度的监督信号。

### 2.4 存储与 dataloader 吞吐挑战

数据以**多相机视角编码视频**存储而非原始帧序列（节省空间，符合规模化数据集惯例），但代价是：

- 部分 episode 长达 **469 秒**，单 episode 内 naive 解码全部帧的开销很大；
- 训练时随机 shuffle 采样一个 batch 内的 timestep 会命中大量不同视频文件的不同帧位置，**video decode 成为吞吐瓶颈**而非 GPU 算力；
- 论文在 Appendix B.1 专门讨论 dataloader 优化（如何在编码存储与随机访问之间取得平衡）——这是复现 ABC 训练管线时最容易被低估的工程细节，很多团队会把预算全部规划给 GPU 而忽视 I/O 层。

---

## 3. ABC-Models 架构拆解（重点）

### 3.1 统一 IO 规范

所有 ABC-DiT / ABC-VLA 变体共享同一套输入输出接口，使得架构 ablation 是「apples-to-apples」的：

```
输入:
  image_top      : 3×224×224   (letterbox padding 处理非方形原图)
  image_gripper_L : 3×224×224
  image_gripper_R : 3×224×224
  proprio        : 14D          (双臂关节状态)

输出:
  action_chunk   : 30 steps × 绝对关节位置目标

归一化:
  state, action  → z-score normalize (基于训练集统计量)

执行:
  frequency      = 30 Hz
  mode           = 异步执行 + action prefix
                   (防止 mode switching / 保证轨迹平滑, follow prior work)
```

「action prefix」异步执行的直觉：策略每次前向产出 30 步的 chunk，但执行时不是等一整个 chunk 跑完再重新推理，而是让新 chunk 与旧 chunk 的重叠前缀做平滑衔接，避免相邻两次推理结果不一致导致的动作突变（mode switching）。

训练硬件为 NVIDIA **H200**；需要注意第 3 节的架构 ablation 用的是**内部 7,000 小时语料**（发布数据集之前的开发数据），而第 5 节最终模型才切换到公开的 **ABC-130K**——这一点对复现有重要含义（见第 7 节风险）。

### 3.2 ABC-DiT：三变体信息流对比

统一规格：**32 层 / 24 attention heads / hidden 1536**（刻意大于 DiT-XL，确保容量不是限制架构结论的瓶颈）。

**AdaLN 路径**（信息被压缩成单一 conditioning 向量）：

```
image (per-view) ──CLIP ViT-B──▶ CLS token ─┐
                                              ├─concat─▶ visual repr (紧凑)
image (per-view) ──CLIP ViT-B──▶ CLS token ─┘
language ──CLIP text encoder──▶ MLP ────────┐
proprio ─────────────────────────────────────┼─▶ 单一 conditioning vector
diffusion timestep ───────────────────────────┘        │
                                                          ▼
                                          adaLN(shift, scale, gate)
                                                          │
                                                          ▼
                                   注入 DiT 每层 (信息已被压扁成 1 个 token 的容量)
```

**Cross-Attention 路径**（保留 token 级空间信息）：

```
image (per-view) ──ViT──▶ 视觉 token 序列 (patch-level, 未 pool)
                              │
                    12 个 learned latent query / 图
                              │
              ┌───────────────▼────────────────┐
              │  Cross-Attention layer          │  ← 插在每个 self-attn 层之后
              │  (latent query attend 视觉 token)│
              └───────────────┬────────────────┘
                              │
                  DiT 主干继续处理 (token 级细粒度信息未被压扁)
                              │
        [语言/proprio/timestep 仍走 AdaLN 通道，未变]
```

三变体：

| 变体 | 视觉编码器 | 视觉条件注入方式 |
|---|---|---|
| CLIP-AdaLN（baseline，灵感 LBM） | CLIP ViT-B | 单 token AdaLN |
| CLIP-Cross-Attention | CLIP ViT-B | 12 latent query cross-attn |
| DINO-Cross-Attention | **DINOv3 ViT-B** | 12 latent query cross-attn |

结论分两层：① **pooled cross-attention > AdaLN**（token 级视觉信息对操作任务重要，压成单 token 丢失太多空间线索）；② **DINOv3 > CLIP**（DINOv3 更 pixel-aligned，视觉表示本身是关键设计变量）。**最终 ABC-DiT = DINOv3 + pooled cross-attention，2B 参数**。

训练超参：200k steps，global batch 4608，LR warmup 1000 iter 线性升到 1e-4 后恒定，AdamW（wd 0.01），视觉编码器 LR scale 0.1，grad clip 10，proprio dropout 0.1。

### 3.3 ABC-VLA：三 connector 对比

骨干：**Gemma 3**（4B，SigLIP 视觉编码器 + Gemma 语言解码器）+ **42M 参数的小型 DiT action head**。论文明确指出**冻结 VLM 表现差**，必须用动作预测目标去微调骨干——这直接推翻了「VLM 只当固定特征提取器」的省算力思路。

三种 connector（固定骨干/动作头/训练配方，只变 connector 设计）：

| Connector | 机制 | Mean progress | 备注 |
|---|---|---|---|
| Cross-Attention | VLM 末层 token 投影为 DiT cross-attn 的 K/V，噪声动作 token 做 query，diffusion 梯度直接回流进 VLM | **11.7%**（最差） | 梯度直接冲击 VLM 表示，训练不稳 |
| Cross-Attn + FAST | FAST-tokenized 动作拼进 VLM 序列用 next-token CE 微调；DiT cross-attend 到 **detach** 的末层特征隔离骨干免受 diffusion 梯度 | **32.6%** | 略优于 π0.5 式复用 VLM K/V |
| AdaLN（改自 LBM-2） | VLM 特征经 attention-pooling(QK-Norm) 压成 8 个 512 维 token，flatten+投影产生 DiT adaLN shift/scale/gate；动作 token 不直接 attend VLM token | **61.4%**（最优） | diffusion 梯度并非天然与 VLM 不兼容，关键是接口设计 |

信息流对比（AdaLN 为何更稳）：

```
[Cross-Attn 系] VLM token ──(K,V)──▶ DiT cross-attn ◀──(Q)── noisy action token
                    ▲                                              │
                    └──────────── diffusion 梯度直接回流 ───────────┘
                    （VLM 表示被 diffusion loss 的噪声/多样性直接扰动）

[AdaLN 系]      VLM token ──attention-pool(QK-Norm)──▶ 8×512 压缩表示
                                        │
                              flatten + 投影 ──▶ (shift, scale, gate)
                                        │
                    DiT 主干 ◀── adaLN 调制 (间接注入，梯度回流路径更平滑)
```

结论：AdaLN 把 VLM 输出先"蒸馏"成低维统计量再调制 DiT，相当于给梯度加了一层缓冲，避免 diffusion 训练固有的高方差梯度直接冲击大模型骨干的表示空间。

### 3.4 Variance Reduction「多次采样」trick

动机：diffusion loss 本质是对数据 x、噪声 ε、扩散时间步 τ 的期望 `E_{x,ε,τ}[...]`，标准做法每个样本只采一组 (ε, τ) 做蒙特卡洛估计，方差较大。ABC-VLA 里 DiT action head（42M）远小于 VLM 骨干（4B），可以**摊销单次 VLM forward 到多组 (ε, τ) 上**：

```python
# 伪代码：Variance-Reduction 多次采样
def forward_with_variance_reduction(vlm, dit_head, image, text, proprio,
                                      clean_action, k):
    # 1) VLM 只跑一次前向（成本大头）
    vlm_features = vlm(image, text)                     # 4B 参数，一次前向

    # 2) 将 VLM 特征与干净动作在 batch 维复制 k 份
    vlm_features_rep = repeat(vlm_features, k)           # (B*k, ...)
    clean_action_rep = repeat(clean_action, k)            # (B*k, ...)

    # 3) 每份独立采样一组 (epsilon, tau) —— 这是方差降低的关键
    eps = sample_noise(B * k)
    tau = sample_timestep(B * k)
    noisy_action = add_noise(clean_action_rep, eps, tau)

    # 4) DiT head 对 k 份并行跑（小模型，成本可忽略）
    pred = dit_head(vlm_features_rep, noisy_action, tau)  # 42M 参数
    loss = mse(pred, eps)

    # 5) 反向传播：在 conditioning 接口处对 k 份梯度求平均
    #    再传回 VLM —— VLM 反向成本与 k 无关（只做一次）
    loss.backward()  # autograd 在 vlm_features_rep -> vlm_features 处自动聚合
    return loss
```

关键性质：**VLM 前向 + 反向的成本与 k 无关**（只算一次），只有 DiT head 部分的计算随 k 线性增长，但 DiT head 本身很小，所以整体开销几乎可忽略。实测 H200 上：

| k | 单步耗时 |
|---|---|
| 1 | 1.346 s/step |
| 8 | 1.366 s/step |

即用 8 倍的梯度估计样本量换来更低方差的梯度，额外时间成本仅 **~1.5%**。这是一个「小模块搭便车大模型前向」的通用技巧，适用于任何「大 backbone + 小 diffusion head」的架构。

---

## 4. 三级评测代理

### 4.1 总体流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  L1 离线指标  │────▶│  L2 仿真     │────▶│  L3 真机      │
│  (最便宜)     │     │  (中等成本)   │     │  (最贵/ground │
│              │     │              │     │   truth)      │
└──────────────┘     └──────────────┘     └──────────────┘
   train loss           MuJoCo 240Hz         3 任务
   val action error      Blender 高保真        × 50 trials
   (固定 diffusion步数)   VR 遥操作数据
   val loss(无用)         400+h
```

### 4.2 L1 离线指标（第 3.4 节）

伪代码描述评测方式：

```python
# 离线代理指标计算
def offline_proxy_metrics(model, val_set):
    train_loss = model.train_loss_at_convergence
    val_loss = compute_diffusion_loss(model, val_set)          # 训练目标本身
    # 关键：验证 action error 必须固定 diffusion 采样步数(10 steps)
    # 且不加 action prefix，否则可以 trivially 降低误差而不代表真实能力提升
    val_action_error = compute_L2_error(
        model.sample(val_set, diffusion_steps=10, use_prefix=False),
        val_set.ground_truth_action
    )
    return train_loss, val_loss, val_action_error
```

与真机 strict/progress 的相关系数：

| 离线指标 | 与真机相关性 | 结论 |
|---|---|---|
| Val action error（固定10步） | Pearson r=-0.89 (p=3.8e-6), Spearman ρ=-0.93 | **最强**，但仅在固定步数下有意义 |
| Train loss | Pearson r=-0.84 | 显著负相关 |
| Val loss | Pearson r=-0.04 (p=0.886) | **不显著/无用**——训练早期降后期升，但真机表现持续改善 |

结论：离线指标是「廉价 scalar 代理」，适合快速迭代小设计决策（如学习率、归一化方式），但**无法可靠预测成功率，也无法揭示具体失败模式**（比如策略是卡在抓取还是卡在放置）——这需要更高保真的代理，也就是仿真。

### 4.3 L2 ABC-Sim（第 4 节）

- 渲染两档：**MuJoCo 240Hz**（低保真，训练/评测主力）+ **Blender path-tracing**（高保真重渲染，可选用于训练）。
- 视觉观测以 **60Hz** 推给 VR 头显（Meta Quest / Apple Vision Pro），遥操作用 **GELLO** leader arm 驱动仿真里的 YAM arm；pass-through 渲染看到真实世界背景以缓解晕动症。
- 发布 10 个仿真任务 + **400+ 小时** VR 仿真遥操作数据。
- **Sim-to-real 相关性实验**设计：ABC-DiT/ABC-VLA 在 ABC-130K 上各选 4 个 checkpoint + 4 个「内部 7K + 仿真数据混合」的 ABC-DiT checkpoint，共 **12 个 checkpoint**，每个在仿真和真机各跑 3 任务 × 50 trials，比较排名/分数一致性。

| 指标 | Pearson r | p-value | n |
|---|---|---|---|
| Strict success | 0.85 | 4.2e-4 | 12 |
| Task progress | 0.91 | 5.0e-5 | 12 |

结论：仿真是真机表现的**有意义预测器**，可以在昂贵真机评测前先在仿真里 debug 架构/数据/超参决策，大幅降低对真机时间的依赖——但仍不是完美替代（r 未到 1.0，且仿真渲染本身低保真，见第 7 节风险）。

### 4.4 L3 真机（ABC-Eval，第 5 节）

- 架构 ablation（Table 1）用 3 个真机任务：Bottles（扔瓶入桶）、Dishrack（装盘入架）、Mugs（翻正杯子），每任务 **50 trials**。
- 下游微调评测用 4 个高精度灵巧任务：钱包取信用卡、LEGO 分拣、插笔帽、拧瓶盖。
- 折箱子 DAgger 案例每轮也用固定次数 rollout 评估进展（见第 6 节）。

### Table 1 架构 ablation 完整数据（真机，200k steps，内部 7,000h 语料）

| 模型 | Mean strict | Mean progress | Latency (ms/chunk, 10 diffusion steps) |
|---|---|---|---|
| VLA Pooled adaLN | 32.8% | 61.4% | 17.24 |
| VLA FAST+X-Attn | 3.6% | 32.6% | 19.24 |
| VLA X-Attn | 0.0% | 11.7% | 19.24 |
| DiT DINOv3-xattn | 32.9% | 67.5% | 37.4 |
| DiT CLIP-adaln | 13.4% | 47.3% | 27.5 |
| DiT CLIP-xattn | 24.5% | 58.8% | 37.5 |

注意两个交叉信息：① VLA 系的 latency 显著低于 DiT 系（17-19ms vs 27-37ms），因为 DiT head 只有 42M 而 ABC-DiT 全量 2B；② strict success 上 VLA-AdaLN（32.8%）与 DiT-DINOv3-xattn（32.9%）几乎打平，但 **progress 上 DiT 更高（67.5% vs 61.4%）**——latency 与精度之间存在权衡，具体选型要看部署延迟约束。

---

## 5. 算力缩放规律（第 3.3 节）

背景：领域内 batch size 选择无共识——OpenVLA 用 2048，LBM 用 2560，GR00T N1 用 16,384。ABC 系统扫描：

```
batch size:  1536 ──── 4608 ──── 9216
steps:       100k ──── 200k

           小 batch (1536)          大 batch (9216)
DiT   :    优于 VLA                提升有限（已接近饱和）
VLA   :    弱于 DiT                大幅跃升，反超 DiT
```

- **小 batch 时 DiT 优于 VLA**（VLA 骨干大，小 batch 下有效梯度信号不足以支撑 4B 参数收敛）。
- **scale 到 9216 时 DiT 提升有限，而 VLA 大幅跃升并反超 DiT**——说明 VLA 是「data/batch-hungry」架构，需要更大算力预算才能发挥优势。
- 但**DiT 在同等 FLOPs 下更 compute-efficient**——即如果按 FLOPs 而非 wall-clock 或参数量记账，DiT 的性价比更高。

对实践的含义：预算有限（小算力/小 batch）选 DiT；有充裕大 batch 训练预算且能承受 VLA 的延迟，则 VLA 上限更高。

---

## 6. DAgger 灵巧任务配方：折箱子案例（第 5.3 节）

折箱子（fold cardboard box）是长程、需要中途精细纠正的任务，预训练 ABC-DiT 直接零成功，是全文唯一详细展开的 DAgger 案例。

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: 预训练 ABC-DiT (ABC-130K)                            │
│         真机折箱子 → 0% 成功                                  │
└───────────────────────┬───────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 采集 10h "collapsed data"                             │
│         （更严格 SOP，减少轨迹发散）                            │
│         微调 75k steps, batch 720                             │
│         → 仍仅 24% 成功                                        │
│         诊断：策略"理解"任务，但卡在中途微调整/纠正动作           │
└───────────────────────┬───────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Round 1 DAgger:                                                │
│   rollout 当前策略；卡住时人接管遥操作直到完成                    │
│   记录整段 rollout（非仅干预片段！）                              │
│   耗时 ~1-1.5h/round；干预数据占比 ~30%                          │
│   采集条件：微调数据在 cage 内，DAgger 干预在无 cage              │
│                         │                                       │
│   混合训练 (80:10:10):                                          │
│     80% 上一轮 base 数据                                        │
│     10% 当前轮干预数据                                           │
│     10% 当前轮其余(非干预)数据                                    │
│   从上一轮 base 继续训 75k steps                                 │
└───────────────────────┬───────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Round 2 DAgger:                                                │
│   同样流程，干预占比降到 ~15%（策略变强，卡住更少）                │
│   继续 80:10:10 混合训 75k steps                                 │
└───────────────────────┬───────────────────────────────────────┘
                         ▼
                    最终折箱子策略
```

三个关键设计决策，缺一不可：

1. **只采恢复行为而非整段重采**：诊断发现瓶颈是中途纠正而非整体理解，所以 DAgger 目标是让人示范"如何从卡住状态恢复"，而不是重新示范整个任务。
2. **记录整段 rollout 而非仅干预段**：如果只训练干预片段，策略会学会利用「cage 内 vs 无 cage」这种采集条件差异形成的**伪相关**（分布偏移线索）而不是真正学会恢复行为，导致性能反而变差。
3. **80:10:10 比例而非等权混合**：隐式过采样干预数据（干预只占轮次数据的 10-15%，但在 80:10:10 里被更高频次呈现），同时保留大部分原始 base 分布，避免灾难性遗忘。

---

## 7. 复现风险与陷阱

| 风险点 | 具体表现 | 复现时的应对提示 |
|---|---|---|
| **Dataloader 视频解码瓶颈** | Episode 长达 469s，多视角编码视频存储，随机 shuffle 采样导致大量视频文件随机 seek + 解码，I/O/CPU 解码可能先于 GPU 成为瓶颈 | 预算和工程投入不能只倾斜给 GPU；需要专门优化的视频解码/预取管线（论文见 Appendix B.1），否则实际吞吐远低于理论算力 |
| **H200 算力门槛** | 训练与算力缩放实验（batch 1536-9216 × 100k/200k steps，2B 参数 DiT / 4B VLM）默认在 H200 上完成 | 学术组或小团队若只有消费级/上一代数据中心卡，需要重新评估 batch size 与训练步数的可行组合，直接照搬论文超参可能不现实 |
| **架构 ablation 用内部 7,000h 而非公开数据** | Table 1 全部架构对比结论建立在**未公开的内部语料**上，第 5 节最终模型才切到公开 ABC-130K 训练 | 复现者用 ABC-130K 重跑 Table 1 式 ablation 时，**结论的绝对数值甚至相对排序都未必完全一致**——7,000h 内部数据的任务分布/质量与 130K 公开数据不保证同分布，这是论文本身留下的可比性缺口，需要谨慎对待「复现失败」到底是自己实现问题还是数据集差异 |
| **VLA 冻结骨干失效** | 论文明确提到冻结 VLM 直接当特征提取器效果差，必须用动作预测目标微调骨干 | 试图省算力冻结 Gemma 3 骨干来加速训练/减少显存，会直接损失性能，这不是可选项而是必要设计 |
| **DAgger cage/无cage 伪相关** | 折箱子案例中，若只训练 DAgger 干预片段（而非整段 rollout），策略会学到「cage 内 vs 无 cage」这一采集条件差异作为捷径特征，性能不升反降 | 复现 DAgger 流程时必须保持采集条件(cage 状态等环境线索)与训练数据混合方式一致于论文描述，否则容易在自己的任务上复现出「加了 DAgger 数据反而变差」的反直觉结果，且难以定位原因 |
| **Sim 低保真** | ABC-Sim 默认 MuJoCo 渲染保真度低（Blender path-tracing 是可选的高保真补充，非默认） | Sim-to-real r=0.85~0.91 是在论文特定任务集和特定 checkpoint 选择（12个）下测得的相关性，不能想当然地认为"仿真里更好=真机一定更好"对所有任务/所有架构改动都成立，尤其是涉及精细接触/柔性物体（如折纸箱这类任务）时仿真物理保真度本身就是短板 |

---

*本文档基于 `/tmp/claude-paper-downloads/FACTS_ABC.md` 提供的论文事实撰写，所有数字均来自原始 FACTS 底稿，未做推测或编造。*
