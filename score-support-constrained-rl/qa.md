# Q&A — SCORE 自测 15 题

> 5 基础 · 5 进阶 · 5 高阶。先自己想，再展开答案。

---

## 基础（Basic）

### 1. SCORE 是哪几个词的缩写？它要解决的一句话问题是什么？

<details>
<summary>Answer</summary>

**S**upport-**C**onstrained **O**ff-Domain **RE**inforcement（支撑集约束离域强化）。

要解决的问题：**离域策略提升（Off-Domain Policy Improvement）**——给定一个在真实世界学到的策略 π_base，**仅用在相关源域（仿真）中的交互**，得到一个真实世界表现更好的策略 π̂，使 J_real(π̂) > J_real(π_base)。核心：真机零额外 RL 交互，全靠仿真提升。

</details>

---

### 2. 论文对比了哪三种优化范式？各自的一句话结局是什么？

<details>
<summary>Answer</summary>

1. **无约束 RL**：钻仿真漏洞拿高分 → reward hacking → 真机崩溃（如 FPO 学会用手砸桌子夹球）。
2. **分布约束 RL**：`D(π, π_base) < ε` → 陷入"提升 vs 迁移"权衡（松则不迁移，紧则不提升）。
3. **支撑集约束 RL**：`supp(π) ⊆ supp(π_base)` → 在先验支持的动作里挑最优模式 → 快、鲁棒、可迁移（SCORE）。

</details>

---

### 3. SCORE 的主结果数字是多少？

<details>
<summary>Answer</summary>

8 个真机灵巧操作任务上，平均成功率 **37.8%（Base）→ 89.9%（SCORE）**，最佳基线 59.5%；执行速度比 Base 快 **36.8%**（Residual-RL 只快 19.0%）。硬件：Franka FR3 + LEAP Hand。

</details>

---

### 4. SCORE 在训练中会更新基座策略 π_base 的权重吗？

<details>
<summary>Answer</summary>

**不会。** π_base 被**冻结**，全程不接受梯度更新。RL 只学一个轻量的 steering 策略 π_steer(z|o)，作用在 π_base 的隐噪声输入 z 上。部署时也不蒸馏、不真机微调，直接把 π_steer 与 π_base 组合上真机。

</details>

---

### 5. 基座策略 π_base 用什么生成式模型？动作是怎么产生的？

<details>
<summary>Answer</summary>

用 **flow matching**。训练时学一个速度场 v_θ(x_τ, τ, o)，把高斯先验 p₀ 输运到行为分布（目标 Eq.4：`E‖v_θ(x_τ,τ,o) − (a−x₀)‖²`）。推理时抽 z~p₀，从 x₀=z 积分 ODE `dx_τ/dτ = v_θ` 到 x₁，得动作 `a = x₁ = π_base(o, z)`。给定 o，它是**从噪声 z 到动作 a 的确定性映射**。

</details>

---

## 进阶（Intermediate）

### 6. 为什么分布约束会陷入"提升 vs 迁移"的根本矛盾，而支撑集约束不会？

<details>
<summary>Answer</summary>

分布约束 `D(π, π_base) < ε` 把两件事耦合了：
- ε 大 → 可远离 base → 提升空间大，但会跑到 base 没覆盖、仿真才有的动作 → **不迁移**。
- ε 小 → 贴着 base → 迁移好，但被 base 的次优模式拖住 → **不提升**。

支撑集约束 `supp(π) ⊆ supp(π_base)` 只管"动作在不在先验的可能集合里"，**不管概率如何分配**。因此集合内 RL 可任意重分配概率（把慢模式压到 0、快模式拉到 1），既保证每个动作可迁移（base 能生成），又完全放开提升自由。**它把"用哪些动作"和"概率怎么分"解耦了。**

</details>

---

### 7. Flow steering 为什么能"免费"实现支撑集约束？

<details>
<summary>Answer</summary>

因为给定 o，`a = π_base(o, z)` 是从 z 到 a 的确定性映射，其像集 `A_base(o) = {π_base(o,z) | z∈Z}` 就是基座能生成的全部动作。**只要只调 z、不改 flow 网络**，无论 π_steer 把 z 分布搞成什么样，输出动作**永远落在 A_base(o) 内**。于是支撑集约束不需要判别器、不需要改 Q 值、不需要额外正则项——它由"只在噪声空间做 RL"这个动作结构**自动保证**。这是 SCORE 极简的根源。

</details>

---

### 8. DSRL 和 RFS 有什么区别？为什么 SCORE 选 RFS？

<details>
<summary>Answer</summary>

- **DSRL**（`SCORE-DSRL`）：纯 latent steering，`a_k = π_base(o_k, z_k)`，动作严格 ∈ A_base(o) → **硬支撑集约束**。
- **RFS**（`SCORE`）：额外加小残差，`a_k = π_base(o_k, z_k) + a_r` → **软支撑集约束**，在支撑集附近局部精修。

选 RFS 是因为**灵巧操作需要那点额外精度**——DSRL 的动作被限制在基座能生成的离散模式集合里，精度不够。实验中 SCORE(RFS) 略优于 SCORE-DSRL。注意 a_r 必须"小"，否则退化成 Residual-RL 甚至无约束，破坏迁移保证。

</details>

---

### 9. 非对称 actor-critic 是什么？为什么对 SCORE 重要？

<details>
<summary>Answer</summary>

- **Actor**：只接收**部署观测**（点云 + 本体感觉 + 历史），输出 latent steering 分布 → 保证部署时可用。
- **Critic**：训练时额外使用**特权仿真状态**（privileged state）→ 更强 value 估计。

重要性：灵巧操作 + 稀疏奖励下成功接触序列稀有，样本效率是瓶颈。特权 critic 提供更强学习信号，显著提升样本效率与最终仿真成功率（Fig.11），同时**不改变部署策略**（critic 只训练期用）。若改用对称 critic（也只给点云+本体感觉），性能明显下降。

</details>

---

### 10. "retry 数据"和"play 数据"对 base 策略无益，为什么却能提升 SCORE？

<details>
<summary>Answer</summary>

因为它们**扩大了支撑集**。retry 数据（故意失误后恢复）和 play 数据（次优交互）不改善基座策略的平均表现（对传统 IL 是噪声），但给 A_base(o) 增加了新选项——比如"失败后恢复"的动作模式。SCORE 能在仿真里学会**稳定触发**这些原本罕见的选项。实证：Cube Pinch retry 让 SCORE 40%→100%；play 数据让 SCORE 30%→64%。由此得出论文的数据观：**覆盖度 > 最优性**。

</details>

---

## 高阶（Advanced）

### 11. SCORE 的提升能力有一个根本上限，它是什么？跨物体实验如何印证？

<details>
<summary>Answer</summary>

上限 = **基座策略的支撑集覆盖度**。SCORE 只能在 A_base(o) 内挑动作，**基座根本做不到的动作它也变不出来**——这既是它的安全来源（不会臆造不可迁移动作），也是它的天花板（无法获得 OOD 新技能）。

跨物体实验印证：把 **bottle-grasp** 先验 steer 去抓胡萝卜，成功率 22%→67%——因为该先验里**偶尔含捏取动作**（如抓瓶颈），SCORE 能把它放大；而 **cup** 先验**根本不含捏取**，steer 去抓胡萝卜则**失败**。两个对照精确地说明：支撑集里"有没有那个模式"决定了 SCORE 能不能提升。

</details>

---

### 12. 为什么会出现"恢复行为遗忘"？论文用什么补丁对抗？这个补丁揭示了什么隐忧？

<details>
<summary>Answer</summary>

**成因**：训练后期策略学会稳定的首次成功后，就**很少再访问失败状态**，于是逐渐遗忘"失败后如何恢复"的行为（分布漂移导致的灾难性遗忘的一种）。

**补丁**：**人工失败注入（artificial failures）**——在 bottle/cube 任务里，接触后按随机间隔位移物体，强制策略反复经历失败并练习恢复。去掉它真机成功率仅从 100% 降到 95%（因多数评估首次即成功），但**定性上恢复明显变差**。

**隐忧**：这是个**手工设计的课程/干预**，说明 SCORE 的"免 reward engineering"卖点并不彻底——工程复杂度从奖励设计转移到了环境/课程设计（域随机化、失败注入、仿真构建）。鲁棒性依赖这些人工干预是可扩展性的一个软肋。

</details>

---

### 13. 从形式上看，SCORE 和 offline RL 里的支撑集约束是什么关系？它做了什么关键"迁移"？

<details>
<summary>Answer</summary>

**思想同源**：offline RL 的支撑集约束（[Singh 2022], [Mao 2023], REFORM [33] 等）限制策略只用"数据集覆盖的动作"，防止 value 函数在 OOD 动作上外推出错。

**关键迁移**：SCORE 把这个约束从"**纯离线、无新交互**"的设定，搬到"**离域在线**"设定——**在线交互可用，但只在仿真里**。这带来两点不同：
1. offline RL 常需 discriminator / 保守 Q / 多轮真实交互来实现约束，超参多、不稳定；SCORE 用 **flow steering** 在大规模并行**在线**仿真里优雅实现，无需这些机制。
2. 目标从"防 value 外推"扩展为"防 sim-real 迁移失败"——支撑集约束在这里同时起到了**可迁移性保证**的作用（动作 ∈ base 可生成 ⇒ 真实数据里出现过 ⇒ 可迁移）。

</details>

---

### 14. 若把 RFS 的残差 a_r 逐渐放大，SCORE 会退化成哪些已知方法？说明这条"约束强度谱"。

<details>
<summary>Answer</summary>

存在一条**约束强度连续谱**：

```
硬 ←──────────────────────────────────────────→ 软/无
DSRL          RFS(小 a_r)=SCORE      Residual-RL(有界大 a_r)      无约束 RL(FPO)
纯支撑集       软支撑集局部精修         Wasserstein-1 分布约束         钻仿真漏洞
```

- **a_r = 0**：退化为 **DSRL**，纯硬支撑集约束，动作严格 ∈ A_base(o)。
- **a_r 小**：**SCORE(RFS)**，在支撑集附近局部精修，兼顾迁移与精度。
- **a_r 有界但较大**：接近 **Residual-RL**，等价于 Wasserstein-1 分布约束——能局部扰动但无法切换到 base 的另一个模式，提升受限（速度只 +19%）。
- **a_r 无界**：退化为 **无约束 RL（FPO 类）**，可任意臆造动作 → reward hacking → 真机崩。

这条谱说明 SCORE 的关键不是"有没有残差"，而是**把残差保持得足够小**，让支撑集的迁移保证不被破坏，同时借"重选 z"（而非"扰动 a"）获得 Residual-RL 拿不到的模式切换能力。

</details>

---

### 15. 作者暗示 SCORE 会"substantially benefit from large-scale multi-task settings"。基于论文机制，为什么这个方向合理？它把机器人学习的什么范式往前推了一步？

<details>
<summary>Answer</summary>

**为什么合理**：SCORE 的提升上限 = 支撑集覆盖度。多任务实验已显示：单一多任务 base 虽逊于单任务 base，但 steer 后不仅恢复强性能，还能**跨任务复用覆盖**（Cube Pinch 在更宽 Bottle-Grasp reset 下 55%→95%）。即：**更多任务的数据合并 → 更广的共享支撑集 → 每个任务可借用的动作模式更多 → SCORE 提升更大**。大规模多任务预训练正是"最大化行为覆盖度"的天然途径，与 SCORE"覆盖度 > 最优性"的数据观完美契合。

**范式推进**：它把预训练目标从"**训出最强的零样本策略**"重新定义为"**提供最广的、可被 steer 的行为先验**"。这与 LLM/VLA 领域"预训练大而广的基座 + 轻量后训练对齐"的范式同构——SCORE 相当于把"post-training / steering"这套思路带进了机器人操作，且 steering 发生在**仿真**而非真机，几乎零边际数据成本。作者明确把 future work 指向"为 steering 而设计数据集与预训练算法，而非为零样本性能"。

</details>

---

## 复盘提示

答完后自检：你能不看答案说清楚 **(a)** support vs distribution 的区别、**(b)** flow steering 为何自动满足支撑集约束、**(c)** 覆盖度为何比最优性更重要 这三点吗？能，则本篇核心已掌握。
