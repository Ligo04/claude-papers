# 自测题 — GPC（15 题）

> 5 基础 · 5 进阶 · 5 高级。先自己答，再展开对照。

---

## 基础（Basic）

### Q1. GPC 的全称是什么？它把什么范式引入到运动控制？

<details>
<summary>答案</summary>

**Generative Pretrained Controllers（生成式预训练控制器）**。它把 NLP 的「tokenization + 下一个 token 预测 + 预训练/微调」范式引入物理仿真角色的运动控制：先学一套离散运动词表，再用 GPT 式自回归 Transformer 建模，最后用 PEFT 迁移到下游任务。

</details>

---

### Q2. FSQ 是什么？它的量化公式长什么样？

<details>
<summary>答案</summary>

**Finite Scalar Quantization（有限标量量化）**。对编码器输出 `z` 的每一维独立量化：

```
ẑ = ⌊ (L/2) · tanh(z) ⌉
```

`tanh` 把每维压到 (-1,1)，乘 `L/2` 取整落到 `L` 个格点之一。`d` 维联合形成隐式码本 `L^d`（本文 `9^40`）。它**不需要显式码本**。

</details>

---

### Q3. GPC 三个阶段分别是什么？

<details>
<summary>答案</summary>

1. **Skill Quantization（§5）**：用 FSQ + 端到端 RL 构建离散运动词表。
2. **Generative Controller Training（§6）**：用 GPT 式自回归 Transformer 建模技能 token 分布。
3. **Task Adaptation（§7）**：用 CoLA（PEFT）+ SFT/RLFT 适配下游任务。

</details>

---

### Q4. GPC 的跟踪成功率是多少？成功如何判定？

<details>
<summary>答案</summary>

**99.98%**。判定标准：一个 episode 中，**全帧平均关节位置误差 < 0.5m** 即算成功。数据集为 Bones（680h）和 AMASS（40h）。

</details>

---

### Q5. 解码器输出的动作是什么？如何驱动角色？

<details>
<summary>答案</summary>

解码器输出**各关节的目标关节旋转**，作为 **PD（比例-微分）控制器**的目标。PD 控制器把目标转成关节力矩，施加到物理仿真角色上驱动运动。这沿用 DeepMimic 的做法。

</details>

---

## 进阶（Intermediate）

### Q6. FSQ 相比 VQ-VAE 解决了什么核心痛点？

<details>
<summary>答案</summary>

VQ-VAE 需要维护显式码本，训练要三项损失（重构+码本+commitment），并依赖 EMA 更新、死码周期重初始化等启发式，容易**码本坍缩 / 利用率低**。FSQ 把离散集合固定成规则网格（隐式码本），无码本、无辅助损失、无坍缩概念——把「学离散集合」这个难题转化为「编码器把连续特征摆到网格上」的普通连续优化，训练稳定得多。尤其适合大规模 + 高方差 RL 梯度场景。

</details>

---

### Q7. 为什么要做 token 分组（grouping）？G 太大或太小分别有什么问题？

<details>
<summary>答案</summary>

`d=40` 个 9 进制 token 若逐个自回归，序列长（40），自注意力昂贵。分组把每 `G` 个打包成一个 `L^G=9^G` 词表的大 token，得到 `d'=d/G` 个 token，缩短序列、降低算力/内存。

- **G 太小**（如 G=1）：序列长 40，建模长依赖难、算力高。
- **G 太大**（如 G=8）：词表 `9^8≈4.3×10⁷` 超过现代 LLM，内存爆炸不可行。
- **甜点 G=5**：词表 59049、序列 8，定性质量与多样性（APD）最佳。

类比 NLP 从 character-level 到 subword（BPE）。

</details>

---

### Q8. 端到端 RL 训练 FSQ 为什么比「先监督预训编码器再冻结」好？给出数据。

<details>
<summary>答案</summary>

Table 2：FSQ-K（监督预训 kinematic 编码器后冻结）成功率 99.03、MPJPE 78.26、利用率 76.34%；端到端 FSQ 为 99.98、34.90、82.15%。

原因：监督学到的码为「重构 kinematic 轨迹」优化，但一个运动学合理的目标不一定**动力学可执行**。端到端 RL 让离散码直接对「能否被物理仿真真实完成」负责，学到的表征天然对齐控制能力。**可执行性必须进入表征的目标函数。**

</details>

---

### Q9. GPC 的涌现恢复行为（摔倒爬起、被推调步）从哪来？为什么 CVAE 没有？

<details>
<summary>答案</summary>

GPC 是**生成式**的——每步从 `p(z̃|s)` 采样 token，建模的是**整个技能分布**（包含摔倒→爬起、被推→调步这些转移）。当角色进入「快摔倒」状态 `s`，条件分布自然给出恢复技能的 token，**无需专门的恢复奖励**。

CVAE 式控制器**确定性地选一个潜码**完成任务，在训练分布边缘会脆断、行为坍缩（Fig 8/9 显示 CVAE 重复几乎相同动作，GPC 多样）。把控制器从「函数」升级为「分布」是鲁棒性与多样性的来源。

</details>

---

### Q10. CoLA 由哪两个已有技术组合而成？各起什么作用？

<details>
<summary>答案</summary>

**DoRA + FiLM**。

- **DoRA**：把权重更新拆成**幅度 `m`** 与**方向（低秩 `BA`）**，解耦更新更稳定、更有表达力。
- **FiLM**：`γ(c), β(c)` 由轻量 MLP 从任务观测 `c` 生成，在低秩空间 `R^r` 做**任务条件调制**，使同一套低秩子空间按任务上下文动态变化。

冻结主干、仅训 CoLA 参数，新增 <1% 参数，既迁移新任务又保留预训练自然行为。

</details>

---

## 高级（Advanced）

### Q11. 梯度如何穿过不可导的量化操作 `⌊·⌉`？写出 STE 的实现。

<details>
<summary>答案</summary>

用 **STE（Straight-Through Estimator，直通估计器）**：前向取整，反向把量化当恒等映射，梯度直接传给 `tanh` 前的连续 `z`。实现技巧：

```python
z_disc = round((L/2) * tanh(z_cont))
z_ste  = z_cont + (z_disc - z_cont).detach()   # 前向=z_disc，反向梯度=∂/∂z_cont
```

`.detach()` 让 `(z_disc - z_cont)` 不参与反传，于是前向值等于 `z_disc`，反向梯度等于对 `z_cont` 的梯度。

</details>

---

### Q12. 写出生成控制器的自回归分解式，并解释因果自注意力如何保证 train/inference 一致。

<details>
<summary>答案</summary>

分解（式 5）：

```
p_θ(z̃_t | s_t) = p_θ(z̃_t^0 | s_t) · Π_{j=1}^{d'-1} p_θ(z̃_t^j | s_t, z̃_t^{<j})
```

GPT 式 Transformer 用**因果掩码**：token `j` 只 attend 到 `z̃_t^{<j}`（之前的 token）。训练用 teacher forcing + 交叉熵（式 6），推理逐 token 生成。因为训练时每个位置的可见上下文（因果掩码）与推理时逐步生成时的可见上下文完全一致，避免了 train/inference 失配。

</details>

---

### Q13. 写出 CoLA 的前向公式（式 7），并说明每个符号的作用。

<details>
<summary>答案</summary>

```
Wx = W₀x + m · [ B(diag(γ(c))·Ax + β(c)) ] / ‖ B(diag(γ(c))·Ax + β(c)) ‖_F
```

- `W₀`：冻结的预训练权重。
- `A ∈ R^{r×d_in}`, `B ∈ R^{d_out×r}`：可训练低秩矩阵，`r ≪ min(d_in,d_out)`。
- `m`：学习到的幅度向量（DoRA 幅度/方向解耦）。
- `γ(c), β(c)`：轻量 MLP 从任务观测 `c` 生成的 FiLM 缩放/偏移，在低秩空间做任务条件调制。
- `‖·‖_F`：Frobenius 归一化方向分量，配合 `m` 实现幅度/方向解耦，稳定优化。

新增参数 <1%。

</details>

---

### Q14. RLFT 微调时如何防止 RL 把角色带成「高分但不自然」的策略？SFT 又带来什么取舍？

<details>
<summary>答案</summary>

**RLFT 防崩**：探索时用**无条件生成模型引导 nucleus（top-p）采样**，只从预训练分布的高概率（自然）区域采 token，把探索约束在自然行为流形附近，避免 RL 找到抽搐式高分策略。

**SFT 取舍**（Table 4）：SFT 用示例动作做交叉熵，把控制器偏向特定技能子集（如蹲走），降低预测 logits 的熵 → 行为更风格一致，但**多样性下降**（APD 0.24 vs 0.26、ADE 0.15 vs 0.27、熵 2.57 vs 5.08）。好处是不损害任务完成度与扰动鲁棒性，实现可控的风格化。

</details>

---

### Q15. 论文最大的两块「支柱」贡献分别独立贡献了什么？如果只保留一个会怎样？

<details>
<summary>答案</summary>

两块支柱：**(a) FSQ 替代 VQ-VAE**、**(b) 端到端 RL 替代蒸馏**。

- 从 Table 1：VQ-VAE 端到端训练也能到 99.94/37.92，说明**端到端 RL 贡献了大部分性能**；FSQ 相对 VQ-VAE 主要提升是训练简洁性、无需码本启发式、利用率更高，以及略好的 MPJPE（34.90 vs 37.92）。
- 从 Table 2：去掉端到端 RL（FSQ-K）掉到 99.03/78.26，**性能大幅下降**。

结论：**端到端 RL 是性能的主要来源，FSQ 是稳定性/工程简洁性的主要来源**。二者互补——FSQ 让大规模离散表征可稳定训练，端到端 RL 让离散表征物理可执行。其余（GPT 自回归、分组、CoLA）是把成熟 NLP 工具按需搬运。

</details>

---

> 全部答对说明你已掌握 GPC 的核心。想更进一步可看 `reflection.md` 的开放问题。
