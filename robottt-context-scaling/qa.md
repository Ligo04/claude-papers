# 自测题：RoboTTT: Context Scaling for Robot Policies

## 基础题（5 道）

### 问题
RoboTTT 把机器人策略的视觉运动上下文扩展到多长？相比之前的 SOTA 机器人基础模型，这大约是多大的提升？
<details>
<summary>答案</summary>
8K 个时间步（30Hz 控制下约 4-5 分钟）。之前的 SOTA VLA/世界动作模型大多只用单步观测，或最多 2-8 帧的短历史，所以 8K 相比它们是三个数量级（约 1000 倍）的提升，且这个扩展没有增加推理延迟。
</details>
---

### 问题
TTT 层被插在 RoboTTT 架构的什么位置？为什么放在那里，而不是替换掉原有的注意力层？
<details>
<summary>答案</summary>
TTT 层被插在 DiT action head 的自注意力/交叉注意力层之后，共 16 层 DiT 每层加一个 TTT 层。原因是二者分工不同：attention 处理的是单个时间步内部（跨 token，如 register token、proprioception token、noised action token 之间）的信息交互；TTT 层处理的是跨时间步的信息（历史的压缩与检索）。二者是互补而非替代关系，所以保留原有 attention 结构，在其后叠加 TTT。
</details>
---

### 问题
什么是 "fast weights"（快权重）？它和常规的模型参数（"slow weights"）有什么本质区别？
<details>
<summary>答案</summary>
Fast weights 是 TTT 机制中，在训练和推理阶段都会通过梯度下降持续更新的一小部分参数（对应一个小型网络 f_W，本文中是两层 MLP）。区别在于更新时机：slow weights（常规模型参数）只在训练阶段更新，推理时冻结不变；fast weights 在推理时依然按 W_t ← W_{t-1} − η∇W ℒ_FW(f_{W_{t-1}}(K_t), V_t) 持续更新，本质上是把历史信息在线压缩进这组参数里，随后用它做预测（apply 步：O_t = f_{W_t}(Q_t)）。
</details>
---

### 问题
RoboTTT 在三个评测任务上的平均任务完成分数是多少？分别比单步基线 GR00T N1.7 和最强基线 GDN 高多少？
<details>
<summary>答案</summary>
RoboTTT 平均任务完成分数为 79%。相比单步基线 GR00T N1.7（42%）高 87%；相比最强基线 GDN（Gated DeltaNet，56%）高 41%。RoboTTT 也是在 Gear Bot（十阶段、五分钟的最长任务）上唯一取得完整成功（2/10）的方法。
</details>
---

### 问题
tanh 门控机制（O = tanh(α) ⊙ O_TTT + O_attn）在 RoboTTT 中起什么作用？α 初始化为什么是接近 0 的小值（0.001）？
<details>
<summary>答案</summary>
它用于保护预训练模型（GR00T N1.7）已经学到的能力：RoboTTT 从预训练权重初始化，如果一开始就让新加入的 TTT 层输出以较大权重叠加到 attention 输出上，会在训练早期扰乱预训练模型的既有计算，破坏其能力。将 α 初始化为接近零的小值（0.001），使 tanh(α) ≈ 0，让 TTT 分支在训练刚开始时几乎不参与计算，模型近似退化为原始预训练模型；随着训练推进，α 被学习着逐渐调大，TTT 的贡献才平滑地"长出来"。
</details>
---

## 中级题（5 道）

### 问题
Sequence action forcing 具体做了什么？如果不采用它（即整条序列共享同一个 flow-matching 噪声水平），会出现什么问题，论文给出的解释是什么？
<details>
<summary>答案</summary>
Sequence action forcing 是指在序列训练时，为序列中每个 action chunk **独立采样** flow-matching 的噪声水平 τ_t = s(1−u)，u ~ Beta(1.5, 1)，s = 0.999，而不是整条训练序列共用一个 τ。如果共用一个噪声水平（即"full-sequence diffusion"），会导致整条序列在某次迭代中要么全部处于低噪声（因而全部"过易"），要么全部处于高噪声（因而全部"过难"），训练信号在不同 batch 间的难度分布过于极端、不均衡，导致训练不稳定。论文类比了 Chen et al. 提出的 diffusion forcing 工作中的类似发现。消融实验显示，去掉这一技术会导致模型动作显著不精确，机器人几乎无法有效推进任务。
</details>
---

### 问题
Truncated Backpropagation Through Time（TBPTT）如何让训练上下文可以扩展到 8K 步而不爆显存？关键设计是什么，为什么这个设计是必需的？
<details>
<summary>答案</summary>
TBPTT 把长训练序列切分成若干个 segment，梯度反传只在每个 segment 内部进行，在 segment 边界处被截断（detach）。这样，无论总序列多长，一次训练迭代需要保存的中间激活值只取决于单个 segment 的长度，GPU 显存开销与总上下文长度解耦。关键设计是：虽然梯度在 segment 边界被截断，但 fast weights 本身（W_t 的数值）会原样跨 segment 边界继续传递、持续更新，也就是说 TTT 的"更新-应用"这个前向计算流程贯穿整个长序列，只是反向传播被分段截断了。这个设计是必需的，因为如果不让 fast weights 跨 segment 传递，模型将无法真正学习跨越整个长序列的上下文利用，退化为若干条独立的短序列训练。
</details>
---

### 问题
论文中的两种"从上下文学习"应用（一次性人类视频模仿、DAgger Distillation）共享的核心机制设计是什么？
<details>
<summary>答案</summary>
共享机制是"把 flow-matching 损失从某些时间步上 mask 掉"，让这些时间步只更新 fast weights（提供上下文信息）而不提供模仿目标（不产生梯度反传到输出预测）。在人类视频模仿中，视频段的动作损失被 mask，只有机器人轨迹段的动作损失参与训练；在 DAgger Distillation 中，机器人的次优动作段的损失被 mask，只有人类纠正动作段的损失参与训练。二者本质上都是解耦"更新 fast weights 的输入"与"计算模仿损失的目标"，让模型能从异构、甚至次优的信息源（人类视频、失败的机器人动作）中学习，而不要求这些信息源本身是要模仿的目标。
</details>
---

### 问题
消融实验显示，仅仅给 GR00T N1.7（无 TTT）加上和 RoboTTT 数量相同的 register tokens 并没有带来提升，但给 RoboTTT 加 register tokens 却带来了 18% 的相对提升。这说明了什么？
<details>
<summary>答案</summary>
这说明 register tokens 本身并不自带"记忆/信息携带"的能力——它们只是一组可学习的额外 token，注意力机制可以让它们在每个时间步内与其他 token 交互，但如果没有跨时间步的机制，它们无法把信息带到未来的时间步。register tokens 真正发挥作用，是在与 TTT 层配合时：由于论文出于计算效率，没有让完整的视觉语言 token Φ_t 直接通过 TTT 层，而是依赖数量更少的 register tokens 作为"信息中转站"跨时间步携带视觉语言信息，TTT 层的跨时间步更新机制给了 register tokens 一个可以被利用的时间通道。因此，register tokens 的价值是和 TTT 的时间建模能力绑定的，二者需要协同才能生效。
</details>
---

### 问题
论文用 GDN（Gated DeltaNet 替换 TTT 层）作为一条重要对照基线。GDN 和 RoboTTT 的相同点与不同点分别是什么？实验中二者表现出的差异说明了什么？
<details>
<summary>答案</summary>
相同点：二者都维持一个固定大小（不随时间增长）的循环状态，推理成本恒定，都属于"用循环状态压缩历史"的序列模型范式。不同点在于状态更新规则：GDN 使用的是线性关联式（linear associative）的 delta rule 更新，不涉及测试时的梯度下降；而 RoboTTT 的 fast weights 通过在训练和推理阶段都执行真正的梯度下降（对 fast-weight loss 求梯度）来更新，并且这组更新动态本身（包括初始化 W_0）是被外层任务损失元学习出来的。实验中 GDN 在主任务上整体不如 RoboTTT，在上下文缩放律实验中完全没有表现出随预训练上下文增长而提升的趋势，在一次性人类视频模仿上则完全失败（0/10）。这说明：拥有一个固定大小的循环状态并不足以支撑长上下文能力，能否被端到端元学习、能否用非线性模型做更强的压缩，才是能否有效利用长上下文的关键。
</details>
---

## 高级题（5 道）

### 问题
为什么 GDN（Gated DeltaNet）在上下文缩放实验中没有表现出随预训练上下文长度增长而提升的趋势，而 RoboTTT 却表现出清晰、无饱和的缩放律？请从"元学习 W_0"这个角度解释。
<details>
<summary>答案</summary>
RoboTTT 的 fast weight 初始化 W_0 以及投影矩阵 θ_Q, θ_K, θ_V 都是模型参数的一部分，随着外层任务损失（flow-matching loss）通过"梯度的梯度"（gradient of gradients）被联合优化——也就是说，训练序列越长，W_0 就要经历越多次内层梯度下降更新（Eq. 1）才能产生最终预测，这使得外层梯度对 W_0 的塑造更加"感知"到长程更新动态，长序列训练本质上在教会模型一个更适合长期演化的初始化点和更新节奏。GDN 的状态更新是线性关联式的 delta rule，没有一个类似"可学习、可被外层损失塑造的初始化与更新动力学"的结构去随着训练序列变长而被进一步优化——它的状态转移规则是相对固定的线性形式，不存在这种随着上下文变长而被元学习得更精细的自由度。因此，更长的训练上下文对 RoboTTT 而言意味着"元学习信号更丰富"，而对 GDN 而言只是"处理更长但结构不变的序列"，二者对长上下文的"利用效率"截然不同，导致缩放曲线形态迥异。
</details>
---

### 问题
为什么 sequence action forcing（逐时间步独立采样噪声水平）对于序列长上下文训练是必要的，而在非序列（单步）的 flow-matching/diffusion 训练中通常不需要类似机制？
<details>
<summary>答案</summary>
单步训练中，每次前向只涉及一个动作 chunk 的一个噪声水平，batch 内不同样本本来就天然采样不同的 τ，不存在"整条序列难度一致"的问题。但在长序列 TTT 训练中，序列内部的时间步是相互耦合的：fast weights 在序列内部逐步更新，后面时间步的预测质量依赖前面时间步 fast weights 的更新质量。如果整条序列共享一个噪声水平 τ（即所谓 full-sequence diffusion），那么某次迭代中，序列里所有时间步的去噪难度会同步地"全易"或"全难"：全易时几乎没有学习信号（模型只需微调），全难时监督信号噪声占主导、几乎学不到有效的去噪方向；无论哪种情况，都无法为 fast weights 提供覆盖多样难度梯度的、稳定连续的训练信号，从而拖累模型学到稳健的更新动态。而逐 chunk 独立采样噪声水平，能保证序列内部始终混合有不同难度的监督信号，让 fast weights 的更新轨迹在训练过程中始终收到有效、多样化的梯度反馈，训练因此更稳定。
</details>
---

### 问题
DAgger Distillation 中，"失败作上下文、纠正作目标"这个非对称设计具体是怎么实现的？为什么论文强调次优机器人动作"作为模仿目标毫无价值，但价值在于作上下文"？这如何验证？
<details>
<summary>答案</summary>
实现上：给定一条 DAgger 风格的交互轨迹 ξ_DAgger = {(l, o_t, q_t, A_t)}，其中每个执行的动作 chunk A_t 要么是机器人自己的（次优）动作 A_t^R，要么是人类的纠正动作 A_t^H。训练时，整条轨迹（包含 A_t^R 和 A_t^H）都参与 fast weights 的更新（作为 Key/Value 更新 W），但 flow-matching 的模仿损失只在人类纠正动作 A_t^H 对应的时间步上计算（机器人自身动作对应的损失被 mask 掉）。验证这一非对称设计价值的关键对照实验是：把 GR00T N1.7（没有 fast weights、无法把上下文与目标解耦）分别用"完整轨迹（含机器人动作）微调"和"只用人类纠正微调"两种方式训练，结果两者性能完全相同（都是 57%），说明次优机器人动作单独作为模仿目标（即让模型直接模仿它）没有任何附加价值——模型学不到东西，因为它本身就是"错误示范"。但当同样的次优动作被用作 RoboTTT/GDN 的 fast weight 更新上下文（而不是模仿目标）时，DAgger Distillation 带来了 33% 的平均额外提升，明显超过标准 DAgger 的 9%。这证明了次优动作的价值不在于"教模型怎么做"，而在于"让模型据此判断当前处于什么失败状态，从而学会在类似情境下主动纠正"——这正是"失败作上下文"想要实现的效果。
</details>
---

### 问题
在上下文缩放律实验中，为什么预训练上下文低于 1K 时，RoboTTT 的性能反而明显下降（虽然仍优于同长度的 GDN）？论文给出了哪两个互相关联的解释？
<details>
<summary>答案</summary>
论文给出两个解释，二者本质上都是"训练时上下文窗口与推理时实际所需窗口不匹配"的不同侧面：第一，1K 个时间步（30Hz 下约半分钟）比论文最短的任务 episode 还要短，这意味着在真实部署 rollout 时，fast weights 会被持续更新到远超训练时序列长度的地步——模型在训练阶段从未经历过"fast weights 已经积累了这么久的更新"这种状态，因此在这种状态下的行为缺乏训练信号支撑，表现变差。第二，位置编码（RoPE）在推理时会被外推到训练时从未见过的位置索引上，超出训练分布的位置编码通常会导致模型行为退化，这是长上下文序列模型的共性问题。这两点共同导致：当训练上下文短于实际部署所需的时长时，模型在部署中段以后进入"未训练区域"，性能因此下滑。
</details>
---

### 问题
论文认为 register tokens 与 TTT 层之间存在协同效应，而不是各自独立起作用。结合模型架构设计（VL tokens 不直接过 TTT 层），推导这种协同的因果链条是什么？
<details>
<summary>答案</summary>
因果链条大致是：(1) 出于计算效率考量，完整的视觉语言 token Φ_t（数量较多、维度较高）没有被直接送入 TTT 层做跨时间步处理；(2) 于是模型需要一种更紧凑的"信息中转载体"来把每个时间步的视觉语言语义压缩、并跨时间步传递下去，这正是 N=16 个 register tokens 的角色——它们在每个时间步的注意力层里与 Φ_t、q_t、Ã_t 等交互，吸收该时间步的语义信息；(3) 但 register tokens 要真正实现"跨时间步携带信息"，必须依赖某种跨时间步的更新与检索机制——如果只有单步注意力，register tokens 每个时间步都是从零重新计算，无法真正累积历史；(4) TTT 层恰好提供了这个机制：register tokens 的表示进入 TTT 层的 K/V 更新 fast weights，之后的时间步通过 Query 检索这些 fast weights，从而让 register tokens 携带的历史语义真正跨越时间步生效。这解释了为什么单独给 GR00T N1.7（无 TTT）加 register tokens 没有用——没有 TTT 提供的跨时间步更新-检索机制，register tokens 就只是普通的、每步重新计算的可学习 token，起不到携带历史的作用；只有二者组合，"紧凑表征（register tokens）+ 跨时间步压缩与检索机制（TTT）"才能共同构成一个有效的长上下文信息通道。
</details>
---
