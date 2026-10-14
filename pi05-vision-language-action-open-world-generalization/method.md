# 方法拆解

## 1. 输入、输出和联合分布

观察为

\[
o_t = [I^1_t,\ldots,I^n_t,q_t],
\]

其中 `I` 是多相机图像，`q_t` 是关节、夹爪、躯干和底盘状态。模型表示联合分布

\[
\pi_\theta(a_{t:t+H},\hat\ell\mid o_t,\ell)
= \pi_\theta(a_{t:t+H}\mid o_t,\hat\ell)\,\pi_\theta(\hat\ell\mid o_t,\ell),
\]

`ℓ` 是总体任务命令，`\hat{ℓ}` 可以是 high-level subtask 或 web/VLM 任务的文本答案。分解意味着先采样 `\hat{ℓ}`，再条件化生成动作块。

## 2. 统一 token 序列

Transformer 输入序列中混合：

- 文本 token：语言命令、子任务、VQA/caption 答案。
- 图像 patch：经过视觉 encoder。
- 离散 FAST action token：预训练时的动作块。
- 连续 flow token：post-training 时的部分去噪动作。

注意矩阵 `A(x)` 控制不同 token 类型的可见性；文本、图像和连续 action token 使用双向 attention，而语言输出仍按自回归方式生成。动作 token 使用独立 expert 权重，类似 mixture-of-experts 的 action expert。

## 3. 两阶段训练

### Stage 1：Pre-training

- Backbone 初始化自网页数据训练的 VLM。
- α = 0，只做文本/FAST 离散 token 的 next-token prediction。
- 数据包含 MM、ME、CE、HL、WD。
- 所有动作按每个数据集的 1%/99% 分位数归一化到 `[-1,1]`；低维机器人动作 zero-pad 到统一维度。
- 训练 280k steps。

### Stage 2：Post-training

- 在 MM + ME 的成功 episode 上专门化到 mobile manipulation。
- 加入 WD、与 ME 对应的 HL 片段、以及专家 verbal instruction（VI）。
- 新增随机初始化的 action expert，联合优化 token cross-entropy 和 flow-matching MSE。
- `α = 10.0`，训练 80k steps。
- 推理时文本/子任务自回归，然后对动作做 10 次 denoising；动作 horizon 为 H=49（正文有时简称约 50）。

联合损失可写成：

\[
\mathcal L = \mathbb E\left[
\mathcal H(x_{1:M},y^\ell_{1:M})
+ \alpha\left\|\omega-a_{t:t+H}-f^a_\theta(a^{\tau,\omega}_{t:t+H},o_t,\ell)\right\|_2^2
\right].
\]

其中
`a^{\tau,\omega}=\tau a+(1-\tau)\omega`，`\omega\sim\mathcal N(0,I)`，`τ∈[0,1]`。

## 4. 运行时算法

```text
输入：高层命令 ℓ，图像 I1...In，机器人状态 q
循环：
    1. 用四相机和 ℓ 自回归生成语义子任务 ŝ
    2. 用腕部/前向相机、q 和 ŝ 作为条件
    3. 从噪声动作块开始，执行 10 次 flow denoising
    4. 得到连续 action chunk（关节/末端、夹爪、底盘速度、升降）
    5. 以 50 Hz 将目标送给 PD controller
    6. 更新观察，重新预测下一个 ŝ
```

## 5. 评测设计

- 真实家庭：3 个未参与训练的 homes，厨房和卧室，展示长时程行为。
- Mock homes：控制变量，四个任务分别为 items in drawer、dishes in sink、laundry basket、make bed；标准评测为每个 policy/task 10 次。
- 位置扩展：MM 训练位置为 3、12、22、53、82、104；每个 post-training 实验固定 40k steps，让位置数量变化尽量不与样本量混淆。
- 语言跟随：五个候选物体中根据命令选择目标，理论 shortcut 上限约 20%；分别报告 follow rate 与 task success，并拆分 in-distribution/OOD 物体。

## 6. 实现陷阱

1. **动作空间对齐**：不同机器人需要明确 joint/end-effector control mode、量纲、分位数归一化和 zero-padding；否则跨 embodiment 的 token 迁移会混入坐标系偏差。
2. **高低层相机不等价**：高层使用四相机获得场景语义，低层只用腕部和前向相机；复现时不能把所有相机简单拼接后宣称等价。
3. **attention mask**：离散动作和 flow action 不应互相泄漏，否则联合训练会失去论文想要的表示隔离。
4. **action chunk 与控制频率**：chunk 长度、denoising 步数和 50 Hz 输出节奏要一起校准；只复现网络结构不代表复现闭环行为。
5. **高层标签质量**：子任务粒度过粗会让低层条件含糊，过细则增加标注成本和推理步数。

## 7. 超参数敏感性与复现风险

论文明确给出的关键超参数是 280k/80k steps、`α=10.0`、10 denoising steps 和 H=49；宽度、深度等结构参数见 Appendix E，但 batch、学习率、action chunk 的采样配比、采样温度和各数据源的精确权重仍会影响结果。真实机器人还受抓手、相机外参、底盘控制和安全策略影响。复现时应把“静态 loss/离线 action 误差”“mock-home rubric”“真实家庭闭环”分成三层证据，不能用前一层替代后一层。
