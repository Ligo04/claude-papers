# 方法拆解

## 四类对象各自负责什么

| 对象 | 输入和输出 | 训练时角色 |
| --- | --- | --- |
| MuJoCo Warp | 实际执行动作，返回下一状态与奖励 | 数据来源，不反向传播 |
| 世界模型 `fω` 及各个 head | 状态、动作、上下文；预测状态、观测、奖励和终止 | 由仿真日志监督训练 |
| 策略 `πθ` | 有噪观测历史、目标；输出关节设定值 | 通过想象回报反向传播训练 |
| critic `Vψ` | 预测状态、目标、上下文；输出长期价值 | 使用想象轨迹的 lambda-return 训练 |

训练循环可读作：

```text
策略 + 动作噪声 → 标准仿真 → replay buffer
                                   │
                        8 步日志窗口 + 上下文
                                   ↓
                        自回归拟合世界模型 ω
                                   │
                        replay 中采样起始状态
                                   ↓
                 策略 ↔ 冻结参数的模型，展开 32 步
                                   │
                    回报 + 末端 critic 价值
                                   ↓
                    BPTT 更新 θ，随后拟合 ψ
                                   │
                             返回仿真采样
```

原文图 2 位于第 4 页，算法 A.1 位于第 15 页。

## 1. 记录真实执行的动作

策略输出确定性动作 `a_t=πθ(o_t,g_t)`，仿真器实际执行 `a_t+ε_t`。replay 必须保存后者。若保存没有噪声的动作，监督数据就会把不同控制输入错误地对应到同一个标签，增加模型无法解释的随机性。

本文把状态、动作、仿真上下文，以及相应监督信号用于模型训练。实现还需要保存 reset/termination 边界、参考时间索引和无噪观测等信息，才能正确重建训练窗口；论文未给完整 buffer 字段布局，应在复现中显式定义。不能跨 reset 拼接窗口而仍假设物理连续。

## 2. 条件化的确定性世界模型

状态转移和各个 head 写作：

$$
\hat s_{t+1}=f_\omega(\hat s_t,a_t,\xi),\qquad
\hat r_{t+1}=r_\omega(\hat s_t,a_t,g_t),
$$

$$
\hat{\bar o}_t=g_\omega(\hat s_t,\xi),\qquad
\hat d_t=d_\omega(\hat s_t,g_t).
$$

观测分为无噪部分和已知分布的传感器噪声：`o_t=bar_o_t+η_t`。上下文 `ξ` 包含摩擦、关节校准偏移和 torso 质心偏移。目标影响回报和终止，不直接改变机器人动力学。

每个 head 可学习，也可解析计算。解析 head 没有对应的监督损失项，不应从总公式推断所有 head 都是神经网络。论文没有公开逐个 head 的完整实现，不能仅凭符号重构它们。

## 3. 用 8 步自回归监督拟合模型

从 replay 取一个连续窗口，初始状态使用记录值。后续每一步使用模型自己的预测状态，动作使用对应的日志动作：

$$
\mathcal L_{\mathrm{WM}}
=\mathbb E_{\mathcal B}\sum_{j=1}^{k}
\left[
\|\hat s_j-s_j\|^2
+(\hat r_j-r_j)^2
+\|\hat{\bar o}_j-\bar o_j\|^2
+\operatorname{BCE}(\hat d_j,d_j)
\right],\qquad k=8.
$$

这是原文公式 1。与单步 teacher forcing 相比，它暴露累计误差，让模型在自己的预测分布上学习。论文报告长期 rollout 稳定，但这属于经验观察，不是从 8 步损失推出的长期稳定性保证。

状态包含姿态等不同单位的量，归一化与旋转表示会影响训练。文中报告的预测误差按 nominal error 归一化，不能当作米、弧度或统一百分数。具体状态存储、归一化、姿态更新、解析 head 的实现仍需要源码或补充说明。

## 4. 用软生存权重处理终止

仿真器记录硬终止 `d∈{0,1}`，模型预测概率 `hat_d∈[0,1]`，并递推：

$$
\hat c_0=1,\qquad
\hat c_{t+1}=\hat c_t(1-\hat d_{t+1}).
$$

`c_t` 表示轨迹到该步仍存活的权重。可微概率避免在 imagination 中用硬阈值切断终止附近的导数。它不保证终止概率校准，也不防止策略寻找模型过度乐观的区域。

注意原文索引：`r_{t+1}` 在公式 2 中乘 `c_t`，末端价值乘 `c_H`。实现时应该明确终止发生在哪个状态、奖励何时结算，避免平移一位。

## 5. 32 步 imagination 与 actor 目标

每次从 replay 的状态和上下文开始，在预测无噪观测上重新加上传感器噪声。模型参数固定，策略随预测状态闭环地产生动作。目标为原文公式 2：

$$
\hat J(\theta)
=\mathbb E\left[
\sum_{t=0}^{H-1}\gamma^t\hat c_t\hat r_{t+1}
+\gamma^H\hat c_HV_\psi(\hat s_H,g_H,\xi)
\right],\qquad H=32.
$$

最大化该目标时，梯度穿过预测状态、动作、观测转换、奖励、终止概率，以及末端 critic 对状态的依赖。`ω` 与 `ψ` 的参数在这次 actor 求导中固定；计算图中依然保留输入导数。

PyTorch 的实现模式如下。这里只展示梯度边界，完整人形机器人状态与损失仍需另行实现。

```python
world_model.requires_grad_(False)
critic.requires_grad_(False)
actor_optimizer.zero_grad(set_to_none=True)
objective = imagine_with_terminal_value(actor, world_model, critic, replay_start)
(-objective).backward()
actor_optimizer.step()
# 不在 imagination 外包 torch.no_grad()，不逐步 detach 预测状态。
```

关于 `requires_grad` 与 `no_grad` 的区别见 [PyTorch 官方 autograd 文档](https://docs.pytorch.org/docs/stable/notes/autograd.html)。本包独立演示会用中心差分核对这一机制。

## 6. critic 的目标和更新

令 `V_t=Vψ(hat_s_t,g_t,ξ)`，从 `R_H=V_H` 向后递推原文公式 3：

$$
\hat R_t=\hat r_{t+1}
+\gamma(1-\hat d_{t+1})
\left[(1-\lambda)V_{t+1}+\lambda\hat R_{t+1}\right].
$$

$$
\mathcal L_V(\psi)=\mathbb E\sum_{t=0}^{H-1}(V_t-\hat R_t)^2.
$$

targets 每条 rollout 计算一次，并在拟合 critic 时固定。默认 `γ=0.99`、`λ=0.95`。论文没有 EMA target critic。梯度与参数更新要分开管理，避免残留梯度让 actor 步骤修改世界模型，或 critic 更新意外回传 actor。

## 观测、动作和网络

单帧本体观测由 29 维关节位置、29 维关节速度、3 维 base 角速度、3 维 projected gravity、29 维上次动作组成，合计 93 维。策略使用 10 帧历史，按直接拼接计为 930 个标量。这不包含目标输入，也不意味着最终网络实际采用某种固定内存布局。

motion encoder 读取 10 个未来参考帧。控制动作映射为 `q_sp=α a+q0`；由于采用低 PD gain，设定值起到关节力矩代理的作用，而非要求关节瞬时达到该位置。

| 网络 | 隐藏层宽度 | 说明 |
| --- | --- | --- |
| 世界模型 dynamics | 2048 / 2048 / 2048 | 输入状态、动作、上下文 |
| motion encoder | 2048 / 1024 / 512 / 256 | 输出 32 维、tanh 有界 latent |
| velocity encoder | 256 / 256 | 输入速度命令与观测历史特征 |
| 共享 decoder | 2048 / 1024 / 512 / 256 | 输入 latent 和观测历史，输出动作 |
| 每个接口的 critic | 256 / 128 | 使用状态、目标和上下文 |

来源为附录 A、C，表 A.1/A.4。

## 关键训练配置

| 配置 | 取值 | 需要注意 |
| --- | --- | --- |
| 控制周期 | 0.02 s | 仿真器用四个 5 ms 子步对应一个控制步 |
| 世界模型更新 | 每次 10 批，每批 2048 个 8 步窗口 | 不应误当作只有 2048 个单步样本 |
| imagination | 每次 4096 条、32 步 | 两卡使用数据并行，梯度跨卡平均 |
| 学习率 | 单卡 `2e-4`，双卡 `sqrt(2)*2e-4` | actor、model、critic 使用相同基准 |
| actor Adam | `β=(0.9,0.95)` | model/critic 表中仅写 Adam |
| 梯度裁剪 | global norm 1.0 | actor 再逐元素裁到 0.1 |
| critic | 每次 5 epochs，每个 epoch 4 个打乱 minibatch | 目标在 rollout 内固定 |
| 仿真环境 | 每卡 4096 | PPO-Sim 对照为每卡 8192 |
| 动作探索噪声 | 每维 `N(0,0.1²)` | 实际执行动作写入 replay |
| replay 大小 | 每个环境 1000 步 | 旧数据被新数据替换 |
| update-to-data ratio | 0.4 | 论文给出该数值，具体调度单位需核对实现 |
| 学习率调度 | 250k iterations 内 cosine 到 0.03 倍，然后常数 | 总长度 320k iterations |

数据来自表 A.5，第 16 页。不能根据这些配置自行承诺消费级显卡的训练时间或显存需求。

reset 将 motion 时间线分成 1 s bins，混合 50% 均匀采样、40% 按低回报 softmax 采样、10% 按历史 softmax 权重采样。困难片段获得更多练习，已经学好的困难片段仍保留部分权重。公式为 `softmax(-25 G_b/G_max)`，来源为附录 D.1。

## 下游 velocity encoder 的特殊梯度边界

接口策略写成 `a=decθ(encθ,i(g),o)`。velocity 回报可以穿过共享 decoder 传到 velocity encoder，但 decoder 的参数不接受该速度目标的更新；decoder 仍由 motion-tracking 目标训练。RL 版本在参考命令 rollout 上还使用权重 0.1 的 latent 对齐损失。

所以本文“共同训练接口”并不等于所有目标共同优化全部参数。论文也承认，在这些梯度边界下，velocity encoder 可能没有必要与 motion tracker 同时训练。详见附录 B/G 和第 9 页局限。

## 复现前必须补齐的信息

- 状态定义、单位归一化、rotation 表示与每个 head 的解析/学习选择。
- 观测历史如何随想象状态和上次动作更新，reset 后如何初始化。
- 日志中的动作裁剪、执行器 gain、action scale、参考对齐和 termination mask。
- replay 采样边界、更新调度与多卡批量统计口径。
- 同一个训练回报是否来自偏难的 reset 分布，还是从首帧完整播放的检查点评估。

这些缺口决定真实复现难度。本文学习材料给出方法解释和独立教学程序，没有把尚未公开的实现补成作者代码。

来源：[固定版本论文](https://arxiv.org/pdf/2610.28480v1)，第 4 至 5 页和附录 A 至 G。
