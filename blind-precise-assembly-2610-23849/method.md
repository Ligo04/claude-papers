# 方法拆解与复现注意事项

原文依据：[§2 与附录 A–C](https://arxiv.org/html/2610.23849v1)。下面的公式和伪代码用于整理信息流，未注明的实现选择不能当作作者源码。

## 观测、表示与动作

插销、齿轮的单步输入为：

```text
o_t = [q_t(7), qdot_t(7), gripper_pose_t(6),
       gripper_twist_t(6), action_(t−1)(7)]      共 33 维
z_t = TemporalConv(o_(t−29), ..., o_t)          32 维
a_t = MLP([o_t, z_t])                          7 维
q_target,t = q_measured,t + action_scale × a_t
```

夹爪 pose/twist 由正运动学得到，仍属于编码器可导出的观测。螺母策略用旋转矩阵两列的 6 维旋转表示代替 3 维 Euler 角，故单步为 36 维。

动作增量加到**测得的关节位置**上。若误改为累加到上一个目标角，会改变接触时的误差累积和运动行为。

actor 隐藏层为 `(256, 256, 128)`。论文没有完整列出 temporal convolution 的核宽、层数、padding、归一化和历史初始化，教学代码会显式标记自选部分。

## 训练和部署的数据边界

| 数据或模块 | actor 部署可用 | 训练用途 |
|---|---|---|
| q、qdot、正运动学夹爪状态、上一动作 | 是 | 策略输入 |
| 30 帧本体感知历史 | 是 | 编码成 latent |
| 零件相对夹具位姿、真实接触标签 | 否 | critic、辅助标签 |
| 接触 decoder 的预测 | 否 | 算辅助 loss，分析表示 |
| 图像、力或力矩观测 | 否 | 不是本体感知 actor 输入 |
| 系统辨识参数 | 不作为任务观测 | 用于构建接近真实机械臂的仿真 |

“策略不读力矩”不等于机器人没有力矩控制，也不等于训练不能对 commanded torque 施加惩罚。

## 接触辅助损失

插销在底边采样 64 点。高于夹具顶面超过 1 mm 为 free，接近台面或孔边为 contact，位于开口内为 inside。decoder 隐藏层为 `(128, 128)`，从 latent 输出每点分类。

```text
L_contact = weighted_cross_entropy(decoder(z), simulated_labels)
L_training = L_PPO + 接触辅助项
```

原文列出的类别权重是 free 0.25、contact 3.0、inside 等有效进入类别 2.0。总损失的辅助系数与实现 reduction 必须在复现时确认，不能把类别权重当作总损失系数。

齿轮监督包括轴口 64 点和齿顶 64 点。螺母使用 64 点、free / bolt-tip contact / around bolt / thread-caught 四类，不能直接复用插销的三类输出。

## PPO 配置

| 超参数 | study policy |
|---|---|
| 环境 / rollout | 12,288 / 32 步 |
| epochs / minibatches | 5 / 插销 20，齿轮螺母 24 |
| 学习率 | 3e−4 |
| PPO clip | 0.2 |
| gamma / GAE lambda | 0.98 / 0.95 |
| entropy / value loss coefficient | 0.003 / 1，value clipping |
| gradient norm clip | 1 |
| 初始动作标准差 | 0.5 |

训练使用成功与阶段首次达成奖励。插销首次被孔捕获和最终成功各得 +1。不是连续几何距离 shaping，也不应描述成只有一次终局奖励。

```text
为每个 world 生成零件几何
循环训练：
    reset 时随机化夹具姿态、抓取姿态与接触参数
    采集 32 步 rollout：
        只用本体感知历史计算 actor 动作
        用关节阻抗控制器执行目标角，推进仿真
        计算阶段奖励、成功判定与特权接触标签
        critic 使用额外特权信息估值
    用 PPO 更新策略与 critic
    用接触分类损失更新 decoder 和历史编码器
部署：
    删除 critic 与 decoder
    按 50 Hz 使用同样的本体感知预处理、历史和 actor
```

## study 与 deployed policy 必须分开

| 设置 | study 插销 | deployed 插销 |
|---|---|---|
| 截面 | stadium | 随机凸、多边形/平滑/混合轮廓，带手柄 |
| 初始平面偏移，每轴 | ±12.5 mm | ±25 mm |
| 夹持倾斜 | ±10° | ±20° |
| Kp / Kd | 80 / 20 | 40 / 20 |
| 动作尺度 | 0.02 rad | 0.04 rad |
| 板面摩擦 | 0.5–2.0 | 0.25–4.0 |
| 板面接触时间常数 | 3.5–7.1 ms | 2.5–10 ms |
| 速度与力矩惩罚 | 无 | 学会任务后分阶段加入 |
| minibatches | 20 | 24 |
| 探索末期 | 固定学习率设置 | exploration anneal，target KL 0.01 |

来源：表 1、表 2、附录 A.2。部署阶段惩罚从硬件速度与 commanded torque 上限的六分之一开始。完整调度和系数没有在本文给齐。

几何每个 world 生成一次；位姿与接触参数每次 reset 重采样。表 1 明确写没有添加 observation noise，不能自动把“domain randomization”扩展为传感噪声随机化。

## `mjfit` 的几何流程

```text
指定孔二维轮廓、可选 islands、间隙 c
    → 圆孔/stadium 外接离散化；Bézier 采样为多边形
    → 把孔外实体区域划分成凸多边形
    → 挤出为凸棱柱，构成夹具
    → 孔轮廓向内偏移 c，得到配对插销
    → 输出 MuJoCo 可碰撞的 MJCF
```

islands 可以表示连接器中的凸起；对应插销要留凹槽。螺纹和齿轮使用相应凸截面/齿形构造。它不是任意复杂 CAD 的无条件精确凸分解器。

圆形配置下最大额外游隙为 `c(sec(π/n) − 1)`。注意这是特定构造的几何量，不能直接当成 MuJoCo 最终沉降误差，后者还有求解器和接触搜索的影响。

## 碰撞基准应该怎样读

表 4 的 CPU 为单环境单线程；GPU 为 RTX 5090 上 **512 个世界的总实时因子**。图 6b 则用 256 个世界，并在 Warp 上测误差。二者不能交换，也不能把 GPU 总实时因子当成单场景速度。

表 4 示例：mjfit 方孔 CPU 93×、GPU 合计 970×，20 μm 间隙的最大沉降误差 0.2 μm。7–32 边圆孔 CPU 82–29×、GPU 1120–1070×，误差不超过 2.6 μm。其他 CoACD / V-HACD 参数甚至达到 GPU 1900×，但填掉了间隙；只比较速度会得出错误选择。

共同设置是 1 ms timestep、implicit-fast、Newton solver、`solref=(2 ms,1)`、`solimp=(0.99,0.999,0.001)`、摩擦 1、margin 0。GPU contact/constraint buffers 按表示校准并检查溢出。这里的 Newton solver 是 MuJoCo 求解器选项，不是 Newton 机器人仿真项目。

## 复现前需要补齐的内容

- temporal convolution、历史 reset、观测标准化、动作裁剪和控制时间对齐。
- armature、damping、friction 的辨识数值与真实机器人控制接口。
- 接触 loss 系数、PPO 细节与部署安全惩罚的完整调度。
- 广泛随机化情况下的碰撞/约束缓冲容量和溢出检测。
- 未报告的随机种子方差，尤其齿轮和螺母泛化只有一个训练 seed。
- 真机失败分类、异常中止定义及 240 个计划 reset 与 237 次报告试验的差异。

公开链接确认的是 `mjfit` 几何生成库。网页有可运行策略演示，不等于训练代码、完整环境和检查点均已完整发布。本文没有运行原作者 PPO、MuJoCo GPU 训练或真机实验。
