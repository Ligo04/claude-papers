# 核心洞见：把“动作后果”放回可度量的空间

## 一句话理解

UNITAS 的创新不是再造一个更大的 WAM，而是把 WAM 的共同接口从 image/latent 换成：

    current observation
    + world coordinates
    + point identity
    + physical time
    + action-conditioned displacement

因此“机器人要怎么动”和“世界会怎么变”不再只是两个 loss，而是共享同一套空间锚点和轨迹 token。

## 为什么 2D video latent 不够

2D 图像是 view-dependent projection。若相机换位：

- 同一把夹爪移动 5 cm，pixel displacement 会改变；
- 同一个像素移动量，可能对应不同深度下完全不同的 metric displacement；
- 物体接触、遮挡和静态背景漂移很难从 pixel loss 中区分。

UNITAS 的 world-aligned positional embedding 将视觉 token 绑定到 camera calibration 和 depth back-projection 得到的 world position。没有 depth 时，它不简单地把深度设成常数，而是沿 camera ray 采样 64 个候选深度并由 feature routing 聚合 world-frame positional feature。

## 为什么 point flow 是好的中间接口

点轨迹同时满足三个需求：

1. **跨 embodiment**：机器人 gripper 与 human hand 都可表示为一组 point trajectories。
2. **跨帧率/时长**：trajectory tokenizer 使用 seconds 而不是 frame index。
3. **可解释的空间后果**：scene flow 直接回答哪些环境点移动了多少，而不是只比较图像纹理。

一条轨迹压成一个 token，还避免了长序列每个 timestep 都占用一个 transformer token。代价是 trajectory decoder 必须保留 point identity 和时间条件。

## UNITAS 的“分工”不是 Astra 式 Hybrid

UNITAS 有两个 coupled experts：

- point dynamics expert：生成 action flow 和 scene flow；
- action expert：生成可执行的 end-effector action chunk。

训练时 scene flow 看 clean action flow；推理时 scene flow 看 generated action flow 或 externally supplied trajectory。policy-only inference 则跳过 point-flow branches，只保留 action chunk。

这是一种**模型内部的结构耦合**，不是“一个高级模型审核另一个低级 policy”的外部 Hybrid。它的强项是让动作和后果共享几何表征；它的弱点是 scene-flow 预测会带来额外 latency 和 query-density 成本。

## 3D grounding 带来的证据

LIBERO-Plus ablation 清楚显示：

- 去掉所有 visual 3D PE：67.58%；
- 保留 action-flow/history 等但去掉 3D PE：81.35%；
- routed PE（测试时无 depth）：84.43%；
- ground-truth depth PE：84.68%。

这说明 3D grounding 的价值不是“拥有深度传感器”本身，而是让视觉特征具有可迁移的空间参考。无 depth 推理只比 ground-truth depth 低 0.25 points，是一个重要的工程边界。

## 与视频 WAM、VLA 和 PointWorld 的关系

- 相比 video WAM：UNITAS 直接把 geometry 放进 shared interface，减少 viewpoint-dependent pixel reasoning。
- 相比纯 VLA：它增加 action flow、scene flow 和 spatial/temporal supervision，而不只学习 action chunk。
- 相比 PointWorld：PointWorld 以 externally supplied robot motion 为条件预测 scene flow；UNITAS 还能从 instruction 直接生成 executable action。
- 相比 planner + simulator：UNITAS 让 action 和 consequence prediction 在同一 representation 里互相约束，但 policy mode 可以跳过场景预测以降低 latency。

## 取舍与边界

| 选择 | 得到的能力 | 付出的代价 |
| --- | --- | --- |
| World-aligned 3D PE | viewpoint / initial-state robustness | 需要 calibration、depth 或 ray routing |
| Action flow | 跨 embodiment 的运动先验 | 需要 URDF/MANO query 定义与点身份 |
| Scene flow | action-conditioned 后果预测 | 需要 scene queries，推理成本随密度上涨 |
| Physical-time token | 对齐不同数据源帧率 | 时间窗口有限，长时程仍要分块 |
| Policy-only mode | 55.77 ms/chunk action-only inference | 运行时跳过了显式 scene prediction |
| Mixed pretraining | 跨数据集迁移与 robustness | 多坐标系、mask、数据质量和采样规则更复杂 |

## 实践含义

若要把 UNITAS 思路迁移到机器人项目，优先固定：

- world frame、camera extrinsics、robot-base frame；
- query-point 采样和 point identity；
- action chunk 的 physical time；
- policy mode 与 world-model mode 的独立 latency budget；
- action-scene consistency 的闭环指标。

不要先把目标定成“预测好看的点云”；先问预测的 scene displacement 是否与实际执行的 action 产生一致的物理后果。

