# NeuralActuator 论文摘要

- **标题**: NeuralActuator: Neural Actuation Modeling for Robot Dynamics and External Force Perception
- **作者**: Zhiyang Dou, John U. Onyemelukwe*, Hangxing Zhang*, Heng Zhang, Minghao Guo, Yunsheng Tian, Michal Piotr Lipiec, Joshua Jacob, Chao Liu, Peter Yichen Chen (MIT), Yuri Ivanov (Amazon Robotics), Wojciech Matusik (MIT)
- **arXiv**: 2607.11734v1 [cs.RO]，2026-07-13

## 背景

可微仿真已经在策略学习和基于模型的控制中广泛应用，但**执行器动力学**长期被低估——尤其是在低成本舵机平台上。这类平台普遍用简化的线性电流-力矩近似 τ = K_t·I 来建模执行器，但实际上齿轮箱摩擦、迟滞 (hysteresis)、背隙 (backlash)、电流饱和、热漂移等非线性效应会让这个近似严重失真，成为 sim-to-real 误差的主要来源之一。论文用实测数据直接证伪了线性假设：在 OpenManipulator-X 上逐关节拟合 K_t，有两个关节在电流饱和 (clamp) 前就拟合出了**负斜率**。

与此同时，低成本平台普遍**没有力/力矩传感器**，机器人无法感知外部接触力，这构成了"无感力觉 (sensorless force perception)"的核心需求——既要在没有传感器的情况下估计外力，又要判断当前是否真的处于接触状态（避免误报）。

## 问题陈述

如何在没有力矩传感器标签、也没有力/力矩传感器硬件的前提下，同时做到：
1. 学出一个能驱动可微仿真器前向传播的"力矩代理"，弥补线性电流-力矩模型的失效；
2. 从执行器遥测（电流、PWM、电压、温度等）中感知外部接触力，并给出可靠的接触概率判断；
3. 诊断执行器自身的机械健康状态（例如某个关节是否被卡住/受限）。

## 主要贡献

1. **NeuralActuator 模型**：四头多任务 Transformer（力矩代理头、原始外力头、接触概率门、逐电机状态分数头），通过带门控注意力的 Transformer encoder 处理历史遥测窗口；力矩代理头**没有直接力矩标签**，而是通过可微仿真（MuJoCo/MJX）前向积分、用真实位姿轨迹做 BPTT 反向传播训练。
2. **双臂遥操作系统**：leader-follower 双臂 teleoperation 平台（2× OpenManipulator-X，DYNAMIXEL XM430-W350 舵机），同步采集关节状态、执行器遥测与 6 轴力/力矩传感器标签，用于生成训练数据。
3. **NAD (Neural Actuation Dataset)**：由上述遥操作系统采集的执行器-力感知数据集，覆盖自由运动、已知重量负载、力传感器交互、电机机械受限等多种场景。
4. **三平台实验验证**：在 OpenManipulator-X、SO-101（LeRobot 生态，STS3215 舵机）、Franka Emika Panda（7-DoF 工业臂）三个差异很大的平台上验证模型的可迁移性，并与经典外力观测方法做基线对比。

## 关键结果与定量指标

### 数据集规模 (NAD, OMX 子集)
- 总时长约 **94.52 分钟**，8:1:1 划分训练/验证/测试
- 自由运动（无外力）：119,933 帧，约 34.15 分钟
- 力标注交互：约 46.24 分钟 = 已知重量负载 (200/300/400/500 g) 88,611 帧 + 轴对齐力传感器交互（最大 9 N）74,020 帧
- 电机机械受限（橡皮筋捆 Joint 3）：49,898 帧，约 14.13 分钟

### Rollout 精度（Table III）
- 平均关节 MAE **2.5–3.2°**（100→600 步 rollout），夹爪滑动坐标误差 0.2 mm；600 步约等于 10 秒

### 力估计精度
- 力传感器基准（Table IV）：接触时 F MAE **0.36–0.39 N @100 步**，**0.38–0.57 N @500 步**；无接触参考轨迹误差近零 (0.00–0.02 N)
- 负载基准（Table V）：平均 F MAE **0.08–0.12 N**，单任务最大 0.24 N
- 推力计验证（Table VI/Fig 1）：三个高度前推轨迹 MAE 分别为 0.037/0.015/0.028 N

### 与经典基线对比（Table VII，负载基准 @100 步）
- ID-Linear: 1.41 N
- ID-Friction (Stribeck + 背隙模型): 1.23 N
- GMO (广义动量观测器): 0.66 N
- **NeuralActuator: 0.12 N**（约为 GMO 误差的 1/5.5）
- 无接触时经典方法误报接触率 **33–59%**，NeuralActuator 误报率近零
- 注意：基线每步使用真值状态，而 NeuralActuator 用自己的仿真 rollout 状态——两者评测条件并非完全公平对等

### 电机状态分类（Table XI）
- 准确率 **91.0%**，精确率 84.5%，召回率 96.2%，AUC-ROC 0.95
- 基线：阈值法 58.6%/0.45，SVM(RBF) 59.9%/0.62，随机森林(200 树) 67.1%/0.72

### 运行时性能（Table XII）
- 参数量 1.44M，5.46M FLOPs，FP32 权重 5.50 MiB
- 平均推理延迟 0.25 ms（P95 0.31 ms），吞吐 4,019 Hz（batch=1）/ 10,992 Hz（batch=32）
- 在 60 Hz 控制率下可实时运行

### 下游模仿学习提升（Table XIII，40 trials）
- 冻结 NeuralActuator 提供力反馈输入后：
  - Pick-and-Place 成功率：80% → **92.5%**
  - Go Up-and-Stay 成功率：85% → **95%**

### 在线自适应（IV-E）
- 12 条新轨迹（7,810 帧），70 epochs，356 秒（约 5.1 秒/epoch），跟踪误差平均下降约 **44%**

### 架构消融（Table XIV，参数量对齐约 1.4M，500 步）
- 力估计 MAE：MLP 0.47 N，GRU 0.49 N，LSTM 0.41 N，**Transformer 0.23 N**（在 6 项指标中 4 项最优或并列最优）

### 跨平台验证（IV-F）
- Franka Panda（7-DoF，>$30k）：离线负载力估计 F MAE 0.26–0.42 N，但指令位姿依赖未来记录状态（future-state-conditioned），是离线基准而非真正在线评估
- SO-101（6-DoF，LeRobot，STS3215 舵机）：500 步 F MAE 0.47–0.73 N，最大关节误差约 9.8°（Joint 3），语料 65,916 帧（约 63.5 Hz）
