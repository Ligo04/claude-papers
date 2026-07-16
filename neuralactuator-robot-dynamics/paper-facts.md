# NeuralActuator — 核心事实卡（供材料生成使用，所有数字均来自论文原文，禁止编造）

## 基本信息
- 标题: NeuralActuator: Neural Actuation Modeling for Robot Dynamics and External Force Perception
- 作者: Zhiyang Dou, John U. Onyemelukwe*, Hangxing Zhang*, Heng Zhang, Minghao Guo, Yunsheng Tian, Michal Piotr Lipiec, Joshua Jacob, Chao Liu, Peter Yichen Chen (MIT), Yuri Ivanov (Amazon Robotics), Wojciech Matusik (MIT)
- arXiv: 2607.11734v1 [cs.RO], 2026-07-13
- 项目页: https://frank-zy-dou.github.io/projects/NeuralActuator/index.html

## 问题动机
- 低成本舵机平台上线性电流-力矩近似 τ = K_t·I 不可靠：齿轮箱摩擦、迟滞 (hysteresis)、背隙 (backlash)、电流饱和、热漂移
- 执行器动力学是 sim-to-real 误差的主要来源；且低成本平台没有力/力矩传感器 → 无感力觉 (sensorless force perception) 需求
- 实测：在 OpenManipulator-X 上逐关节拟合 K_t，有两个关节在 clamp 前拟合出负斜率——线性假设彻底失效的证据

## 方法（四头多任务 Transformer + 可微仿真训练）
- 输入序列 X_t：8 帧历史 + 当前帧 = 9 token；每帧特征 = 指令目标 (q_cmd, q_cmd_g)、本体感知 (q, q̇, q_g)、执行器遥测 (u 电流/负载寄存器/指令力矩, 电压 V, 温度 T)、跟踪误差 e = q_cmd − q
- 架构：Transformer encoder ×4 层, d=192, 4 heads, d_ff=384；pre-norm 残差块 + query 依赖的逐元素注意力门控 A = [MHA(U) ⊙ σ(U·W_g)]·W_o；GELU FFN；末层后时序均值池化 → h_pool；四个 SiLU-MLP 输出头
- 四个头:
  1. 力矩代理 (torque surrogate) τ_pred ∈ R^{n_m}：clip 后 τ̄ 作为可微仿真器的广义控制输入，s_{t+1} = DiffSim(s_t, τ̄_t)
  2. 原始外力 f̂_raw ∈ R³
  3. 接触概率门 g ∈ [0,1]；最终力 f̂ = g · f̂_raw；门标签 g_gt = I[‖f_gt‖ > ε], ε=0.01 N
  4. 逐电机状态分数 c ∈ [0,1]^{n_m}（实验只监督/评估 Joint 3；c₃≈0 = 机械受限）
- 关键训练思想：力矩代理头**没有直接力矩标签**——通过可微仿真（MuJoCo/MJX, JAX+JIT）前向积分把预测力矩映射为未来位姿，用**位姿轨迹**做监督（BPTT，式 11-12）；力/门/状态头用直接标签监督
- 损失：Huber (Smooth L1) 回归 L_joint、L_grip (κ_g=1000 米→毫米)、L_force (β_f=0.15 力传感器/已知重量微调阶段, 否则 1)；BCE 的 L_gate、L_cond；可选速度损失 L_vel (λ_vel=5 或 10, SO-101/Franka)；AdamW，最多 100k epochs，实际最佳 checkpoint 3 万 epochs 内出现
- 梯度诊断（Table II）：rollout 视野 H∈{64,128,256,320,500}，仿真器输入梯度范数保持 O(1e-2)，参数梯度余弦相似度 ≥0.96 (相对 H=128)，‖∇θL‖ 在 H=256–320 达到平台 → 用于课程学习设计
- 物理解释（式 4）：τ_ID = M(q)q̈ + C q̇ + g(q) = τ_act + J_v^T f_ext，但**不作为强制分解约束**；训练/评估用无负载模型且 MJX 接触关闭 → 代理是"集总的仿真等效广义输入"，会吸收未建模接触/负载效应
- 夹爪运动学（式 19）：旋转电机驱动对称双滑块，a(α) = 2(√(l²−r²sin²α)+r·cosα) − 2δ；准静态夹持力 F = T·cosβ, τ = 2Tr·sinγ

## 数据集 NAD (Neural Actuation Dataset)
- 双臂 leader-follower 遥操作，2× OpenManipulator-X，DYNAMIXEL XM430-W350 舵机；同步记录关节位姿/速度、电流、PWM、电压、线圈温度；6 轴力/力矩传感器提供 3 通道力标签（基座系，环境作用于机器人符号约定）
- 组成（OMX 实验子集，总计 ~94.52 分钟）:
  - 自由运动 (无外力): 119,933 帧, ~34.15 min（圆周运动、单关节扫掠覆盖 ~90% 行程、全臂原语）
  - 力标注交互: ~46.24 min = 已知重量负载 200/300/400/500 g (完整集另有 100 g), 88,611 帧 + 轴对齐力传感器交互 ±X±Y±Z (最大 9 N), 74,020 帧（含配对的无接触参考轨迹）
  - 电机受限 (橡皮筋捆 Joint 3): 49,898 帧, ~14.13 min
- 8:1:1 训练/验证/测试划分；GUI 人工校验对齐

## 实验结果（关键数字）
- Rollout 精度 (Table III): 平均关节 MAE 2.5–3.2°（100→600 步），夹爪滑动坐标 0.2 mm；600 步 ≈ 10 s
- 力估计:
  - 力传感器基准 (Table IV): 接触时 F MAE 0.36–0.39 N @100 步, 0.38–0.57 N @500 步；无接触参考近零 (0.00–0.02 N)；≤50 g (~0.5 N) 负载接近平台噪声下限
  - 负载基准 (Table V): 平均 F MAE 0.08–0.12 N，单任务最大 0.24 N
  - 推力计 (Table VI/Fig 1): 三高度前推 trace MAE 0.037/0.015/0.028 N；表内平均 F 0.08–0.10 N
  - 未见物体泛化: 261 g 与 226 g 未见形状物体上评估预训练模型
- 基线对比 (Table VII, 负载基准@100 步): ID-Linear 1.41 N, ID-Friction (Stribeck+背隙) 1.23 N, GMO 广义动量观测器 0.66 N, NeuralActuator 0.12 N（GMO 的 1/5.5）；无接触时经典方法误报接触率 33–59%，NeuralActuator 近零。注意基线每步吃真值状态，NeuralActuator 用自己的仿真 rollout 状态
- 在线自适应 (IV-E): 12 条新轨迹 (7,810 帧), 70 epochs, 356 s (~5.1 s/epoch)，跟踪误差平均降 ~44%
- 跨平台 (IV-F):
  - Franka Panda (7-DoF, >$30k): 离线负载力估计基准，F MAE 0.26–0.42 N；注意指令位姿用平滑代理 q̃_cmd = 1.03·q_rec(t+5)（依赖未来记录状态）→ 是 future-state-conditioned 离线基准而非在线评估；仿真端保留原生 affine PD 位置执行器
  - SO-101 (6-DoF, LeRobot, STS3215 舵机, 用符号负载寄存器做 effort 输入): 500 步 F MAE 0.47–0.73 N，最大关节误差 ~9.8° (Joint 3)；语料 65,916 帧 (~63.5 Hz)
- 电机状态分类 (Table XI): 准确率 91.0%, 精确率 84.5%, 召回率 96.2%, AUC-ROC 0.95；基线: 阈值 58.6%/0.45, SVM (RBF, 20 手工特征) 59.9%/0.62, 随机森林 (200 树) 67.1%/0.72
- 运行时 (Table XII): 1.44M 参数, 5.46M FLOPs, FP32 5.50 MiB；平均推理 0.25 ms (P95 0.31 ms)，吞吐 4,019 Hz (b=1) / 10,992 Hz (b=32)；60 Hz 控制率下实时
- 下游 BC 模仿学习 (Table XIII, 40 trials): 冻结 NeuralActuator 提供力反馈输入 → Pick-and-Place 80% → 92.5%，Go Up-and-Stay 85% → 95%
- 架构消融 (Table XIV, ~1.4M 参数对齐, 500 步): 力 MAE — MLP 0.47, GRU 0.49, LSTM 0.41, Transformer 0.23；Transformer 在 6 指标中 4 项最优或并列
- 视觉监督 (IV-J): 可微渲染 + 轮廓损失（VGGT 估内参、SAM3 分割、EasyHeC 式），平均 IoU 0.8515（只评对齐质量）
- 额外后端: NVIDIA Warp (FeatherstoneIntegrator, wp.Tape 反向), 60 Hz 环境率 ×5 substeps, armature 0.01

## 局限（论文自述）
1. 长视野记录遥测条件下的轨迹传播误差累积（在线自适应缓解）
2. 力矩代理是"仿真器等效"而非可辨识的物理力矩——未建模接触/负载效应会被吸收进去
3. 在线推理需要实时 effort 遥测；**不能对未执行的候选指令做反事实 rollout**（MPC 需要额外的 effort 响应模型）
4. 力头只估计末端单一 3D 合力：不能定位多点/分布式接触，不能估计完整 6D wrench，且需要显式力标签
5. ≤50 g 负载低于平台噪声下限，可靠检测是未来工作

## 与先前工作的区别
- Hwangbo et al. 2019 (ANYmal actuator net): 需要力矩传感标签；本文无需
- Schwendeman et al.: 需要标定的电流→力矩估计；本文无需
- 经典外力估计 (GMO 等): 依赖可靠力矩反馈/线性电流映射，在齿轮舵机上失效
- NeuralSim/NeRD 等神经增强仿真: 关注残差动力学；本文专攻低成本无芯 DC 舵机执行器 + 多任务感知头
