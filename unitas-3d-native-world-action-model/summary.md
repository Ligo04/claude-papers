# UNITAS 论文摘要

## 1. 背景问题

视频式 WAM 往往用 future image 或 visual latent 表示世界变化。但机器人交互发生在 metric 3D space：

- 相同的物理位移，在不同 camera view 下可能产生不同 pixel displacement；
- 相同的 pixel displacement，不代表相同的物理距离；
- camera intrinsics、camera-to-robot pose 和 embodiment 改变后，2D 表征难以保留接触与几何关系；
- 图像预测看起来逼真，也可能让静态区域漂移、物体位移错误或接触关系变形。

UNITAS 的问题是：能否让 observation、action 和 action-conditioned scene dynamics 共享一个物理坐标接口？

## 2. 核心方法

UNITAS 在一个固定 interaction window 中使用：

- **World-aligned visual tokens**：有 depth 时将像素反投影到 world frame；无 depth 时沿 camera ray 采样 64 个候选深度，在 0.1–2.0 m 范围内由 feature routing 推断位置。
- **Action flow**：每只机器人手臂从 URDF gripper geometry 采样 20 个点；人手使用 21 个 MANO keypoints。每个点表示一条 3D displacement trajectory。
- **Scene flow**：对环境 query points 预测点位移，并以 action flow 为条件。
- **Physical-time trajectory tokenizer**：时间用真实秒数而不是 frame index 表示；每条 point trajectory 压成一个 token，Tmax = 3 s。
- **Coupled experts**：point dynamics expert 负责 action/scene flow；smaller action expert 负责 executable end-effector action chunk。
- **Two modes**：policy mode 只生成 executable action；simulator/world-model mode 额外生成 point flow 和场景后果。

联合因子分解为：

    p(A, u, S | C, q0)
      = p(A | C) · p(u | C, q0) · p(S | A, C)

其中 A 是 action flow，u 是可执行 action chunk，S 是 scene flow，C 是 instruction、multi-view observation、calibration 和历史 action flow，q0 是当前 end-effector state。

## 3. 数据与训练

预训练混合九个来源：

- RoboTwin、LIBERO、VLABench：仿真操作；
- DROID、AgiBot、RoboMIND、RH20T、W1：真实机器人；
- VITRA：人手/egocentric hand data。

时间采样按 physical time 对齐：

| 数据源 | 采样率 | future intervals H | history Hhist | 覆盖时长 |
| --- | ---: | ---: | ---: | ---: |
| RoboTwin | 15 Hz | 20 | 20 | 1.33 s |
| LIBERO | 20 Hz | 20 | 20 | 1.00 s |
| VLABench | 10 Hz | 15 | 15 | 1.50 s |
| DROID / AgiBot / RoboMIND / W1 | 15 Hz | 20 | 20 | 1.33 s |
| RH20T | 20 Hz | 20 | 20 | 1.00 s |
| VITRA | 6/8/10 Hz | 10 | 10 | 10/f s |

模型配置：

- Florence-2-large VLM：570.7M；
- point dynamics expert：28 layers, width 1280, 20 heads，826.3M；
- action expert：28 layers, width 640, 10 heads，256.0M；
- tokenizer：1.9M；
- total：1,694.5M parameters；
- pretrain：200K steps，total batch 1,024，64 BW1000 H DCUs；
- action flow、scene flow、action chunk loss 权重为 1，3D PE loss 权重为 0.1；
- inference：10 Euler steps。

## 4. 主要结果

### 4.1 Simulation policy

| Benchmark | UNITAS | 论文含义 |
| --- | ---: | --- |
| LIBERO average | 99.8% | 包括 Spatial/Object/Goal/LIBERO-10 |
| LIBERO-Plus overall | 89.1% | 7 类 observation/instruction/initial-state perturbation |
| RoboTwin Clean | 92.94% | 50 bimanual tasks |
| RoboTwin Randomized | 93.04% | scene randomization |
| VLABench SR | 56.2% | Progress Score 67.6，Intention Score 69.3 |

混合数据预训练带来的总体增益是 2.6 percentage points；robot initial-state perturbation 从 84.8% 提升到 94.2%，language perturbation 从 77.6% 提升到 85.0%。

### 4.2 Real-world

每个任务每个方法 20 trials：

| 任务 | π0.5 | Fast-WAM | UNITAS |
| --- | ---: | ---: | ---: |
| Organize Books | 15/20 = 75% | 13/20 = 65% | 18/20 = 90% |
| Pack the Ball | 11/20 = 55% | 12/20 = 60% | 16/20 = 80% |

这证明的是两个真实任务上的 transfer，不是任意真实操作场景的通用成功率。论文引言有一句把 Pack the Ball 的 UNITAS 结果写成 65%；摘要、§4.3 正文和表 2 一致给出 80%，这里按表 2 记录。

### 4.3 Action-conditioned scene prediction

RoboTwin 550 test clips，指标单位为 cm：

| 方法 | Moving ADE ↓ | Static ADE ↓ | Moving FDE ↓ | Moving Acc@1cm ↑ |
| --- | ---: | ---: | ---: | ---: |
| PointWorld | 0.656 | 0.084 | 1.050 | 88.22% |
| Ctrl-World + DA3 + STv2 | 3.310 | 0.369 | 5.503 | 41.54% |
| X-WAM + STv2 | 2.286 | 0.122 | 3.886 | 49.69% |
| UNITAS | 0.427 | 0.043 | 0.825 | 92.31% |

闭环 action–scene consistency 中，UNITAS 的 Moving ADE/FDE 为 1.211/2.020 cm，Moving Acc@1cm 为 64.26%，优于比较方法。

## 5. 论文结论的准确边界

UNITAS 说明：把 physical interaction geometry 作为 action learning 和 world modeling 的共同接口，能同时改善 manipulation performance、viewpoint/initial-state robustness 和 action–scene consistency。它没有证明：

- 所有 scene dynamics 都能被准确预测；
- policy mode 在长时程、多接触任务中始终稳定；
- real-world scene flow 已有定量 ground truth 验证；
- 55.77 ms action-only latency 可以直接等同于完整机器人闭环 latency。

