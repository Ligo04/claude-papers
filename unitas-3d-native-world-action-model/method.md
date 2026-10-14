# 方法详解：3D-native WAM 的五个接口

## 1. Context 与目标变量

在一个 prediction window 中，context C 包含：

    instruction ℓ
    multi-view observations O_t
    camera intrinsics K_i
    camera-to-world transforms T^W_t,i
    historical action flow H_A
    current end-effector state q0

模型输出三类变量：

- A：交互主体的 action flow；
- u：可执行的 end-effector action chunk；
- S：环境 scene flow。

联合分解：

    p(A, u, S | C, q0)
      = p(A | C) · p(u | C, q0) · p(S | A, C)

policy mode 只估计 p(u | C, q0)；simulator/world-model mode 还生成 A 和 S。

## 2. World-aligned observations

### 有 depth

对图像像素做反投影：

    P_camera(u,v) = D(u,v) K^{-1} [u,v,1]^T
    P_world(u,v)  = T_camera_to_world [P_camera(u,v), 1]^T

再将 P_world 送入 Fourier positional map 与 3D encoder，加入 visual token。

### 无 depth

UNITAS 沿 camera ray 采样 M = 64 个候选深度，范围 0.1–2.0 m。feature 计算候选深度的 attention weight，再把 camera-frame keys 和 world-frame values 聚合成 routed world positional embedding。这样训练可以混合 RGB 和 RGB-D，推理时可以没有 depth。

## 3. Physical-time point flow

机器人每只手臂从 URDF gripper geometry 采样 20 个点；人手采用 21 个 MANO keypoints。对每个 query point：

    Δx_n,k = x_(t+τ_k,n) - x_(t,n)

时间变量是物理秒数 τ，而非 frame index。Tmax = 3 s。一个 point trajectory 编成一个 token，decoder 在指定 τk 上重建 displacement。

Scene query points 从 point cloud 采样；仿真数据使用 scene state 保留对应关系。缺少某种 supervision 时用 mask 忽略该 loss 或 action slot。

## 4. Coupled experts

- Point dynamics expert：28 layers，width 1280，20 heads；处理 world-aligned visual tokens、history、action flow 和 scene flow。
- Action expert：28 layers，width 640，10 heads；输出每个未来时刻的 translation、rotation、gripper openness。
- VLM：Florence-2-large，570.7M。
- Tokenizer：先独立重建 action/scene/history trajectories，训练后冻结。

Attention 约束：

- scene tokens 可以看 observation、clean action flow 和其他 scene tokens；
- scene tokens 不看 noisy action flow；
- action chunk tokens 只看 observation KV、current EEF state 和自身 token；
- policy-only inference 可跳过 future action-flow 和 scene-flow tokens。

## 5. Training objective

使用 rectified flow。对 clean target y 与 Gaussian noise ε：

    y_γ = (1 − γ) ε + γ y
    v_target = y − ε

联合目标：

    L = λA LA + λs Lscene + λctrl Lctrl + λPE LPE

其中：

- λA = 1；
- λs = 1；
- λctrl = 1；
- λPE = 0.1。

可用标注缺失时采用 masked loss。

## 6. Policy evaluation

训练设置分两种：

1. mixed-dataset pretraining，再对每个 benchmark/dataset fine-tune；
2. no-pretraining，直接在目标 benchmark/dataset fine-tune。

评测覆盖：

- LIBERO；
- LIBERO-Plus 七类扰动；
- RoboTwin 2.0 Clean/Randomized；
- VLABench 五条 track；
- Organize Books、Pack the Ball 两个 real-robot tasks；
- action-conditioned scene prediction 与 closed-loop action–scene consistency。

## 7. Inference cost

RTX 5090 D 上：

| 输出模式 | Latency / chunk |
| --- | ---: |
| UNITAS actions only | 55.77 ms |
| actions + action flow | 122.51 ms |
| actions + both flows, 1024 scene queries | 305.49 ms |
| actions + both flows, 3000 scene queries | 561.60 ms |

因此 policy mode 和 simulator mode 必须分开报告；不能用 world-model latency 替代 action-only control latency，也不能把 action-only latency 直接宣称为完整系统实时率。

## 8. 复现风险

1. camera calibration 与 world frame 不一致会直接破坏 3D PE。
2. URDF gripper points、MANO keypoints 与 action labels 的 point identity 必须稳定。
3. 不同数据源的 coordinate convention、frame rate 和 physical horizon 不能直接拼接。
4. 只复现 action loss 而省略 action-flow/history/scene-flow supervision，会改变模型。
5. scene query 数影响 world-model latency，必须固定或报告。
6. real-world scene-flow 目前主要是 qualitative transfer，不能写成定量真机 scene-prediction benchmark。

