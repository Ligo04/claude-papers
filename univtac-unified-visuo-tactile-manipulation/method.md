# 方法拆解：让视触觉数据、表示和评测形成闭环

> 这份方法笔记同时参考论文 PDF、arXiv HTML 和公开仓库 `05bcd3ed`。凡是来自仓库的实现观察都会单独标注；它们不自动等同于论文中未展开的训练细节。

## 1. 总体架构

```text
┌────────────────────────────────────────────────────────────────────┐
│ UniVTAC Platform                                                   │
│  TacEx + Isaac Sim + 传感器模型 + Grasp/Move/Place/Probe/Rotate API │
└───────────────┬────────────────────────────────────────────────────┘
                │ 自动化接触 episode + 仿真特权标签
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 数据样本                                                            │
│ I_marked, I_pure, D, M, p + RGB/深度/动作/场景元数据                  │
└───────────────┬────────────────────────────────────────────────────┘
                │ 多任务 MSE 预训练
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ UniVTAC Encoder                                                    │
│ ResNet-18 共享编码器 → Shape / Contact / Pose 多个训练头             │
│ 部署时丢弃 decoder，只保留 tactile-centric latent                   │
└───────────────┬────────────────────────────────────────────────────┘
                │ 视觉特征 + 触觉 latent
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ ACT / 其他策略 → 八项 UniVTAC Benchmark 任务 → 真机 sim-to-real     │
└────────────────────────────────────────────────────────────────────┘
```

核心不是单独的传感器仿真或单独的 encoder，而是让同一平台负责产生“可学习的接触变化”、训练具备物理归因的 latent，并用物理有效性标准评估下游策略。

## 2. Platform：传感器与操作 API

### 2.1 传感器异构性

论文平台建模三类视触觉传感器：GelSight Mini、ViTai GF225、Xense WS。各传感器的光学和机械差异通过内部相机内参、gelpad 网格和渲染方法调节，因此理论上可以在统一接口下比较不同硬件。项目页还把 Xense WS 描述为 force/torque 类型；论文正文主要以视触觉图像、marker 和深度信号为例。

### 2.2 五个原子 API

| API | 作用 | 关键实现 |
|---|---|---|
| `Grasp` | 接近并闭合夹爪 | 依据触觉深度自适应限速，防止穿透/剪裁 |
| `Move` | 生成精确位姿轨迹 | 使用 cuRobo 的碰撞感知规划 |
| `Place` | 稳定放置物体 | 轨迹优化与终点放置 |
| `Probe` | 安全开始接触 | 让传感器获得有效触觉读数 |
| `Rotate` | 接触后小范围旋转 | 制造剪切力、marker flow 和滑移线索 |

`Move`/`Place` 解决“把机器人带到那里”，`Grasp`/`Probe` 解决“怎样安全地产生有意义的接触”。这一区分避免把低层物理积分、轨迹规划和触觉数据采集揉成一个不可复用脚本。

### 2.3 抓取反馈方程

抓取阶段的关节速度由最小深度 `d_min`、无接触深度 `d_max` 和 reaching threshold `δ_th` 决定：

```text
q̇ = v_fast                                      , d_min = d_max
q̇ = min(|d_min − δ_th|, v_slow)                  , d_min < d_max
```

无接触时快速移动；一旦触觉深度偏离无接触值，就切换到受限速度。这里的 `v_fast` 与 `v_slow` 是速度上限，`δ_th` 控制最终压入程度。论文把它作为运行时数据生成安全机制，而不是学习策略。

## 3. Contact-rich 数据生成

### 3.1 为什么“完美 expert 轨迹”不够

自动 expert 程序很容易合成无碰撞、一次成功的插入轨迹，但这种轨迹中的接触通常瞬时且单一，策略看不到“偏了一点后怎样修正”的证据。对触觉学习而言，失败和纠正不是噪声，而是监督信号的载体。

### 3.2 论文中的 episode 过程

1. 选取 14 种压头几何（球、锥等凸形，以及星形、十字等非凸形），安装到标准 prism base。
2. 夹爪接近 prism 并初始化接触。
3. 随机化 reaching depth threshold `δ_th`，覆盖轻触到深压入。
4. 接触后用 `Move`/`Rotate` 做随机小旋转，产生法向压入、切向剪切和 marker 位移。
5. 同步记录带 marker RGB、无 marker 的 counterfactual RGB、gelpad 深度、marker 2D 投影和物体相对位姿。
6. 在 benchmark 的接触技能里额外引入随机失败，让 expert 根据仿真 ground-truth 接触反馈采取纠正动作。

报告的规模是每种形状约 14,000 个 interaction frames，总计 **205,826 samples**。这里的“约”和总数都应作为论文作者报告的采集量，而不是从仓库快照重新统计的结果。

### 3.3 样本变量

| 名称 | 论文记号/形状 | 语义 |
|---|---|---|
| `I_marked` | `3×H×W` | 带传感器 marker 纹理的 RGB 触觉图 |
| `I_pure` | `3×H×W` | 去掉 marker 伪影的纯接触图 |
| `D` | `H×W` | gelpad 法向形变/压入深度 |
| `M` | `2×N` | `N` 个 marker 在图像平面的 2D 位置 |
| `p` | `7` | 物体相对位姿：3D 平移 + 4D 四元数 |

真实传感器通常只直接给出 RGB/marker 图；`I_pure`、`D`、`M` 和 `p` 是仿真提供的特权标签，用于训练和评测。

## 4. UniVTAC Encoder

### 4.1 共享编码器与训练头

论文采用 ResNet-18 作为共享 backbone，把一张带 marker 的触觉图压缩为 latent。训练阶段连接四类输出头：

```text
marked RGB ─┐
pure RGB   ─┤→ RGB 转置卷积 decoder（Shape pathway）
depth      ─┤→ 单通道转置卷积 decoder（Contact pathway）
marker     ─┤→ 轻量 MLP，输出 2D marker 位置
pose       ─┘→ 紧凑 MLP，输出 7 维相对位姿（Pose pathway）
```

所有头共享同一个 latent，因此每条路径的梯度都要求表示保留可被其他任务利用的物理因素。部署时只保留 ResNet-18 编码器，decoder 不产生额外推理成本。

### 4.2 三条感知路径的归因

**Shape pathway** 同时重建 `I_marked` 与 `I_pure`。如果 latent 只记住 marker 纹理，它无法生成没有 marker 的图；双视图目标迫使模型把物体固有形状与传感器伪影分开。

**Contact pathway** 预测 `D` 和 `M`。深度图表达法向压入，marker flow 表达切向剪切和相对滑移；联合目标把像素变化连接到接触力学，而不是只追求漂亮的 RGB 重建。

**Pose pathway** 回归 `p∈R^7`，把局部触觉图放进外部度量空间。没有这一条，表示可能知道“碰到了什么形状”，却不知道物体相对夹爪偏向哪一侧，难以做插入和放置纠正。

### 4.3 损失函数

论文把所有路径的基本误差设为 MSE：

```text
L_shape   = MSE(Î_marked, I_marked) + MSE(Î_pure, I_pure)
L_contact = MSE(D̂, D) + MSE(M̂, M)
L_pose    = MSE(p̂, p)
L_total   = λ_s L_shape + λ_c L_contact + λ_p L_pose
```

实验权重为：

```text
λ_s = 1.0,  λ_c = 0.5,  λ_p = 0.5
```

这些权重是论文明确给出的超参数；它们不是根据表 II 的结果反推出来的。

### 4.4 公开仓库的实现对应

在仓库 commit `05bcd3ed` 的 `encoder/network.py` 中可以直接看到：

- `Tactile` 默认使用 `resnet18(num_classes=latent_dims)`；
- `RGBDecoder` 由全连接层、四次转置卷积、上采样和卷积组成，深度头把输出通道设为 1；
- `MarkerDecoder` 是 `latent → 256 → marker_nums×2` 的 MLP；
- `PoseDecoder` 是 `latent → 256 → 7` 的 MLP；
- `loss()` 对各监督键使用 MSE，并按传入权重相加；代码快照的 marker head 默认 `marker_nums=63`。

仓库代码是可读的参考实现，但不能据此断言论文所有训练实验都使用完全相同的初始化、数据切分或配置；例如项目页宣传的 ImageNet 初始化并没有在该构造函数里以 `pretrained=True` 形式显式出现。

## 5. Benchmark 设计

### 5.1 三个感知维度

```text
Pose reasoning       → Lift Bottle / Lift Can / Put Bottle in Shelf
Shape perception     → Grasp Classify
Contact-rich         → Insert Hole / Insert HDMI / Insert Tube / Pull-out Key
```

任务难度从“能否通过接触辨认形状”到“在视觉遮挡和连续接触中持续修正位姿”。每项任务同时提供标准视觉观察与触觉观察，图示中为清晰起见只展示一侧指尖。

### 5.2 物理有效性门槛

一次 trial 若最大穿透深度超过安全阈值，或 gelpad 与物体之间出现显著相对滑移，就判为 invalid。这个门槛比“最终物体是否到了目标坐标”更严格，能排除穿模、硬推和数值漏洞带来的伪成功。

## 6. 策略训练与评测流程

论文正文的高层流程如下：

```text
自动生成 50 条完整轨迹/任务
          ↓
ACT（vision-only）       VITaL       ACT + UniVTAC Encoder
          └───────────────┬──────────────────────────────┘
                          ↓
                 100 次测试 rollout/任务
                          ↓
              成功率 + 任务/路径消融 + 数据规模曲线
```

附录 B 给出的 ACT 训练细节是：总计 4,000 个优化步、batch size 64、视觉和触觉 encoder 学习率 `1×10^-5`、weight decay `1×10^-4`；Transformer encoder 4 层、decoder 7 层，视觉使用固定正弦余弦位置编码、触觉使用可学习位置编码，预测未来 50 个动作并用 temporal aggregation 执行。它们是策略实验设置，不应与 encoder 多任务损失权重混为一谈。

## 7. 真机迁移流程

```text
仿真接触数据 → 预训练 UniVTAC Encoder ──┐
                                          ├→ ACT + diffusion policy head
VR/Meta Quest 专家示范 ───────────────────┘
                                          ↓
                         Marvin 7-DoF + 腕部 RGB + 双 GF225
                                          ↓
                         不再微调的 20-rollout/任务评测
```

真机使用 Insert Tube、Insert USB、Bottle Upright 三项任务；每项 150 条遥操作示范，触觉传感器 30 Hz。Encoder 只在仿真预训练，部署时不做 fine-tuning，目的是测表示本身的跨域可迁移性。

## 8. 可复现的教育版伪代码

```python
# 这是流程伪代码，不是官方训练脚本。
episodes = collect_contact_episodes(
    shapes=14,
    depth_threshold="random",
    actions=["grasp", "probe", "move", "rotate"],
    corrective_failures=True,
)

for batch in episodes:
    latent = encoder(batch["marked_rgb"])
    pred = {
        "marked_rgb": shape_head_marked(latent),
        "pure_rgb": shape_head_pure(latent),
        "depth": depth_head(latent),
        "marker": marker_head(latent),
        "pose": pose_head(latent),
    }
    loss = (
        1.0 * mse(pred["marked_rgb"], batch["marked_rgb"])
        + 1.0 * mse(pred["pure_rgb"], batch["pure_rgb"])
        + 0.5 * mse(pred["depth"], batch["depth"])
        + 0.5 * mse(pred["marker"], batch["marker"])
        + 0.5 * mse(pred["pose"], batch["pose"])
    )
    update(encoder, heads, loss)

# 部署时只保留 encoder；把 latent 与视觉特征交给策略。
action = policy(vision_obs, encoder(tactile_rgb))
```

## 9. 实现陷阱与复现风险

1. **把特权标签当作真实输入**：真实部署不一定有 `I_pure`、`D` 或精确 `M`；它们的作用是训练监督和分析。
2. **把随机失败当作无效数据删掉**：插入纠正所需的触觉线索正来自这些有意的偏差；但失败必须由物理有效性门槛约束。
3. **只用终点成功判断**：穿透和滑移会制造虚高成功率，必须复现 benchmark 的 invalid 规则。
4. **忽略坐标系**：pose 是物体相对 gelpad 中心局部坐标的 3D 平移 + 四元数；marker 是图像平面投影，不能直接和世界坐标混用。
5. **混淆训练阶段**：附录的 ACT 学习率/步数属于策略训练；encoder 的多任务权重是另一组配置。
6. **假设三类传感器都已可运行**：论文/项目页描述三类支持，但公开仓库 TODO 明确当前采集和评测只支持 GelSight Mini。
7. **忽略仓库依赖变更**：安装需要 Linux、Python 3.10、Isaac Sim 4.5、Isaac Lab 2.1.1、cuRobo 和项目修改过的本地 TacEx；直接安装公共 TacEx 可能得到不兼容 API。
8. **把在线数据集预览当作原始数据真相**：Hugging Face 当前 viewer 报 schema `CastError`；应下载原始文件并按 HDF5 结构核验。
9. **把许可证说明照抄 README**：README 写 MIT，仓库 `LICENSE` 是 Apache-2.0，法律使用应以许可证文件为准。

## 10. 复现实验的最小分层

| 层级 | 可验证目标 | 需要的资源 |
|---|---|---|
| L0 | 运行本目录概念 demo，理解五 API/三路径 | Python 标准库 |
| L1 | 复现 encoder 的 MSE 头与路径消融 | 仓库、PyTorch、触觉 HDF5 |
| L2 | 复现八任务 ACT 成功率 | Isaac Sim/Lab、TacEx、cuRobo、GPU |
| L3 | 复现真机 25 个百分点平均增益 | Marvin、双 GF225、VR 遥操作、20 rollout/任务 |

论文的数字结果位于 L2/L3；只完成 L0/L1 时，不应把结果写成“复现了 UniVTAC”。
