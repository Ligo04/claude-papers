# 摘要与事实卡片

## 1. 背景与问题

视觉语言动作（VLA）策略在很多操作任务上进步很快，但接触丰富的操作仍有一个“看不见”的环节：末端执行器会遮挡目标，近距离深度不准，物体一旦接触就无法仅靠相机直接观察界面的力、形变和滑移。视触觉传感器能在接触界面提供局部几何、压力分布和相对运动，因此对插入、对齐、探测和纹理分类特别重要。

真正的基础设施瓶颈有两层：

1. 真实触觉数据采集昂贵，传感器设计不统一，难以产生大规模且覆盖丰富接触状态的数据。
2. 缺少统一 benchmark，无法在相同任务、传感器和成功标准下比较策略，也难以判断一个表示是否真的学到了接触物理。

UniVTAC 的回答不是再提出一个更大的策略，而是把 **仿真数据生成 → 触觉表示预训练 → 统一策略评测 → 真机迁移** 串成一个闭环。

## 2. 三项贡献

### 2.1 UniVTAC Platform

基于 TacEx 和 NVIDIA Isaac Sim，平台建模三类视触觉传感器：

- GelSight Mini；
- ViTai GF225；
- Xense WS。

传感器差异通过内部相机内参、gelpad 网格和渲染方法表达，使同一数据/算法框架可以跨硬件规格验证。平台提供五个原子操作 API：`Grasp`、`Move`、`Place`、`Probe`、`Rotate`；其中 Move/Place 使用 cuRobo 做精确轨迹，Grasp/Probe 使用触觉反馈控制接触过程。

### 2.2 UniVTAC Encoder

共享 ResNet-18 编码器把带标记的触觉图像映射到紧凑 latent。训练时挂载多个解码头，部署时只保留编码器：

| 路径 | 预测目标 | 希望学到的物理因素 |
|---|---|---|
| Shape | 带 marker RGB 与无 marker RGB | 物体全局形状，去除 marker/光照伪影 |
| Contact | gelpad 法向深度图与 marker 位置 | 局部压入、剪切、滑移和接触动力学 |
| Pose | 7 维相对位姿（平移 3 + 四元数 4） | 外部度量空间中的空间定位 |

具体头部是：RGB 转置卷积 decoder、单通道 depth decoder、轻量 marker MLP、7 维 pose MLP。所有头在部署时丢弃，所以不会给策略推理增加额外解码开销。

### 2.3 UniVTAC Benchmark

八项任务覆盖三个感知维度：

| 维度 | 任务 | 需要什么 |
|---|---|---|
| Pose reasoning | Lift Bottle、Lift Can、Put Bottle in Shelf | 估计夹爪与物体的相对位置/方向 |
| Shape perception | Grasp Classify | 通过接触探索区分相似外观、不同纹理的物体 |
| Contact-rich interaction | Insert Hole、Insert HDMI、Insert Tube、Pull-out Key | 在连续接触、遮挡和错位中做纠正 |

Benchmark 使用自动化 API、标注资产和 expert program；对插入等技能故意加入随机失败，再让 expert 根据接触反馈纠正，从而产生比“完美无碰轨迹”更有信息量的训练数据。

## 3. 数据生成细节

平台准备 14 种几何压头，从球、锥等凸形到星形、十字等非凸图案，安装在标准 prism base 上。每个 episode 中：

1. 夹爪接近带压头的 prism 并开始接触；
2. 随机化 reaching depth threshold `δ_th`，覆盖轻触到深压入；
3. 接触后通过 Move/Rotate 执行随机小旋转，制造剪切力和 marker 位移；
4. 同步保存带 marker RGB `I_marked`、无 marker 的纯接触图 `I_pure`、深度图 `D`、marker 的 2D 投影 `M` 和物体在 gelpad 中心局部坐标系下的位姿 `p`。

报告称每种形状约生成 14,000 个 interaction frames，总计 **205,826 个样本**。这些“仿真特权”信号用于监督表示学习，不要求真实传感器在部署时提供同样的 ground truth。

### 自适应抓取速度

为避免夹爪穿过 gelpad 或产生不现实的 clipping，抓取阶段使用基于最小深度的反馈：

```text
q̇ = v_fast                                  , d_min = d_max
q̇ = min(|d_min − δ_th|, v_slow)              , d_min < d_max
```

`d_max` 是无接触深度，`d_min` 是当前传感器最小深度；无接触时快速接近，接触后限速并朝阈值收敛。这个控制器不是策略本身，而是数据生成安全机制。

## 4. Encoder 的多任务目标

令 `Î_marked`、`Î_pure`、`D̂`、`M̂`、`p̂` 为各头输出，报告使用 MSE：

```text
L_shape   = MSE(Î_marked, I_marked) + MSE(Î_pure, I_pure)
L_contact = MSE(D̂, D) + MSE(M̂, M)
L_pose    = MSE(p̂, p)
L_total   = λ_s L_shape + λ_c L_contact + λ_p L_pose
```

实验设置为 `λ_s = 1.0`、`λ_c = 0.5`、`λ_p = 0.5`。双视图重建迫使 latent 同时解释“物体本身”和“传感器 marker 伪影”；深度与 marker flow 将像素变化连接到接触形变；位姿回归则防止表示只停留在局部纹理而失去空间尺度。

## 5. Benchmark 成功标准

任务不只检查最终坐标。若最大穿透深度超过安全阈值，或 gelpad 与物体间出现显著相对滑移，trial 会被判为 invalid。这样可以避免策略利用数值/物理漏洞取得虚假的成功率，尤其适合评价插入和抓取稳定性。

## 6. 仿真评测设置

- 策略：ACT（vision-only）与接入 UniVTAC Encoder 的 ACT；VITaL 作为已有视触觉预训练基线。
- 每项任务收集 50 条自动轨迹用于策略训练，每个策略评估 100 次测试 rollout。
- 成功率单位是百分比；平均值是八项任务的算术平均。

### 表 I：八项任务成功率（%）

| 方法 | Lift Bottle | Pull-out Key | Lift Can | Put Bottle in Shelf | Insert Hole | Insert HDMI | Insert Tube | Grasp Classify | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACT（无触觉） | 42.0 | 28.0 | 20.0 | 28.0 | 19.0 | 15.0 | 45.0 | 50.0 | 30.9 |
| VITaL | 72.0 | 47.0 | 8.0 | 32.0 | 25.0 | 6.0 | 34.0 | 100.0 | 40.5 |
| ACT + UniVTAC | **71.0** | 46.0 | **29.0** | 31.0 | 24.0 | **28.0** | **56.0** | 99.0 | **48.0** |

接入编码器后平均成功率从 30.9% 到 48.0%，绝对提升 **17.1 个百分点**。提升并非每个任务都同样大：Insert HDMI、Insert Tube 等连续接触任务受益明显，而 Grasp Classify 上 VITaL 已达到 100%。

### 表 II：路径消融（平均成功率）

| 变体 | Scratch | Contact | Shape | Contact+Shape | Full |
|---|---:|---:|---:|---:|---:|
| Average | 34.0 | 41.9 | 41.6 | 43.9 | **48.0** |

Full 同时使用形状、接触和位姿路径；Contact+Shape 为 43.9%，说明位姿路径对需要空间意识的任务仍有增益。相对 scratch，Contact 提升 7.9 个百分点，Shape 提升 7.6 个百分点。

## 7. 数据规模实验

Encoder 预训练数据量取 `0、1K、5K、10K、50K、100K、200K`。八任务平均成功率依次为：

| 数据量 | 0 | 1K | 5K | 10K | 50K | 100K | 200K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Average (%) | 34.2 | 38.3 | 38.3 | 44.1 | 45.0 | 44.3 | **48.0** |

作者将整体趋势描述为随数据量增加而改善；逐任务数值并不严格单调（例如某些任务在 100K 略降），因此更稳妥的解释是“平均趋势支持规模扩展，任务收益受难度和随机性影响”。

## 8. 真机实验

- 平台：Tianji Robotics Marvin，7-DoF 机械臂、平行夹爪、腕部 RGB 相机。
- 触觉：两个 ViTai GF225，指尖安装，30 Hz 高分辨率形变图。
- 数据：每项任务 150 条 VR（Meta Quest）遥操作专家轨迹。
- 策略：ACT + diffusion policy head；输出关节位置直接动作。
- Encoder 先在仿真合成数据上预训练，部署到真机时**不再微调**。
- 每个任务 20 次 rollout，成功由人工观察者判定。

### 表 IV：真机成功率（%）

| 任务 | Vision only | Vision + UniVTAC | 绝对增益 |
|---|---:|---:|---:|
| Insert Tube | 55.0 | **85.0** | +30.0 |
| Insert USB | 15.0 | **25.0** | +10.0 |
| Bottle Upright | 60.0 | **95.0** | +35.0 |
| Average | 43.3 | **68.3** | +25.0 |

结果支持一个具体结论：仿真中学到的形状/接触/位姿表征，在这三项真机任务上能帮助策略做微小对齐和稳定性修正；它不等于已经解决了任意传感器、任意物体的 sim-to-real。

## 9. 公开实现与论文的边界

论文、项目页和仓库不是完全同一层面的证据：

- 论文描述三类传感器；当前公开 README 的 TODO 写明采集/评测暂只支持 GelSight Mini。
- 项目页列出 Hugging Face 与 ModelScope 数据集，但 Hugging Face 当前数据查看器报告 `DatasetGenerationError / CastError`，说明在线预览 schema 仍有问题；不能据此断言原始文件不可下载。
- 仓库 `encoder/network.py` 可见 ResNet-18、RGB/depth/marker/pose 头；`encoder/train.py` 可见路径权重和 Adam 训练代码，但这只是公开快照，不是对论文所有实验的独立复现。
- 仓库 README 声称 MIT，仓库 `LICENSE` 实际是 Apache-2.0；使用时应以许可证文件为准并核对上游更新。

## 10. 论文引用

```bibtex
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```
