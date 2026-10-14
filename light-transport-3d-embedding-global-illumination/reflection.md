# 反思与可延伸方向

## 最值得继续验证的方向

1. **动态场景的增量 embedding**：只重编码移动物体、光源变化或受影响区域，避免每次场景更新都跑完整 PTV3。
2. **更强的方向表示**：用 spherical Gaussian、anisotropic spherical Gaussian 或低秩 basis 取代 32×32 离散方向直方图，降低 glossy target 的存储和推理成本。
3. **遮挡感知的局部解码**：把 visibility、ray direction 或 occlusion feature 加入 KNN attention，减少 under-carpet / occluded point 的 latent 泄漏。
4. **时序约束**：为动态序列加入 temporal consistency，抑制逐帧独立 query 产生的 flickering。
5. **场景域扩展**：加入 outdoor、participating media、透明材质和大尺度光源，检验 global scale normalization 是否足够。
6. **CUDA 融合与蒸馏**：把 vector cross-attention 从 Python 移到 fused kernel，评估能否把 368 ms 解码降到交互式范围。

## 脆弱假设

- scene point sampling 已经覆盖影响当前 query 的几何和纹理频率；
- 固定 `K=32` 的邻域包含足够的局部光传输线索；
- PTV3 patch + NNE 的 receptive field 能传递远距离光源影响；
- 数据集级尺度归一化适用于不同室内空间；
- diffuse embedding 可以用有限 fine-tuning 迁移到 glossy directional radiance。

## 可能失败的场景

- 光源位于相邻房间、强遮挡或极端间接光路径；
- 训练集中没有出现的纹理色调和材质组合；
- 被几何遮挡但仍参与路径贡献的点；
- 高频 specular、caustics、玻璃和参与介质；
- 动态物体或光源变化导致旧 embedding 失效；
- 点数、embedding 维度和 KNN 规模同时增大，显存超过设备上限。

## 一个可检验的研究假设

如果把 KNN 邻域从固定欧氏距离改成“距离 + 可见性 + 预估光传输贡献”的稀疏选择，是否能在相同 `K=32` 下减少相邻房间光源和遮挡点造成的错误？实验需要同时报告 MSE、SSIM、FLIP/LPIPS、KNN 时间、embedding 更新时间和失败场景分组。
