# 论文摘要

## 1. 背景问题

Global illumination 需要模拟光在几何、材质和光源之间的多次反弹。Monte Carlo path tracing 质量高但代价昂贵，现有学习方法各有瓶颈：

- per-scene neural GI 需要针对每个场景训练，跨场景泛化弱；
- screen-space denoising/deep shading 依赖视角和分辨率，观察不到屏幕外的几何；
- 直接在 mesh vertex 上做 global transformer，attention 对场景 primitive 数量是二次复杂度，难以处理百万三角形的真实室内场景。

论文要回答的是：能否从原始 3D scene configuration 直接学习一个可泛化的 light transport representation，让任意世界空间 query 都能得到 GI 量？

## 2. 核心贡献

1. **Point-based 3D intermediate representation**：每个 scene point 记录位置 `p`、法线 `n`、albedo `c`、emissivity `e`；点密度与原始 mesh topology 解耦，也能统一 area light、point light 和 emissive geometry。
2. **Light Transport Encoder**：用 Nearest Neighbor Embedding（NNE）先做 FPS 与邻域聚合，再用 Point Transformer V3 建模长距离关系。论文将其复杂度写成 `O(CM)`，其中 `M` 是 scene points、`C=256` 是固定 patch 相关常数。
3. **Local Query Decoder**：每个 query 只检索 `K=32` 个 embedding 邻居，用相对位置编码和 vector cross-attention 聚合，解码复杂度为 `O(KN)`，与全局场景规模解耦。
4. **World-space supervision**：每个场景均匀采样约 2M 个 query points，用 path tracing 生成 irradiance/radiance target；训练不依赖某一台相机或固定分辨率。
5. **可迁移 encoder**：在 diffuse irradiance 上预训练后，只替换 decoder 并小规模 fine-tune，就能预测 glossy material 的 spatial-directional radiance field。

## 3. 模型与数据

### 场景表示

- 数据集：13,900 个 procedural indoor scenes，覆盖不同 floorplans、家具、家电、装饰、纹理和区域光源。
- 每个场景约 20k scene points；每个点为 `(p, n, c, e)`。
- 训练 query：每场景约 2M 个 world-space surface points；每个点用 1,024 条 stratified cosine-weighted rays、最大 path depth 5 做 ground truth。
- 数据生成：约 250 CPU/GPU days；场景文件约 500 MB–2 GB。

### 训练配置

- backbone：Point Transformer V3。
- hidden dimension `D=128`；patch size 1024；NNE/decoder neighbor count 32；decoder 6 层；FFN 512；dropout 0.1。
- 主模型约 42M 参数；训练 AdamW，learning rate `2×10⁻⁴`，weight decay `1×10⁻⁴`，batch size 3，2M steps。
- 论文正文写四张 A10 训练约 7 天；每 GPU batch size 为 3。
- 损失：对预测与 ground-truth irradiance/radiance 使用 log-space relative L2。

## 4. 主要结果

### 4.1 跨场景 diffuse GI

Table 1 在 105 个测试场景上报告：

| 方法 | MSE ↓ | SSIM ↑ | 解释 |
| --- | ---: | ---: | --- |
| Deep Shading（overfitting） | 0.071 | 0.882 | 将测试视图放回训练，属于过拟合参考 |
| Improved Hermosilla et al. | 0.171 | 0.775 | 改进的 point-convolution baseline |
| Path Tracing（64 spp） | 0.051 | 0.425 | 实时预算参考，不是学习模型竞争者 |
| Ours | **0.048** | **0.912** | 跨场景、3D query supervision |

视觉结果展示了多次反弹的 color bleeding、soft shadows 和遮挡区域的间接光；仍可见 toilet base / bowl 下方 light leak 与 living room 色偏。

### 4.2 Glossy radiance field transfer

作者从 diffuse 模型初始化 encoder，加入 roughness 和 Directional Radiance Decoder，在 GGX glossy subset 上 fine-tune。每个 query 预测 1,024 个半球方向（`32×32` bins），每个方向用 64 条样本生成 path-traced target。这个实验说明 embedding 可以换 decoder 承担不同 rendering quantity，但仍是 preliminary result，不能等同于完整 production-quality specular renderer。

### 4.3 可扩展性与消融

- Vanilla global transformer：20k scene points、batch 3 时训练显存超过 80 GB。
- Light Transport Encoder：约 21 GB；完整 Local Query Decoder：约 22 GB。
- 512×512 图像的平均耗时：encoder 208 ms/scene update，local decoder 368 ms；其中 KNN GPU 约 25–36 ms，其余主要是 Python attention。
- `D=64/128/256` 时参数量约 10.5M/42M/167.7M，训练显存约 10.7/21.3/43.1 GB。
- `K=8/32/48` 的 PSNR 为 26.96/34.57/34.71；论文选 `K=32` 作为质量和成本平衡。
- scene points 从 5k/10k/20k/40k 增加时 PSNR 为 28.51/30.94/34.57/36.25，20k 是论文默认点数。

## 5. 论文结论的边界

论文证明的是：点云中间表示、线性复杂度 encoder、固定局部解码和 3D supervision 的组合能把 GI 学习扩展到复杂室内场景，并提升跨场景的视角与分辨率独立性。论文没有证明：

- 当前实现已经达到实时渲染；
- 高频 specular、caustics、participating media 或 outdoor scenes 都能同样工作；
- 动态场景可以只局部更新而无需重新编码；
- 纯数据先验能够替代 path tracing 的生产级质量保证。
