# 方法详解：Light Transport Encoder + Local Query Decoder

## 1. 输入、输出和符号

场景被转成 `M≈20k` 个 scene points：

```text
X_i = (p_i, n_i, c_i, e_i)
```

其中 `p` 是世界空间位置，`n` 是法线，`c` 是 albedo，`e` 是 emissivity。渲染时的 query point 为：

```text
Q_j = (p_j, n_j, c_j)
```

主实验在场景表面均匀采样约 `N≈2M` 个 query points 作为 3D supervision；推理时 query points 来自当前相机 ray intersections。

## 2. Scene discretization

论文不用原始 mesh vertex 作为唯一 primitive，而是从几何表面采样点。这个转换带来三件事：

- topology-independent：点不依赖 mesh connectivity；
- density-controllable：纹理和几何复杂区域可以增加点；
- light-unified：发光几何与普通几何都进入同一组 point attributes。

训练场景不被缩放到各自 unit cube。作者先减去场景 bounding-box minimum，再除以数据集级固定尺度 `s`，把各场景放入 `[0,1]^3`，同时保持绝对尺度关系对 inverse-square falloff 和 occlusion 的影响。

## 3. Nearest Neighbor Embedding（NNE）

直接对所有 `M` 点做 transformer 会让点数和 attention 负担一起增长。NNE 的伪代码是：

```text
X_i = F(p_i, n_i, c_i, e_i)
centers = FPS({X_i})                  # 两次 halving，m = M/4
for center in centers:
    neighbors = KNN(center, raw_points)
    local = concat(neighbor - center, center)
    center_feature = max_pool(G(local))
F_tilde = center_feature
```

论文实现中 FPS 做两次，每次把点数减半，最终约 `m=M/4` 个中心。每个中心不只保留自身特征，还聚合原始点邻域的相对差分，因此扩大 PTV3 patch 的空间覆盖。

## 4. Light Transport Encoder

NNE 输出送入 Point Transformer V3：

```text
{F_l}^m = PTV3({F_tilde_l}^m)
```

PTV3 将 3D 点按 space-filling serialization 排成序列，切成 1024 点 patch，只在 patch 内做 attention，并在不同层改变 serialization order。它以线性复杂度处理大点集；NNE 则补足 patch attention 的空间覆盖。

默认配置：

| 参数 | 值 |
| --- | ---: |
| hidden dimension `D` | 128 |
| patch size | 1024 |
| encoder depths | `[2,2,2,6,2]` |
| channels | `[128,128,128,256,256]` |
| heads | 4 (`D/32`) |
| dropout | 0.1 |
| 参数量 | 42M |

## 5. Local Query Decoder

对每个 query `Q_j`：

1. 用 MLP `H` 将 `(p_j,n_j,c_j)` 投到 query latent `G̃_j`；
2. 在 scene embeddings `{F_l}` 中取 `K=32` 个 KNN；
3. 计算 query 到邻居的相对位置 `Δp_{jκ}=p_j-p_κ`，再经过 `γ` 得到 `P_{jκ}`；
4. 用 vector cross-attention 聚合：

```text
Gq_j = Wq(G̃_j)
Gk_jκ = Wk(KV_jκ)
Gv_jκ = Wv(KV_jκ)
A_jκ = Gq_j - Gk_jκ + P_jκ
G_j = Σκ A_jκ · (Gv_jκ + P_jκ)
I_out = W_out({G_j})
```

`I_out` 可以是 diffuse irradiance，也可以由任务专用 decoder 变成 directional radiance。query 之间彼此独立，所以单个像素的成本只依赖固定 `K`，不依赖全场景点数。

## 6. 训练监督与损失

每个场景约 2M 个 query point 通过 path tracing 得到 target。主任务使用 irradiance；glossy transfer 使用每点 1024 个半球方向的 incident radiance histogram。损失是 log-space relative L2：

```text
L = 1/N Σ_j [ ( log(I_hat_j + 1) - log(I_gt_j + 1) )²
              / ( log(I_hat_j + 1) + ε ) ]
```

这样可以压缩高动态范围，同时避免低能量区域完全被高亮区域主导。

## 7. 推理路径和复杂度

```text
scene assets
  → point sampling + attributes
  → NNE + PTV3                     # one scene update
  → latent code anchored at points
  → ray intersection query
  → KNN retrieval + vector cross-attention
  → irradiance / radiance
```

论文结构表对比：

| 部分 | RenderFormer | 本文 |
| --- | --- | --- |
| Encoder | `O(M²)` global attention | `O(CM)`，`C=256` |
| Decoder | `O(MN+N²)` | `O(KN)`，`K=32` |
| 输入 | mesh vertices | sampled point cloud |
| supervision | 2D images | sampled 3D query points |
| 视角/分辨率 | view-dependent / fixed | view-independent / flexible |
| 参数量 | 205M / 483M | 42M |

## 8. 实现注意事项

- **尺度**：不能每个场景单独归一化到 unit cube，否则绝对距离与光照衰减关系被破坏。
- **点采样**：20k 是论文默认平衡，不是普适上限；大型户外场景可能需要更多点或分块。
- **KNN**：论文用 cudaKDTree 加速，GPU KNN 仅约 25–36 ms，但 Python attention 仍占主要解码时间。
- **缓存**：相机移动不改变场景 embedding，应复用 encoder；动态几何或光源变化需要重新编码或增量更新。
- **监督**：直接 3D target 与 2D image target 的能力边界不同，不应把本方法的 MSE 与 screen-space 论文无条件横比。
- **高光**：diffuse 预训练后迁移到 GGX directional radiance，需要新的 decoder、roughness 输入和方向采样。

## 9. 复现风险

1. 数据生成的 PBRT、纹理烘焙和 area-light heuristics 影响分布，难以只靠正文复刻。
2. 2M query × 1024 rays/point 的 ground truth 生成成本很高，论文报告约 250 CPU/GPU days。
3. PTV3 serialization、NNE 邻域半径和 KNN 实现会改变有效 receptive field。
4. 使用 24 GB GPU 时 `D=128` 已接近 21.3 GB；更大维度需要更大显存或分片。
5. 论文没有给出当前 PDF 中的官方代码仓库地址；复现应等待作者承诺的 code/data/weights 发布。
6. 论文的结果是 neural approximation，不是 path tracing 的物理保证；失败案例包含纹理色调偏移、相邻房间光源和被遮挡点泄漏。
