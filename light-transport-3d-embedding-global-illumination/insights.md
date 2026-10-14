# 核心洞见：先把全局光照烘焙进 3D embedding，再让每条射线局部读取

## 一句话理解

这篇论文的核心不是“把 Transformer 变小”，而是重新拆开渲染问题：

```text
复杂 3D 场景 + 几何/材质/光源
          ↓  一次性全局编码
scene-point light transport embeddings
          ↓  每个 query 只读固定 KNN 邻域
irradiance / directional radiance
```

Encoder 承担跨房间、遮挡、多次反弹的全局建模；Decoder 承担 query-specific 的局部读取。重计算被摊到 scene update，渲染分辨率只影响 query 数量。

## 为什么 point cloud 是关键中间接口

mesh vertex 看似自然，但它把 embedding 密度绑死在原始拓扑上；要表达高频纹理，就要继续细分 mesh，场景 token 数和材质表达同时膨胀。论文改为从表面采样 point cloud：

- 可以按重要性调整采样密度；
- 不需要 mesh connectivity，扫描数据也能进入同一接口；
- 几何点和发光点共享表示，区域光、点光源和 emissive strip 不需要换模型；
- query point 可以来自任意 shading ray intersection，因此训练和推理的分辨率没有固定绑定。

代价是点采样质量决定覆盖率，空间尺度、法线和材质属性必须保持一致，邻域检索也必须足够快。

## 为什么需要 NNE，而不是直接把 20k 点扔进 PTV3

PTV3 用 serialization-and-patch attention 获得线性复杂度，但固定 1024 点 patch 只覆盖局部空间；一个点的光照却可能由远处窗户、灯和多次反弹共同决定。NNE 先用 FPS 选中心，再对每个中心聚合原始点邻域，把更大的 3D halo 压进 token。这样一个 patch 里的 1024 个 token 代表更宽的空间范围，远程依赖更容易跨层传播。

NNE 解决的是 **patch receptive field**，PTV3 解决的是 **attention scaling**；两者承担不同职责。

## 为什么 decoder 必须独立处理 query

如果 decoder 让所有场景点和所有 query 彼此 global attention，代价会变成 `O(MN+N²)`，场景复杂度和渲染分辨率会争抢显存。Local Query Decoder 只取 `K=32` 个邻居：

```text
query (p, n, albedo)
  + relative geometry to KNN scene embeddings
  → vector cross-attention
  → MLP
  → irradiance / directional radiance
```

不同 query 之间没有 attention 依赖，所以可以自由改变图像分辨率、批处理大小和 ray 数量。这正是 resolution-agnostic 的来源。

## vector cross-attention 的含义

标准 dot-product attention 将每个 query-key 对压成一个标量权重；论文改用逐通道的 subtraction relation：

```text
A = Gq(query) - Gk(neighbor) + P(relative position)
G = Σκ A · (Gv(neighbor) + P)
```

这种做法保留每个 feature channel 的差异，让不同通道可以分别表达距离、遮挡线索和材质影响。它不是物理可解释的光传输求解器，而是给神经聚合器更细粒度的关系输入。

## 3D supervision 带来的交换

训练 target 是世界空间 query point 的 irradiance/radiance，而不是某台相机看到的 2D image。得到的好处是：

- 换相机轨迹时，同一个 world point 仍有同一个监督坐标；
- 输出分辨率改变时，只需重新生成 ray intersections 并查询 embedding；
- 屏幕外几何可以通过 embedding 影响当前点。

交换是高频 view-dependent effects 的表达能力下降。RenderFormer 的 2D loss 更容易拟合 caustics 和 specular highlights，但它的 training resolution、view frustum 和全局 attention 更难扩展。两者优化的是不同目标。

## 论文真正展示的“泛化”

- **跨场景**：约 14k 室内场景中留出 test scenes，不做 per-scene retraining。
- **跨视角/分辨率**：query 在 world space 定义，decoder 不依赖固定 image grid。
- **跨任务**：encoder 从 diffuse irradiance 迁移到 glossy directional radiance。
- **跨光源类型**：点云接口统一 geometry 和 emissive lights。

这不等于完全解决了 domain shift。论文自己展示了 underrepresented texture tones、相邻房间光源、遮挡点的失败案例。

## 与 prior work 的关系

| 方法 | 主要接口 | 强项 | 结构性限制 |
| --- | --- | --- | --- |
| Per-scene neural GI | 单个场景的 latent/mesh | 可针对场景拟合 | 泛化差，需要重训 |
| Screen-space deep shading | G-buffer / 2D image | 推理路径短 | 视角依赖，看不到屏幕外几何 |
| Hermosilla et al. | 3D point convolution | 早期 3D light transport | 长距离关系弱，复杂场景泛化差 |
| RenderFormer | mesh vertices + global attention | 小 mesh 的高频效果 | `O(M²)`/`O(MN+N²)`，难扩展 |
| 本文 | point cloud + linear encoder + local query | 大场景、视角/分辨率独立 | 当前 diffuse 优先、推理仍非实时 |

## 工程含义

如果把这个思路迁移到实时渲染系统，先固定四件事：

1. 统一 world-space 坐标和场景尺度；
2. 设计可增量更新的 scene point / embedding 管理；
3. 用 GPU KNN 和 fused vector attention 消除 Python 解码开销；
4. 把 encoder scene-update latency 与 per-frame query latency 分开预算。

否则“线性复杂度”只会停留在理论级，实际瓶颈仍可能落在点采样、KNN、Python kernel launch、材质高频和动态更新上。
