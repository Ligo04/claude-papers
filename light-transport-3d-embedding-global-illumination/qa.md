# 自测题

## 基础题

### 1. 论文要解决什么问题？

<details>
<summary>答案</summary>

从原始 3D 场景配置直接预测可泛化的 global illumination，减少 per-scene retraining、screen-space view dependence 和 3D global attention 的尺度瓶颈。

</details>

---

### 2. Scene point 包含哪些属性？

<details>
<summary>答案</summary>

位置 `p`、法线 `n`、albedo `c` 和 emissivity `e`。这些属性共同描述 geometry、材质和光源。

</details>

---

### 3. 论文默认每个场景使用多少 scene points？

<details>
<summary>答案</summary>

约 20k 个 scene points。论文还分析了 5k、10k、20k、40k 的点数影响。

</details>

---

### 4. 主实验的 Ours MSE 和 SSIM 是多少？

<details>
<summary>答案</summary>

在 105 个测试场景上，MSE 为 0.048，SSIM 为 0.912。

</details>

---

### 5. 本文的两个主要模块叫什么？

<details>
<summary>答案</summary>

Light Transport Encoder 负责把全局多次反弹关系编码进场景点 embedding；Local Query Decoder 负责用固定 KNN 邻域回答任意 query point。

</details>

---

## 中级题

### 6. 为什么不直接使用原始 mesh vertices？

<details>
<summary>答案</summary>

mesh vertex 会把表示密度绑定到原始拓扑和 tessellation。采样 point cloud 可以独立调整点密度、支持扫描数据，并统一普通几何与发光几何。

</details>

---

### 7. NNE 解决了 PTV3 的什么问题？

<details>
<summary>答案</summary>

PTV3 的固定 patch 虽然线性，但单个 patch 的 1024 个 raw points 只覆盖小范围空间。NNE 先用 FPS 选中心，再聚合邻域，扩大每个 token 的空间 halo，使远距离光照影响更容易传播。

</details>

---

### 8. Local Query Decoder 为什么让分辨率更自由？

<details>
<summary>答案</summary>

每个 ray intersection 被视为独立 query，只读取固定 `K=32` 个 scene embeddings。query 之间没有 `N²` attention，因此增加图像分辨率只增加 query 数量，不改变单个 query 的场景依赖。

</details>

---

### 9. vector cross-attention 和普通 dot-product attention 有什么差别？

<details>
<summary>答案</summary>

普通 dot-product 把 query-key 关系压成一个标量权重；本文用逐通道 subtraction，并加入相对位置编码，让每个 feature channel 保留自己的 relation signal。

</details>

---

### 10. 为什么训练 target 要采样约 2M 个 3D query points？

<details>
<summary>答案</summary>

为了让模型直接学习 world-space light transport，而不是只拟合某个相机视角和固定分辨率的图像。这样测试时可以换相机和 ray resolution，但数据生成成本很高。

</details>

---

## 高级题

### 11. 本文的“线性复杂度”具体指什么？

<details>
<summary>答案</summary>

结构表将 encoder 写成 `O(CM)`，`C=256` 是固定 patch 相关常数；decoder 写成 `O(KN)`，`K=32` 固定，分别避免 RenderFormer 的 `O(M²)` encoder 和 `O(MN+N²)` decoder。线性是关于点数的结构复杂度，不等于整套系统已经实时。

</details>

---

### 12. 为什么 Ours 的 decoder 仍可能很慢？

<details>
<summary>答案</summary>

复杂度虽然是 `O(KN)`，但 `N` 是图像中所有 ray queries。论文报告 512×512 解码约 368 ms，其中 KNN 约 25–36 ms，剩余时间主要消耗在 Python attention；kernel fusion 和蒸馏仍是工程瓶颈。

</details>

---

### 13. 为什么论文不把 Path Tracing 64 spp 当作普通 baseline？

<details>
<summary>答案</summary>

Path tracing 是物理上更准确的参考，但 64 spp 的误差主要是高频采样噪声，SSIM 只有 0.425。论文把它作为实时预算质量参考，而不是与神经模型公平竞争的学习 baseline。

</details>

---

### 14. 3D supervision 和 RenderFormer 的 2D supervision 如何取舍？

<details>
<summary>答案</summary>

3D supervision 带来 view consistency 和 resolution independence，并能利用屏幕外几何；2D supervision 更容易拟合相机看到的高频 specular 和 caustics。两者优化目标不同，不能只按单个 SSIM 数字判断优劣。

</details>

---

### 15. 论文当前最明显的能力边界是什么？

<details>
<summary>答案</summary>

模型以 diffuse GI 为主，glossy 结果只是有限 fine-tuning 的迁移示范；相邻房间光源、低频纹理、遮挡点会产生错误，动态序列可能 flicker，而且当前 208 ms encoder + 368 ms 512² decoder 尚未达到实时渲染。

</details>
