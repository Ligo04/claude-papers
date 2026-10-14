# 心智模型：全局求解一次，局部查询很多次

## 它解决哪类问题？

这是一个 neural global illumination / neural rendering 架构。输入是离散化的 3D 场景资产，输出是任意世界空间 query 的光照量。它同时处理三个尺度：

- **场景尺度**：房间、走廊、远处光源和遮挡关系；
- **局部尺度**：当前 query 与附近表面点的相对几何；
- **渲染尺度**：任意相机、任意分辨率带来的 query 数。

## 需要哪些先验知识？

建议先理解：

- global illumination、irradiance、radiance、BRDF 与 path tracing；
- point cloud、KNN、FPS、Point Transformer；
- self-attention 的复杂度和 GPU 显存；
- screen-space 与 world-space supervision 的区别。

## 一张图

```text
geometry + normals + albedo + emissivity
                 ↓
         sampled scene points
                 ↓
      NNE + Point Transformer V3
        global light transport embedding
                 ↓  KNN(K=32)
query point + relative geometry
                 ↓
        vector cross-attention
                 ↓
   irradiance / directional radiance
```

## 如何分类

可以把它记成：

```text
3D scene representation
+ global linear encoder
+ independent local query decoder
+ world-space supervision
= scalable generalizable neural GI
```

## 阅读时的三个问题

1. 这个模块是在处理全局反弹关系，还是只在读取某个 query 的局部邻域？
2. 这个结果使用的是 3D query supervision，还是 2D image supervision？
3. 报告的成本是一次 scene embedding 更新，还是每张图像的 query decoding？

回答这三个问题，就不会把 208 ms encoder 和 368 ms 512² decoder 混成一个固定 FPS，也不会把 view-independent irradiance 结果误解成高频 glossy renderer。
