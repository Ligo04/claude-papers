# A Generalizable Light Transport 3D Embedding for Global Illumination

> SIGGRAPH Conference Papers ’26 · DOI: 10.1145/3799902.3811095 · 18 pages

这篇论文学习一个问题：能不能只从几何、材质和光源构成的 3D 场景，直接预测跨场景、跨视角、跨分辨率的 global illumination（GI）？作者把场景采样成带位置、法线、albedo 和 emissivity 的 point cloud，用 Point Transformer V3 构成 Light Transport Encoder，把长距离、多次反弹的光照关系“烘焙”进锚定在场景点上的 latent embedding，再用固定 KNN 邻域的 Local Query Decoder 回答任意 shading query。主模型 42M 参数，在 105 个测试场景上报告 MSE 0.048、SSIM 0.912，并将同一个 encoder 迁移到 glossy spatial-directional radiance field。

## 难度与阅读路线

- 难度：Advanced。
- 性质：architecture-based empirical system。
- 建议顺序：先读 `summary.md` 了解问题和数字；再读 `method.md` 跟着 NNE、PTV3、vector cross-attention 走一遍；用 `insights.md` 理解“全局编码、局部读取”的设计；最后看 `reflection.md` 和 `qa.md`。
- 预计时间：通读 60–90 分钟；连同公式、附录和 demo 约 2 小时。

## 五个关键结论

1. 点云中间表示同时解耦原始 mesh topology、纹理密度和光源类型，支持约 20k scene points 的复杂室内场景。
2. Light Transport Encoder 用 NNE + PTV3 把全局建模成本降到线性量级；Local Query Decoder 只访问固定 32 个邻居，使 query 数量和场景规模分开。
3. 直接在 world-space 采样约 2M query points 做 3D supervision，带来 view consistency 与 resolution independence，但会牺牲一部分高频、强视角相关的 specular 表达。
4. 论文报告 Ours 的 MSE/SSIM 为 0.048/0.912；相同设计路线中 Vanilla Transformer 为 0.068/0.885，加入 Light Transport Encoder 后为 0.084/0.870，再加入 Local Query Decoder 后为 0.048/0.908。
5. 论文明确承认当前实现还不是实时渲染器：encoder 约 208 ms/scene update，512×512 local decoding 约 368 ms；主要贡献是跨场景泛化和内存可扩展性。

## 文件结构

- `README.md`：导航、难度和关键结论。
- `summary.md`：背景、方法、数据集、结果和限制。
- `insights.md`：核心洞见、设计取舍和与 prior work 的关系。
- `method.md`：公式、模块、伪代码、超参数和复现风险。
- `mental-model.md`：把论文放进 neural GI / 3D learning 地图。
- `reflection.md`：开放问题、脆弱假设和可延伸方向。
- `qa.md`：5 个基础、5 个中级、5 个高级自测题。
- `code/light_transport_embedding_demo.py`：标准库教学 demo；不依赖官方 checkpoint。
- `index.html`：单文件交互式学习页。
- `paper.pdf`、`paper.txt`、`meta.json`：原始 PDF、文本抽取和元数据。
- `images/`：PDF 图像、关键图重命名副本和架构页截图。
- `sources.md`、`source-status.md`：来源和验证边界。
