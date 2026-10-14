# Heat Kernel Textures

**论文**：*Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat*  
**作者**：Simone Foti、Caner Korkmaz、Stefanos Zafeiriou、Tolga Birdal（Imperial College London）  
**会议/年份**：ECCV 2026  
**难度**：高级；离散微分几何、可微渲染与显式纹理表示交叉  
**类型**：架构/表示方法 + 实验密集型

## 这篇论文解决什么问题

传统 UV 纹理把曲面摊到二维图像上，因此会产生接缝、畸变、空白区域、接缝顶点复制和不均匀分辨率。HKTex（Heat Kernel Textures）直接在三角网格的流形表面上放置可优化的各向异性热核：核的位置沿测地线移动，核的形状由各向异性热扩散定义，渲染时在射线与网格的连续交点处查询局部核并混合颜色。它保留了显式局部基元的表达能力，同时移除了 UV 参数化依赖。

## 如何阅读

1. 先看 [summary.md](summary.md) 建立问题、贡献和实验结论。
2. 再看 [mental-model.md](mental-model.md) 与 [insights.md](insights.md)，理解“为什么热核是曲面上的高斯”。
3. 需要实现时阅读 [method.md](method.md) 和 `code/heat_kernel_demo.py`。
4. 用 [index.html](index.html) 调整核数、各向异性和阈值，观察核心机制。
5. 用 [qa.md](qa.md) 检查是否掌握了数学、系统和实验层面的细节。

## 关键结论

- 热核是流形上的扩散核；在欧氏平面上它退化为高斯，但在一般网格上沿测地距离传播。
- 各向异性拉普拉斯–贝尔特拉米算子（ALBO）控制核的方向和拉伸；论文用 7×7 参数网格预计算谱基，并通过对齐、双线性插值支持连续查询。
- 双调和距离 KNN 将每个交点的候选核限制为 `K_s` 个，使评价复杂度从 `O(P×S)` 降到 `O(P×K_s)`。
- Riemannian-SGD 将位置梯度投影到切平面，再用指数映射沿表面更新；克隆/拆分也用指数映射保持在流形上。
- 在 313 个 Objaverse 网格的 UV 拟合中，HKTex（约 4.8k 核）达到 44.8 dB PSNR、0.08×10^-3 MSE、99.5×10^-2 MS-SSIM 和 96.6 KB 存储。

## 预计学习时间

首次通读约 60–90 分钟。演示只需 Python 标准库；完整复现需要 GPU 环境、网格数据、谱预计算和训练，耗时取决于资产与硬件，未在此次学习中执行。

## 来源与验证边界

以 [用户提供的 PDF](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/2392.pdf) 为主要依据：方法见第 5–10 页，设置见第 10–11 页，表 1/2 见第 14/15 页。[项目页](https://circle-group.github.io/research/HeatKernelTextures/)与[官方代码](https://github.com/circle-group/hktex)已确认可访问。笔记中的研究建议和公式检查项属于阅读分析，不是论文已验证结论。此目录的代码是可运行教学代理，未复现论文基准。

## 文件结构

```text
paper.pdf                 原始 PDF
meta.json                 元数据与两个语义标签
summary.md                背景、问题、贡献、结果
insights.md               核心直觉、取舍、限制和研究启示
method.md                 方法、公式、伪代码、实现风险
mental-model.md           研究地图与心智模型
reflection.md             可扩展方向和开放问题
qa.md                     5 基础 + 5 中级 + 5 高级问题
index.html                无外部依赖的交互式概念探索器
code/heat_kernel_demo.py  纯 Python 可运行的概念演示
images/                   从论文 PDF 提取的图片
```
