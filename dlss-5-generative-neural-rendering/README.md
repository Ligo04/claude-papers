# DLSS 5：实时生成式渲染

这份资料围绕 NVIDIA 的技术报告《DLSS 5: Real-Time Generative Rendering》建立。论文把 DLSS 从“重建渲染器本来会产生的更高成本结果”推进到“渲染器之后的生成式外观增强”：引擎仍负责几何、材质、相机、动画和基础光照，DLSS 5 读取当前 RGB 帧，在真实照片学习到的外观先验中选择一个由艺术家控制的、与原场景对齐的显示外观。报告强调四个同时成立的要求——保留艺术意图、因果流式处理、严格帧预算、可编排的艺术指导——并用一步像素空间扩散、训练时场景属性一致性监督、跨帧状态和面向 Blackwell 的 Transformer 实现它们。

## 论文档案

- 难度：Advanced（需要图形学、扩散模型、视频时序和 GPU 推理的交叉背景）
- 论文性质：系统设计 + 实证评测
- 方法复杂度：新型受渲染器约束的生成模型；训练和部署细节有意保持在产品级抽象
- PDF：24 页，创建时间 2026-09-01，SHA-256 见 [`meta.json`](meta.json)
- 官方项目页：[NVIDIA ADLR / DLSS 5](https://research.nvidia.com/labs/adlr/DLSS5/)
- 预计学习时间：90–120 分钟；若要读完评测指标与批判性问题，预留 2 小时

## 推荐阅读顺序

1. [`summary.md`](summary.md)：先掌握问题、系统、数据和数字结果。
2. [`mental-model.md`](mental-model.md)：把它放入“渲染器 + 外观先验 + 时序控制”的研究地图。
3. [`method.md`](method.md)：沿输入、训练监督、一步推理、流式状态和艺术控制逐层拆解。
4. [`insights.md`](insights.md)：理解论文真正改变的抽象，以及哪些结论不能从实验推出。
5. [`reflection.md`](reflection.md)：用开放问题和失败模式检验是否真的理解。
6. [`qa.md`](qa.md)：15 道分级自测题；每题的答案都折叠在题目下。
7. [`code/dlss5_concept_demo.py`](code/dlss5_concept_demo.py)：运行一个不依赖模型权重的概念性 CPU demo。
8. [`index.html`](index.html)：在浏览器里调节结构、色调、色度和掩码，查看概念输出与论文真实指标。

## 最重要的四个结论

- **目标变了**：DLSS 5 的目标不是估计物理渲染参考，而是在保留渲染器定义内容的邻域内，朝照片外观分布移动。
- **约束来自两处**：运行时的 RGB 帧提供密集、像素对齐的内容锚点；训练时的反照率、法线和光照监督抑制结构漂移。这些额外缓冲区不是发布模型的输入。
- **实时性是模型定义的一部分**：一步去噪、像素空间、因果逐帧状态和 FP8/TMA 协同，使 4K 推理落入单帧预算；分块视频模型的等待和 O(K) 工作集不适合交互式显示。
- **可控性不是装饰**：外观模型选择、结构/色调强度、色度、手工区域掩码和自动皮肤掩码共同决定“哪里改、改多少、往哪个外观改”。

## 资料夹结构

```text
paper.pdf                         原始 PDF
paper-layout.txt                  pdftotext -layout 全文（核验用）
pdfinfo.txt                       PDF 元数据
meta.json                         论文元数据与恰好两个语义标签
summary.md                        背景、方法、贡献、数字结果
method.md                         组件、流程、伪代码、复现风险
mental-model.md                   研究地图与心智模型
insights.md                       核心洞见、比较与边界
reflection.md                     延伸问题与实验提案
qa.md                             5 基础 + 5 中级 + 5 高级问题
sources.md                        来源、抽取和证据边界
code/dlss5_concept_demo.py        可运行的确定性 CPU 概念 demo
index.html                        零外部依赖的交互式探索器
images/                           原始图像抽取与命名后的关键图
```

## 证据边界

这是 NVIDIA 发布的技术报告，不是带有公开训练代码和权重的可复现实验论文。报告披露了输入、训练监督的类别、模型规模、精度和评测设置，但没有给出完整数据集、损失权重、层数/宽度、训练步数、权重或 SDK 接口。因此本资料把作者报告的数字和樵樵写的概念 demo 明确分开：demo 用来理解约束与控制，不实现也不声称复刻 DLSS 5。

## 关键图像

PDF 中的多面板图已按主题保留在 [`images/`](images/)；优先查看：

- [`figure-1-rendered-vs-dlss5.png`](images/figure-1-rendered-vs-dlss5.png)：输入/输出示例
- [`figure-2-streaming-latency-memory.png`](images/figure-2-streaming-latency-memory.png)：分块与逐帧流式比较
- [`figure-6-tone-structure-control.jpeg`](images/figure-6-tone-structure-control.jpeg)：结构/色调控制
- [`figure-9-qualitative-comparison.png`](images/figure-9-qualitative-comparison.png)：与离线基础模型的定性比较
- [`figure-11-low-quality-input.png`](images/figure-11-low-quality-input.png)：输入质量限制

学习时建议同时打开 [`sources.md`](sources.md)，这样不会把“感知真实感”“物理正确性”“内容对齐”混成一个指标喵。
