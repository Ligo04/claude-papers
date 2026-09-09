# 来源、抽取与证据边界

## 一手来源

1. **PDF 报告**：`DLSS 5: Real-Time Generative Rendering`，NVIDIA Corporation，Technical Report，2026，24 页。
   - 本地副本：[`paper.pdf`](paper.pdf)
   - 下载地址：<https://research.nvidia.com/labs/adlr/files/DLSS5_Report.pdf>
   - SHA-256：`e5441f35ab11f7e43d2166e0538d79c7dd83d90cec37b8e299c4cacc9b0e6498`
   - PDF 创建时间：2026-09-01 19:48:27 CST
2. **NVIDIA ADLR 项目页**：<https://research.nvidia.com/labs/adlr/DLSS5/>
   - 页面标题为 “DLSS 5: Generative Neural Rendering”，发布日期为 2026-09-01。
   - 页面概述与 PDF 一致地说明：3D-guided neural rendering、一步像素空间扩散、RGB/运动向量/时序状态/艺术指导条件、最高 4K 实时渲染和 GeForce RTX 50 系列本地运行。

## 本地抽取产物

- [`paper-layout.txt`](paper-layout.txt)：使用 `pdftotext -layout` 提取的完整可搜索文本，约 72 KB。
- [`pdfinfo.txt`](pdfinfo.txt)：Poppler 对页数、尺寸、创建者和文件大小的检查。
- [`parse-output.json`](parse-output.json)：技能自带 `parse-pdf.js` 的原始输出。
- [`extract-images.log`](extract-images.log)：PyMuPDF 图像提取日志。
- [`images/`](images/)：原始图像资源及按主题生成的页面预览。

## 解析可靠性说明

`parse-pdf.js` 能读出标题、正文和页数，但其 stdout 前面混入了字体/TrueType 警告，导致原始输出不是严格 JSON；它还把作者误识别为 “Model Optimizations”，并因 PDF 的换行格式没有抽到 Abstract。为避免把错误字段写进知识库，本资料以 `pdftotext -layout`、`pdfinfo` 和 PDF 内的引用页为事实依据，手工建立 [`meta.json`](meta.json)。这是一个可重复的核验边界，不是对报告内容的独立复现。

## 结果如何分级

- **报告直接披露**：109 场景、794 有效判断、表 1–3 数字、154M 参数、FP8/FP16、约 8 ms、731 MB、四类要求和局限性。
- **官方项目页重复确认**：产品定位、3D 引导神经渲染、一步像素扩散、运行时条件和 RTX 50 系列部署。
- **本资料的解释**：公式直觉、ASCII 流程、比较表、工程建议和概念 demo；这些是教学重述，不是 NVIDIA 的额外实验。
- **尚未证实**：公开权重、完整训练数据/损失、层数宽度、状态张量、SDK 接口、跨 GPU 性能和独立用户研究。

## 论文中的基线提示词

报告为两个离线图像基线固定了不同提示词，并强调它们是为平衡照片感与内容保留而选择的。提示词本身属于评测条件，不应被误读为 DLSS 5 的运行时 prompt；DLSS 5 的产品接口是渲染帧、运动/状态和艺术控制。

## 引用

```bibtex
@misc{dlss5,
  author       = {NVIDIA Corporation},
  title        = {{DLSS 5: Real-Time Generative Rendering}},
  howpublished = {\url{https://research.nvidia.com/labs/adlr/DLSS5/}},
  note         = {Technical Report},
  year         = {2026}
}
```
