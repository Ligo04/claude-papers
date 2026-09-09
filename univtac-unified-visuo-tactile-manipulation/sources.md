# 来源、抽取与证据边界

## 一手论文来源

1. **arXiv 摘要页**：<https://arxiv.org/abs/2602.10093>
   - 标题：UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking
   - 版本：v1，提交时间 2026-02-10，分类 `cs.RO`
   - 页面列出 16 位作者、项目主页和 PDF/HTML/TeX 入口。
2. **PDF**：<https://arxiv.org/pdf/2602.10093>
   - 本地副本：[`paper.pdf`](paper.pdf)
   - 13 页，未加密，PDF 1.7，文件大小 3,519,346 bytes
   - SHA-256：`073c5bdbef9ff06d21c7217c0b25fc059281632f6eb3f0e871c811539303b688`
3. **arXiv HTML**：<https://arxiv.org/html/2602.10093v1>
   - 用于交叉核对表格、附录任务描述和公式排版；论文 PDF 仍是本资料的主要事实来源。
4. **项目主页**：<https://univtac.github.io/>
   - 提供代码、数据集、三类传感器说明、八任务概览和 BibTeX。

## 公开实现与数据

- 代码：[https://github.com/univtac/UniVTAC](https://github.com/univtac/UniVTAC)
  - 本次只读快照：`05bcd3edb92237107efa40105292a24f1a9fd761`（2026-09-02 检查）
  - 可见 `encoder/network.py`、`encoder/train.py`、`docs/Collection.md`、`eval_policy.sh` 等实现线索。
- 数据集：[Hugging Face / byml/UniVTAC](https://huggingface.co/datasets/byml/UniVTAC)
- 数据集：[ModelScope / byml2024/UniVTAC](https://modelscope.cn/datasets/byml2024/UniVTAC)

## 公开状态差异

### 传感器支持

论文与项目主页把 GelSight Mini、ViTai GF225、Xense WS 作为平台支持的三类传感器；仓库 README 的 TODO 则写明“数据采集和评测目前只支持 GelSight Mini，另外两类将在后续加入”。因此本资料将“三类接口/仿真建模”与“当前公开脚本可端到端运行的范围”分开记录。

### 数据集在线预览

Hugging Face 页面在本次检查时显示 `DatasetGenerationError` / `CastError`，原因是展示 schema 与实际列名不匹配（页面错误信息指出 `actor/atom/embodiment/observation/tactile` 等字段的 cast 失败）。这只证明在线 viewer 当前有 schema 问题，不能单独证明原始数据文件不可下载；下载后应按 HDF5 字段和 shape 自行核验。

### 许可证

仓库 README 的许可证段落写的是 MIT，但仓库根目录的 `LICENSE` 文件内容是 Apache License 2.0。法律使用、修改和再发布应以实际 LICENSE 文件及最新上游状态为准；本学习包不重新分发仓库源码或数据集。

## 本地抽取产物

- [`paper-layout.txt`](paper-layout.txt)：`pdftotext -layout` 的原始版面文本。
- [`paper-clean.txt`](paper-clean.txt)：移除 PDF 文本中的 NUL 字节后用于检索的副本。
- [`paper.txt`](paper.txt)：兼容资料库/WebUI 的全文别名。
- [`pdfinfo.txt`](pdfinfo.txt)：页数、作者、PDF 版本和大小。
- [`parse-output.json`](parse-output.json)：技能自带解析器原始输出。
- [`parse-output.clean.json`](parse-output.clean.json)：去掉解析器 stdout 前置警告后的 JSON。
- [`extract-images.log`](extract-images.log)：PyMuPDF 图像抽取日志。
- [`images/`](images/)：原始图像资源、整页预览和图像索引。

## 解析可靠性说明

技能自带 `parse-pdf.js` 能读出正文和页数，但对该 PDF 把标题截成第一行、作者只识别出第一名，Abstract 也为空；输出还混入 TrueType 警告。因而 `meta.json` 的作者、标题、摘要和日期以 PDF 元数据、`pdftotext -layout`、arXiv 摘要页和项目主页交叉核对后手工整理。表格数字则逐项从 PDF/HTML 对照，不从解析器的截断正文推断。

## 结果分级

- **论文直接报告**：205,826 样本、14 种几何、五个操作 API、三路径损失、八项任务、表 I–IV、真机设置与成功率。
- **官方仓库观察**：ResNet-18/decoder 结构、公开训练脚本、当前传感器 TODO、安装依赖、数据字段和许可证文件。
- **本资料解释**：三层心智模型、比较、工程建议、概念 demo 与交互式图表；这些不是额外实验。
- **尚未独立验证**：真实 Isaac Sim/TacEx 全链路、跨三类传感器支持、论文表格的随机种子置信区间、真机重复实验和数据集所有下载镜像。

## 引用

```bibtex
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```
