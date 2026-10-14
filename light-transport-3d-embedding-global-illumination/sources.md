# 来源与证据

## 主来源

- PDF：<https://bingxu.tech/papers/light_transformer_sig_2026.pdf>
- DOI：<https://doi.org/10.1145/3799902.3811095>
- 论文题目：A Generalizable Light Transport 3D Embedding for Global Illumination
- 会议标识：SIGGRAPH Conference Papers ’26, July 19–23, 2026

## 相关依赖

- Point Transformer V3：论文引用 Wu et al. 2024；当前学习包只记录其在本文中的使用，不把外部实现当作本文官方代码。
- Falcor：PDF 中的 CUDA interop 和 rendering integration 相关引用为 <https://github.com/NVIDIAGameWorks/Falcor>。

## 证据层级

- 论文事实：PDF 正文、表格、公式和附录中的架构、数据、指标与限制。
- 阅读推断：`insights.md`、`mental-model.md` 中对设计原因和研究地图的解释。
- 教学 demo：`code/light_transport_embedding_demo.py`，用标准库重现 NNE/KNN/vector relation 的计算骨架，不是作者 checkpoint 或正式训练复现。
