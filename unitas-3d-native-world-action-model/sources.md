# 来源与证据边界

- **主论文**：arXiv:2610.12099v1，提交于 2026-10-08；本目录的 paper.pdf 为 27 页版本，paper.txt 为布局文本抽取。
- **HTML 版**：https://arxiv.org/html/2610.12099v1。它包含正文、附录、表格和公式，是核对 PDF 字体抽取的主要来源。
- **摘要页**：https://arxiv.org/abs/2610.12099。
- **代码链接**：https://github.com/DexForce/UNITAS。

## 证据等级

1. **论文事实**：模型结构、数据源、训练配置、表格指标和 latency 直接来自正文/附录。
2. **阅读推断**：insights.md 中关于“metric frame 作为接口”的解释，是对方法和消融的归纳，不是额外实验。
3. **教学代理**：code/world_aligned_point_flow_demo.py 只演示 back-projection、world-frame motion 和 scene-flow 条件关系；不调用 UNITAS checkpoint，也不是官方 benchmark 复现。
