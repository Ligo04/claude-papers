# 复现状态

- **已验证**：PDF 下载完整（`pdfinfo` 显示 46 页、35,445,248 bytes）、`pdftotext -layout` 文本抽取、arXiv HTML/API 元数据、图片抽取（101 个嵌入图像）。
- **未验证**：未调用 GPT-6 Astra API，未运行论文的 RoboDojo/RoboCasa/DexJoCo/HumanoidBench 官方评测，未复现论文的 token 账本和成功率。
- **可运行教学内容**：Python demo 不依赖第三方包，用确定性 toy state 展示 proposal → review/correction → execution → feedback 闭环。
