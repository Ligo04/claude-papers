# 复现状态

- **已验证**：PDF 下载完整（18 页、46,314,643 bytes、SHA256 `e340fa3d7682219ed7a2487c0e5c000836bae55bdcba127c184ba9abf5b490ed`）、`pdfinfo`、`pdftotext -layout`、136 个嵌入图像、正文表格和附录配置。
- **解析边界**：`parse-pdf.js` 正确识别标题但只返回第一作者且摘要为空；作者名单和摘要按 PDF 首页正文人工核对后写入 `meta.json`。
- **代码边界**：论文写明将公开 code/data/weights，但当前 PDF 未提供本文专属 GitHub 仓库；解析到的 Falcor 链接是渲染框架依赖，不等于本文实现。
- **未验证**：未运行作者 checkpoint、未重建 13,900 场景数据集、未重跑 105 测试场景、未验证 RTX 5880 Ada latency，也未复现 250 CPU/GPU days 的 ground-truth 生成。
- **可运行教学内容**：Python demo 只用标准库，验证点云属性、NNE 风格邻域聚合、KNN vector relation 和复杂度对比；输出中的论文指标是引用展示，不是 demo 预测结果。
