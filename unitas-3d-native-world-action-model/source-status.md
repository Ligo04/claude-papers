# 复现状态

- **已验证**：PDF 下载完整（pdfinfo 显示 27 页、24,911,135 bytes）、arXiv HTML 章节与表格、pdftotext 抽取、205 个嵌入图像。
- **解析边界**：PDF 内嵌字体触发 Poppler syntax warnings，标题/作者被 parse-pdf.js 错误识别；标题、作者、摘要和表格以 arXiv HTML/API 与人工核对为准。
- **论文内部勘误**：引言一句把 Pack the Ball 的 UNITAS 成功率写成 65%；摘要、§4.3 正文和表 2 都给出 16/20 = 80%，本包与交互页按后者记录，并保留该不一致。
- **未验证**：未运行 UNITAS 官方 checkpoint、未重跑 LIBERO/VLABench/RoboTwin/真机评测、未验证 GPU latency。
- **可运行教学内容**：Python demo 只用标准库，验证 metric back-projection、point trajectory 和 paper metrics 的展示逻辑。
