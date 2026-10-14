# 来源与证据边界

- **主论文**：arXiv:2609.38537v1，提交于 2026-09-29；本目录 `paper.pdf`（46 页）和 `paper.txt`。
- **HTML 版**：<https://arxiv.org/html/2609.38537v1>，用于核对标题、摘要、表格和项目链接。
- **API 元数据**：<https://export.arxiv.org/api/query?id_list=2609.38537>，用于核对作者、日期和摘要。
- **代码/项目**：<https://github.com/anonymous-report-421/GPT-as-Policy>、<https://galaxygeneralrobotics.github.io/astra-policy/>。

## 证据等级

1. **论文事实**：表格中的样本数、成功率、Score、returns、token 和 latency 直接来自 PDF/HTML。
2. **阅读推断**：`insights.md` 中关于“分工边界”的解释，是由 Direct/Hybrid 对比和失败案例归纳出的机制判断，已明确标注。
3. **教学代理**：`code/closed_loop_control_demo.py` 只复现控制逻辑和指标计算的形状，使用论文公开的真实汇总数字，不能代替 Astra API、机器人仿真或官方 benchmark。
