# SDF 的实时碰撞处理：中文学习包

（翻开笔记）这篇论文把两个 SDF 的碰撞处理组织成一个持续更新的多点搜索：保留上一帧有效候选，用曲率引导的新种子寻找其他接触区域，再用排斥项减少接触点聚集。重点是有限查询预算下的接触流形质量喵。

原文：Chris Giles、Sheldon Andrews，*Real-Time Collision Handling for Signed Distance Fields via Caching and Importance Sampling*，Pacific Graphics 2026，12 页。[PDF](https://savant117.github.io/sdf-dcd-demo/paper.pdf) · [项目页](https://savant117.github.io/sdf-dcd-demo/) · [公开 C++ demo](https://github.com/savant117/sdf-dcd-demo)

难度为进阶。需要了解 SDF、刚体接触法向、局部优化与 warm start。它是计算几何与仿真系统论文，没有训练阶段。

建议先花 15 分钟读 [summary.md](summary.md)，再花 30 分钟读 [insights.md](insights.md) 和 [method.md](method.md)。用 [交互页面](index.html) 对照真实表格，运行二维演示，再用 [qa.md](qa.md) 的 15 个问题检查理解。完整学习约 1.5–2.5 小时，这是阅读建议。

需要记住的结论：一个最深点不足以稳定支撑复杂刚体；时间缓存、空间覆盖和新区域探索共同影响稳定性。方法属于 DCD，仍会有穿透和漏检；论文的毫秒数是碰撞检测耗时。

| 文件 | 用途 |
| --- | --- |
| `summary.md` | 问题、贡献与实测结果 |
| `insights.md` | 直觉、推导、方法限制与工程判断 |
| `method.md` | 公式、算法、实现顺序和参数缩放 |
| `mental-model.md` | 前置知识与研究位置 |
| `qa.md` | 5 道基础、5 道进阶、5 道研究问题 |
| `reflection.md` | 可检验的后续研究想法 |
| `source-audit.md` | 论文与公开 demo 的差异、原文不一致之处 |
| `index.html` | 零外部依赖的实验数据与算法浏览器 |
| `code/` | 标准库二维演示、运行说明和实际输出 |
| `sources/metrics.json` | 表 1–5 的结构化转录 |
| `sources/upstream/` | 固定 commit 的相关源码与原始许可证 |
| `images/` | 原图、公式和表格截图 |
| `paper.pdf`、`paper.txt`、`meta.json` | 原文、文本与校正后的元数据 |
| `validation.json` | 本地验证范围与结果 |

本学习包阅读的是 PDF 内容及 commit `08539dd3ad5fa9b41fdd160c730747de0b510b4e` 的公开代码。二维教学演示不等于论文的 CPU/GPU 仿真实验复现。
