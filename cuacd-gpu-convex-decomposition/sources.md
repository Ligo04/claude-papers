# 来源与证据地图

## 原始来源

- [arXiv 摘要及版本](https://arxiv.org/abs/2609.28731)：作者、提交日期、v1。
- [PDF](https://arxiv.org/pdf/2609.28731)：15 页，11,556,614 bytes。
- [HTML 正文与附录](https://arxiv.org/html/2609.28731v1)：表格文本交叉核对。
- [官方仓库](https://github.com/eliphatfs/cuacd)：源码与使用说明。
- [固定源码版本](https://github.com/eliphatfs/cuacd/tree/13a00e25477f53a3533938b0325e2d575e3fd68b)：本包观察的依据。

PDF SHA256：`42e12686861ad6c89f45a380032b0ea0c891f3ea9d85f1acee915cd02bcca461`。

## 事实定位

| 事实 | PDF 位置 |
|---|---|
| warp 任务分工、四类原语 | §3.1，pp.3–4 |
| heap、64 arena、分桶与合并 | §3.2，pp.4–5；Appendix E，p.12 |
| 凹度 Eq.1、候选与 look-ahead | §4.1，p.5 |
| 切割封口 | §4.2、Algorithm 1，pp.5–6；Appendix F，p.12 |
| D&C、Hybrid hull、Hausdorff | §4.3，p.6 |
| CC-Split 与 hull merge | §4.4，p.6 |
| 硬件、数据集、PaMO 不计时 | §5.1–5.2，pp.6–7 |
| 三数据集所有主结果 | Table 1，p.7 |
| CPU 控制实验、occupancy | §5.3，p.7 |
| 256 组点集的凸包时间 | Table 2，p.7 |
| 阈值、规模与 Bistro | §5.5–5.6，pp.8–9 |
| heap/寄存器/搜索局限 | §6，p.9 |
| 参数、消融、输出分布 | Tables 3–5，p.11 |
| 排序比较与原语细节 | Table 6、Appendices C–I，pp.12–15 |

## 图像归属

images/page_* 是通过技能脚本从论文抽取的 140 张嵌入图像，不代表有 140 幅独立 figure。decomposition-gallery、pipeline、benchmark-table、search-ablation、warp-fusion 为 PDF 指定区域渲染，分别对应 Fig.1、Fig.4、Table 1、Table 4、Fig.14。论文为 CC BY 4.0；原图归 Shi 等作者，页面附有来源与原文链接。界面中的图表根据 metrics.json 重绘，不添加未报告的测量点。

## 解析降级

parse-pdf.js 识别标题正确，但作者变为 Additional Key、摘要为空、content 截断到约 50k 字符。原输出保存在 sources/parser-output.json；meta.json 的作者来自 PDF 首页与 arXiv，摘要为标明类型的中文转述。paper.txt 保存 pdftotext -layout 全文，不能把 parser 的 content 当作完整论文。

## 证据类型

论文事实：作者报告的方法/实验。源码观察：固定 revision 的静态证据。阅读推导：公式展开、工程意义和后续设想。教学示例：本地 CPU 模拟，不是官方 CUDA 复现。四者在笔记和页面中分别标明。
