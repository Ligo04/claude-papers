# 来源与证据边界

## 论文身份

| 字段 | 核对结果 |
|---|---|
| 标题 | Convergent Incremental Potential Contact |
| arXiv | `2307.15908v1` |
| 提交时间 | 2023-07-29 06:29:04 UTC |
| 分类 | `math.NA`（主），`physics.comp-ph` |
| 作者 | Minchen Li；Zachary Ferguson；Teseo Schneider；Timothy Langlois；Denis Zorin；Daniele Panozzo；Chenfanfu Jiang；Danny M. Kaufman |
| 页数 | PDF 26 页 |
| PDF SHA-256 | `75b294a61c5a49ac39cd68fdf9f1bc51c767efb789b748b7484166e01789fece` |

## 一手来源

- arXiv 摘要与版本历史：<https://arxiv.org/abs/2307.15908v1>
- 用户指定的实验性 HTML：<https://arxiv.org/html/2307.15908v1>
- 精确 v1 PDF：<https://arxiv.org/pdf/2307.15908v1>
- 本地原文：[paper.pdf](paper.pdf)
- PDF 布局文本：[paper.txt](paper.txt)

## 交叉核对过程

1. 直接下载带 `v1` 的 PDF，而没有使用会丢失版本后缀的通用下载脚本。
2. 用 `pdfinfo` 核对标题、作者简写、页数与 PDF 生成时间。
3. 用 `pdftotext -layout` 提取完整 26 页文本，并逐节核对公式、实验设置和图注数值。
4. 用 arXiv Atom API 与摘要页核对完整作者、提交时间、分类和版本。
5. 运行技能自带 `parse-pdf.js` 作为辅助，但没有盲信其作者和摘要解析。

## 已知来源异常

### arXiv HTML 日期错误

实验性 HTML 的标题脚注显示“Submitted to arXiv August 24, 2026”，与同页 watermark、arXiv 版本历史和 v1 PDF 冲突。元数据采用官方版本历史中的 2023-07-29；PDF 页脚正文写的是 2023-08-01，可能是作者稿内部日期，也没有替代 arXiv 提交日期。

### `parse-pdf.js` 解析限制

- `pdf-parse` 输出了 `Could not find a preferred cmap table` 警告，导致原始重定向结果不是合法 JSON。
- 标题识别正确，但作者误识别为 `Incremental Potential`。
- 摘要正则没有匹配成功。
- 正文被截断到 50,000 字符。

因此本包作者和摘要来自 arXiv 元数据，正文事实来自原始 PDF / `pdftotext -layout`。

## 代码来源边界

v1 正文没有给出 GitHub 或专属实现 URL，`meta.json` 的 `githubLinks` 和 `codeLinks` 保持为空。`ipc-sim/IPC` 与 `ipc-sim/ipc-toolkit` 是理解原始 IPC 生态的相关项目，但没有证据证明它们是这篇 v1 的完整 EIPC 参考实现，因此未列为论文代码。

## 教学材料边界

- `code/eipc_barrier_demo.py` 直接实现论文障碍公式、摩擦平滑公式和一个共线双边重复计数例子。
- 交互页使用论文图注报告的真实收敛阶；由 `2^{-pℓ}` 生成的曲线明确标为“由报告收敛阶推导的归一化趋势”，不是论文原始误差样本。
- 本包没有声称复现 FEM、CCD、完整 Projected Newton、摩擦外迭代或论文所有实验。

