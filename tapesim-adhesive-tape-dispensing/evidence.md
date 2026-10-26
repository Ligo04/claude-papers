# 来源与验证边界

学习日期：2026-10-26，Asia/Shanghai。

## 固定的原始材料

| 来源 | 版本/用途 | 本地快照 |
| --- | --- | --- |
| [arXiv 摘要](https://arxiv.org/abs/2609.28766v2) | 2026-09-23 首发，2026-09-28 v2 修订 | PDF metadata 与首页核对 |
| [PDF v2](https://arxiv.org/pdf/2609.28766v2) | 9 页，正文、公式、图表与参考文献 | `paper.pdf` |
| [HTML v2](https://arxiv.org/html/2609.28766v2) | 辅助核对公式与章节 | `sources/arxiv-2609.28766v2.html` |
| [作者项目页](https://genesis-embodied-ai.github.io/tapesim-website/) | 展示、评估说明和代码入口状态 | `sources/project-page.html`、`project-app.js` |
| [作者统计 JSON](https://genesis-embodied-ai.github.io/tapesim-website/assets/data/evidence.json) | 可比运动成本、逐例混淆矩阵与回放均值 | `sources/evidence-official.json` |

PDF SHA-256：

```text
fc3600439f1a75f3a6ad948e41f33c85443e250450effc8abb2f362ec67553f3
```

技能中的下载脚本会把带版本的 arXiv 地址规范化为无版本 PDF URL。因此最终原文通过明确的 `/pdf/2609.28766v2` 地址另行下载，并核对首页 `arXiv:2609.28766v2 [cs.RO] 28 Sep 2026` 水印。没有仅凭下载命令的输入声称版本已固定。

`parse-pdf.js` 保留在 `sources/parser-raw.json` 中的结果丢失了部分标题、仅识别首位作者、摘要为空。最终 `meta.json` 已根据 PDF metadata、首页和项目页校正为 13 位作者；中文摘要为本包整理。`sources/paper-layout.txt` 保存完整文本，而非脚本最多 50000 字符的截断结果。

## 数据从哪里来

- 论文 Fig.3/IV-B 支持固定容差与可比运动两类成本。可比运动的完整 2/8/16/32 圈均值取自作者网页数据。固定容差 Coh-F 未给精确正文步时，本包留空，保留论文报告的约 3.4×。
- 混淆矩阵来自作者公开统计，并与论文 Fig.6 的 agreement 和 IV-D 的 BA/计数核对。指标均重新从 TP/TN/FP/FN 计算。
- 回放采用 `active_mean_px`，每个 episode 先平均两次重复，再等权平均五段。没有使用包含初始静止期的 `full_mean_px` 代替论文指标。
- 作者统计中的 `progress_pairs` 是另一个字段。本包配对成败指标使用 `n` 与 TP/TN/FP/FN，不把该字段当成任务评估样本量。
- 黏附敏感性百分比抄录自 PDF Fig.7 的标签，按原文四舍五入。98 例敏感性试验与 100 例主 Peel 评估分别处理。
- 网页 JSON 含三个内部统计文件的 provenance digest，但本包未获得那些原始文件，未声称可以独立重建作者全部实验。

## 本地做过哪些检查

| 层次 | 已完成内容 | 证据 |
| --- | --- | --- |
| 来源与元数据 | 固定 v2、页数、作者、PDF SHA-256、原图提取 | `meta.json`、`sources/pdfinfo.txt`、`sources/source-manifest.json` |
| 数字核对 | 16 个混淆矩阵、派生指标、回放均值、封箱时间与索引保留 | `sources/study-validation.json` |
| CPU 教学数学/状态行为 | 27 项检查：解析梯度、刚体不变性、力协变、时间步缩放、压缩门槛、前沿快照、质量分类及速度投影/恢复 | `code/demo-validation.json` |
| 浏览器 | 控件改变可见内容、默认/边界/任务切换、固定容差缺数说明、全失败基线、sandbox 运行 | `sources/browser-checks.json`、`images/study-page-preview.png` |

可复制的核对命令：

```bash
python3 /home/ligo/claude-papers/papers/tapesim-adhesive-tape-dispensing/code/releasable_bond_and_front.py
python3 /home/ligo/claude-papers/papers/tapesim-adhesive-tape-dispensing/code/verify_study.py
```

教学程序的归一化梯度误差是全部梯度分量的最大绝对差除以解析梯度的最大绝对分量，不是每个小分量的相对误差。程序中的合成三角形、材料行数和强制位移均明确标为教学设置。

质量检查覆盖材料行分类的无重叠/无丢失和总质量核算；速度检查覆盖刚体速度投影与恢复。它们不覆盖真实三角薄壳交界的元素质量装配或实际求解中的动量误差。

## 没有完成的复现

没有运行作者 TapeSim、GPU/IPC/CCD/PCG 求解器、薄壳放带、重网格切割、机器人 VLA 或真机实验。没有重新测量论文的速度、剥离力、回放误差与策略成功率。全部这类数字均为作者报告。

本包没有做手机 viewport 覆盖测试。浏览器检查包含默认桌面布局和 sandbox iframe，不能据此宣称全部设备兼容。

截至查阅，项目页明确显示 Code Coming soon；`githubLinks` 与 `codeLinks` 为空。这里的教学 Python、交互页面和工程建议不是作者源码，也未修改任何仿真项目仓库。

## 保存与索引

原索引先保存为 `sources/index-before.json`，新增条目前重新读取现有索引，仅追加本论文，原条目和索引额外字段保持不变。新增条目的标签严格为 `胶带操作仿真` 与 `释放前沿降阶`。文件摘要见 `sources/source-manifest.json`，原始数据不会被教学派生结果覆盖。

## 深入数学与动态 collar 的扩展

用户要求两部分都深入后，增加 `deep-dive-bond-math.md`、`dynamic-collar-implementation.md` 和两份 NumPy 教学程序。初版文件在修改前逐项与 manifest 的 SHA-256 核对，结果没有额外改动；原 manifest 与原 HTML 分别保存在 `sources/source-manifest-initial.json` 和 `sources/index-initial.html`。没有再新增论文索引条目。

- `code/bond_hessian.py` 通过 18 项检查，精确局部 Hessian 与梯度中心差分的最大归一化误差约 `4.60e-10`；比较了 Gauss–Newton、压缩负曲率和参考态刚体零模态。
- `code/triangular_collar.py` 通过 80 项检查，用 93 顶点、120 三角形展示元素级 `<N` 规则和材料状态转移。检查覆盖质量、线/角动量及动能在重分类瞬间一致，位置是预先指定的。
- 归档接口在网格教学中是每元素一个合成记录，用来验证历史恢复；没有生成真实胶带接触对或接触搜索。
- `sources/deep-dive-data.json` 与第五个交互章节使用这些程序实际输出。法向曲线根据明确列出的合成几何解析计算，属于教学推导，不是论文实验曲线。
- 新界面的浏览器与 sandbox 记录保存在 `sources/deep-dive-browser-checks.json`。初版的 `sources/browser-checks.json` 保留为初版检查记录。

新增程序仍没有全局 Newton/PCG、薄壳运动、CCD 或真实机器人。局部 Hessian 检查不能代替全局求解器正确性，重分类瞬间守恒不能代替时间积分守恒。
