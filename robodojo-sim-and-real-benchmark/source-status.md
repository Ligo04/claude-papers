# 来源与验证状态

## 论文主来源

- 本地原始输入：`/home/ligo/Downloads/2607.04434v3.pdf`
- 资料库副本：`paper.pdf`
- SHA-256：`d189503d21e7c4cf89f819797ded798923ceffeda50a0b787e71ac57da66aa8b`
- PDF 元数据标题正确，共 46 页、44 位作者；arXiv 页面确认 v1 于 2026-07-05 提交，v3 于 2026-07-08 修订。
- skill 自带 `pdf-parse` 把首页日期 `2026-7-5` 误判为标题，并只识别出第一位作者，因此标题/作者改用 PDF metadata、页面文字和 arXiv 元数据三方核对；正文再用 `pdftotext -layout` 与 PyMuPDF 检查。

## v1 归档

- 论文库原先保存的是 v1（45 页，SHA-256 `9a3413db17b50aacef42a0d5679cec983c8e2114d01f1b920ac04e192a46c08e`）。
- v1 PDF 和旧图片保存在 `archive/v1/`，当前根目录材料统一按 v3 修订。
- v3 核心榜单数值没有变化；版本更新重点见 `FACTS.md` 的“v1 到 v3 的实质变化”。

## 官方源码快照

- RoboDojo：`https://github.com/RoboDojo-Benchmark/RoboDojo`，检查提交 `57b8c52424ff1198ade242505656511515c58792`（2026-08-19）。
- XPolicyLab：`https://github.com/XPolicyLab/XPolicyLab`，检查提交 `85ff90d4092a9530860c9efe96a732d9f9150954`（2026-08-18）。
- 源码快照晚于论文 v3，所有路径观察写在 `implementation-map.md`，不用于篡改论文冻结实验口径。
- 论文写 30 个策略；当前 RoboDojo README 写 40+，当前 XPolicyLab `policy/` 有 41 个一级 adapter 目录。这说明仓库继续演进，不表示论文当时评了 41 个策略。

## 许可证注意事项

- 检查提交中，RoboDojo `README.md` 声明“RoboDojo Non-Commercial Research License”，但同一提交的 `LICENSE` 文件正文是 MIT License；两者存在直接冲突。
- XPolicyLab 根 `LICENSE` 是 Apache License 2.0。
- 本材料只记录观察，不作法律解释。若用于商业场景，应让维护者书面澄清 RoboDojo README 与 LICENSE 的优先关系。

## 教学产物边界

- `code/heterogeneous_parallel_demo.py` 是离散事件教学模型，不运行 Isaac Sim，也不复现论文的 1,640,000 帧实验。
- `code/capability_dimension_diagnosis_demo.py` 使用论文表 1 的真实榜单数据；其中“等权维度均值 vs 按任务数加权”部分使用明确标注的构造分数。
- `index.html` 只嵌入论文 v3 的冻结数据，无外部 fetch、无 localStorage，可在 sandboxed iframe 中运行。
