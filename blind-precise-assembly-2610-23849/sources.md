# 来源与证据边界

核对日期：2026-10-20。论文固定为 v1，2026-10-16；原 PDF 共 21 页，所有页码均为 PDF 页码。

| 资料 | 链接 / 本地记录 | 用途 |
|---|---|---|
| arXiv 摘要与版本 | https://arxiv.org/abs/2610.23849v1 | 标题、作者、版本、发布日期 |
| 原始 PDF | [paper.pdf](paper.pdf) / https://arxiv.org/pdf/2610.23849v1 | 正文、图、表与附录 |
| HTML 全文 | https://arxiv.org/html/2610.23849v1 | 交叉核对结构与数字 |
| 作者项目页 | https://younghyopark.me/blind-proprioceptive-assembly/ | 官方介绍、浏览器策略演示与仓库链接 |
| mjfit 仓库 | https://github.com/younghyopark/mjfit | 已公开几何生成器、README、MIT 标识 |
| 读取时仓库 HEAD | `eee986cc087d14458337fdc195d5cf03abc6aa6d` | `git ls-remote` 与下载源码 HEAD 一致；后续几何专题运行范围见文末 |

原 PDF SHA-256：

```text
5bc66905c70d821a0d70a728e66ba80acb2269f6b4231f2e57de0b3470f122fb
```

## 结论到原文的对应

| 内容 | 原文位置 |
|---|---|
| 纯本体感知与局部接触任务 | §1–2，pp.1–3 |
| 凸片构造、外接多边形误差公式 | §2.2，p.4 |
| actor 与接触监督信息流 | §2.3、图 7，pp.5–6 |
| 螺旋搜索、横向距离 | §3.2、图 9，p.7 |
| 表征、probe 与 latent 替换 | 图 10 p.8，附录 C pp.18–19，图 15 p.19 |
| 真机分母、间隙、成功次数 | 图 12、§3.4，p.9 |
| 局部任务与已接触起点的限制 | §5，p.10 |
| 网络和 PPO 超参数 | 附录 A.1，p.14 |
| study / deployed 随机化差异 | 表 1–2，pp.15–16，附录 A.2 pp.14、17 |
| 接触几何探针与 CPU/GPU 性能 | 附录 B、表 3–4，pp.17–18 |
| 时间、轨迹距离的统计口径 | 表 5，p.18 |
| 接触监督的步数倍数 | 附录 C p.18、图 14 p.19 |
| OOD 各列定义和全部成功率 | 表 6–7，p.20 |
| 真机物体尺寸、计划 reset、20 秒 horizon | 附录 C，pp.20–21 |

## 数据处理

- `sources/paper-layout.txt` 来自 `pdftotext -layout`，完整保留提取结果。
- `sources/parser-raw.json` 为技能脚本原始输出。它把标题错读为 Preprint，作者仅识别一人，正文截断。`meta.json` 根据 arXiv 与 PDF metadata 校正，完整正文来自 PyMuPDF。
- `sources/paper-data.json` 手工转录表 7、图 12、表 5 与附录 C 的数字，表 7 和图 12 对照页面渲染检查过。
- 齿轮 `>99.9` 保留不等号；图表高度按 99.9 下界绘制，不能误写成精确值。
- 真机图 12 的仿真柱没有数值标签，因此交互页面不从像素估读这些柱。
- 交互页总成功率与 Wilson 区间由报告计数计算；几何误差由原文公式计算。两者均标明为推导数据。
- 全部学习说明用中文，保留原论文标题、作者和技术标识。

## 已检查与未检查

初次学习读取了原论文、附录、项目页及仓库公开 README，并下载 PDF、校对关键图表。公开仓库被确认是 `mjfit`，不能据此声称完整策略训练代码已发布。没有运行作者网站中的真实策略，也没有执行其 MuJoCo benchmark、PPO 训练或真机实验。

初次学习的两个示例是独立教学实现。代码运行结果和 HTML 交互检查单独记录在 `validation.md`，不与论文作者的结果合并。

## 后续几何专题

`sources/mjfit/` 保存上述 SHA 的只读源码副本。查阅 `_polygon.py`、`_peg.py`、`_mjcf.py`、`shapes.py`、`_thread.py`、`_gear.py` 及相应官方 docs 和测试。`code/inspect_mjfit_geometry.py` 实际调用该版本生成器，用 MuJoCo 3.16.0 编译星形销孔，并检查凸性、分块面积、间隙和凸包体积。15 项聚焦源码测试通过；没有推进动力学。

代码运行环境在本学习目录 `.geometry-venv/`，新增依赖安装在该 venv，未写入原有 Python 环境。具体命令、结果、版本边界见 [几何专题](deep-dive-mjfit-geometry.md)。
