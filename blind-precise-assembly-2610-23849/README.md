# Blind Precise Assembly 中文学习笔记

这篇论文把关节运动当作接触信息源：给机械臂一个关节目标，观察实际运动及其历史，再推断零件是否碰到边缘、部分进入或卡住。作者用仿真强化学习训练插销入孔、齿轮啮合和螺母起扣，并把插销策略部署到真机。配套的 `mjfit` 从指定间隙构造凸碰撞几何，为大量不同零件提供训练环境。

原文：Antonia Bronars、Younghyo Park、Kushal Kedia、Pulkit Agrawal，MIT，2026-10-16，arXiv:2610.23849v1。学习材料核对日期：2026-10-20。难度为**高级**，属于实验研究与机器人系统设计；需要理解基本机器人运动学、阻抗控制与 PPO，不要求复杂数学证明。

建议先打开 [交互学习页面](index.html)，再读 [核心理解](insights.md)。完整学习约 90–150 分钟，这是学习安排估计，并非论文实验时间。

| 材料 | 内容 | 建议时间 |
|---|---|---|
| [summary.md](summary.md) | 研究问题、贡献与实验数字 | 15 分钟 |
| [insights.md](insights.md) | 接触可观测性、泛化机制和结论边界 | 20 分钟 |
| [method.md](method.md) | 观测、动作、训练损失、几何生成和复现风险 | 25 分钟 |
| [mental-model.md](mental-model.md) | 先修知识与研究定位 | 10 分钟 |
| [qa.md](qa.md) | 5 道基础、5 道进阶、5 道研究题，答案可折叠 | 20 分钟 |
| [reflection.md](reflection.md) | 研究扩展与适用于 ManaEnv / Newton 的实验提案 | 按需 |
| [deep-dive-mjfit-geometry.md](deep-dive-mjfit-geometry.md) | 源码核对：偏移、凸片、螺纹、齿轮和几何检查 | 25 分钟 |
| [sources.md](sources.md) | 每个核心结论的原文位置与核验边界 | 按需 |

抓住四点：

- 动作历史使关节反馈有意义。实际运动对指令的响应携带接触信息。
- 接触监督训练编码器，接触预测不进入 actor；部署时移除 decoder。
- 碰撞几何必须保留间隙，否则策略学习的是错误装配任务。
- 真机结果为 173/237，实验从已接触且靠近孔的位置开始。齿轮与螺母仅有仿真结果。

两个示例可直接运行：

```bash
python3 ~/claude-papers/papers/blind-precise-assembly-2610-23849/code/clearance_geometry_demo.py
python3 ~/claude-papers/papers/blind-precise-assembly-2610-23849/code/proprioceptive_information_flow.py
```

第一个仅依赖 Python 标准库，输出几何图和 CSV。第二个依赖 PyTorch，用合成输入检查辅助损失的梯度去向。两者均为教学演示，没有训练论文策略或复现真机成功率。运行结果和实际检查记录见 [validation.md](validation.md)。

```text
paper.pdf                 原始 21 页 PDF
meta.json                 校正后的元数据、恰好两个标签、完整提取正文
index.html                无外部依赖的交互页面
summary.md / insights.md / method.md
mental-model.md / reflection.md / qa.md
sources.md / validation.md
code/                     两个可运行教学演示
images/                   PDF 原图及关键页面渲染
sources/                  原文文本、原始解析结果、论文数据、提取日志
outputs/                  教学演示输出和验证结果
```

原论文的图表与作者报告的结果保留其来源；本文笔记中的推导、工程建议和教学实现均明确区分。交互页面只绘制可核对数字，不从柱状图猜测未报告值。
