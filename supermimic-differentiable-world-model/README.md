# SuperMimic 中文学习包

**一句话理解：不同动作共享同一个机器人的动力学，因此先把动作后果学成可微模型，再沿着模型梯度改进控制策略。** SuperMimic 将标准 MuJoCo Warp 当作数据来源，把短期策略优化移到学习出的世界模型中。模型与策略持续交替训练，最终部署的是目标条件策略。

原文：[arXiv:2610.28480v1](https://arxiv.org/abs/2610.28480v1)，2026-10-20，19 页。作者为 Qiayuan Liao、Yiyang Shao、Junfeng Long、Kevin Zakka、Roberto Horowitz、Pieter Abbeel、Koushil Sreenath，UC Berkeley。官网：[SuperMimic](https://supermimic.github.io/)。本学习包整理日期为 2026-10-22。

难度为高级。论文属于方法架构和实证系统研究，需要了解强化学习、自动微分、动力学和 actor-critic；不需要先掌握可微接触求解器。推导复杂度中等，实现和复现实验的复杂度高。建议学习 2.5 至 4 小时。

## 推荐阅读顺序

网页查看器中可从左侧文件树选择材料。下方相对文件链接也可在本地 Markdown 阅读器使用。

1. 用 10 分钟阅读 [mental-model.md](mental-model.md)，区分仿真器、世界模型、策略和 critic。
2. 用 25 分钟阅读 [summary.md](summary.md) 和 [insights.md](insights.md)，理解收益来自哪里，以及结果证明了什么。
3. 用 40 分钟阅读 [method.md](method.md)，对照原文第 4 至 5 页公式和第 13 至 16 页附录。
4. 打开 [交互探索页](index.html)，切换训练阶段、想象时域和失败分组，核对计算口径。
5. 运行 [CPU 教学演示](code/README.md)，检查冻结模型参数后梯度仍能传给策略。
6. 用 [15 道问答](qa.md) 自测，再看 [工程应用分析](engineering.md) 与 [研究延伸](reflection.md)。

## 需要记住的结论

- 真正共享的是机器人和接触的动力学。目标改变奖励和动作选择，不要求每个动作另学一套物理。
- 策略更新用学习模型的状态、动作 Jacobian。原始仿真器不需要提供反向传播。
- 冻结模型参数不等于截断计算图。世界模型仅在策略更新内部冻结，后续循环仍继续拟合。
- 默认模型拟合窗口为 8 步，想象时域为 32 步。critic 为超过该时域的回报提供估计。
- 论文的零失败来自训练动作集的确定性仿真评估。它不等于未见动作泛化或真机全部动作零失败。
- 约 7200 倍卡时差距含硬件和数据条件差异，不能解释为物理引擎或推理速度提高了同样倍数。

## 文件说明

```text
paper.pdf                 固定版本原文
paper.txt                 完整布局文本
meta.json                 校正后的元数据，恰好两个语义标签
data.json                 有来源和口径说明的实验数据
index.html                无外部依赖的单文件交互页面
summary.md                背景、贡献和实验数字
insights.md               核心机制、价值与限制
method.md                 公式、训练流程和实现注意点
mental-model.md           概念关系与先修知识
engineering.md            对 Newton / ManaEnv / UIPC 的工程含义
qa.md                     5 基础 + 5 中级 + 5 高级问题
reflection.md             可检验的延伸研究
evidence.md               页码、来源、数据处理与验证边界
code/                     独立 CPU 教学演示
images/                   原文图片提取与关键图表裁图
sources/                  来源记录、解析输出与本地验证结果
```

本地演示验证的是教学系统的梯度和控制行为。未运行作者训练、129,514 动作 benchmark、H100 吞吐实验或 G1 真机实验。官方网页 demo 可在线体验；其专有许可与公开论文是不同交付物，不能据此标为开源训练实现。详见 [evidence.md](evidence.md)。
