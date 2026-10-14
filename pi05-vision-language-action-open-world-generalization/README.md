# π0.5：具备开放世界泛化能力的 Vision-Language-Action Model

这篇论文提出 π0.5：在 π0 的 VLA 基础上，把不同机器人、语义子任务、人工口头指令和网页视觉语言数据放进同一个 co-training 配方，再用“先预测语义子任务、再生成连续动作块”的层级推理控制移动操作机器人。论文报告了在训练中未出现的新厨房和卧室里执行收拾餐具、整理衣物、铺床等多阶段任务的结果；重点不是单个动作的更高精度，而是如何用异构知识覆盖开放世界的长时程决策。

- **难度**：Advanced；需要 VLA、imitation learning、flow matching 和真实机器人评测背景。
- **论文性质**：system design + empirical-heavy；核心贡献是训练数据配方、两阶段模型和真实环境验证。
- **预计学习时间**：首读 45–60 分钟；结合代码和图表复盘 90–120 分钟。
- **建议路径**：先看 `summary.md` 建立全貌，再看 `method.md` 理清数据与推理链，最后用 `insights.md`、`reflection.md` 和 `qa.md` 检查是否真正理解。

## 关键结论

1. 只有约 400 小时移动操作数据并不足以覆盖开放世界；来自其他 embodiment、其他环境、语义预测和 web 的数据能把知识迁移到目标移动机器人。
2. π0.5 用同一个 transformer 表示高层文本/子任务和低层动作，但在训练上把离散 FAST action token 与 post-training 的 flow-matching action expert 结合起来。
3. 推理时先输出诸如“pick up the plate”的子任务，再条件化地产生连续 action chunk；在论文的四任务 mock-home 指标中，完整模型约 79% 平均进度，优于去掉交叉 embodiment 数据的变体。
4. 论文演示的是研究原型的真实机器人证据，不是可直接下载复现的开源 checkpoint；本目录的代码是教学代理实现。

## 文件导航

- `summary.md`：问题、贡献、训练和结果速览
- `insights.md`：为什么异构 co-training 有效，以及它改变了什么设计取舍
- `method.md`：模型、损失、两阶段训练和推理伪代码
- `mental-model.md`：把论文放入 VLA / hierarchical control 的研究地图
- `reflection.md`：脆弱假设、失败模式和可延伸方向
- `qa.md`：5 道基础、5 道中级、5 道高级自测题
- `code/hierarchical_policy_demo.py`：纯 Python 教学 demo；展示高层子任务 + 低层 action chunk + 混合损失
- `index.html`：无外部依赖的交互式学习页
- `images/`：论文图表和评测环境截图
- `paper.pdf`：下载的 arXiv v1 PDF

## 证据边界

正文中明确给出的训练步数、数据规模、动作维度、频率、评价协议和 p 值按论文原文记录。图 7、10、12、13 的柱高和图 8、9 的曲线点没有附数字表格，学习页把它们标为“图中约读数”；这些数值用于理解趋势，不应当当作作者发布的原始日志。
