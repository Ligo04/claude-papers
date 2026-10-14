# 摘要与结果

## 背景

传统 VLA 在训练分布附近的短任务上表现突出，但“把机器人带出实验室”要求它同时处理新房间、新物体、新布局、不同 embodiment 和长时间的多阶段任务。单纯扩大目标移动机器人的演示数据成本很高，也难覆盖所有可能的家庭情境。π0.5 的问题是：能否把机器人动作、语义监督、口头指导和网页知识放进同一个 VLA 训练体系，让中等规模的目标机器人数据获得更宽的开放世界泛化？

## 方法概要

π0.5 继承 π0 的 vision-language-action backbone，采用两个训练阶段：

1. **Pre-training**：把所有任务统一成离散 token 序列，进行 next-token prediction。输入包括多相机图像、语言和离散化 proprioception；输出既可为文本/位置 token，也可为 FAST 编码的动作 token。
2. **Post-training**：用移动操作相关的 MM 与 ME 数据、web 数据、高层子任务数据和 verbal instruction 数据继续训练；保持文本 token 预测，同时新增随机初始化的 action expert，用 flow matching 产生连续 action chunk。
3. **运行时**：模型先根据高层命令和图像输出语义子任务，再把该子任务作为条件生成低层连续动作。高层使用四个相机，低层使用腕部与前向相机；控制目标以 50 Hz 送给简单 PD controller。

## 训练数据

- **MM（Diverse Mobile Manipulator）**：约 400 小时，约 100 个家庭环境，最接近最终评测。
- **ME（Multi-Environment non-mobile）**：固定在平台上的单臂/双臂机器人，提供更多家庭环境。
- **CE（Cross-Embodiment）**：实验室中多机器人、多任务数据，如整理桌面、折叠衣物。
- **HL（High-Level）**：从长命令和观察预测短语义子任务，并可先预测相关 bounding box。
- **WD（Web Data）**：caption、VQA、object localization 等多模态网页任务。
- **VI（Verbal Instruction）**：专家用户用语言逐步引导机器人完成移动操作；主要用于 post-training。

预训练第一阶段中，来自移动操作机器人的数据只占 2.4%，其余 97.6% 来自其他机器人或 web 等来源。

## 论文贡献

1. 给出一个能承载文本、离散动作和连续动作的统一 π0.5 架构。
2. 提出“异构 co-training + 两阶段训练 + 层级子任务推理”的完整 recipe。
3. 在训练未见过的三套真实家庭（厨房与卧室）和 mock 环境上，评估场景数量、数据源消融、VLA 对比和高层推理。
4. 证明跨 embodiment、语义子任务、口头指令和 web 数据分别在不同泛化维度上发挥作用。

## 关键设置

| 项目 | 论文设置 |
|---|---|
| 预训练步数 | 280k gradient steps |
| Post-training 步数 | 80k steps |
| Flow-matching 权重 | α = 10.0 |
| 动作推理 | 10 个 denoising steps |
| 机器人状态/动作维度 | 18 或 19 DoF |
| 相机 | 四相机高层；腕部 + 前向低层 |
| 控制频率 | 50 Hz，action chunking |
| 评测位置数 | 3、12、22、53、82、104（泛化曲线） |
| 标准 mock 评测 | 4 tasks × 10 trials × 多个位置 |

## 结果

- **真实家庭**：在 3 个未见过的真实家庭中执行 items in drawer、dishes in sink、laundry basket 等任务；单个 episode 通常持续约 2–5 分钟，论文还展示了 10–15 分钟的厨房/卧室清理行为。
- **位置扩展**：移动操作训练位置从 3 增加到 104 时，mock-home 平均进度持续上升；104-location 模型在不看测试家庭数据时，接近直接在测试家庭训练的 control。
- **数据源消融**：图 10 中完整 π0.5 的平均任务进度约 79%；去掉 CE 约 52%、去掉 ME 约 54%、同时去掉二者约 40%，差异标注为 `p < 0.001`。去掉 WD 的 mock 任务约 72%，差异 `p = 0.385`，说明 web 数据在该总体指标上影响较小，但对 OOD 物体和高层推理很重要。
- **VLA 对比**：图 12 中完整 π0.5 在 dishes in sink、items in drawer、laundry basket、make bed 四任务上约为 91%、86%、80%、58%，高于 π0-FAST+Flow、π0（300k）和 π0（80k）的对应柱高；图中数值为柱高近似读取。
- **高层推理**：图 13 中完整模型约 79%，implicit HL 约 71%，no HL 约 62%，no VI/no WD 约 60%，GPT-4 high-level 约 58%，human HL 约 63%；作者报告完整模型优于 human HL oracle，且 no VI、no WD 显著下降。
- **语言跟随**：随着训练位置增多，in-distribution 和 out-of-distribution 的 follow rate、success rate 都提高；OOD 对 web 数据更敏感。

## 结论

论文的核心证据链是：增加目标机器人场景可以提升泛化，但“异构数据 + 语义中间层 + 合适的离散/连续动作训练”让有限的 MM 数据发挥出更大范围的迁移效果。作者同时明确指出，陌生抽屉把手、遮挡、易分心的高层子任务、短上下文和简单 prompt 仍是限制。
