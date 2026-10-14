# Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies

> arXiv:2609.38537v1 · Galbot Team · 2026-09-29 · cs.RO

这篇论文把 GPT-6 Astra 放进六类具身任务中，系统测量它能否从视觉、本体状态和反馈中产生数值机器人动作。论文同时比较两种职责分配：Direct 让 Astra 直接给目标或动作，Hybrid 让 Astra 审核、接受或修正已训练策略/whole-body controller 的提议。核心结论是，Astra 已经能做有价值的任务判断和局部修正，但“判断正确”还没有等价为“稳定完成物理控制”；接触重构、长时间运动、推理延迟和接口校准仍是硬边界。

## 难度与阅读方式

- 难度：Advanced。
- 性质：跨仿真 benchmark 的 empirical-heavy system evaluation。
- 建议顺序：先读本页和 summary.md，再读 method.md；用 insights.md 理解论文真正的因果边界，最后用 qa.md 自测。
- 预计时间：第一次通读 60–90 分钟；结合表格和 demo 复习约 2 小时。

## 先记住的五件事

1. Direct 与 Hybrid 测量的是不同的控制责任，Hybrid 的成功不能直接写成 Astra 独立能力。
2. RoboDojo 中 Hybrid 达到 48% 成功率、native Score 62.60；50 条轨迹里 85.6% 的执行步仍由 π0.5 完成，Astra 只改动 14.4%。
3. 导航是最强区域之一：固定 50 集合上 RxR 成功率 92%、HM3D ObjectNav 82%，但 HM3D 平均路径效率 SPL 只有 43.69。
4. 直接灵巧手控制暴露了接触重构问题：Astra 保得住部分抓握，却很难让手指重新协调以持续旋转或完成转移。
5. 物理时钟和推理时钟必须分开：一次 30 秒 locomotion 运行需要 250 次模型调用，平均每次 39.86 秒，推理期间 physics 暂停。

## 文件结构

- README.md：导航、结论和阅读路线。
- summary.md：背景、问题、六域实验和主要数字。
- insights.md：核心机制、取舍、边界和与先前范式的比较。
- method.md：Direct/Hybrid 控制路径、反馈闭环、伪代码和复现风险。
- mental-model.md：把这篇工作放进具身策略研究地图。
- reflection.md：可延伸问题和脆弱假设。
- qa.md：5 个基础、5 个中级、5 个高级问题。
- code/closed_loop_control_demo.py：只用 Python 标准库的教学 demo。
- index.html：无外部依赖的交互式论文导览。
- paper.pdf、paper.txt：下载并抽取的 v1 论文。
- images/：按论文页抽取的 101 个嵌入图像。
- sources.md、source-status.md、meta.json：来源、证据等级和元数据。

