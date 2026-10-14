# 心智模型：一条带责任边界的闭环

## 它是什么问题？

这是一个“通用推理模型能承担多少机器人控制责任”的测量问题，不是一个新型神经网络结构论文。实验把同一个模型放进六类 physical interfaces，再观察：

- 任务理解和空间决策；
- 数值动作构造；
- 对 learned proposal 的审核；
- 对接触和全身动作的持续协调；
- 延迟、token 和重复试验稳定性。

## 一张图

观察 → Astra 生成/审核 → action interface → low-level controller / policy → 物理状态变化 → 新观察

Direct 让 Astra 走“生成”分支；Hybrid 让 Astra 走“审核/修正”分支。越往右，动作越连续、频率越高、误差越容易累积；越往左，语义和阶段决策越重要、但需要强接口把意图翻译成可执行动作。

## 研究地图中的位置

- 上游：VLM/VLA、LLM agent、技能规划、world model。
- 中间层：action proposal review、subgoal rewriting、feedback-guided control。
- 下游：IK、operational-space control、PD、RL policy、whole-body controller、simulator/robot。
- 论文贡献：提供跨域评测和 failure taxonomy，展示推理模型的能力曲线，而不是宣称单一模型替代整条控制栈。

## 阅读时的三个问题

1. 本段结果是 Astra 独立结果，还是 Hybrid 系统结果？
2. 该指标是 binary success、partial Score、native return、path efficiency 还是 token accounting？
3. 物理运动是否在等待 Astra，或由 policy/controller 持续执行？

只要这三个问题没有回答，单看一个百分比就容易得出错误结论。

