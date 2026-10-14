# 心智模型

## 这是什么类型的问题？

这是一个**开放世界移动操作的多模态 imitation learning / hierarchical VLA** 问题。目标不是在固定桌面上把一个物体抓得更准，而是让机器人从自然语言任务出发，在陌生家庭里连续完成一串可验证子任务。

## 需要的先验知识

- Vision-Language Model：图像 patch、文本 token、预训练语义。
- VLA：把观察、指令和 action chunk 放进统一序列。
- Imitation learning：从专家轨迹学习条件动作分布。
- Flow matching：以向量场把噪声动作逐步变成连续动作。
- Hierarchical control：高层 subtask 负责“做什么”，低层 policy 负责“怎么做”。
- Embodiment transfer：不同机器人需要统一控制语义和动作尺度。

## 在研究地图中的位置

```text
Web/VLM 语义知识 ─┐
其他机器人动作 ───┼─> 统一多模态序列模型 ─> 高层语义子任务
目标移动机器人 ───┘                         └> 低层连续 action chunk
                                                     └> 50 Hz PD 控制
```

它连接了三条路线：

1. **Generalist robot policy**：用跨任务、跨环境数据获得更宽的技能覆盖。
2. **Language-conditioned planning**：用语言/语义子任务缩短长时程 credit assignment。
3. **Expressive action decoding**：用 FAST token 获得训练效率，用 flow expert 获得连续实时动作。

## 如何在脑中分类

把 π0.5 看成一个“共享 trunk + 两种动作头 + 可选高层文本头”的系统：

- 共享 trunk 吸收 web、机器人和语义数据。
- 文本头输出子任务或答案。
- FAST 路径负责预训练阶段的离散 action token。
- action expert 负责 post-training/部署阶段的连续 chunk。
- 评测核心不是某个 loss，而是新环境中的任务进度、语言跟随和长时程稳定性。

## 最容易误读的地方

- “end-to-end” 不等于没有层级：π0.5 仍在同一模型内部显式生成高层子任务。
- “97.6% 非移动机器人数据” 不等于目标数据无关；400 小时 MM 是把迁移落到移动操作上的锚点。
- “优于 human HL” 是论文特定 rubric 的结果，不是机器人整体能力优于人类。
- “新家”只表示评测位置不在训练位置，不能外推到所有开放世界场景。
