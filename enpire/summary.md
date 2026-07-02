# Summary：ENPIRE 速览

## 背景上下文

真实世界灵巧机器人操作（dexterous manipulation）长期受制于两个瓶颈：

1. **人类监督密集**：复位场景、判断成败、调参、写训练代码都需要研究员亲力亲为。
2. **算法工程量大**：从 heuristic policy、行为克隆（BC）、离线/在线 RL 到工具调用，每条路线都需大量试错与基础设施搭建。

与此同时，**编程智能体（coding agent）**——如 Codex、Claude Code、Kimi Code——已能在纯数字环境（写代码、跑单元测试、调 bug）中自主完成"算法搜索"。但一旦把目标换成真实机器人，这些智能体就"够不着"了：它们没法复位桌上的零件，没法判断扎带是否真的穿过了卡头。

> ENPIRE 的判断：**缺失的抽象是一个可重复的真实世界策略改进反馈闭环**。

## 问题陈述

如何把"让真实机器人策略变好"这件事，转变成一个**编程智能体可以自主管理、可优化、可公平消融**的过程，而最小化人类介入？

具体子问题：
- 怎么让任务"自复位 + 自验证"，使智能体可以反复调用？
- 怎么把策略改进（PI）的多种范式（启发式、BC、离线/在线 RL、工具调用）统一成一个智能体可编排的接口？
- 怎么用多机器人机队并行加速，并度量其效率？
- 怎么公平比较不同智能体、不同训练 recipe？

## 主要贡献

1. **ENPIRE 框架**：四模块闭环 EN-PI-R-E，把真实世界机器人学习变成智能体可管理的可控优化过程。
2. **真机自主达到 99% pass@8**：在前沿编程智能体驱动下，PushT、Pin Insertion、GPU 插拔、扎带切割等灵巧任务几乎全自动达到高成功率。
3. **多 PI 范式统一**：智能体可在 heuristic learning、tool calling、BC、offline/online RL 之间自主选择与切换。
4. **机器人机队加速**：提出并行多机器人 rollout，并给出 scaling 行为分析。
5. **两个新效率指标**：
   - **MRU（Mean Robot Utilization）**：机器人平均利用率
   - **MTU（Mean Token Utilization）**：token 平均吞吐
6. **AutoEnvBench 评测协议**：跟踪智能体在 wall-clock 时间上的研究进度（而非仅看最终策略成败），横向比较 Codex GPT-5.5、Claude Code Opus 4.7、Kimi Code K2.6。
7. **仿真验证（RoboCasa）**：把"智能体研究行为"与"真机硬件吞吐"解耦，验证 recipe 迁移性。

## 关键结果（定性）

- 三个前沿编程智能体在 ENPIRE 闭环内都能持续改进策略，**team-avg 成功率随研究 wall-clock 时间单调爬升**。
- 论文展示了"**idea tree**"（假设 git-tree）：每条分支对应一个智能体，每个节点对应一个尝试过的想法；绿色环标记提升了 team-avg 的想法，绿色曲线标记跨智能体的灵感借鉴。
- 典型提升点示例（来自项目页 idea tree）：
  - `I37 BC regularization`：**+10.8 pp**
  - "Re-evaluate"：**+12.5 pp**
  - `I16 Online RL mix Demo`：+3.8 pp
  - `I76 Compensate controller`：+1.3 pp
  - `I66 Tune batch size 1024→512`：+0.9 pp
  - `I56 Tweak BC term weight`：+0.4 pp
- 机队 scaling：1/4/8 智能体团队对比；更大机队**更快到达成功**，但**总 token 预算随规模上升**。
- Push-T 阶段式进度（fleet 设置下，Codex 视觉消融）：time-to-first-move / range-ok / orientation-ok / first-success，对比 native vision / VLM-as-vision-tool / GoFE vision 三种视觉方案。

## 定量指标

| 指标 | 数值 / 说明 |
|------|------------|
| 真机 pass@8 成功率 | **99%**（PushT、Pin Insertion、GPU Insertion、Tie/Cut Zip-tie） |
| pass@8 含义 | 单次长程 rollout 内最多 8 次上下文重试（条件化于前次失败），**非 best-of-8 i.i.d. 采样** |
| pass@1 vs pass@8 | 13% 策略重试后仍 ~13%（说明重试只在策略可恢复时才有效；爬到 99% 本身是能力） |
| 比较智能体 | Codex GPT-5.5 / Claude Code Opus 4.7 / Kimi Code Kimi K2.6 |
| 机队规模 | 1 / 4 / 8 智能体团队 |
| 新指标 | MRU（机器人利用率）、MTU（token 吞吐）、Token-to-Success、Time-to-Success |
| 仿真平台 | RoboCasa（含 Coffee Setup Mug、Open Cabinet、Open Drawer、Open Stand Mixer、Counter→Cabinet、Sink→Counter、Turn Off Stove、Turn On Sink 等任务） |
| Push-T 仓库 | `huggingface/gym-pusht`（智能体被要求写无神经网络的启发式策略，且不得改环境代码——"No cheating"） |

## 一句话总结

> ENPIRE 把"做机器人研究"本身变成了一个 AI 编程智能体可以自主跑通的闭环优化问题，靠的不是更强的策略模型，而是**一个可被智能体反复调用的真实世界反馈闭环（EN-PI-R-E）**，并用 MRU/MTU 度量这条"物理自研究"路线的效率与扩展性。
