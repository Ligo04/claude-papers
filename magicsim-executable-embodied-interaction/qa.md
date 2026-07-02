# 自测题(15 题)

> 5 基础 / 5 进阶 / 5 高级。先自己想,再展开答案。

## 基础题(Basic)

### Q1. MagicSim 一句话定位是什么?它和 Isaac Sim 是竞争关系吗?

<details>
<summary>答案</summary>

MagicSim 是一个**具身交互基础设施(embodied interaction infrastructure)**,围绕"一个确定性批处理运行时 + 一个共享 MDP + 三个驱动器"构建。它**不是**新物理引擎或渲染器,而是**建立在 Isaac Sim / IsaacLab 之上的 manager 中心编排层**,负责确定性世界构建、异构对象、planner-in-the-loop 执行、同步标注、数据采集、回放与服务。因此它与 Isaac Sim 是**互补**而非竞争。

</details>

---

### Q2. 副标题"One Runtime, One MDP, Three Drivers"里的三个 driver 分别是什么?

<details>
<summary>答案</summary>

1. **RL 训练**(benchmark / RL 评测)——policy 直接 step `TaskBaseEnv`(current)。
2. **脚本 AutoCollect**(数据采集)——经 `Command→Skill→Planner→Robot→Record` 流水线生成成功门控轨迹(current)。
3. **Inference / Replay**(Agent 推理 / 回放)——用 policy 或回放动作流 step 同一 MDP,走 RobotManager 动作空间(**planned**,规划中)。

三者 step 的是**同一个任务 MDP**,而不是三个不同环境。

</details>

---

### Q3. 论文反复强调"设计单元是 episode 而不是 frame",这是什么意思?

<details>
<summary>答案</summary>

一个 **frame(帧)** 只记录"长什么样"(像素),无论交互是否成功都能产出。一个 **episode(交互片段)** 保留交互的**因果结构**:可复现的初始条件、任务成功状态、规划与技能历史、grounded affordance(在哪儿抓的)、以及**只有成功才保留**的轨迹。forward-only 渲染管线会丢弃这些交互依赖的状态,所以无法复现/评测/标注同一个 episode。

</details>

---

### Q4. 什么是"魔法动作(magic action)"?MagicSim 如何避免它?

<details>
<summary>答案</summary>

"魔法动作"指高层指令**不经过真实机器人执行,直接把仿真器状态编辑/瞬移到目标状态**(常见于高层 Agent 评测)。MagicSim 通过 `Command→Skill→Planner→Robot→Record` 流水线避免:高层指令实例化 task command → 推进 AtomicSkill 状态机 → 提交规划请求 → 执行机器人动作 → 观测 → 更新任务状态。运行时**只记录由 grounded 机器人动作产生的数据**,因此数据物理可实现。

</details>

---

### Q5. MagicSim 的"配置优于代码(configuration over code)"原则体现在哪?

<details>
<summary>答案</summary>

场景、布局、资产、机器人、相机、任务、规划器、技能、随机化、录制选项全部通过 **YAML-style 结构化规范**暴露。好处:① 研究者无需重写仿真器内部就能构建/修改环境;② **编码 Agent 可以程序化地检查、生成、编辑**这些规范。规范刻意解耦"有什么 / 放在哪 / 怎么行为 / 怎么暴露"四件事。

</details>

---

## 进阶题(Intermediate)

### Q6. "物理时间同步,语义时间异步"具体指什么?为什么重要?

<details>
<summary>答案</summary>

**物理时间同步**:所有活跃子环境跨越**同一个批处理 `sim_step()` 屏障**——它们共享同一个物理 tick。**语义时间异步**:每个环境的语义状态(采样指令 / AtomicSkill 阶段 / GlobalPlanner 动作 / 成功判定 / 录制 / reset)**独立推进**——任意时刻不同环境处于不同控制阶段。

重要性:planner-backed 行为**不是常数时间**。若每个规划请求都同步在主循环求解,一个慢环境会冻结整个 batch。异步让单个慢规划不拖垮 batch,同时把请求聚合成更大批量求解。

</details>

---

### Q7. submit-and-poll 契约是怎么工作的?它解决什么问题?

<details>
<summary>答案</summary>

技能或 global planner **提交(submit)** 一个规划请求 → 立即收到一个 **future**(非阻塞)→ 批处理仿真**继续步进**其它环境 → 当 future **就绪(poll done)** 时再**消费(consume)** 结果并执行。底层由 cuRobo 实现为**异步微批求解农场**。

解决的问题:① 仿真主循环**永不被慢解阻塞**;② per-agent 的零散请求被**聚合成更大的批量 GPU 求解**(提高利用率)。代价:future 未就绪时该环境要执行"保持姿态 / 安全动作"。

</details>

---

### Q8. MDP 边界(`TaskBaseEnv`)在系统里扮演什么角色?为什么说运行时拓扑是"刻意非对称"的?

<details>
<summary>答案</summary>

**MDP 边界**(`TaskBaseEnv` 包裹 `SyncRobotEnv`)定义 obs/action/reward/termination/success。它是**不同研究模式的会合点**:边界**之下**复用同一世界/机器人/传感器/规划器/确定性;边界**之上**挂 RL 策略、脚本采集栈、或策略/回放 driver。

**非对称**:RL 训练是直接 step `TaskBaseEnv` 的外部循环;脚本采集是在同一任务**之上叠 wrapper**(`AsyncRobotEnv` 加技能/规划,`AutoCollectEnv` 加采样+录制);Inference/Replay 走 RobotManager 动作空间、**跳过** Collect 层的 AtomicSkill/GlobalPlanner。三条路径结构不同,但共享同一 MDP。

</details>

---

### Q9. 论文说 driver 和 exposure 是"两条正交的轴",请解释并举例。

<details>
<summary>答案</summary>

- **Driver(怎么 step MDP)**:RL training / AutoCollect / Inference-Replay。
- **Exposure(episode 怎么被使用)**:benchmark rollout / dataset trajectory / agent-facing interaction。

二者正交:例如同一个 AutoCollect driver 产出的 episode 既可作 dataset trajectory(供模仿学习),其成功信号又同时是 MDP 的评测信号(benchmark)。把两轴分开,系统就**不会退化成一堆专用管线**——driver 变,批处理运行时不变。

</details>

---

### Q10. "可复现性是运行时契约而非日志工具"包含哪些具体机制?

<details>
<summary>答案</summary>

包括:**确定性 reset 顺序**、**不相交的种子化随机流(disjoint seeded randomization streams)**、**状态快照(snapshot)**、**`reset_to` 回放**、以及 **P1/P2/P3 动态对象生命周期处理**。这些被设计为运行时契约的一部分,而不是事后加的日志工具。这样许多并行世界能从相同初始条件 reset、快照、回放。

</details>

---

## 高级题(Advanced)

### Q11. 为什么"成功门控(success-gating)"既是优点也是潜在偏差来源?

<details>
<summary>答案</summary>

**优点**:任务成功信号扮演双重角色——MDP 评测信号 + 数据写入门控。只保留成功 episode 保证数据集质量(无失败噪声),且数据天然对齐语言/动作/视觉/任务状态。

**偏差**:失败 rollout 被 flush 丢弃,数据集**没有失败 / 恢复样本**。对于需要鲁棒性(从失败中恢复)、需要负样本、或需要建模失败分布的下游学习,这是信息损失。复现鲁棒策略时需额外设计负样本/失败轨迹采集。

</details>

---

### Q12. 如果你要在 MagicSim 上训练一个高层 VLM 规划器做闭环 RL,论文当前支持到什么程度?有哪些缺口?

<details>
<summary>答案</summary>

论文把这定位为**接口的"计划用途(planned use)"而非已完成能力**:
- **当前可用**:同样的异步、批处理、agent-addressable 运行时,理论上能让 VLM 对许多并行世界发出 task/skill 指令,而不只是回放预录轨迹。
- **缺口**:① Inference/Replay driver 与 Agent/VLM serving 接口仍是 **planned**;② 当前 VLM layout hook 只是一个**集成点(integration point §14)**;③ 高层 planner 的闭环 RL 后训练(verl / RLinf)是**计划中**的用途。所以现在不能假设 VLM 闭环 RL 开箱即用。

</details>

---

### Q13. 微批窗口(microbatch window)大小是一个关键调参旋钮,它在权衡什么?

<details>
<summary>答案</summary>

异步规划服务把多个 env 的规划请求聚合进一个微批窗口再交给 cuRobo 批量求解。
- **窗口太小**:聚合的请求少 → GPU 求解利用率低,失去批量化收益。
- **窗口太大**:等待聚合的时间长 → future 就绪延迟高 → 提交请求的环境长时间处于"保持姿态/安全动作"挂起态,语义延迟上升,可能影响紧耦合反馈任务的行为质量。

本质是**吞吐(GPU 利用率)与实时性(语义延迟)**的权衡。

</details>

---

### Q14. 多子环境共享一个 Isaac Sim 进程,相比"一 agent 一进程"有何利弊?MagicSim 如何缓解弊端?

<details>
<summary>答案</summary>

**利**:不必为每个 agent / 规划请求 / 任务尝试开一个 Isaac Sim 进程;许多 agent、任务、规划请求、轨迹缓冲**共存于一个批处理进程**,物理/渲染批处理,GPU 利用率与吞吐大幅提升,这是论文的**核心系统贡献**。

**弊**:共享 stage 的隔离性差,一个 env 的坏状态可能影响共享 stage;单进程也是单点。

**缓解**:① 每个 **manager 拥有 per-env 的状态切片**(world/command/planner future/task status/record buffer 都 per-env 独立);② reset/replay/snapshot 在选定 `env_ids` 上操作;③ 多进程仍可作为**外层 serving / 集群扩展层**——核心贡献在单批处理运行时内部,外层扩展正交。

</details>

---

### Q15. 仿真族数量被写成变量 `K=|F|`(当前 K=15),这个设计选择反映了什么架构哲学?如果要新增一个"烟雾-布料"耦合族,架构上需要改什么?

<details>
<summary>答案</summary>

**哲学**:论文刻意把 `K` 写成变量而非硬编码本体(ontology),表达"**仿真族是当前实现的属性,不是引擎定义的一部分**"——引擎抽象用集合 `F` 表述,新族可加入而**不改架构**。这是"capability taxonomy(能力分类)而非内部 registry 镜像"的设计:对用户呈现统一仿真表面,内部由多个 manager + 专用 backend 实现。

**新增耦合族需要**:① 定义新族 `f_{K+1}` 的求解器表示与 `θ_solver`(方法特定参数,如烟雾的源位姿/方向/密度 + 布料的约束/刚度);② 把它接入参数化构造路径 `x_e=(a_e,f(e),T_e,s_e,θ_vis,θ_phys,θ_solver,h_e)`,提供导入/生成、材质绑定、观测 hook;③ 实现与现有族的**物理耦合**接口(跨求解器边界的相互作用),并接入 manager 的 reset/seed/replay 契约。**不需要**改 MDP 边界、driver 或运行时拓扑——这正是该设计的目的。

</details>

---
