# 自测题

## 基础题（5）

### π0.5 解决的核心问题是什么？

<details>
<summary>答案</summary>

让移动操作机器人在训练未见过的家庭环境中，依据高层自然语言命令完成厨房/卧室里的长时程、多阶段和灵巧操作任务。

</details>

---

### π0.5 的两个训练阶段分别做什么？

<details>
<summary>答案</summary>

预训练把异构数据统一成离散 token 的 next-token prediction；post-training 面向移动操作专门化，并加入 flow-matching action expert 来生成连续动作块。

</details>

---

### MM、ME、CE、HL、WD、VI 分别代表什么？

<details>
<summary>答案</summary>

MM 是多环境移动操作数据，ME 是多环境非移动机器人数据，CE 是实验室跨 embodiment 数据，HL 是高层子任务预测，WD 是 web 多模态数据，VI 是专家用语言逐步指导机器人的示范。

</details>

---

### 运行时为什么要先预测 high-level subtask？

<details>
<summary>答案</summary>

它把“clean the kitchen”这类长命令压缩成当前可执行的短目标，如“pick up the plate”，为低层 action chunk 提供条件，降低长时程任务的分解难度。

</details>

---

### 论文的目标机器人控制频率是多少？

<details>
<summary>答案</summary>

目标以 50 Hz 送给简单 PD controller；action chunking 用于把模型输出和控制循环连接起来。

</details>

---

## 中级题（5）

### 为什么预训练使用 FAST 离散 action token，而部署阶段使用 flow matching？

<details>
<summary>答案</summary>

FAST token 适合高效的自回归序列训练，能把动作和文本/视觉任务放进同一个目标；flow matching 能以少量 denoising steps 生成细粒度连续动作，更适合实时控制。两阶段组合了训练效率和部署表达能力。

</details>

---

### 联合损失中的 α 控制什么？论文使用多少？

<details>
<summary>答案</summary>

α 平衡文本/离散 token 的 cross-entropy 与连续动作 flow-matching MSE；post-training 使用 α = 10.0。

</details>

---

### 为什么要对不同机器人动作做分位数归一化和 zero-padding？

<details>
<summary>答案</summary>

分位数归一化把不同数据集的动作尺度压到可比较的 `[-1,1]`，zero-padding 则把不同 DoF 的动作向量放进统一输出维度；同时用 control-mode 标记区分 joint 与 end-effector 目标。

</details>

---

### 图 10 说明了什么？

<details>
<summary>答案</summary>

完整 π0.5 约 79% 平均任务进度；去掉 CE 或 ME 会降到约 52%/54%，同时去掉二者约 40%，且差异显著。交叉 embodiment 数据对 mock-home 总体任务进度很关键；WD 对这个总体指标影响较小，但对 OOD 物体和高层推理仍重要。

</details>

---

### implicit HL 为什么有意义？

<details>
<summary>答案</summary>

implicit HL 在训练中保留高层子任务数据，但运行时不显式生成子任务。它仍明显优于 no HL，说明语义监督会塑造共享表征；显式高层推理是在此基础上的额外收益。

</details>

---

## 高级题（5）

### π0.5 的统一分布分解如何避免高层命令直接泄漏到低层动作？

<details>
<summary>答案</summary>

模型把联合分布分解为 `π(a|o, l̂)π(l̂|o,l)`：高层输出 `l̂` 先由总体命令 `l` 和观察决定，低层动作只以观察和预测的 `l̂` 为条件。实现上还用 attention matrix 控制 token 可见性，并让 action token 使用独立 expert 路径。

</details>

---

### 如果只增加 MM 数据而去掉 CE/ME，为什么不能等价复现开放世界泛化？

<details>
<summary>答案</summary>

MM 提供目标 embodiment 的直接经验，但环境和任务覆盖有限。CE/ME 提供不同机器人、更多场景和更多操作语义；消融显示去掉它们会显著降低 mock-home 任务进度，因此“同平台更多轨迹”不能替代跨 embodiment/跨环境迁移。

</details>

---

### web 数据在图 10 和图 11/13 中的作用为什么看起来不同？

<details>
<summary>答案</summary>

图 10 的指标是四个完整 mock-home 任务的平均进度，WD 去掉后的差异不显著；图 11 专门看语言跟随和 OOD 物体，图 13 看高层策略，WD 的语义和物体知识更直接，所以影响明显。不同评测测量的是不同泛化轴。

</details>

---

### “优于 human HL oracle”应如何解读，不能如何解读？

<details>
<summary>答案</summary>

它表示在论文定义的 high-level policy 实验、固定任务词表和低层 policy 条件下，π0.5 的平均任务进度高于人类提供的高层子任务序列。不能解读成机器人整体能力超过人类，也不能脱离该 rubric 比较一般规划能力。

</details>

---

### 设计一个有说服力的复现实验，需要至少控制哪些变量？

<details>
<summary>答案</summary>

固定测试家庭和物体、每 task 的 trial 数、动作/相机接口、位置拆分、训练 steps 与数据样本量；单独报告 token loss、离线 action 误差、mock rubric 和真实闭环结果；对图中近似值不冒充原始 trial 数据，并记录硬件与控制器差异。

</details>

---
