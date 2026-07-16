# Mental Model — 如何在脑中给 SCORE 归位

## 1. 这是什么类型的问题？

**核心问题类型：约束型策略优化（Constrained Policy Optimization）在分布偏移下的实例。**

更具体地，它坐落在三个领域的交叉点：

```
        Sim-to-Real 迁移
              ╱  ╲
             ╱    ╲
   生成式策略 ──── Offline/支撑集约束 RL
   (flow/diffusion)      (support constraint)
              ╲    ╱
               ╲  ╱
          灵巧操作 (dexterous manip)
```

- 从 **sim-to-real** 看：它是 real→sim→real，把仿真当"提升器"而非"训练器"。
- 从 **offline RL** 看：它把"限制到数据支撑集内以防 value 外推"的思想搬到**在线仿真**里（online interaction available, but only in sim）。
- 从 **生成式策略** 看：它利用 flow/diffusion 的隐空间作为可控旋钮，做"不改权重的策略提升"。

## 2. 假设的先验知识（Prerequisites）

要顺畅读懂，你需要：

| 概念 | 为什么需要 | 最小够用理解 |
|------|-----------|-------------|
| **Flow matching** [Lipman 2023] | 基座策略类；隐噪声→动作映射是全篇机制核心 | 学一个速度场把高斯噪声输运到目标分布；采样 = 从噪声积分 ODE |
| **MDP / 策略梯度 / PPO** | 优化框架 | 状态-动作-奖励；PPO 是稳定的 on-policy 策略梯度 |
| **Reward hacking / reality gap** | 论文要解决的失败模式 | 仿真高分真机崩，因为钻了 sim-real 差异 |
| **支撑集 vs 分布** | 全篇最核心的概念区分 | support = "可能取到的值的集合"；distribution = "概率如何分配" |
| **KL / JS / Wasserstein 散度** | 分布约束基线的统一视角 | 都是"分布间距离"，对应 BC/GAIL/bounded-residual |
| **Offline RL 支撑集约束** [Singh 2022 等] | 思想源头 | 别让策略跑到数据没覆盖的动作上 |
| **DSRL [11] / RFS [12]** | 两个 steering 实例 | DSRL 纯调噪声；RFS 加残差 |
| **灵巧操作 / 点云 / PointNet** | 应用域与观测 | 多指手高维接触密集控制 |

如果只想抓思想，**support vs distribution** + **flow 隐空间可 steer** 这两点就够了。

## 3. 它在研究地图中的位置

### 上游（它站在谁的肩膀上）
- **Flow matching / Diffusion Policy** [4,5,6,7]：提供生成式基座策略类。
- **DSRL [11]、RFS [12]、REFORM [33]**：flow steering 的算法工具（原本为真实在线 RL 设计，SCORE 把它们搬进仿真）。
- **RialTo [3]**：real-to-sim-to-real + BC 正则的直接前身/对照。
- **Offline RL 支撑集约束** [31,32,33]：概念源头。
- **ExpertGen [24]、Su et al.[12]**：从行为先验出发做仿真优化的近邻工作。

### 下游 / 它开启了什么
- 把"数据集设计"从"追求最优演示"转向"追求最广覆盖"——一个新的数据工程方向。
- 大规模多任务预训练 + steering 的组合（作者明说的 future work）。
- "冻结大策略、只在隐空间做 RL" 作为部署后安全提升的通用范式。

### 同期竞争 / 对照
- **无约束**：FPO [9]（flow RL SOTA，正文反例）。
- **分布约束**：RialTo（KL）、Residual-RL（Wasserstein-1）。
- **支撑集约束**：SCORE-DSRL（硬）、SCORE=RFS（软）。

## 4. 如何归类记忆（One-liner Categorization）

> **"DeepMimic/AMP 用分布约束把 RL 拽向先验；SCORE 反其道，用支撑集约束把 RL 关进先验的'可能动作牢笼'里，然后在牢笼内自由选最优——牢笼由 flow 的隐空间免费提供。"**

或者更短：**"Frozen flow policy + RL on the noise = 提升与迁移的解耦。"**

## 5. 常见误解澄清（Misconception Guards）

- ❌ "SCORE 在真机上做 RL" → ✅ **全程在仿真做 RL，真机零额外交互**（真机只用来采初始演示 + 最终评估）。
- ❌ "SCORE 微调/蒸馏了基座策略" → ✅ **基座权重从不更新**，只在其隐输入上加一个 steering 头。
- ❌ "支撑集约束 = 更紧的分布约束" → ✅ 两者**正交**：支撑集管"能用哪些动作"，分布管"概率怎么分"。支撑集约束在集合内**完全放开**概率分配自由。
- ❌ "更好的演示数据 → 更好的 SCORE" → ✅ 是**更广覆盖**（含失败/次优数据）→ 更好的 SCORE；单纯"更优"演示对 SCORE 增益有限。
- ❌ "SCORE 能学出基座做不到的新技能" → ✅ 它只能**放大/组合基座已支持的模式**，OOD 新能力它变不出来（这正是其安全性的代价）。

## 6. 一张"何时该想到 SCORE"的决策卡

```
你有一个真机生成式策略（flow/diffusion），能用但慢/糙/脆？
        │ 是
        ▼
你能搭一个够用的仿真（真实扫描 + 域随机化）？
        │ 是
        ▼
真机 RL 太贵/太危险，但你想安全提升、且不敢重训基座？
        │ 是
        ▼
   ★ 用 SCORE：冻结基座，只在噪声空间 PPO ★
        │ 否（需要全新技能 / 无仿真 / 无生成式基座）
        ▼
   考虑别的路线（真机 RL / 重新采数据 / 蒸馏式 sim 专家）
```
