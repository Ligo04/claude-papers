# 论文摘要：Astra 作为具身策略

## 1. 背景

传统 VLA 或 policy 通常把视觉/语言映射到动作，低层执行依赖专门训练的策略、IK、PD 或 whole-body controller。GPT-6 Astra 的特殊之处是：它可以直接输出数值机器人命令，并利用执行后的视觉和本体反馈继续决策。论文要回答的不是“语言模型会不会说出动作”，而是：

- 它在哪些具身任务上能产生有效决策？
- 什么时候应该让它直接控制，什么时候应该让它配合已训练控制器？
- 成功来自 Astra 的判断、动作接口，还是下游 policy/controller？
- 推理 token 与 latency 是否允许它进入真实闭环？

## 2. 控制定义

- **Direct**：Astra 独立生成末端位姿、关节目标、抓爪/底盘命令、导航 primitive 或 PD target，再由 analytic control / PD / 离散动作执行。
- **Hybrid**：Astra 审核 learned policy 的动作 proposal，接受 prefix、替换为局部修正，或向冻结 whole-body controller 提供参考/速度/稀疏目标。RoboCasa 还允许重写 subgoal。
- **反馈回路**：observe → Astra 生成或审核 → robot moves → observe。每个研究的观察、动作接口和反馈预算不同，不能把六域数字简单视为同一个 benchmark。

## 3. 实验范围

| 域 | 评测条件 | 论文报告的主结果 |
| --- | --- | --- |
| Gripper manipulation | RoboDojo 10 tasks × 5、RoboLab 10 tasks × 5 | RoboDojo Hybrid 24/50 = 48%，Direct 13/50 = 26%；Hybrid native Score 62.60 vs Direct 37.81。 |
| Dexterous manipulation | 10 个 Sharpa 任务，另有 10 次 DexJoCo 双手经验引导试验，以及 4 个 Allegro in-hand 任务 | Sharpa Hybrid 平均 Score 61.6，π0.5 为 44.2，Direct 为 16.6；DexJoCo 成功 5/10 = 50%。 |
| Mobile manipulation | RoboCasa365 的 15 tasks × 5 episodes，共 75 个 shared starts | π0.5 17/75 = 22.7%，Direct 25/75 = 33.3%，Hybrid 29/75 = 38.7%。 |
| Navigation | R2R、RxR、MP3D、HM3D，各 50 episodes | Astra SR：78%、92%、56%、82%；对应 SPL：65.27、77.25、22.43、43.69。 |
| Locomotion | 一个 matched open-clutter course 上 5 次连续 Astra 尝试；另有无 Astra 的接口诊断 | 5 次均未到达目标；最后一次 30 s 内只前进 2.598 m，末端误差 6.519 m。PASSAGE 13.18 s 到达。 |
| Humanoid loco-manipulation | HumanoidBench 30 tasks；SIMPLE L2 6 tasks × 10 scenes | HumanoidBench 平均 Astra return 581.9，高于 DreamerV3 338.2、TD-MPC2 42.7、SAC −41.9；13/30 task 超过三种 baseline 的最好值。SIMPLE 成功 50/60 = 83.3%。 |

## 4. 关键量化结果

### 4.1 Gripper

RoboDojo 是 paired starts、10 tasks、每个方法 50 instances。Hybrid 的 24/50 成功和 62.60 Score 高于 Direct 的 13/50、37.81，也高于重加权后的 π0.5（15.67%、24.43）。但 Hybrid 的 42,750 个已执行控制步中，36,576 步（85.6%）沿用 π0.5，6,174 步（14.4%）由 Astra 生成或修正。RoboLab 的单臂零样本子集则是 Direct 49/50 = 98%、Hybrid 46/50 = 92%、π0.5 18/50 = 36%，说明 policy prior 是否适配任务会改变排名。

### 4.2 Dexterous

在十任务 Sharpa benchmark 中，Hybrid 比 Direct 在每个任务都高，且比 π0.5 在 8 个任务上高、2 个持平。Hybrid 的平均 Score 61.6 仍是部分完成分数，不是 61.6% binary success。Astra 的修正决策最多用于 missed grasp / dropped object recovery（33.33%），其次是 failed placement recovery（20.90%）；已执行步只有 11.98% 是 Astra correction。

在 DexJoCo 双臂 disk stacking 中，Astra + π0.5 完成 5/10 个 experience-guided trials，执行步中 73.6% 是未修改 policy action、26.4% 是 Astra correction。成功案例显示 Astra 能判断“还没有接触到支撑面”并延迟 release；失败案例显示过度修正也会把物体推离目标。

在 Allegro in-hand rotation 中，Astra 直接控制的 at-goal rate 只有 cylinder 0.51%、cuboid 4.40%，而 RL 为 76.90%、63.50%。翻译任务 Astra 1/5、RL 4/5；translation + rotation Astra 0/5，RL 在 matched endpoint 4/5、完整 15 s 为 5/5。论文把瓶颈归因到 finger contact reconfiguration，而非单纯“抓不住”。

### 4.3 Mobile manipulation

RoboCasa 的 Hybrid 比两个单独条件都高，但 paired outcome 的 Hybrid vs Direct 精确 McNemar p = 0.5235，不能据此宣称显著胜出；Hybrid vs π0.5 的探索性 p = 0.0227，且论文提醒这些 p 值没有校正多重比较，也未处理 task 内依赖和 harness 差异。Hybrid 执行步中 55.2% 接受 policy action，44.8% 是 Astra-authored action；66/75 episodes 使用了 language subgoal。

### 4.4 Navigation

Astra 只看 RGB，不使用 pose 或 map。RxR 的 92% SR、80.42 nDTW、77.25 SPL 是本地固定 50 episodes 的结果；HM3D ObjectNav 的 82% SR 伴随 43.69 SPL。搜索要同时解决“往哪里走”和“什么时候停止”，所以成功率高不代表路径短。论文还保留了单视角与三视角方法的观察差异，不能把所有行当成同协议排行榜。

### 4.5 Locomotion 与 humanoid

Dense motion reference 很难由推理模型一次性产生。五次 Astra sequential attempts 都没到 goal；其中第一次摔倒，后四次接近 no progress 或 time horizon。无 Astra 的 compact-reference 接口诊断也全部 strict success = 0，说明参考表示到执行控制之间本身就可能失败。HumanoidBench 的成绩依赖冻结的 HumanoidGPT/ScaleBFM whole-body controller：Astra 选择任务命令，但全身协调能力部分来自 controller。SIMPLE 的 83.3% 是同样的 Hybrid 系统结果，且 repeated handover 仍显示成功保持脆弱。

### 4.6 资源成本

- RoboDojo 50 instances/condition：Hybrid 624.8M total tokens，Direct 1,132.3M；大部分是 cached input。
- RoboCasa 75 episodes/condition：Direct 7,910、Hybrid 8,941 条 retained-chain model requests；中位数每 episode 85/108。
- Locomotion 最后一次 30 s simulated motion：250 次同步 model calls，平均 39.86 s/call，physics 在推理时暂停；PASSAGE planning 约 0.08 s/call。

## 5. 论文结论的准确表述

最稳妥的表述是：Astra 已能在多种接口上承担任务级选择、纠错和反馈驱动的局部决策；它与强 learned policy 或 whole-body controller 结合时更有用。但论文没有证明 Astra 单独提供了可靠的全身接触控制，也没有证明这些结果满足实时机器人控制的 latency 要求。

