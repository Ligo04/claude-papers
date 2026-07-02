# 代码 Demo · Embodied-R1.5

三个**纯标准库、零依赖、可独立运行**的教学 demo，对应论文最核心的三个机制。

| 文件 | 对应论文 | 演示什么 |
|------|---------|---------|
| `multitask_balanced_rl_demo.py` | §4.2.1 | **多任务平衡 RL 配方**的优势归一化：① 分母用 σ_batch 而非 σ_group → 让梯度量级∝学习信号、中等难度主导；② 分子用 μ_group 而非 μ_batch → 组内零中心化的正确 baseline；③ 退化组动态过滤 |
| `reward_functions_demo.py` | §4.2.2 | **五类奖励函数** + 统一**分段线性衰减** φ(d; τ_p, τ_z)（ASCII 曲线）+ 防 reward-hacking 护栏 + 格式奖励加权 |
| `pgc_closed_loop_demo.py` | §5 | **PGC 闭环框架**状态机：单模型异步扮演 Planner/Grounder/Corrector，复现 Figure 3 "做奶茶"中"按开关首次失败→自主重试成功" |

## 运行

```bash
python multitask_balanced_rl_demo.py   # 多任务平衡 RL
python reward_functions_demo.py        # 五类奖励函数
python pgc_closed_loop_demo.py         # PGC 闭环
```

无需 GPU、无需 pip install。Python 3.8+ 即可。

## 重要说明

这些 demo 旨在**讲清机制与控制流**，而非复现训练。其中：
- RL demo 用合成奖励数据，展示归一化公式的**行为差异**，不是真实策略梯度训练。
- 奖励 demo 的"语义相似度"用词重叠模拟（真实系统用 Skywork-Reward 奖励模型）。
- PGC demo 的 `plan()/ground()/correct()` 是**脚本化的桩**，用于展示框架调度逻辑；真实系统中它们都是 Embodied-R1.5 的模型前向。

配合 [`../method.md`](../method.md) 与 [`../insights.md`](../insights.md) 阅读效果最佳。
