# Code Demos — SCORE

两个自包含、可运行的概念 demo，帮你直观感受 SCORE 的核心机制。**只依赖 numpy**（第一个可选 matplotlib 出图），CPU 秒级可跑。

## 1. `support_vs_distributional_demo.py` — 三范式对比

复刻论文 Fig.2 "灯泡"玩具例子的核心直觉，用一维"选模式"问题揭示：
- **无约束 RL** → 钻仿真漏洞 a_hack，仿真满分、真机崩溃（reward hacking）
- **分布约束 RL** → 提升 vs 迁移的权衡（松则碰漏洞，紧则被慢模式拖住）
- **支撑集约束 RL（SCORE）** → 概率全压到"快好模式"，既高真机回报又零漏洞

```bash
python support_vs_distributional_demo.py           # 打印数值对比
python support_vs_distributional_demo.py --plot    # 额外存对比图（需 matplotlib）
```

**预期输出**：支撑集约束真机回报 1.00 / 碰漏洞 0%；无约束真机 0.00 / 碰漏洞 100%；分布约束居中且无法两全。

## 2. `flow_steering_demo.py` — flow steering 免费实现支撑集约束

用 numpy **手写一个迷你 flow matching 基座策略**（学"快 +1 / 慢 -1"双模态），然后演示 SCORE 的关键操作：**冻结 flow，只在隐噪声 z 上 steer**。

- 证明"只调 z ⇒ 动作恒 ∈ A_base(o)"（支撑集约束由 flow 前向结构**自动保证**）
- steering 学会永远挑快模式：真机回报 0.15 → 0.97，且**从不越界**到漏洞动作
- 对比无约束 RL 直接优化动作 → 跑到支撑集外、真机崩溃

```bash
python flow_steering_demo.py
```

**预期输出**：基座未 steer 真机回报 ≈0.15 → SCORE steering ≈0.97（100% 落在快模式，动作范围仍在支撑集内）；无约束 RL 真机回报 0.00。

## 说明

这些 demo **刻意简化**（1D、固定观测、用 softmax 重加权代替 PPO、用 numpy 代替 torch），目的是让核心概念**可读、可验证、零环境依赖**。它们不复现真实的灵巧操作/仿真——那需要 Franka FR3 + LEAP Hand + Isaac 类仿真 + DSRL/RFS 实现（见论文附录 D 与项目页 https://weirdlabuw.github.io/score/）。

抓住两个直觉即可：
1. **支撑集 ≠ 分布**：约束"能用哪些动作"而非"概率怎么分"。
2. **flow 的隐空间是免费的支撑集旋钮**：只调 z 就把 RL 关进"基座可生成集合"这个安全牢笼里。
