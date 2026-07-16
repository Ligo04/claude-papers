# SCORE：支撑集约束 RL，无需真实世界经验即可提升真实策略

> **Support-Constrained RL Enables Real-World Policy Improvement without Real-World Experience**
> Raymond Yu\*, William Huey\*, Mustafa Mukadam, Anusha Nagabandi, Abhishek Gupta
> University of Washington · Amazon FAR · arXiv:2606.27475v1 (2026-06-25)
> 项目主页：https://weirdlabuw.github.io/score/

---

## 这篇论文在讲什么（一段话）

真实世界里用模仿学习训出来的机器人策略往往**慢、不精确、抗扰动差**。想用强化学习（RL）去打磨它，但直接在硬件上训练又贵又危险。改在**仿真里**提升策略虽然便宜，可无约束的 RL 会去"钻"仿真与现实之间的接触/动力学漏洞（reward hacking），学出一堆没法迁移到真机的危险动作；而传统的**分布约束**（如 BC 正则、KL 散度）又会把策略死死拽在那个"不完美的先验"附近，限制了提升空间。SCORE 的关键洞见是：应当把仿真里的策略提升**约束到真实基座策略的"支撑集"（support）**上——即基座策略"有可能生成"的动作集合内，而不是约束到它的"分布"上。这样 RL 可以在先验支持的动作里挑出**更快、更精确、更鲁棒**的模式并放大它，同时抑制那些不可迁移的臆造动作。工程上，SCORE 冻结真实世界的 flow matching 基座策略，只在它的隐噪声（latent noise）空间上学一个轻量 **steering 策略**，用稀疏奖励 + PPO 非对称 actor-critic 在仿真里训练，最后**直接部署**、无需蒸馏或真机微调。

## 难度等级

**Advanced（进阶）** —— 需要 flow matching / 生成式策略、sim-to-real、offline/online RL（支撑集约束）、PPO、灵巧操作等背景。数学主体不重（严格分析在附录 E），但概念交叉多。

## 一句话核心贡献

把"real-to-sim-to-real 策略提升"从**分布约束**范式切换到**支撑集约束**范式，并用 **flow steering**（只调 flow 策略的隐噪声）给出一个简单、免蒸馏、可直接部署的落地方案，在 8 个真机灵巧操作任务上把成功率从 37.8% 提到 89.9%。

---

## 如何使用这套学习材料（导航）

| 文件 | 内容 | 什么时候看 |
|------|------|-----------|
| `README.md` | 全局导航（本文件） | 最先看 |
| `summary.md` | 背景、问题、贡献、关键结果与量化指标 | 想快速抓全貌 |
| `insights.md` | **最重要**：核心思想为何成立、概念转变、权衡与局限 | 想真正"看懂"这篇 |
| `method.md` | 方法拆解：flow steering、DSRL/RFS、支撑集约束、PPO、算法流程、伪代码、复现坑 | 想动手实现/复现 |
| `mental-model.md` | 心智模型：这是什么类型的问题、先验知识地图、如何归类 | 想把它放进你的知识网络 |
| `qa.md` | 15 道自测题（5 基础 + 5 进阶 + 5 高阶） | 检验理解 |
| `reflection.md` | 反思：如何扩展、开放问题、脆弱假设、何处会失效 | 想做研究/找 idea |
| `code/` | 可运行代码 demo（支撑集 vs 分布约束的 2D 玩具例子 + flow steering 概念实现） | 想直观感受机制 |
| `index.html` | 交互式探索器（真实指标、约束对比、消融面板） | 想边玩边学 |
| `images/` | 论文关键插图（框架图、玩具例子、任务集、结果、消融） | 配合文档看图 |

## 关键要点（Key Takeaways）

1. **支撑集约束 > 分布约束**：分布约束在"提升"与"可迁移"之间存在根本矛盾（松→不迁移，紧→不提升）；支撑集约束避开了这个矛盾，允许在先验支持范围内自由挑最优模式。
2. **Flow steering 是支撑集约束的优雅实现**：冻结 flow 基座策略，只在隐噪声 z 上学 steering 策略，天然保证动作落在"基座能生成"的集合里。
3. **DSRL（硬约束） vs RFS（软约束）**：DSRL 纯调噪声，动作严格落在 A_base(o)；RFS 额外加一个小残差 a_r，做局部精修——灵巧操作需要这点精度，所以 SCORE = RFS 实例。
4. **免蒸馏、免真机微调、稀疏奖励、一套超参**：整套流程极简，基座策略原封不动。
5. **数据"覆盖度"比"最优性"更重要**：retry 数据、play 数据虽然不能提升基座策略本身，却能**扩大支撑集**，让 SCORE 提升更多（Cube Pinch 40%→100%）。
6. **系统设计关键**：非对称 actor-critic（critic 用特权状态）大幅提升样本效率；人工失败注入（artificial failures）防止遗忘恢复行为。

## 预计学习时间

- 快速通读（README + summary + insights）：**约 40–50 分钟**
- 深入理解（+ method + mental-model + 跑 code demo + 玩 HTML）：**约 2.5–3.5 小时**
- 研究级消化（+ reflection + 读附录 E 形式化分析 + 对照 DSRL/RFS 原文）：**约 6–8 小时**

## 目录结构

```
score-support-constrained-rl/
├── README.md              # 本文件：全局导航
├── summary.md             # 背景 / 问题 / 贡献 / 结果
├── insights.md            # 核心洞见（最重要）
├── method.md              # 方法与算法细节
├── mental-model.md        # 心智模型
├── qa.md                  # 15 道自测题
├── reflection.md          # 反思与扩展
├── meta.json              # 元数据
├── paper.pdf              # 原始论文
├── index.html            # 交互式探索器
├── code/                  # 可运行 demo
│   ├── support_vs_distributional_demo.py
│   ├── flow_steering_demo.py
│   └── README.md
└── images/                # 论文插图（fig1~fig11 已命名）
```
