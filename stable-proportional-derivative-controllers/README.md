# Stable Proportional-Derivative Controllers (SPD)

> Jie Tan, Karen Liu, Greg Turk — Georgia Institute of Technology
> IEEE Computer Graphics and Applications, 2011

## 这篇论文讲什么？

物理仿真中的角色动画普遍使用 **PD 控制器**（比例-微分控制器）来跟踪目标姿态：它就像每个关节上装了一个弹簧 + 阻尼器。但传统 PD 有个致命缺陷——**当你想精确跟踪（用大增益 $k_p$）时，仿真就会数值爆炸**，除非把时间步长 $\Delta t$ 缩到极小（论文对比的工作用了 0.67 ms）。这使得动画师必须在「跟踪精度」和「仿真效率」之间二选一。

本文提出 **Stable PD (SPD)**：把控制力的计算从「当前状态 $q_n$」改为「**下一时刻状态 $q_{n+1}$**」。由于下一时刻状态未知，用一阶泰勒展开近似，最终得到一个把 $K_d\Delta t$ 加到质量矩阵上的隐式求解式。结果是：**任意大的增益（甚至 $k_p=\infty$、$10^{16}$）配合大时间步长（33 ms）都保持稳定**。SPD 可以无缝替换任何现有 PD 控制器，不改变底层控制/仿真机制。

## 难度等级

**Intermediate（中等）** — 核心思想极其优雅，1D 推导只需高中数学；但要真正吃透需要理解显式/隐式积分、广义坐标下的多体动力学、以及稳定性分析。

## 关键收获（TL;DR）

1. **核心一招**：用 $q_{n+1}, \dot q_{n+1}$ 而非 $q_n, \dot q_n$ 计算控制力，再用泰勒展开闭合方程。
2. **数学结果**：加速度 $\ddot q_n = (M + K_d\Delta t)^{-1}(-C - K_p(q_n + \dot q_n\Delta t - \bar q_{n+1}) - K_d\dot q_n + \tau_{ext})$。
3. **关键机制**：$K_d\Delta t$ 被加进了质量矩阵 $\to$ 等价于隐式地"预阻尼"，避免过冲。
4. **稳定条件**：$K_d \ge K_p\Delta t$（即 $\alpha\ge1$）。1D 下 $k_d=k_p\Delta t$ 时收敛最快（两步到位）。
5. **与隐式积分的区别**：SPD 不需要线性化 $M, C$（隐式积分需要，且涉及旋转奇异性，很难），更高效；隐式积分无条件稳定但过阻尼，SPD 有条件稳定但不过阻尼。
6. **三大应用**：① 运动捕捉跟踪 + 物理响应；② 关键帧插值生成次级运动；③ 替代罚函数法做约束仿真。

## 如何阅读这些材料

| 文件 | 内容 | 建议顺序 |
|------|------|---------|
| `README.md` | 总览（你在这里） | 1 |
| `summary.md` | 背景、问题、贡献、量化结果 | 2 |
| `insights.md` | **最重要** — 为什么这招有效、概念性转变、权衡 | 3 |
| `method.md` | 完整推导、伪代码、实现陷阱 | 4 |
| `mental-model.md` | 如何在研究地图中定位这篇工作 | 5 |
| `qa.md` | 15 道自测题（基础/进阶/高级各 5） | 6 |
| `reflection.md` | 扩展方向、脆弱假设、失效场景 | 7 |
| `code/` | 可运行的 Python 演示（1D toy + 双摆对比） | 实践 |
| `index.html` | 交互式探索器（拖动滑块看稳定性边界） | 实践 |
| `images/` | 论文原图 | 参考 |

## 预计学习时间

- 快速理解核心思想：**20 分钟**（README + insights 的"核心一招"）
- 完整吃透推导与应用：**1.5–2 小时**
- 跑通代码 + 玩交互演示：再 **30–45 分钟**

## 目录结构

```
stable-proportional-derivative-controllers/
├── README.md              # 总览
├── summary.md             # 摘要式总结
├── insights.md            # 核心洞察（最重要）
├── method.md              # 方法与推导细节
├── mental-model.md        # 心智模型
├── qa.md                  # 15 道自测题
├── reflection.md          # 反思与扩展
├── meta.json              # 元数据
├── paper.pdf              # 原始论文
├── index.html            # 交互式探索器
├── code/
│   ├── spd_1d_toy.py            # 1D 玩具例子：PD vs SPD 稳定性
│   └── double_pendulum_demo.py  # 双摆跟踪对比（复现 Figure 4 思想）
└── images/                # 论文图片
```
