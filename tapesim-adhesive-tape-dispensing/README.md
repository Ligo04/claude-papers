# TapeSim 中文学习包

TapeSim 研究机器人拉胶带时，卷筒如何移动、胶带如何逐步放出，以及放出的胶带如何再次黏附。它把卷内大部分材料作为一个刚体，把剥离前沿附近保留为可变形 collar，已放出的部分始终保留为薄壳。可选的 releasable bond 进一步简化部分黏附界面。理解重点是自由度分配、界面历史和有限求解精度之间的关系。

论文固定为 [arXiv:2609.28766v2](https://arxiv.org/abs/2609.28766v2)，2026-09-28 修订，共 9 页。学习日期为 2026-10-26。难度为进阶，需要弹性、隐式积分、Newton 法和接触优化的基础。方法属于物理仿真系统设计，包含受控计算实验和机器人真实/仿真配对评估。

## 阅读顺序

1. 用 `index.html` 浏览三种材料区域、四种配置和真实实验数字，约 15 分钟。
2. 读 `mental-model.md` 与 `insights.md`，弄清为什么减少变形自由度也可能改善运动，约 20 分钟。
3. 读 `method.md`，推导 bond 梯度并跟踪一次前沿更新，约 35 分钟。
4. 运行教学 demo，再用 `qa.md` 自测，约 30 分钟。
5. 对照 `engineering.md` 和 `evidence.md` 判断迁移到自有仿真器所需的工作。

需要同时深入数学与实现时，继续读 `deep-dive-bond-math.md` 和 `dynamic-collar-implementation.md`。交互页第五页把合成 bond 的能量/力/曲率与三角网格状态更新并排展示。

## 需要记住的结论

- 主要收益来自随释放前沿推进的空间降阶，bond 是独立的界面模型选择。
- 卷内刚体 cluster 使用 6 个自由度。virtual tetrahedron 的 ABD-type 能量不意味着 cluster 使用 ABD 的 12 个自由度。
- 同一 Newton tolerance 不保证各表示拥有相同的运动误差。固定容差加速和可比运动加速应分别解释。
- 总成功率接近真机，不说明每个案例的成败预测正确。Stick 的配对结果就是反例。
- 截至查阅，官方项目页标注代码 Coming soon；本包没有作者求解器实现。这里验证的是教学公式、状态更新和公开统计的一致性。

## 运行

```bash
python3 /home/ligo/claude-papers/papers/tapesim-adhesive-tape-dispensing/code/releasable_bond_and_front.py
python3 /home/ligo/claude-papers/papers/tapesim-adhesive-tape-dispensing/code/bond_hessian.py
python3 /home/ligo/claude-papers/papers/tapesim-adhesive-tape-dispensing/code/triangular_collar.py
```

仅依赖 NumPy，不需要 GPU。程序检查刚体变换不改变 bond 能量、解析梯度与有限差分一致、压缩不触发拉伸断裂、前沿每次只根据旧自由区更新，以及材料分类和刚体速度转移。它没有实现薄壳、IPC/CCD、Newton-PCG、真实卷绕或机器人。

后两份程序分别增加 18 项二阶导数检查和 80 项三角网格状态检查。三角网格示例按元素角点 `<N` 规则恢复 collar，并核对重分类瞬间的质量、线/角动量与动能；位置是程序指定的教学输入，没有进行真实放带动力学积分。

## 文件

| 文件 | 内容 |
| --- | --- |
| `summary.md` | 背景、贡献、结果和数字口径 |
| `insights.md` | 核心机制、取舍与容易误解的地方 |
| `method.md` | 能量、状态更新、伪代码和推导 |
| `mental-model.md` | 必要知识及研究定位 |
| `qa.md` | 5 道基础、5 道进阶、5 道高级问题 |
| `engineering.md` | 面向 IPC/UIPC 类系统的设计建议 |
| `reflection.md` | 可检验的研究问题 |
| `deep-dive-bond-math.md` | bond 梯度、Hessian、Gauss–Newton、断裂与尺度分析 |
| `dynamic-collar-implementation.md` | 元素/顶点分类、质量、速度、历史和完整伪代码 |
| `evidence.md` | 来源、版本、已验证范围和未复现内容 |
| `index.html` | 无外部依赖的交互学习页 |
| `code/` | 教学程序与运行报告 |
| `images/` | PDF 原图提取和关键图裁切 |
| `sources/` | PDF 文字、作者网页与原始统计快照 |
| `paper.pdf` / `meta.json` | 固定版本原文与校正后的元数据 |

交互页中方法示意图不执行物理仿真。数值图只使用论文或作者项目数据；页面标明实验、硬件、计时范围和派生计算。所有解释和工程建议均为本学习包的整理，不能视为作者未发布源码的实现细节。
