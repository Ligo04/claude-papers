# 面向 IPC/UIPC 类仿真器的工程阅读

本文件是根据论文机制提出的设计建议。没有审计任何本地 UIPC/Newton/ManaEnv 分支，也没有证据表明公开 TapeSim adapter 已存在。官网代码入口在 2026-10-26 查阅时为 Coming soon。

## 应由哪些模块负责

| 模块 | 应负责的状态或操作 | 首个可核验行为 |
| --- | --- | --- |
| 材料网格 | flat rest geometry、材料坐标、sticky side、元素质量 | 卷绕与释放均不修改 rest geometry |
| 运动表示 | shell 顶点、6-DOF cluster、独立 hub、导数映射 | 刚体变换及边界受力一致 |
| 黏附接口 | cohesive history 或 bond reference/live support | 内步固定，步末创建与断裂 |
| 释放前沿 | 旧自由区快照、候选、图距离、collar | 一次更新不递归放出整段 collar |
| 状态转移 | 元素质量归属、恢复速度、原界面历史 | 没有重复质量、速度归零或重新退火 |
| 碰撞系统 | pair exclusion/re-enable、rigid trajectory CCD | bond 源对只在有效期间替换能量 |
| 切割 | kerf、局部重网格、约束重映射 | 两块持续参与接触且不产生重合面 |

不要为了减少未知数丢掉外部碰撞几何。cluster 表示改变的是顶点如何运动，外部接触仍需要知道胶带表面在哪里。

## 建议的实施与验证顺序

1. 固定一个无机器人摆动输入，建立 Coh-F 的 tolerance/成本/轨迹误差曲线。
2. 加静态 cluster，验证外力、惯性和旋转轨迹；单独检查完整 Hessian 或明确标注的近似。
3. 加动态 collar，并验证材料分类单调性、rest state、质量和速度转移。
4. 把 bond 当作独立开关，比较四种配置，防止把表示收益归给界面模型。
5. 用独立 peel-force 实验约束桌面黏附，再做开环/闭环配对评估。
6. 最后加入切割，核对面积、质量损失、碰撞和抓取约束映射。

这一顺序是工程建议，不是作者实现历史。所有项目修改仍需在实际目标仓库确认类型、所有权和已有测试后决定。

## 计时需要保留的几列

应分别记录 motion solve、front/history update、碰撞数据重建、控制/观测、渲染和总墙钟。每项包含 GPU 同步策略、步长、材料参数、collar 图宽与网格尺寸。比较不同表示时，既报告固定 tolerance，也报告固定轨迹误差标准。

论文的 RTX PRO 6000 物理步成本不能直接代表 RTX 5090 封箱演示的总吞吐。`1 / ms_per_step` 也不能直接叫“实时倍速”，必须再乘仿真步长并说明计时范围。

## 仍需作者源码解答的问题

- bond 对的创建、邻域过滤、参考 gap 及病态四面体处理细节。
- cluster 与 shell 交界元素的质量和弹性能如何组织到稀疏结构中。
- front 更新是否引起稀疏模式、预条件器或 GPU 数据结构重建。
- 刚体螺旋轨迹 CCD 的界与实际保守推进细节。
- 切割后历史、contact cache、grasp、渲染拓扑如何同步。
- 内部 VLA 权重、机器人控制器和真实/仿真配准数据是否会随代码发布。

教学 demo 的正确性结果只能支持它实际覆盖的公式和状态行为，不能替代这些实现问题，也不能证明 GPU 性能或物理真实性。

来源：[论文 III–IV 节](https://arxiv.org/html/2609.28766v2)、[官方项目页](https://genesis-embodied-ai.github.io/tapesim-website/)。
