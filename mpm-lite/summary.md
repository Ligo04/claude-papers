# 摘要与要点：MPM Lite

> Xiang Feng*, Yunuo Chen*, Chang Yu*（共同一作）, Hao Su, Demetri Terzopoulos, Yin Yang, Joe Masterjohn, Alejandro Castro（Toyota Research Institute）, Chenfanfu Jiang。
> SIGGRAPH 2026 / arXiv 2602.07853。项目主页：https://mpmlite.github.io

---

## 1. 背景脉络

物质点法（Material Point Method, MPM）自 Sulsky 等人 1995 年提出以来，已成为图形学与计算力学中模拟**大变形、断裂、多相流**的主力方法。它是一种混合 Lagrangian/Eulerian 方案：

- **粒子（Lagrangian）** 携带质量、体积、变形梯度等物质历史；
- **背景规则网格（Eulerian）** 用于求解动量方程、处理拓扑变化（自接触、分裂、合并）。

人们常说"MPM 本质是伪装的 FEM"：粒子是求积样本（充当单元），背景网格是参考空间（充当顶点网格）。然而标准 FEM 并没有 MPM 的那些负担——FEM 在**几何感知的固定位置**使用（通常是 Gauss）求积、用紧致的**节点形函数**、边界语义清晰。这种"MPM 范式"与"FEM 实践"之间的落差，正是本文的出发点。

## 2. 问题陈述

作者指出，主流 MPM 变体继承了**两个结构性负担**：

### 负担一：宽支撑的 B-spline 传输核

现代 MPM 普遍用高阶 B-spline（二次/三次）核来缓解**穿胞不稳定**（cell-crossing instability，线性基函数下粒子穿越单元边界时力的不连续）。但宽核带来三个代价：

1. 复杂化大规模并行（散射 scatter 涉及更多节点）；
2. **隐式求解器变贵**（宽模板 → 稀疏矩阵更稠密、每次 Hessian-向量积更昂贵）；
3. **边界条件含糊**：B-spline 在网格节点上不具备 Kronecker-delta 性质，无法干净地施加节点边界条件。

### 负担二：粒子求积的成本

力通过**粒子求积**积分。这从弱形式推导很干净，但计算成本**直接正比于每胞粒子数 PPC**。在隐式设置下，每一次梯度或 Hessian-向量积评估都会触发一次对**全部粒子**的 grid→particle→grid（G2P2G）循环。随着分辨率或 PPC 增大，会出现**性能悬崖**——而实践中为抑制数值断裂常常需要 PPC ≥ 20。

### 核心问题

> **"MPM 的粒子必须是求积点吗？"（Must MPM particles be quadrature?）**

作者的回答是**否**。在最接近 MPM 的 FLIP/APIC 流体里，粒子主要充当**携带运动学状态、并辅助平流的标记物（marker）**，并不在求解时充当积分点。固体确实需要记住更多（应变、应力、材料历史），但"在求解时把粒子用作积分点"只是逼近弱形式应力积分的**一种选择**。如果能让粒子去**追踪、采样并忠实地重采样**状态到一个小的、网格对齐的求积集合上，就能把力装配与隐式积分**整体搬到网格上**，避免反复的、正比于 PPC 的粒子循环。这个视角同时也解决了 B-spline 问题：穿胞本质上是粒子-网格传输的病态；若求积**不再驻留在粒子上**，力装配就对高频粒子运动不敏感，于是可以用紧致的线性核做"通信"，同时由构造保证稳定。

## 3. 主要贡献（论文明列三条）

1. **线性核传输方案。** 在粒子与网格之间通信动量与应力。速度通信对二次 B-spline APIC **二阶一致**（误差 $O(\Delta x^2)$）；应力通信由构造避免穿胞不稳定。

2. **时空力积分视角，不需要 MPM 粒子参与。** 计算焦点从粒子转移到**单元**。把背景笛卡尔网格视为体素六面体网格，在固定胞心求积点上完成力/Jacobian 装配，使**求解复杂度与粒子无关**。

3. **定制的、基于优化的增量势公式。** 在更新 Lagrangian 线性化中对各向同性材料采用**旋转无关的拉伸参考（rotation-free stretch reference）**。该方案兼容常见材料模型与现有 MPM 变体。

由于去掉了积分中的粒子（粒子仅用于信息追踪），MPM Lite 类似体素网格上的**单点六面体有限元**：求解复杂度不再与粒子相关，现成的 FEM 非线性求解器、预条件子、子空间积分方法都能开箱即用，边界条件无歧义。它可被实现为**独立的粒子重采样模块 + 可扩展的 FEM 积分模块**。

## 4. 关键结果（定量）

**硬件**：NVIDIA RTX Pro 6000 GPU + Intel Core i9-9980XE CPU。

### 显式对比 —— Jelly Falling
- 1.14M 粒子，StVK 模型（$E=5\times10^3$ Pa，$\nu=0.4$），$\Delta x=1/256$ m，$\Delta t=6\times10^{-5}$ s，3 秒 @ 120 fps。
- 总运行时间：**MPM 404.1 s / CK-MPM 247.7 s / MPM Lite 215.4 s** → 相对显式 B-spline MPM **1.88× 加速**，并优于 CK-MPM。

### 隐式对比 —— Cantilever Beams（验证正确性）
- 三根梁，$E$ 从 $5\times10^3$ 到 $5\times10^5$ Pa。
- MPM Lite 的弹性响应与 **Romero 等人 2021** 的悬臂梁重力-弯曲主曲线（以无量纲参数 $\Gamma$ 表征 $H/W$）高度吻合，并与传统隐式 MPM 视觉一致。

### 隐式 PPC 加速曲线 —— 扭转 "faceless toy"
3 秒 @ 50 fps，$\Delta t=1\times10^{-3}$ s，$E=1\times10^4$ Pa，$\nu=0.3$，$\rho=1000$ kg/m³。

| PPC | MPM Lite | 隐式 MPM | 加速 |
|-----|----------|----------|------|
| 8   | 63.5 s   | 300.3 s  | 4.7× |
| 12  | 68.1 s   | 496.1 s  | 7.3× |
| 16  | 70.5 s   | 726.7 s  | 10.3× |
| 20  | 72.8 s   | 954.6 s  | 13.1× |
| 24  | 77.1 s   | 1225.4 s | 15.9× |

> 关键观察：MPM Lite 的运行时间随 PPC **几乎持平（63→77 s）**，而隐式 MPM 随 PPC **近线性暴涨**。这正是"求解成本与 PPC 解耦"的最有力证据。

### 与现成求解器耦合 —— VBD
- 双材料扭转棒，5 秒 @ 100 fps，$\Delta t=1\times10^{-3}$ s。
- **VBD 197.1 s vs Newton-PCG 401.7 s → 2× 加速**，视觉行为相近。

### 多材料 / 规模化演示（Table 2）

| 场景 | 粒子数 | 积分 | 材料 | 备注 |
|------|--------|------|------|------|
| Stuffed Toys | 5.23M | VBD | StVK | 3.70 s/帧 |
| Noodles | 2.89M | PCG | von Mises | 圆柱模具挤出 |
| Candy Fracture | 2.04M | PCG | NACC | 脆性断裂 |
| Rolling Snowball | 8.04M | Explicit | Snow | 雪累积+碰撞破碎 |
| Sand and Water | 1.93M | PCG | Drucker-Prager + $J$-water | 砂水耦合 |
| Wheel | 1.23M | PCG | von Mises | 金属塑性压扁 |
| Cream on Brownie | 2.44M | PCG | Herschel-Bulkley | 黏塑性奶油 |

## 5. 一句话总结

MPM Lite 通过"粒子只做信使、求积固定在胞心、积分纯在网格上做 FEM"的架构重构，把 MPM 的隐式求解成本从 PPC 中解放出来，既保留了 MPM 的鲁棒性与材料多样性，又获得了显著的隐式加速与更干净的 FEM 工程接口。
