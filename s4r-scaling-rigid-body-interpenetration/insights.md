# 深入洞见：scale 不是一个单一旋钮

> 本文把论文、作者的研究笔记和公开实现放在一起读。论文结论只归于 [S4R 论文（arXiv:2609.20524）](https://arxiv.org/abs/2609.20524)；几何解释和延伸例子来自作者的博客 [To Scale or Not to Scale](https://frank-zy-dou.github.io/blog/to-scale-or-not-to-scale/)；实现细节来自 [Dynamics-Modeling/Penetration_Solving](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/main/Penetration_Solving)。博客中的未来方向在本文中都标成“建议”，不当作论文已完成的工作。

## 1. 先把三个 scale 分开

作者在博客中用一句很容易被忽略的观察把问题切开：**整个场景同比缩放不会改变交叠关系**。如果物体半径和物体中心之间的距离都乘以同一个正数，原来相交的物体仍然相交，只是长度单位变了。真正改变约束的，是“固定了什么、缩放了什么”。

| 语义 | 被固定的量 | 被改变的量 | 目的 | 不能推出的结论 |
| --- | --- | --- | --- | --- |
| 全场景同比缩放 | 相对布局、交叠拓扑 | 所有长度 | 改变单位或显示尺寸 | 不能修复已有穿透 |
| Coverage Axis 的 dilation | 表面采样、候选中心、原始球半径 | 选择阶段的有效半径 `r'_i=r_i+δ` | 让一个候选球覆盖一片表面采样，形成紧凑骨架 | 不等于把输入表面变大，也不保证拓扑或重建误差单调 |
| S4R 的 per-body homothety | 参考中心 `p_i`、姿态 `R_i`（默认阶段固定） | 物体相对中心的尺度 `s`，从 `s_min` 恢复到 `1` | 在物体之间制造 clearance，把深穿透转化为逐步出现的接触约束 | 不等于求得最小最终位移，也不等于完成物理动力学 |

Coverage Axis 的 `δ` 是抽象选择的容差。它只出现在“候选是否覆盖表面采样”的关系中，选中之后保留球又恢复原始半径，表面本身没有移动（博客 §1；Coverage Axis §4）。因此增大 `δ` 会让可行 set-cover 选择的数量不增加，但候选身份可能突然改变；薄分支、连续曲面和重建误差仍须单独检查。`δ` 是有长度单位的，输入缩放后如果不同时缩放 `δ`，两件同形但不同尺寸的物体就不再有可比的容差。博客还提醒，加性 `δ` 对小球的相对影响 `δ/r_i` 更大，而乘法缩放给予每个球相同的相对增幅。

S4R 的 `s` 则改变真实碰撞几何。论文将身体顶点写成类似 `v_i(s)=p_i+sR_i\bar v_i` 的形式（论文 §3.2）：中心不随尺寸缩放，物体边界向中心收缩或恢复。对半径为 `R_1,R_2`、中心距为 `d` 的两球，博客给出分离余量

```text
g(s) = d - s (R_1 + R_2).
```

当 `s` 变大，余量下降，接触约束才逐步变得活跃。这里的“scale”是求解路径上的几何参数，而不是把最终物体改成小尺寸。

## 2. S4R 为什么可能有效：把难问题拆成一串局部问题

深度穿透时，直接在全尺寸 mesh 上求解会遇到两个问题：初始状态可能不满足无穿透求解器的假设；接触法向和最近特征可能还没有稳定定义，大修正又会改变下一次接触。S4R 的数值 continuation 把一个目标问题换成一族带参数的问题：

1. 以物体参考中心为缩放中心，先达到一个经过检查的 `s_min`（论文实验使用约 `0.01` 的起始尺度）。不同中心的有界物体在足够小的尺度下可分离；近重合中心则需要额外的起点处理。
2. 从当前尺度向前看一个 `ds`，用 bounding sphere 支持半径估计哪些 pair 会在这次 inflation 中接触。保守 event bound 的价值是少做精确 mesh 查询，而不是替代精确碰撞检查。
3. 对被激活的 pair，用当前 witness/法向把非穿透条件线性化，求一个 minimum-norm 的凸 QP。论文的增量约束可概括为
   `nᵀ(Δp_j-Δp_i) ≥ b`，其中 `b` 同时包含当前 signed distance、目标 margin `d_hat` 和这一步的尺度增量支撑项（论文 §3.3，Eq. 3–4）。
4. 应用位置修正后继续 inflation。当前解成为下一步的 warm start；接触信息按规则刷新，避免把一次旧 witness 当成永久真值。
5. 到 `s=1` 后再做 tail refinement，并用 full-size mesh/FCL evaluator 评分。返回“到达 full scale”与“full-size 无穿透”是两个条件，不能用中间尺度的干净状态代替最终证据。

这个拆分同时解释了工程收益：每个局部 QP 是凸的、变量通常只有平移（可选姿态变量），小步的线性模型比从深穿透一次性线性化更可靠；缓存和 broad phase 减少昂贵的三角网格查询；最终 FCL 或共享 mesh evaluator 把近似过程和验收标准分开。仓库 README 的结果契约明确区分 `continuation_complete`、`native_converged`、`status` 和 `fcl_verified`，这是读实现时比单看“solver returned”更重要的信号。

## 3. 局部最小修正不等于全局最佳布局

论文的全尺寸目标包含相对初始布局的位移代价和（可选的）旋转代价，同时要求 pair clearance 达到 `d_hat`（论文 §3，Eq. 1）。但 continuation 中每一步求的是当前 contact 集合下的最小增量；这些增量的和并没有自动等于全局目标的最小解。早期一次很小但方向不理想的移动，会改变后续最近特征、活动约束和可行通道。博客 §2 将其概括为“local minimum motion differs from minimum final displacement”。

因此要区分三个层次：

- **局部模型可解**：本步 QP 找到满足当前线性化 margin 的 `Δp`；
- **路径完成**：尺度确实走到 `1`，没有因步数、OSQP 状态、容器约束或数值问题提前退出；
- **全尺寸结果合格**：在原始 mesh、容器/关节限制和任务约束下重新检测，没有 residual penetration。

小 `ds` 也不是数学上的安全保证。若真实修正很大，或最近 feature 在本步切换，旧法向的局部模型会失真。论文没有给出“每一步必然可行”或“只要达到 `s=1` 就必然零穿透”的定理；零穿透是受限步数、碰撞 oracle 和 tail-refinement 共同得到的经验结果。论文和实现都把这点写进了输出状态，而不是隐藏在一个布尔成功值里。

## 4. 失败模式：为什么一个 clean intermediate state 不够

- **受限拥挤 packing**：论文实验显示，场景拥挤度升高后，约 `φ≈0.18` 的受限配置会成为困难区；一个身体被多个方向同时夹住时，最小增量可能把系统带入死胡同。
- **嵌套 cavity / containment**：两个闭合表面可能一个完全在另一个内部，表面三角形没有交叉；仅看最近表面距离会漏报，需要 containment 检查和逃逸方向。
- **近重合中心**：初始 bounding-sphere 分离假设可能被破坏，必须先分开中心或采用确定性的扰动；否则 QP 从不可行起点开始。
- **contact feature switch 与病态活动集**：最近三角形、边或顶点切换会使 witness 法向跳变；多个共线/近共面的约束又会使 QP 条件数变差。
- **缓存和近似 oracle 失效**：frozen-witness/cache 只有在移动界足以证明“不会进入 margin”时才能安全复用；身体移动后仍盲信旧距离，会漏掉新 contact。
- **预算不足**：默认 tail refinement 可能在大场景留下 residual pairs；博客引用 S4R App. A.13 说明延长 full-scale refinement 才能清除这些 pair。`status=converged` 的具体含义要以当前仓库版本的契约为准，不能从动画看起来停止就推断零穿透。
- **几何成功不等于动态成功**：博客 §4 的 SimReady_Gate 20-object 例子满足 mesh verifier 的零穿透和 97 个 predicate，但后续 MuJoCo settling 仍让圆形水果违背 upright 约束。几何修复、物理稳定、抓取成功和任务成功应分别记录。

## 5. 最重要的研究判断

Scale 的作用不是“把任何问题变容易”，而是改变约束出现的顺序和局部模型建立的位置。缩小阶段把初始重叠变成 clearance；恢复阶段让接触逐步加入；恢复后的布局仍携带路径历史。博客 §3 说得很清楚：Coverage Axis 留下的是 coverage tolerance 对 primitive selection 的偏置，S4R 留下的是 continuation path 对最终 layout 的偏置。

这也解释了为什么论文把“几何 repair”放在“simulation-ready”之前：S4R 能提供一个可测量的 full-size nonpenetration 初态，却没有在论文中解决质量、摩擦、速度、关节、柔性体或时间积分的一致性。博客 §5 把 mass weighting、friction/velocity、articulated/deformable bodies 和 batched GPU QP 列为未来方向；这些是合理的后续课题，不是当前 S4R 已有能力。

## 6. 阅读实现时的证据顺序

1. 先看 `s4r/s4r_qp.py` 的结果契约、起点 admissibility、SOI/event bound、QP 和 tail refinement；
2. 再看 `mesh_collision.py` 的 full-size evaluator，确认 signed distance、surface crossing 和 nested containment 的处理；
3. GPU 路径 `s4r_gpu/` 的采样 oracle 只能说明候选/迭代过程，最终仍需 exact FCL re-check（仓库 README 的 GPU 部分）；
4. 运行输出同时记录 `status`、`continuation_complete`、`native_converged`、`fcl_verified`、full-size penetration pairs、RMSD 和 wall/joint violations；
5. 最后才把结果送进 MuJoCo 或机器人控制器，验证 settling、姿态、抓取和任务成功。

论文 v1 的表格和仓库 README 的当前示例存在版本/硬件差异，不能混用数值。例如论文报告 CPU S4R 在 Kubric `N=40→5000` 的 RMSD 约 `0.036→0.037`，`N=5000` 约 `68.1 s`，并称各例最终 penetration 为 0；README 当前示例给出的 seed 42 `N=40` 为 RMSD `0.0400`、`0.35 s`，`N=1000` 为 `0.0365`、`15.7 s`。比较时必须记录 commit、数据集、硬件、solver、tail budget 和 evaluator，而不是只复制一个“实时速度”。
