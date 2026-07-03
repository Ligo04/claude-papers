# ABD 方法拆解(Method)

本文档面向希望复现或扩展 Affine Body Dynamics(ABD)的研究者,按照"运动学 → 势能设计 → 每步优化(Affine IPC)→ 碰撞检测(Affine CCD)→ 碰撞剔除(i-AABB)→ Hessian 装配 → 关节约束 → 混合仿真"的顺序,把论文的技术链条拆成可独立理解、可独立实现的组件。

## 1. 组件分解

### 1.1 ABD 运动学:12 DOF、常数 J、常数质量矩阵、无 Coriolis 力

每个仿真体 $b$ 用一个时变线性变换 $A_b(t) \in \mathbb{R}^{3\times3}$ 和平移 $p_b(t) \in \mathbb{R}^3$ 描述,合并为广义坐标

$$q = [p^T,\ a_1^T,\ a_2^T,\ a_3^T]^T \in \mathbb{R}^{12}, \qquad A = [a_1, a_2, a_3]^T\ (\text{按行存储})$$

体内材料点 $k$ 的静止坐标 $\bar{x}_k$ 到世界坐标的映射是仿射的:

$$x_k = A_b \bar{x}_k + p_b = J(\bar{x}_k)\, q, \qquad J(\bar{x}) = [I_3,\ I_3 \otimes \bar{x}^T]$$

**这是全篇最关键的工程事实**:$J(\bar{x})$ 只依赖静止形状 $\bar{x}$,与当前配置 $q$ 无关,是**常数矩阵**。把它代入动能积分:

$$\frac{1}{2}\int_\Omega \rho\, \dot{x}^T \dot{x}\, d\Omega = \frac{1}{2}\dot{q}^T \underbrace{\left(\int_\Omega \rho\, J(\bar{x})^T J(\bar{x})\, d\Omega\right)}_{M}\dot{q}$$

广义质量矩阵 $M$ 同样只依赖静止几何,是**常数**,且只需在预处理阶段计算一次。这与刚体动力学形成鲜明对比——刚体的惯性张量会随姿态旋转而变化,方程中必然出现 $\omega \times I\omega$ 这类 Coriolis/陀螺项,是非线性 ODE。ABD 的运动方程则是简单的线性二阶 ODE:

$$M\ddot{q} = -\nabla V(q) + f$$

**实现要点**:$M$ 是 12×12 的分块矩阵,由体的一阶矩(质心)与二阶矩(惯量张量的推广形式)拼装而成,只需按体积分一次密度加权的 $J^TJ$;外力 $f = \sum_k J(\bar{x}_k)^T f_k$ 把材料点力直接"提升"到广义坐标力,同样是线性映射,无需任何迭代或分解。

### 1.2 正交惩罚势 $V_\perp$:替代刚性约束的光滑势能

ABD 不对 $A$ 施加硬性正交约束 $AA^T=I$,而是用一个高刚度惩罚势"软"地拉住它:

$$V_\perp(q) = \kappa\, \nu\, \|AA^T - I_3\|_F^2, \qquad \kappa > 100\,\text{GPa}$$

其中 $\nu$ 是体积,$\kappa$ 是刚度参数。展开为纯多项式形式(无需极分解 polar decomposition):

$$V_\perp = \kappa\nu\left(\sum_i (a_i \cdot a_i - 1)^2 + \sum_{i\neq j}(a_i\cdot a_j)^2\right)$$

梯度与 Hessian 也是解析多项式:

$$\frac{\partial V_\perp}{\partial a_i} = 2\kappa\nu\left(2(a_i\cdot a_i - 1)a_i + 2\sum_j (a_j\otimes a_j)a_i\right), \qquad
\frac{\partial^2 V_\perp}{\partial a_i^2} = 2\kappa\nu\left(4 a_i\otimes a_i + 2(\|a_i\|^2-1)I_3 + 2\sum_j a_j\otimes a_j\right)$$

**为什么选它而不是 ARAP 或 neo-Hookean**:ARAP($\|A - R(A)\|_F^2$,$R(A)$ 为 $A$ 的极分解旋转部分)每次求值都要做一次极分解(SVD 或迭代求解),neo-Hookean 更是需要完整的超弹性本构管线。$V_\perp$ 是纯多项式,能量求值比 ARAP 快 43%,比 neo-Hookean 快 178%。代价是它不是精确的旋转不变能量(只在 $\kappa\to\infty$ 极限下逼近严格刚性),但由于接触 barrier 本身已经引入了比 $V_\perp$ 更高阶的刚性,这个近似误差在实践中可忽略。

## 2. Affine IPC:每步的增量势优化

单体的离散增量势(IP):

$$q_b^{t+1} = \arg\min_{q_b} E_b(q_b), \qquad E_b = \frac{1}{2}\|q_b - \tilde{q}_b\|_M^2 + \Delta t^2\, V_\perp(q_b)$$

其中 $\tilde{q}_b = q_b^t + \Delta t\, \dot{q}_b^t + \Delta t^2 M^{-1} f_b^{t+1}$ 是由上一步状态外推得到的已知量(隐式欧拉的预测项)。多体系统的全局 IP 再叠加接触与摩擦势:

$$E(q) = \sum_b E_b(q_b) + V_C(q) + V_F(q)$$

- **接触势** $V_C(q) = \kappa \sum_{i \in \mathcal{C}} B(d_i(q))$,$B(d,\hat{d})$ 是 IPC 的平滑截断对数 barrier(距离 $\ge \hat{d}$ 时为零,$d\to 0$ 时趋于无穷),只处理体间(不含体内自碰撞,因为高刚度已排除自穿透可能)。
- **摩擦势** $V_F(q) = \sum_{j\in\mathcal{F}} \mu \lambda_j m_j(q)$,IPC 变分摩擦(mollified Coulomb),$\lambda_j = -\kappa \nabla_q B(d_j(q))$ 是接触力大小,$m_j$ 是相对滑动速度的光滑范数。

求解这个全局 IP 的极小值(标准 Newton + line search + CCD 过滤)就是一个时间步。**这一节之所以能"直接套用"标准隐式积分,根源就在 1.1 节的常数 $M$**——如果 $M$ 随配置变化(如刚体),就必须引入约束 Lagrangian 或 Poisson 积分子等专门机制。

## 3. Affine CCD:为什么可以用线性 CCD

刚体的痛点在于:Newton 迭代中每一步位移沿着**旋转轨迹**移动,精确的连续碰撞检测(CCD)必须处理这条曲线轨迹,rigid-IPC 的做法是保守细分为分段线段(curved CCD),开销随迭代次数线性放大。

ABD 的解法是观察 Newton 单次迭代 $\ell$ 内,顶点位移方向:

$$\Delta A_b^\ell = A_b^\ell - A_b^{\ell-1}, \qquad \Delta p_b^\ell = p_b^\ell - p_b^{\ell-1}$$

对应的轨迹是 $\alpha \in [0,1]$ 上的**线性插值**:

$$x_j(\alpha) = (A_b^{\ell-1} + \alpha\, \Delta A_b^\ell)\, \bar{x}_j + p_b^{\ell-1} + \alpha\, \Delta p_b^\ell = x_j^{\ell-1} + \alpha\left(\Delta A_b^\ell \bar{x}_j + \Delta p_b^\ell\right)$$

这本质上就是普通线性 CCD 的形式(和 FEM 顶点位移做线性 CCD 完全一样),因此可以直接套用任何现成的线性 CCD 算法——论文采用 Additive CCD(ACCD,[Li et al. 2021b]),完全绕开 curved CCD。**注意**:CCD 是在 Newton 每次迭代(而非每个时间步)都要执行一次,这意味着"线性 CCD 比曲线 CCD 快"这一优势会被迭代次数放大——这正是 §5.5 中把"迭代数更少 + CCD 更快"列为 90% 加速来源的原因。

## 4. i-AABB Culling:利用局部性剔除接触对

传统刚体流水线为每个物体预构建一棵覆盖全表面的 BVH,姿态更新时只需对 BVH 做仿射变换而无需重建。ABD 观察到:**近刚体之间的接触区域通常只占各自表面的一小部分**(不像布料等可变形体那样接触面广泛分布)。于是提出 intersection-AABB(i-AABB):

1. 为每个体计算一个粗粒度 AABB,按接触精度 $\hat{d}$ 做外扩偏移;
2. 只在两个体 AABB 的**重叠区域**内构建浅层(至多三级)二叉 AABB 树;
3. 由于不同体对的 i-AABB 彼此不相交,后续的图元级碰撞检测可以在每个 i-AABB 内部**并行**展开,天然免锁。

单层 i-AABB 对 GPU 和大多数 CPU 场景已经足够;只有像图1齿轮组这种高度纠缠、密集接触的场景才需要升级到三级层级。直升机(116K 三角面)+ 船(227K 三角面)互相穿插的压力测试中,全身 BVH 需要 130 万次 AABB 相交测试,三级 i-AABB 降到 12 万次,减少 90%。

## 5. Contact-aware 两遍并行 Hessian 装配

全局系统 Hessian 是 $12|\mathcal{B}| \times 12|\mathcal{B}|$ 的分块矩阵。若无接触,它是纯分块对角矩阵(每体一个 12×12 块,由常数质量项 $M/\Delta t^2$ 加正交势 Hessian 组成),可平凡地并行计算。一旦引入接触,活跃的体对 $(i,j)$ 会在 Hessian 的 $i,j$ 对角块与 $(i,j)/(j,i)$ 非对角块上产生耦合项,而且这个稀疏模式**逐帧、逐迭代都在变化**。

关键洞察:**i-AABB 剔除的结果直接决定了 Hessian 的稀疏模式**——若体 $i,j$ 的 i-AABB 叶子为空,则二者必然不接触,对应的 Hessian 块保持零;否则保守地为该 12×12 非对角块预留存储空间。由此得到两遍装配策略:

- **第一遍**:并行遍历每个 i-AABB 内部的所有候选图元对,计算并缓存各自的局部 Hessian(利用对称性只存半边,减少内存);
- **第二遍**:按全局 Hessian 的非零元位置,并行把缓存的局部 Hessian 累加进去(不同 i-AABB 相互独立、不相交,因此无需原子操作或加锁)。

相比传统的、不感知接触结构的顺序装配(如 rigid-IPC 采用的方式),该策略在体数量大的场景中最高可获得两个数量级加速(图3 wrecking ball 场景实测 188×)。

## 6. 关节约束的线性化(虚拟四面体)

非退化四面体唯一确定一个仿射变换,因此可以把仿射体的广义坐标 $q$ 反过来映射为一个虚拟四面体 $P \in \mathbb{R}^{3\times4}$(顶点 $p_1,\dots,p_4$),把 ABD 仿真体理解为"单元 FEM"(single-element FEM,简化应变能)。设 $\phi$ 为 $q$ 到 $P$ 的映射,$q=\phi(P)$,则:

$$\phi(P) = \left[\frac{1}{4}\sum_i (p_i-\bar{p}_i)^T,\ \ \mathrm{vec}^T\!\left(P\bar{P}^T(\bar{P}\bar{P}^T)^{-1}\right)\right]^T$$

$\phi$ 是 $P$ 的**线性函数**,雅可比 $\partial\phi/\partial p_i$ 只依赖静止形状 $\bar{P}$,因此系统雅可比 $J_i = \partial x_i/\partial q \cdot \partial\phi/\partial\mathrm{vec}(P)$ 同样是常数。

**旋转轴约束的线性化**:虚拟四面体的顶点选取是自由的,于是可以刻意选择一条边平行于目标旋转轴(如图12所示齿轮绕固定轴转动),把"绕轴旋转自由度"约束简单表述为**固定该边两端点** $p_2, p_4$ 的线性等式约束。这在自由度计数上是"过度约束"的——它消掉了 6 个自由度而不是理论上刚好需要的 2 个(因为顶点还应允许沿轴向平移),但由于 ABD 本身就比严格刚体多用了自由度(12 vs 6)来表示同一个刚体运动,这种冗余恰好被吸收,仿真结果依然正确。类似地,限制旋转角度的约束可以转化为线性剪切约束。

## 7. 混合仿真(刚性 + 变形部件)

把 ABD 看作单四面体 FEM 后,自然可以扩展为混合模型:刚性主体仍用仿射坐标 $q$ 描述,而附着的可变形部件(如帆布、旋翼)用标准 neo-Hookean FEM 描述,并通过约束绑定到刚性主体对应的虚拟四面体上。这样刚性/柔性部件之间的接触可以统一用同一套 barrier-based 势函数处理,无需专门设计刚柔耦合的接触算法。

## 8. 单步时间积分算法(伪代码)

```
function ABD_STEP(bodies B, contacts_prev, dt):
    # 1. 预测项(外推,常数 M 使这一步是纯线性代数)
    for b in B:
        q_tilde[b] = q[b] + dt * qdot[b] + dt^2 * Minv[b] @ f_ext[b]
        q[b] = q_tilde[b]                      # Newton 初始猜测

    converged = False
    while not converged:
        # 2. 碰撞剔除(i-AABB):仅在重叠区域内枚举候选接触对
        candidate_pairs = build_i_AABB_and_query(B, d_hat)

        # 3. 组装增量势 E = sum(E_b) + V_C + V_F,及梯度/Hessian
        for b in B:
            E_b, g_b, H_b = eval_inertia_and_orthogonality(b, q[b], q_tilde[b], dt)
        V_C, V_F, g_contact, H_contact = eval_barrier_and_friction(candidate_pairs, q)
        # 两遍并行装配全局稀疏 Hessian(§5)
        H = assemble_two_pass(H_b, H_contact, candidate_pairs)
        g = assemble_gradient(g_b, g_contact)

        # 4. 线性求解(优先 LLT,数值失败退化为 LDLT,见 §9)
        delta_q = solve_spd_or_indefinite(H, -g)

        # 5. 步长搜索:先做 Affine CCD 求最大无穿透步长 alpha_max(§3)
        alpha_max = affine_ccd(B, delta_q, candidate_pairs)
        alpha = line_search_with_barrier_energy(E, delta_q, alpha_max)

        # 6. 更新配置
        for b in B:
            q[b] = q[b] + alpha * delta_q[b]

        converged = check_convergence(g, delta_q, alpha)

    # 7. 速度更新(隐式欧拉)
    for b in B:
        qdot[b] = (q[b] - q_prev[b]) / dt
    return q
```

**要点提示**:第 2 步(i-AABB)与第 5 步(Affine CCD)都发生在**每一次 Newton 迭代内部**,而不是每个时间步一次——这是 IPC 家族方法保证无穿透的核心机制,也是"迭代次数越少,整体越快"这一结论成立的原因。

## 9. ASCII 架构 / 数据流图

```
                          ┌─────────────────────────┐
                          │   时间步 t → t+1 (dt)    │
                          └────────────┬─────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │  预测项 q~_b = q_b + dt*qdot + dt^2 M^-1 f │  （常数 M，纯线性代数）
                    └──────────────────┬───────────────────┘
                                       │
             ┌─────────────────────────▼─────────────────────────┐
             │                 Newton 迭代循环                    │
             │  ┌────────────┐   ┌───────────────┐   ┌─────────┐ │
             │  │ i-AABB     │──▶│ 候选接触对枚举  │──▶│ V_C,V_F │ │
             │  │ culling    │   │ (edge-edge /   │   │ 势能/梯度│ │
             │  │ (§4)       │   │  vtx-face)     │   │ /Hessian │ │
             │  └────────────┘   └───────────────┘   └────┬────┘ │
             │                                              │      │
             │  ┌────────────────┐   ┌──────────────────────▼───┐ │
             │  │ V_b, V_perp     │──▶│ 两遍并行 Hessian 装配(§5) │ │
             │  │ (每体独立并行)   │   │  Pass1: 局部 Hessian 缓存 │ │
             │  └────────────────┘   │  Pass2: 非零元并行累加    │ │
             │                        └────────────┬─────────────┘ │
             │                                     │                │
             │                     ┌────────────────▼───────────┐   │
             │                     │ 稀疏线性求解 LLT→(失败)LDLT │   │
             │                     └────────────────┬───────────┘   │
             │                                      │                │
             │       ┌───────────────────────────────▼─────────┐    │
             │       │ Affine CCD(§3,线性轨迹) → alpha_max     │    │
             │       │ + barrier line search → alpha           │    │
             │       └───────────────────────────────┬─────────┘    │
             │                                       │               │
             │                     q_b ← q_b + alpha * delta_q_b     │
             │                                       │               │
             │                        收敛? ──否──▶ 回到循环顶部      │
             └───────────────────────────────────────┼───────────────┘
                                                      │是
                                       ┌───────────────▼────────────┐
                                       │ qdot_b ← (q_b - q_b^t)/dt   │
                                       └─────────────────────────────┘
```

## 10. 实现要点与坑

- **刚度取值 $\kappa > 100\,\text{GPa}$**:$\kappa$ 需要远高于接触 barrier 的等效刚度,才能保证体内变形可忽略;但过大的 $\kappa$ 会使正交势 Hessian 的条件数急剧上升。论文的经验是这一代价"边际化",因为接触 barrier 本身已经比 $V_\perp$ 更刚,系统求解器的鲁棒性瓶颈已经由 barrier 决定,$V_\perp$ 再刚一些不会显著恶化整体条件数。
- **场景归一化与 $\hat{d}=1/1000$**:论文所有实验都把场景包围盒归一化到 $1\times1\times1$,并统一设 $\hat{d}=1/1000$(若模型尺度约 1 米,则接触精度约 1 毫米)。复现时必须同步做场景尺度归一化,否则直接套用该 $\hat{d}$ 数值会导致接触过早/过晚激活。
- **LLT 优先、失败切 LDLT**:纯刚性场景中全局 Hessian 通常正定,可用 Cholesky(SimplicialLLT)加速求解;但在混合仿真(§7)中,可变形部件的弹性 Hessian 比 $V_\perp$ 的 Hessian 小好几个数量级,导致整体矩阵实际上数值上接近半正定甚至偶尔非正定,LLT 分解会失败,此时需要退化到 SimplicialLDLT(允许不定/半正定输入但速度更慢)。论文也提到 Schur complement 分解(如 Peiret et al. 2019)是可能的补救方向,但未在本文中实现。
- **正交势 vs ARAP/neo-Hookean 的效率对比**:$V_\perp$ 是纯多项式,无需极分解或 SVD,能量运算比 ARAP 快 43%,比 neo-Hookean 快 178%。若复现时图省事直接套用超弹性本构库计算刚体势能,会白白损失这部分性能。
- **over-constraining 关节为何正确**:直觉上"固定四面体一条边的两个顶点"消除了 6 个自由度而不是理论上刚好需要的 2 个(轴旋转应只锁住垂直于轴的 2 个自由度,沿轴向的平移/伸缩本应保留),看似过度约束;但因为 ABD 用 12 维仿射坐标表示了本质上只有 6 维的刚体运动,冗余的 6 维本来就不影响物理上可观测的刚体轨迹,过度约束吸收的正是这部分冗余自由度,而不会影响真实运动。
- **i-AABB 的适用边界**:单层 i-AABB 对绝大多数场景(含 GPU 实现)已经足够,只有高度纠缠、持续密集接触的场景(齿轮组、复杂模型互穿)才值得引入三级层级——层级越深,构建开销和收益需要根据场景接触密度权衡。

## 11. 复现风险点

1. **常数矩阵的正确实现**:$J(\bar{x})$、$M$、关节约束雅可比 $J_i$ 都应该在预处理阶段一次性算好并缓存;若在运行时反复重新计算,会丧失 ABD 相对刚体方法的结构性优势之一。
2. **CCD 必须逐 Newton 迭代调用**,而不是逐时间步调用一次——这是保证无穿透的必要条件,容易在工程实现中为了"省时间"而被误省略或降频。
3. **i-AABB 与全局 Hessian 稀疏模式的一致性**:两遍装配依赖于"i-AABB 剔除结果 = Hessian 非零模式"这一假设,若剔除逻辑与 Hessian 装配逻辑不同步(例如剔除时用了更松的容差、装配时用了更紧的容差),会导致漏算或错误估计稀疏结构。
4. **LLT/LDLT 切换的触发条件**:需要显式检测分解失败(而不是让求解器静默返回错误解),否则混合仿真场景可能得到看似收敛实则数值错误的结果。
5. **$\kappa$、$\hat{d}$、场景尺度三者是耦合的超参数**:论文强调 ABD"无需逐场景调参",但这建立在统一的场景归一化 + 固定 $\hat{d}$ + 足够大 $\kappa$ 这套约定之上;直接迁移到未归一化的场景可能仍需重新校准。
6. **单元测试建议**:复现时可以先在单体自由落体(无接触)场景下验证 $M$、$V_\perp$ 的解析梯度/Hessian 与数值差分一致,再逐步引入 Affine CCD、i-AABB、接触勢,分层验证比一次性搭好整个管线更容易定位 bug。
