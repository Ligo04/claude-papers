# ABD 论文事实速查(供材料生成使用,所有数字均来自论文原文,严禁编造)

论文:Affine Body Dynamics: Fast, Stable and Intersection-free Simulation of Stiff Materials
作者:Lei Lan (Clemson), Danny M. Kaufman (Adobe), Minchen Li (UCLA), Chenfanfu Jiang (UCLA), Yin Yang (Clemson)
发表:ACM TOG (SIGGRAPH 2022),arXiv 2201.10022
全文文本:同目录 `fulltext.txt`(第 1–8 页)与 `fulltext-p9-end.txt`(第 9–14 页)

## 核心方法
- 关键洞见:刚体模型的真正优势**不是刚性假设,而是紧凑坐标表示**。刚性约束(SE(3))导致轨迹分段刚性(curved),使 rigid-IPC 必须做昂贵的曲线 CCD(保守细分为分段线段),成为瓶颈。
- ABD:每体 12 DOF 仿射坐标 q = [pᵀ, a₁ᵀ, a₂ᵀ, a₃ᵀ]ᵀ ∈ R¹²,A = [a₁,a₂,a₃]ᵀ(按行存储)。
- 世界坐标:x = A x̄ + p = J(x̄) q,其中 J(x̄) = [I₃, I₃ ⊗ x̄ᵀ] 为**常数**矩阵。
- 动能 → 广义质量矩阵 M = ∫ρ Jᵀ J dΩ 为**常数** ⇒ 运动方程无 Coriolis 类非线性力,可直接用标准隐式积分(增量势 IP)。
- 正交惩罚势(替代刚性约束):V⊥(q) = κν‖AAᵀ − I₃‖²_F,κ > 100 GPa,ν 为体积。
  - 纯多项式,无需极分解;能量运算比 ARAP 快 43%,比 neo-Hookean 快 178%。
- 每步求解 IP:E = Σ_b E_b + V_C + V_F,E_b = ½‖q_b − q̃_b‖²_M + Δt² V⊥(q_b),q̃ = qᵗ + Δt q̇ᵗ + Δt² M⁻¹ f。
- 接触 V_C:IPC 平滑截断对数 barrier B(d, d̂);摩擦 V_F:IPC 变分摩擦(mollified Coulomb)。体内高刚度 ⇒ 无需自碰撞。
- Affine CCD:Newton 每次迭代的位移方向 ΔA x̄ + Δp 是**线性**轨迹 ⇒ 直接用线性 CCD(采用 ACCD [Li et al. 2021b]),完全避免 curved CCD。
- i-AABB culling:仅在两体粗 AABB 的**交叠区域**(offset d̂)建浅层(三级)AABB 树。直升机+船(343K 三角形):AABB 相交测试从全身 BVH 的 1.3M 降至 120K(省 90%)。GPU 及多数 CPU 例子单层 i-AABB 即够。
- Contact-aware Hessian 装配:全局 Hessian 为 12|B|×12|B| 块结构;稀疏模式由 i-AABB 决定;两遍并行(先算各 pair 局部 Hessian 并缓存,再按非零元并行累加)。相比顺序装配最高两个数量级加速(Fig.3 例 188×)。

## 关节约束(§5.3)
- 非退化四面体唯一确定仿射变换 ⇒ q 可映射到虚拟四面体 P ∈ R^{3×4}(单元 FEM 视角),φ(P) 线性、Jacobian 常数。
- 旋转轴约束:选四面体一条边平行于轴,固定该边两顶点 ⇒ 线性等式约束(over-constraining 但因 DOF 冗余仍正确)。
- pendulum 371× vs rigid-IPC;octopus(8 触手×5 关节)28×。

## 混合仿真(§5.4)
- 刚性主体 + 变形部件(neo-Hookean FEM)约束到虚拟四面体;船 225K tri / 直升机 116K tri;16 s/帧 @Δt=1/100。
- 数值问题:V⊥ Hessian 远大于弹性 Hessian ⇒ 正定性受损,LLT 偶尔失败时切换 LDLT;可能的补救:Schur complement。

## 实验数据(i9-11900K 8 核 + RTX 3090;Eigen SimplicialLLT + TBB;场景归一化 1×1×1,d̂=1/1000)
- Wrecking ball(Fig.3):575 体,14K tri;**124×** vs rigid-IPC(CPU);ABD 0.14 s/帧 vs 17.6 s;GPU 18 FPS;Δt 1/100→1/25 性能不敏感。
- Small chain net(Fig.5):8×8,144 体,63K tri;**34×**;0.059 s vs 2.1 s;GPU 143 FPS;Δt 1/100→1/50 ABD 仅 0.059→0.065 s,rigid-IPC 慢 350%。
- Big chain net(Fig.4):16×16,673 体,445K tri;平均 **1200×**(0.78 s vs 944 s),峰值超 4000×,rigid-IPC 尖锐碰撞时单帧数小时;GPU 5 FPS,**>10000×**。
- Screw(Fig.6 左,动摩擦):2 体 7.9K tri;**30×**(87 ms vs 2.59 s)。
- Arch(Fig.6 右,静摩擦):101 体;**77×**(8.7 ms vs 0.67 s)。
- House of cards(Fig.7):158 体(155 卡+…);**103×**(86 ms vs 8.9 s);GPU 42 FPS。
- Gear set(Fig.1):28 齿轮,2.45M tri / 3.09M edges,>1/4 表面元每步活跃接触;rigid-IPC(curved CCD 找不到 TOI、Newton 死循环)、Bullet(凸代理丢齿形穿插)、MuJoCo 全部失败;ABD CPU 12 s/步(Δt=1/100,17 iter),GPU 5–10 FPS(7.3)。
- Bone dragon(Fig.9):29 体 1.2M tri,2.8 s/帧;Bullet V-HACD 代理丢失凹陷细节。
- Huge chain net(Fig.13):27,645 体,12M tri / 18M edges;CPU ~5 min/帧(310 s),全仿真 5 天;估计 ≥2000–5000× vs rigid-IPC(后者需数年)。
- Hybrid(Fig.11):19+8 体 1.1M tri,16.4 s/帧,GPU 0.4 FPS(求解器占主导 4.6 s Sol.)。

## vs Bullet(Tab.1,单线程,wrecking ball 变体、球不拴链)
- Light 16 块:Δt=1/100 ABD 3 ms vs Bullet 2 ms;1/240:2.2 vs 1.5;1/1000:2 vs 3(ABD 反超)。
- Full 562 体:1/100:657 vs 629;1/240:328 vs 809;1/1000:102 vs 804。
- Bullet **所有步长都有穿插**(含 1/1000);摩擦实验(screw/arch/cards)全失败,即使 Δt=1/10000。
- Bullet 需 V-HACD 凸分解预处理(慢、需手工调参、丢细节);ABD 直接仿真输入三角网格。
- ABD 步长越小每帧越快;Bullet 不是(小步长暴露更多碰撞)。

## 加速来源分析(§5.5)
- 复杂场景 90% 加速来自两点:迭代数更少(接触密集时远少于 rigid-IPC,gear set 17 vs ∞)+ 线性 CCD 替代 curved CCD(且 CCD 在每次 Newton 迭代都要做,差距被迭代数放大)。
- 轻量场景中 i-AABB culling(复杂几何省 90%,简单形状平均省 30%)与并行 Hessian 装配更重要。
- ABD 系统求解因 DOF 多(12 vs 6)略慢于刚体法,但被迭代少 + CCD 快掩盖。

## 局限
- 12 DOF/体 > 刚体 6 DOF,系统求解略慢;不是严格刚体(近刚性,κ 控制);κ 极大带来数值刚性(但接触 barrier 本来就更 stiff,所以边际代价小);混合仿真正定性问题(LLT→LDLT);无自碰撞处理(有意省略)。

## 结论亮点
- 可微、天然并行、直接吃三角网格、无需逐场景调参;适合作为所有刚体类应用的直接替代;面向计算设计/机器学习/机器人的前景。
