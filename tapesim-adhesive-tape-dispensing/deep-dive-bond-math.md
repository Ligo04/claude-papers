# 从四个顶点推到 bond 力与 Hessian

本节按 [TapeSim v2，III-B / Eq.4](https://arxiv.org/html/2609.28766v2#S3.SS2) 的能量定义展开。梯度、Hessian、法向特例和网格尺度分析是本学习包的独立推导。作者没有公开求解器源码，不能把这里的代数表达当成作者实际装配或正定化策略。

## 1. 四个顶点怎样定义一个界面

三角形顶点为 `x₀,x₁,x₂`，另一表面的接触点为 `xP`。创建 bond 时保存

$$D_b=[X_1-X_0,\ X_2-X_0,\ X_P-X_0].$$

大写 `X` 表示 bond 创建时的位置。求解中的边矩阵为

$$D_s=[x_1-x_0,\ x_2-x_0,\ x_P-x_0],\qquad F=D_sD_b^{-1}.$$

`F` 比较当前四点形状与创建时形状。创建时 `F=I`；四点整体平移，边差不变；整体旋转 `Q` 后，`F` 变成 `QF`。

bond 能量为

$$\Phi_b=h^2V\kappa\|FF^T-I\|_F^2,\qquad V=\frac{|\det D_b|}{6}.$$

`h=Δt`，`κ` 单位为 Pa。`Vκ` 单位是 J，`h²` 是隐式积分的势能缩放。`V` 仅是界面能量的参考权重，没有增加真实体积、质量或新的运动坐标。

若 `Q` 正交，`QFFᵀQᵀ−I=Q(FFᵀ−I)Qᵀ`，Frobenius 范数不变，所以整体旋转不产生额外 bond 应变。

创建参考四面体时不能共面，否则 `D_b` 不可逆。很小的参考间隙也会使逆矩阵放大误差。论文的接触激活长度 `d̂`、胶带厚度与某个 bond 的创建间隙不是可以随意互换的量。

## 2. 逐步求梯度

令

$$A=FF^T-I,\quad s=h^2V\kappa,\quad\Phi_b=s\,\mathrm{tr}(A^TA).$$

对矩阵采用内积 `U:V=tr(UᵀV)`。先对残差求微分：

$$d\Phi_b=2s\,A:dA,$$
$$dA=dF\,F^T+F\,dF^T.$$

`A` 对称，两部分贡献相同，因此

$$d\Phi_b=4s(AF):dF,\qquad G_F=\frac{\partial\Phi_b}{\partial F}=4sAF.$$

再用 `dF=dD_s D_b⁻¹` 得到

$$G_D=\frac{\partial\Phi_b}{\partial D_s}=4sAFD_b^{-T}.$$

把 `G_D` 的三列记为 `g₁,g₂,gP`，顶点梯度便是

$$\nabla_{x_1}\Phi_b=g_1,\quad\nabla_{x_2}\Phi_b=g_2,\quad
\nabla_{x_P}\Phi_b=g_P,\quad
\nabla_{x_0}\Phi_b=-(g_1+g_2+g_P).$$

四顶点梯度之和为零，说明这个内部界面不会凭空产生整体线性合力。这是一个适合直接数值检查的物理不变性。

## 3. 恢复力与断裂

Newton 使用的是含 `h²` 的 `Φ_b`。断裂阈值使用 N，所以应对物理势能求导：

$$f_P=-\frac{1}{h^2}\nabla_{x_P}\Phi_b.$$

论文的 primary release criterion 为

$$d>d_{ref}\quad\land\quad\|f_P\|>f_{break}.$$

两项都要满足。压缩时即使力的模很大，也不通过这个拉伸断裂门槛。纯滑移若没有增加距离，也不一定被这条规则断开。不能把它自动扩展成完整的混合模态断裂准则。

bond 集合在一次运动求解中固定。求解结束后再创建/解除 bond；解除后，源 point–triangle 对从下一次接触搜索恢复原来的处理。其他接触对一直存在。

## 4. 一个能手算的法向特例

固定三角形，保持 P 的平面投影，只让高度从 `h₀` 变成 `λh₀`。这里 `h₀` 是几何参考高，与时间步 `h` 不同。此时

$$F=\mathrm{diag}(1,1,\lambda),\qquad
E_{physical}=V\kappa(\lambda^2-1)^2.$$

若三角形面积为 `A_T`，`V=A_T h₀/3`，法向恢复力为

$$f_z=-\frac{4V\kappa}{h_0}\lambda(\lambda^2-1)
=-\frac{4\kappa A_T}{3}\lambda(\lambda^2-1).$$

教学几何取直角三角形两直角边各 2 mm、参考高 0.2 mm；采用论文摆动实验的 `κ=10⁶ Pa`、`f_break=0.5 N`。这些几何不是论文实测 bond 形状。

| 高度比 λ | 力模 | 是否按该判据释放 |
| --- | --- | --- |
| 0.5 | 1 N | 否，处于压缩 |
| 1.0 | 0 N | 否，参考态 |
| 1.5 | 5 N | 是，张开且超过阈值 |

在这个特例中，临界张开比约为 `1.08310370`。它来自本包的合成几何和标量方程求根，不能当成真实胶带的普适断裂伸长率。

公式还表明力随 `A_T` 缩放。网格细化可能改变单对承担面积与 pair 数量，因此固定单对 `f_break` 的宏观含义需要重新标定。按面积缩放是一种待检验方案，论文没有证明它跨所有网格/接触布局都正确。

## 5. Hessian-vector product

给一个顶点方向增量 `δx`，先得到

$$U=\delta F=\delta D_s D_b^{-1}.$$

继续对 `G_F=4sAF` 求导：

$$\delta G_F=4s\left[(UF^T+FU^T)F+AU\right].$$

之后同样乘 `D_b⁻ᵀ`，再把三列分配到四个顶点。这就是局部 `12×12` Hessian 乘向量，不必先显式构造矩阵。教学程序还用 12 个基方向装配了完整矩阵，便于对称性、特征值和有限差分检查。

第一部分

$$H_{GN}[U]=4s(UF^T+FU^T)F$$

是把残差 `R(F)=FFᵀ−I` 做线性化得到的 Gauss–Newton Hessian。其二次型为

$$\delta F:H_{GN}[\delta F]=2s\|\delta F F^T+F\delta F^T\|_F^2\ge0.$$

精确 Hessian 还包含 `4sAU`，它可能引入负曲率。沿法向标量特例，

$$\frac{d^2E_{physical}}{d\lambda^2}=4V\kappa(3\lambda^2-1).$$

当 `λ<1/√3` 时为负。比如 `λ=0.5`，能量本身仍非负，但局部曲率可以是负的。能量非负不等于 Hessian 半正定。

总系统还包含惯性、薄壳和其他接触项。一个 bond 项存在负曲率，不等于总 Hessian 在该步必然不正定。

论文采用 projected Newton，但没有公开具体实现。本包只比较精确 Hessian 与 Gauss–Newton 的代数性质，没有替作者指定采用哪种特征值裁剪、阻尼或预条件策略。

## 6. 零模态与反射边界

在参考态 `F=I`，`A=0`，有

$$\delta G_F=4s(U+U^T).$$

反对称 `U` 表示无穷小旋转，曲率为零；平移不改变边矩阵。非退化四点的参考态局部 Hessian 因而有 6 个刚体零模态。

这个结论要限定在参考态等平衡点。非平衡形状下，沿有限旋转曲线能量不变，涉及 Hessian 与坐标曲线二阶导数共同抵消，不能直接要求顶点空间 Hessian 把任意旋转切向量都映射为零。

另外，`FFᵀ=I` 也包含反射 `det(F)=-1`。这个单一能量项不能独立证明不翻转或不穿透。不能由代数边界反推作者实际发生了这类失效。

## 7. 与 cluster 的二阶导数怎样接起来

bond 的顶点可能来自薄壳，也可能受 cluster 刚体坐标约束。若 `x=x(q)`，则

$$g_q=J^Tg_x,$$
$$H_q=J^TH_xJ+\sum_j(g_x)_j\nabla_q^2x_j.$$

第二项来自旋转映射的非线性，与 bond 局部 Hessian 的残差曲率是两件不同的事。

一个独立小例子能说明为什么不能默认忽略它。取 `E(x)=½||x||²`，`x(θ)=R_z(θ)y`。能量沿旋转恒定，所以对 θ 的一阶和二阶导数都为零；但 `JᵀHJ=||x'(θ)||²>0`。额外项 `x·x''(θ)` 恰好把它抵消。教学程序验证了这项抵消。

## 运行与验证

```bash
python3 /home/ligo/claude-papers/papers/tapesim-adhesive-tape-dispensing/code/bond_hessian.py
```

18 项检查覆盖 Hessian 与梯度差分、能量的二阶方向差分、对称性、参考态零模态、负曲率、Gauss–Newton 半正定、法向力特例和坐标映射曲率。结果见 `code/bond-math-validation.json`。

继续读 [三角网格 collar 实现](dynamic-collar-implementation.md)，把局部界面数学放进整步状态更新中。
