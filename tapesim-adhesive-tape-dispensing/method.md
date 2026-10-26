# 方法推导与实现阅读

本节以 [论文 v2 的 III 节](https://arxiv.org/html/2609.28766v2#S3) 为准。标为“推导”或“建议”的内容是学习材料，不是作者公开源码。

## 1. 每步到底在优化什么

胶带背材是三角薄壳。膜采用 StVK，弯曲采用 discrete-shell bending；两者使用平坦 rest mesh，卷绕不会把弯曲形状重设为无应力形状。

无黏附时，下一步坐标最小化

$$\Phi_0(q)=\mathcal I(q)+\mathcal E_{el}(q)+\mathcal E_{nc}(q)+\mathcal E_{fric}(q).$$

惯性项约束状态接近含外力的预测位置，弹性项抵抗形变，法向 barrier 抵抗接近，摩擦项抵抗切向滑动。文中的势能项已乘 `Δt²`。Projected Newton 解非线性问题，PCG 解线性化系统，CCD 与线搜索限制更新。

接触半厚度为 `ξ`。相同厚度的两张薄壳中面在 `2ξ` 处接触，不能默认中面到零距离才算接触。

## 2. cohesive 界面

对一个黏附 point–triangle 对，历史变量 `β∈[0,1]` 表示黏附强度。法向距离为 `d`，参考间隙为 `d*`，lagged 接触坐标中的切向滑移为 `Δx_t`：

$$\mathcal E_a=\Delta t^2\beta^2\left[\frac{C_n}{2\hat d}(d-d^*)^2+\frac{C_t}{\hat d}\|\Delta x_t\|^2\right].$$

`Cn,Ct` 的单位为 N，`d̂` 是接触激活长度。背面也参与碰撞，但只有 sticky face 能形成黏附。

运动求解结束后再更新历史：

$$\beta^{k+1}=\operatorname{clip}_{[0,1]}(\beta^k+\Delta t\dot\beta).$$

记 `a_n=4[(d-d*)/d̂]²`，`a_t=4(Ct/Cn)||Δx_t/d̂||²`，则论文给出的 rate 为

$$\dot\beta=\begin{cases}
\frac{10}{\eta}[W-\beta(a_n+a_t)]_-,&p<0,\\
\frac{100\gamma}{p_b}[p-\beta p_0]_+ +\frac{10}{\eta}[W-\beta a_t]_-,&p\ge0.
\end{cases}$$

`[z]_-` 是 `min(z,0)`，不是正的负部幅值。`p` 是压缩量，`p_b` 来自接触带中点的 barrier-only 压缩；`p_b=0` 时增长项为零。完整 `p` 定义见原文 Eq.3 附近。压缩促进黏合，张开/滑移促进脱黏。本教学 demo 没有实现这套 rate law。

## 3. virtual tetrahedron bond

四个顶点仍是原 point–triangle 对的顶点。参考边矩阵 `D_b` 在创建时记录，实时边矩阵为 `D_s`：

$$F_b=D_sD_b^{-1},\qquad V_b=|\det D_b|/6,$$
$$\mathcal E_b=\Delta t^2V_b\kappa_b\|F_bF_b^T-I\|_F^2.$$

`κb` 的单位是 Pa。这个 bond 抵抗张开、压缩和滑移，允许四个顶点一起平移/旋转。源对的 cohesive、normal barrier、friction 三项被替换，其他对保持工作。

推导：令 `A=FFᵀ−I`、`s=Δt² V κ`，则

$$\frac{\partial\mathcal E_b}{\partial F}=4sAF,\qquad
\frac{\partial\mathcal E_b}{\partial D_s}=4sAFD_b^{-T}.$$

若 `D_s=[x_1−x_0,x_2−x_0,x_P−x_0]`，后三个顶点的梯度分别是该矩阵的三列，`x_0` 的梯度是它们的负和。这还提供平移不变性的数值检查。

断裂使用物理恢复力，必须去掉时间步缩放：

$$f_b=-\frac{\partial(\mathcal E_b/\Delta t^2)}{\partial x_P}.$$

同时满足 `d>d_ref` 和 `||f_b||>f_break` 才断开。压缩下，即使力的模很大，也不走这条拉伸断裂判据。断裂在步边界执行，源对从下一次接触搜索开始恢复普通 IPC。

工程建议：创建时拒绝奇异或病态 `D_b`；不要把每步变化的形状当成新参考，否则会抹掉刚度与历史。`||FFᵀ−I||²` 单独允许反射，这是代数性质，不是对作者实际失败的报告。

## 4. cluster 的坐标与导数

成员顶点保存局部坐标 `y_i`：

$$x_i=R_py_i+t_p.$$

cluster 用 6 DOF；hub 是独立物体。cluster 内部弹性和内部接触/黏附项省略，边界壳元素和外部接触仍耦合整个系统。

设顶点坐标对求解坐标的 Jacobian 为 `J`，链式法则给出

$$g_q=J^Tg_x,$$
$$H_q=J^TH_xJ+\sum_j(g_x)_j\nabla_q^2[x]_j.$$

第二项来自非线性旋转映射。直接保留 `JᵀHJ` 属于近似，不能称为完整 Hessian。刚体旋转下顶点走弧线，论文使用 screw trajectory 上的保守推进来约束 CCD，不能把直线顶点 CCD 的保证直接搬过来。

质量按元素归属分配给 cluster；壳顶点仍保留 cluster 外 incident elements 的贡献。形成 cluster 时速度投影到刚体运动。恢复薄壳顶点时继承

$$v_i=v_p+\omega_p\times(x_i-c_p).$$

rest geometry 始终保留，不能把释放瞬间的弯曲形状重设为无应力形状。

## 5. 前沿更新的完整含义

运动求解期间，成员与历史保持固定。求解后依次更新界面、检测前沿、补充 collar。

1. 保存旧自由区 `F_k`；候选只取它一条图边外的 collar 顶点。
2. 比较独立壳位置与刚体携带位置：`g_i=||x_i−(R_p y_i+t_p)||`。
3. `g_i>δ` 才可能离卷。Bond 还要求发生 release event 且没有剩余 live support；cohesive 版本不要求 `β=0`。
4. 所有候选相对同一个旧 `F_k` 判断，再一次性生成 `F_{k+1}`。新释放顶点不能在同一次更新中继续触发下一圈候选。
5. 重算到自由区的图距离；cluster 元素中，只要某个角点距离小于 `N` 条边，就恢复为可变形区域，形成新的 collar。
6. 恢复原来的 cohesive `β`，或原 bond 与参考形状；转移质量、坐标与速度。

因此 `F_k⊆F_{k+1}`，`C_{k+1}⊆C_k`。已释放材料可以再黏到目标上，但不能因此回到原卷的 cluster。

```text
初始化：平坦 rest mesh → 卷绕/稳定 → 保存界面历史 → 建 cluster/collar
for 每个时间步:
    固定本步的 interface set 与 cluster membership
    用隐式薄壳/刚体/接触势能做 Newton-PCG，受 CCD 与线搜索限制
    更新 cohesive history 或 bond create/release
    candidates = old_free 的图邻居 ∩ old_collar
    同步判断 deformation gate 与 support/release 条件
    new_free = old_free ∪ accepted
    按 new_free 图距离恢复下一批 cluster 元素
    恢复历史、质量与刚体速度；更新边界及碰撞数据
    若满足切割条件，在步边界重建受影响拓扑
```

教学代码使用一维材料行图演示候选快照和单调性。它用顶点图层模拟 collar，省略真实三角形“任一角点距离 < N”的元素规则，不能把它当成卷绕网格实现。

## 6. 切割

$$p_{max}=\max_e\frac{\sum_{c\to e}F_{n,c}}{A_e}.$$

刀片接触的 IPC 法向力按 tape 元素分配，面积采用 rest area 或 incident area 份额。阈值连续达到两个时间步才触发。示例阈值是 1 kPa；这是配置的数值触发条件，不是测得的普遍胶带断裂强度。

在步边界把刀刃曲线与局部三角形投影到平面，移除有限宽 kerf，再三角化保留材料。新顶点从父三角形插值状态，质量由保留 rest area 决定；碰撞连接和 grasp 约束也需重建。割缝应超过新边缘的合计接触厚度，避免产生重合接触面。

## 7. 参数及敏感性

| 参数 | 文中值或规则 | 学习时要关注的影响 |
| --- | --- | --- |
| `Δt` | 通常 0.01 s；刀压入 0.005 s；封箱 1/30 s | 时间离散、断裂采样和计时不可混比 |
| collar `N` | 10 graph edges | 物理宽度随网格改变；过窄可能限制弯曲，过宽增加成本，此为工程推断 |
| `δ` | 摆动实验 0.9 mm | 前沿推进门槛；不同任务不可无来源照搬 |
| `κb, f_break` | 摆动实验 1e6 Pa、0.5 N | 弹性与释放阈值应分开理解；单对阈值可能有网格依赖 |
| Newton tolerance | `1e-2` 或 `1e-3 m/s` 等 | 同一值在不同表示下不保证同一运动误差 |
| tabletop `Cn` | 0.675 N；敏感性实验 20×、100× | 只匹配策略总成功率可能选到物理不合理的参数 |

复现时还需要卷绕预处理、接口激活距离、几何厚度、质量归属、数值 Hessian/PCG 细节及控制时序。论文给出了主要机制，不足以还原全部工程默认值。

## 8. 深入推导与三角网格示例

`deep-dive-bond-math.md` 继续推导局部 12×12 Hessian、Gauss–Newton 近似、压缩负曲率和力阈值的法向特例。`dynamic-collar-implementation.md` 区分元素归属与顶点坐标，解释严格 `<N` 规则、边界质量贡献、质量中心变化与历史恢复。

新增 `code/bond_hessian.py`、`code/triangular_collar.py` 可独立运行。它们比原一维材料行 demo 更细，但仍不包含作者的动力学求解器。
