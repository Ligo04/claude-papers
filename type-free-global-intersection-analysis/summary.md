# 摘要与结果

## 背景

IPC/ barrier 求解通常要求进入求解器的初态已经无穿透；用户编辑、姿态 retargeting、场景拼接和前一轮 solver 失败都会破坏这个前提。薄壳还缺少可靠的 inside–outside 语义：同一张布料可以折叠成深层、嵌套、开边界或非流形相交。

传统路线各有短板：GIA 先把 contour 分成 Closed、Eight、BLI 等类别，再按类别 flood-fill；检测噪声、重复/缺失 EF pair 和非流形结构会把它送入错误分支。ICM 只缩短局部交线，深穿透和多层互锁时会掉进局部极小值。

## 论文做了什么

给定一条相交 contour \(\mathcal C\)，论文固定当前节点位置 \(x\)，搜索单位球上的分离方向 \(r\in S^2\)：

1. 用 contour 几何生成候选方向；
2. 沿 \(r\) 做 vertex–face (VF)、edge–edge (EE)、face–vertex (FV) 三类线性路径测试；
3. 在 source/destination 两侧同时满足 primitive adjacency 的 hit 图上做连通分量筛选，只留下锚定于 contour 的 response band；
4. 用最大穿透深度和 support 面积定义位移代价，先离散筛选，再用球面切空间上的 Newton/Cauchy 试探细化；
5. 把最终 hit 写成普通 VF/EE penalty pair，交回同一个隐式 Newton/PCG 系统。

核心目标（论文式 (2)）为

\[
D_\mathcal C(r)=\max_{h\in S(r)}\delta_h(r),\qquad
J(r)=D_\mathcal C(r)^2\sum_{h\in S(r)}A_h,
\]

其中表面积密度 \(\rho\) 被视为常数省略。这样搜索会偏好“能一次分开整个 contour、且移动较轻 support”的方向。

## 贡献

- **Type-free response construction**：同一条评估路径覆盖开/闭 contour、自相交和跨物体穿透，不依赖七类拓扑表。
- **统一 ray evidence**：VF、EE、FV 都化成 \([r\mid a\mid b][\delta,\alpha,\beta]^T=d\) 的 3×3 系统；Cramer 形式直接给出 \(\delta_h(r)=\nu_h/(n_h^Tr)\)。
- **双侧 product-complex cluster culling**：避免远处、切向光线和脱离 contour 的几何碎片被误注入 solver。
- **混合方向搜索**：10 个几何 seed 覆盖 scatter principal axes、两侧中心和 mesh-to-contour 方向；在固定 hit basin 内用 IRLS surrogate 与球面 Newton/Cauchy refinement。
- **solver-compatible response**：用固定 barycentric weights 形成 affine constraint 和常数 Gauss–Newton block，直接融入 IP/Newton/PCG。
- **工程加速与稳定性**：adaptive \(k\)-ring、contour-side restriction 和冲突 suppression 把搜索限制在局部邻域。

## 主要数字（论文表 2–6）

| 场景/指标 | Ours | ICM | GIA/ISIR | 备注 |
|---|---:|---:|---:|---|
| Synthetic | 6,331 → 0 EF，88 iter | 691 EF | 1,977 EF | Ours wall GPU 341.410 s |
| Round Table | 48,686 → 0，320 iter | 1,840 | 13,975 | 105,614 V / 210,157 F |
| Cloth Drop | 49,124 → 0，48 iter | 953 | 72,693 | 111,208 V / 212,636 F |
| Hair Card | 168,914 → 0，889 iter | 1,428 | 55,651 | 32,579 V / 41,668 F |
| 59-case benchmark | 59/59 成功，16.51 avg iter | — | ISIR 25.45% 成功 | Ours GPU 96.5 s，CPU 158.5 s |
| adaptive k-ring | mean raw hits 60,749 → 1,360 | — | 100% valid-hit parity | 152.5× geometric-mean reduction；99.44% median |
| stiffness sweep | \(\kappa=10^8\)：59/59 | \(10^7\)：49/59 | \(10^{12}\)：42/59 | 过小会停滞，过大单精度条件数变差 |

七个 canonical contour（BB/II、BI/BI、BLI、Closed、Cross、Eight、LL）都在论文中达到 0 residual EF，通常 1–2 次迭代；论文表 2 的额外案例中 Klein Bottle 为 11 次、Cloth Cylinder 为 5 次、Cloth Sorento 为 2 次。

## 形状保真与成本

论文使用三道缓解策略：增大 bending stiffness（例如 \(\kappa_b:0.05\to1.0\)）、单步位移上限 5 mm、解缠后 200 步普通 IPC relaxation。59-case 的平均体积比从 soft 0.911 提升到 bounded+200 IPC 的 1.006；这说明“清掉 EF”与“保持形状”是两个目标。

在 SMPL_019（2,082 初始 EF）上，CPU 路径 866.3 ms 中 648.3 ms（74.8%）是 ray casting；GPU ray casting 降至 90.3 ms（7.2×），但 CSR build（204.7 ms）和 host evaluation/assembly（185.1 ms）成为主项。

## 不应过度解读

这些数字是作者在特定硬件（Core Ultra 7 265KF、RTX 5070 12 GB、32 GB RAM）、固定 \(\Delta t=1/60\) s、\(\hat d_{prox}=1\) mm 和场景参数下报告的结果。它们证明“在所评 benchmark 上可用”，不等于任意非流形、自相交或刚体空腔都能解开。
