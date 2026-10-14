# S4R 方法：从深穿透到可验证的全尺度布局

本文按论文 v1（arXiv:2609.20524v1）解释默认的 translation-only S4R。公式页码指 PDF 页码；实现文件名指公开代码 commit `91cbaf2672a6f587075ad75ceed770717d5e6e85`。代码没有在本机完整重跑，因此代码说明是静态核对，不是本轮运行收据。

## 1. 目标问题与符号

有 (N) 个刚体，初始位置与姿态为 (x_0=(p_0,R_0))，其中 (p_i\in\mathbb R^3)、(R_i\in SO(3))。概念上的全局目标是

\[
\min_{p^\star,R^\star}\sum_i\left[\frac12\|p_i^\star-p_{i0}\|^2+\frac\beta2d_{SO(3)}(R_i^\star,R_{i0})^2\right]
\quad\text{s.t.}\quad d_{ij}(x^\star)\ge\hat d,\ \forall i\ne j.
\]

(d_{ij}) 是带符号的 mesh clearance：分离为正、内部重叠为负；(hat d>0) 是内部求解 margin。该目标只作为 reference goal，S4R 不在全局上计算它；默认解只更新平移，姿态固定。[论文 §3.1，PDF p.4](https://arxiv.org/pdf/2609.20524#page=4)

每个 mesh 先重心化到归一化包围盒中心 (c_i=p_i)，其顶点偏移记为 (\bar v)。以 `target_size=0.1` 归一化最长包围盒边，定义包围半径

\[
r_i=\max_{\bar v}\|\bar v\|.
\]

## 2. 缩放初始化

尺度 (s\in(0,1]) 下，顶点轨迹是

\[
v_i(s)=p_i+sR_i\bar v.
\]

两个物体的包围球在尺度 (s_{min}) 下有足够间隔的充分条件为

\[
s_{min}<\min_{i<j}\frac{\|c_i-c_j\|-\hat d}{r_i+r_j}.\tag{2}
\]

论文所有报告实验固定 `s_min=0.01`。若参考中心对违反 Eq.(2)，实现沿 pair 轴确定性扰开，目标距离为 (hat d+s_{min}(r_i+r_j)+\epsilon_{sep})，其中 (epsilon_{sep}=10^{-6})。恰好重合的中心使用固定轴。扰开位移计入 RMSD；论文报告小尺度 mesh evaluator 在所有主实验中无负分，主实验最小 gap 至少 0.014、stress 场景至少 0.009。[论文 §3.2、App.A.6–A.7，PDF pp.5、22–23](https://arxiv.org/pdf/2609.20524#page=22)

要点是：整个 scene 统一缩放不会改变相互交叠；S4R 是每个 body 围绕其中心做 homothetic shrink，因此相对中心距离产生 clearance。作者博客将这点与 Coverage Axis 的 additive dilation 明确区分。

## 3. 一步接触模型

### 3.1 接触方向与单侧支持

对 pair ((i,j))，单位方向 (n_{ij}) 从 (i) 指向 (j)：分离时取最近点连线，穿透时取检测器报告的最深 contact 方向；在一次 cache window 内冻结。对任意方向 (n)，定义单侧支持

\[
e_i(n)=\max\left(0,\max_{\bar v} n^TR_i\bar v\right),
\quad E_{ij}=e_i(n_{ij})+e_j(-n_{ij}).
\]

均匀膨胀 (ds) 时，冻结 witness 在 (n_{ij}) 投影上的闭合量至多为 (dsE_{ij})。若最近点分支稳定，真实 witness 闭合系数不超过 (E_{ij})，所以该项对纯 inflation 是保守的；它不保证 post-step mesh evaluator 的真实分数。[论文 §3.3.1、§3.3.4，PDF pp.5–7](https://arxiv.org/pdf/2609.20524#page=6)

### 3.2 Gap predictor 与 active set

从 (s_k) 到 (s_{k+1}=s_k+ds)，FCL 精确检测或缓存得到分数 (\tilde d_{ij}^k)。冻结 witness 的预测为

\[
\ell_{ij}^{k+1}=\tilde d_{ij}^{k}+n_{ij}^T(\Delta p_j-\Delta p_i)-dsE_{ij}.\tag{3}
\]

强制 (ell_{ij}^{k+1}\gehat d) 后，得到线性右端

\[
b_{ij}=\hat d-\tilde d_{ij}^{k}+dsE_{ij}.
\]

活动集不是按未知 post-step score 决定，而是

\[
A_k=\{(i,j):\tilde d_{ij}^k-dsE_{ij}<\hat d,\text{ 且 witness/normal 有效}\}.
\]

每个 body pair 仅保留一个代表 witness；非凸 mesh 的其他 primitive 分支通过刷新和 tail 再发现。[论文 §3.3.2，PDF p.6](https://arxiv.org/pdf/2609.20524#page=6)

### 3.3 最小范数 QP

默认 translation-only 子问题为

\[
\min_{\Delta p^\star}\frac12\sum_i\|\Delta p_i^\star\|^2
\quad\text{s.t.}\quad
n_{ij}^T(\Delta p_j^\star-\Delta p_i^\star)\ge b_{ij},\quad(i,j)\in A_k.\tag{4}
\]

只为活动接触图中的 body 建变量；矩阵是 block-sparse。论文默认 CPU 使用 OSQP；GPU Warp 版本使用 dual-APGD（float32、每个 QP 固定 60 iterations、CUDA graph），其 GPU oracle 与 CPU FCL 不同，最终仍共享 mesh evaluator。[论文 §3.3、§4.1，PDF pp.6、9–10](https://arxiv.org/pdf/2609.20524#page=9)

可选 6-DOF 版本加入每个 body 的小角增量 (omega_i)，接触行增加杠杆臂项

\[
(y_j\times n_{ij})^T\omega_j-(y_i\times n_{ij})^T\omega_i,
\]

目标增加 (\beta\|\omega_i\|^2/2)，并限制 (|\omega_i\|_\infty\le0.1) rad；论文使用 (\beta=r_{max}^2)。这不是默认路径：Kubric (N=40) 中 RMSD 0.036→0.031，但 0.16→2.4 s。[论文 §4.6、App.A.10，PDF pp.12–13、24–25](https://arxiv.org/pdf/2609.20524#page=13)

## 4. 路径加速

### 4.1 Scale-of-impact（SOI）事件跳跃

固定姿态、只膨胀时，包围球给出保守事件尺度

\[
s_{soi}^{cons}=\frac{\|c_j-c_i\|-\hat d}{r_i+r_j}.
\]

每次 full detection 重新计算；若 (s+ds<0.9s_{soi}^{cons})，可丢弃在下一步前不可能进入 margin 的 pair。默认 `dsmax=0.05`，事件跳最多 (3ds_{max})；无接触步可取 (2ds_{max})。0.9 是 10% safety factor，不是几何证书；QP 平移可能把远处 pair 带入接触，后续 refresh/tail 才能发现。[论文 §3.4.1，PDF p.7](https://arxiv.org/pdf/2609.20524#page=7)

### 4.2 Frozen-witness cache

默认每 `M=3` 个 scale steps 做一次 FCL full detection。其余 (M-1) 步用

\[
\dot d_{ij}^{k+1}=\dot d_{ij}^{k}+n_{ij}^T(\Delta p_j^\star-\Delta p_i^\star)-dsE_{ij}.\tag{5}
\]

缓存的 normal 不更新，prediction 不是新的 detector score；每次 refresh 清空漂移。论文 M 消融在 N=40 中：M=1/2/3/5/10 时间 0.234/0.186/0.157/0.149/0.160 s，RMSD 0.033/0.034/0.036/0.036/0.038；M=10 已有 stale-cache over-push 的迹象。[论文 §3.4.2、Table A14，PDF pp.7、28](https://arxiv.org/pdf/2609.20524#page=28)

### 4.3 Warm start 与 damping

固定 row set、normal 和 closure coefficient 时，参数化 QP 对 (ds) 的解是 piecewise affine，可用上一解的 regularized sensitivity warm-start；这只描述 frozen local trial QP，不说明真实轨迹永远 affine。QP 输出后使用

\[
\alpha=\min\left(1,\frac{\hat d}{\max_i\|\Delta p_i^\star\|}\right),
\quad p_i\leftarrow p_i+\alpha\Delta p_i^\star.
\]

它只限制一次更新的 body displacement，不保证 damped update 满足未阻尼的 QP rows。[论文 §3.4.3、§3.5.2，PDF pp.8–9](https://arxiv.org/pdf/2609.20524#page=8)

## 5. Full-scale tail refinement

尺度达到 1 后，先 full FCL detection，再固定 (ds=0) 反复修正。tail 活动集包含所有 full-scale AABB overlap 且 (\tilde d_{ij}<\hat d) 的 pair，既包括 penetration pair，也包括 near-contact (0\le\tilde d_{ij}<\hat d) 的邻居，以降低“修一个 pair、撞上另一个邻居”的连锁效应。每轮重新检测并求同类最小范数 QP。

停止条件：检测到 Pen.=0；或达到 `K_tail=20`；或 penetration pair 数和最深 violation 连续 3 轮无改善；或 QP 没有 accepted solution。论文明确声明：零穿透是这个 bounded solver 的经验结果，而不是定理。[论文 §3.5，PDF pp.8–9](https://arxiv.org/pdf/2609.20524#page=8)

## 6. 论文 Algorithm 1 的可执行伪代码

```text
normalize meshes; compute r_i; perturb pairs violating Eq.(2)
s = s_min = 0.01; dscap = +∞; build BVHs/event list
full_detect(); cached_gap = detector_score
prev_update = 0; prev_ds = 0
while s < 1 and attempts < K_max (200):
    event_ds = SOI_event_increment()
    ds = min(event_ds, dscap, 1-s)
    if refresh scheduled or forced:
        full_detect(); score = detector_score; cached_gap = score
    else:
        cached_gap = Eq.(5)(cached_gap, prev_update, prev_ds)
        score = cached_gap
    A = {(i,j): score_ij - ds*E_ij < d_hat and witness valid}
    (delta_p, status) = solve_sparse_min_norm_QP(A, optional_walls)
    if container-infeasible: return incomplete
    if status != solved:
        dscap = ds/2
        if dscap < ds_min (1e-6): return qp_failure
        force refresh; continue
    p += damping(delta_p); s += ds
    prev_update = delta_p; prev_ds = ds; dscap = +∞; attempts += 1
if s < 1: return incomplete after evaluator
tail_refine_at_s_1(max 20 iterations, stagnation guard 3)
evaluate full-size poses with strict FCL mesh evaluator
return poses, report, status
```

## 7. 默认超参数与实现对应

| 参数 | 默认值 | 作用/边界 |
|---|---:|---|
| (hat d) | 0.02 | S4R/QP-LCP/Soft-Penalty 内 margin，不是最终 Pen target |
| (s_{min}) | 0.01 | 缩放起点；每场景按 Eq.(2) 检查 |
| (ds_{max}) | 0.05 | 粗尺度步长，非 certified bound |
| `M` | 3 | FCL full refresh interval；GPU pipeline 不用 cache、每尺度重检测 |
| (epsilon_{sep}) | (10^{-6}) | 过近中心扰开 padding |
| `ds_min` | (10^{-6}) | QP 失败时 step-halving floor |
| `K_max` | 200 | scale-loop attempt budget，论文报告中未达到 |
| `K_tail` | 20 | full-scale tail 上限 |
| damping (alpha) | 动态 | 每 body 单步位移不超过 (hat d) |
| OSQP tol | (10^{-6})，tail (10^{-7}) | CPU QP；max iterations 4000，tail 8000 |
| rotation | off | 开启时 (|\omega\|_\infty\le0.1) rad，(\beta=r_{max}^2) |

代码 README 只锁定核心依赖范围（NumPy、SciPy、trimesh、python-fcl、OSQP；GPU 额外需要 Warp/CUDA）。本次静态探针环境曾显示 Python 3.13、NumPy 2.4、SciPy 1.17、OSQP 1.1.1、python-fcl 0.7.0、trimesh 4.11、Warp 1.14，但没有在该环境完成上游基准；实际环境版本、mesh repair、FCL contact depth 约定都会影响速度与边界。[论文 App.A.6–A.8，PDF pp.22–24](https://arxiv.org/pdf/2609.20524#page=22)

## 8. 复现风险与正确的验证顺序

1. **版本风险**：论文 v1 的 Table 2 是 3-seed mean；代码 README commit `91cbaf2` 的 seed-42 示例为 CPU (N=40) RMSD 0.0400、0.35 s，(N=1000) RMSD 0.0365、15.7 s，和论文数字不同。两者不可混算。
2. **数据风险**：HY3D 过滤、decimation、watertight/manifold 条件、Thingi10K 流式下载都会改变 candidate meshes；先记录数据版本和 git SHA。
3. **评估风险**：必须在 full-size bodies 上做共享 mesh evaluator；只看 solver 自己的 `native_converged` 或小尺度状态会误报成功。nested watertight pair 还需 containment audit。
4. **协议风险**：论文时间含 setup + solve，但排除场景生成、mesh loading、最终共享评估；GPU 要包含 runtime/JIT 初始化。换计时边界会改变比较。
5. **求解器风险**：OSQP 的 tolerances 只描述 CPU 版本；GPU dual-APGD 的固定 60 iterations 有 post-solve residual diagnostics，但不把 CPU OSQP 的精度声明套到 GPU。
6. **路径风险**：随机场景与近重合中心会改变扰开位移；密度、容器墙、nested cavity 可能让 continuation 在中间尺度停住，即使 full-scale 另有可行布局。
7. **物理风险**：S4R 只建立几何无穿透初态，不保证重力/摩擦/惯性下稳定，不保证 robot policy 或 manipulation 成功。博客的 SimReady Gate 例子中，97 个几何谓词满足后 MuJoCo settle 仍可能破坏水果 upright 约束。

建议验证顺序：先锁定 commit、环境和数据；运行仓库自带 tests；用 seed 42 的 N=40 demo 检查 `continuation_complete`、full-size `fcl_verified`/mesh Pen.=0 和 `tail_stop_reason`；再扩展 N 与 seeds；最后分别做物理 settle 和任务级验收。不要把“端口/进程运行”“solver status”或“近似 oracle 无接触”当作 mesh-level 全尺度通过。
