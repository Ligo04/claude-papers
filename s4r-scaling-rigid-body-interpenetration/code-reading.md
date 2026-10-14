# S4R 代码阅读笔记（固定提交）

这份笔记对应仓库 `Frank-ZY-Dou/Dynamics-Modeling` 的固定提交 [`91cbaf2672a6f587075ad75ceed770717d5e6e85`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/91cbaf2672a6f587075ad75ceed770717d5e6e85)，提交时间为 2026-09-19，提交说明为 `READMEs: centre the figure tables and their images`。论文是 [S4R: Scaling for Rigid-Body Interpenetration Resolution, arXiv:2609.20524](https://arxiv.org/abs/2609.20524)，对应代码目录 [Penetration_Solving](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving)。固定提交的关键源码副本保存在本目录的 [`sources/code/`](sources/code/) 下，便于离线对照；源码副本来自同一个 commit 的 raw URL，没有修改。

## 仓库定位和最短运行入口

根 README 将项目分成 `NeuralActuator`、`RigidFormer`、`Penetration_Solving` 等。2609.20524 是 S4R，所以阅读和复现入口应从 `Penetration_Solving` 开始；`NeuralActuator` 是另一篇工作（README 标注 arXiv 2607.11734），不要混为同一论文。

CPU 依赖在 [`requirements.txt`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving/requirements.txt)：

```text
numpy>=1.24
scipy>=1.10
trimesh>=4.0
python-fcl>=0.7
osqp>=0.6
```

GPU 额外需要 `warp-lang>=1.5` 和 NVIDIA CUDA；`thingi` 数据集还需要 `thingi10k`。README 的 CPU 快速开始是：

```bash
cd Dynamics-Modeling/Penetration_Solving
pip install -r requirements.txt
python examples/run_kubric.py --N 40 --seed 42
```

GPU 入口：

```bash
python examples/run_kubric.py --N 40 --seed 42 --solver gpu
```

`examples/run_kubric.py` 的调用关系是：

```text
make_scene(dataset, N, seed)
  └─ scenes/make_scene.py
       └─ mesh_collision.generate_kubric_scene / generate_hy3d_scene

solve_s4r_qp(objects, ... contact_backend="fcl_prebuilt")
  └─ s4r/s4r_qp.py
       ├─ PrebuiltFCLOracle.find_contacts
       ├─ OSQP per-step contact QP
       ├─ full-scale tail QP (ds=0)
       └─ evaluate_world_collision_meshes (full-scale final score)

write result centers/rotations back to objects
  └─ evaluate_mesh_object_scene(objects)  # example-side independent re-score
```

场景的默认密度由 [`scenes/make_scene.py`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving/scenes/make_scene.py) 给出：`spawn_xz=0.15*(N/40)^(1/3)`、`spawn_y=0.35*(N/40)^(1/3)`；Kubric 的每个对象目标尺寸为 `0.1`，`max_verts=5000`，允许重复采样；README 的基准 seeds 是 42、123、456。

## CPU solver：从尺度继续到全尺度判定

实现见 [`s4r/s4r_qp.py`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving/s4r/s4r_qp.py)，公开入口是 `solve_s4r_qp`。

1. **固定缩放中心和初始尺度。** 每个顶点按 `x_i(s)=c_i+s R_i v̄_i` 缩放；默认 `s_min=0.01`。代码先用包围球半径 `r_i=max||v̄_i||` 检查论文 Eq. (2) 的充分条件 `||c_i-c_j|| >= d_hat + s_min(r_i+r_j)+eps`。违反时调用 `separate_coincident_centroids`；当前代码会重复多轮（最多 100 轮）对违规中心对称推开，直到没有违规对或达到上限。
2. **事件尺度和候选对。** 包围球给出 first-contact/event scale；`adaptive_ds=True` 时，下一事件可改变 `ds`。`PrebuiltFCLOracle` 先在归一化 `u` 空间重用每个物体的 unit-scale FCL BVH，再以 `center/s` 改变 pose。AABB gap、SOI 事件筛选和 one-sided support `e_i(n)` 形成保守候选集；FCL 对候选对做 closest-point / collision 查询，另加整个 body 或 disconnected piece 的 containment 检查。
3. **每个增尺度步骤。** 对当前检测的 `(i,j,d,n,e_i,e_j)`，约束是
   `nᵀ(Δp_j−Δp_i) >= d_hat − d + ds*(e_i+e_j)`。
   平移默认最小范数目标 `min 1/2 Σ_i ||Δp_i||²`；可选 6-DOF 时加入小角度 `ω`、旋转权重和 `|ω|<=max_omega`，也能加入 container walls、joint rows、attraction。OSQP 求解；不可解/非有限时 `ds` 对半重试，过小则返回 `qp_failure`。
4. **缓存和 warm start。** 当 `revalidate_interval>1`，冻结 witness 的距离按
   `d̃_next = d̃ + nᵀ(Δp_j−Δp_i) − ds*(e_i+e_j)`
   解析推进；达到刷新间隔重新查询。旋转修改会使缓存失效。上一步 binding rows 还用于预测 `dq*/ds` 和 QP warm start。
5. **full-scale tail 和结果契约。** 到 `s=1` 后，锁定 `ds=0`，对 `d<d_hat` 的 full-active-set 做最多 `S4R_MAX_TAIL_ITERS`（默认 20）个 OSQP correction QP，每步阻尼到单体位移不超过 `d_hat`。随后可选 penalty cleanup（论文报告默认关闭）。最终一定用 `evaluate_world_collision_meshes` 在**全尺寸**重评分；`status="converged"` 只有 continuation 到 1、pose finite、full-scale `pen_pairs==0` 且墙/关节满足时才成立。`native_converged` 只表示 continuation 到 1 且 solver tail 自己判定 feasible，二者有意分开。

`mesh_collision.py` 的 evaluator 使用共享 FCL triangle crossing / containment 逻辑，修正 inward winding；FCL 不可用时才退到 trimesh 采样路径，后者是 contact generator 而不是严格 certificate。代码测试特别覆盖 nested body、断开的 piece containment、cavity witness normal、crossing bars 和 representation-invariant broadphase。

## GPU solver：Warp oracle + dual APGD

GPU 入口在 [`s4r_gpu/s4r_gpu_native.py`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving/s4r_gpu/s4r_gpu_native.py)，核心 QP 在 [`dual_apgd.py`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving/s4r_gpu/dual_apgd.py)：

```text
primal: min_{Δp∈R^(3N)} 1/2 ||Δp||²  s.t. A Δp >= b
row k:  -n_k on body i, +n_k on body j

dual:   min_{λ>=0} 1/2 λᵀ(AAᵀ)λ - bᵀλ
recover Δp*=Aᵀλ*
```

`DualAPGD` 在 Warp 上做 FISTA/APGD：`y=λ+θ(λ−λ_prev)`，`λ_new=max(0,y−(AAᵀy−b)/L)`；`Aᵀ` 是 atomic scatter，`A` 是 per-contact dot，避免构造 dense matrix。源码当前的 `_certified_L` 用 actual normals 构造 sparse `A@A.T`，取 Gershgorin row-sum 上界，并与 degree bound 取较小者，再乘 `L_safety=1.05`；它不是 finite power iteration 的 Rayleigh quotient。

接触检测在 [`warp_pair_contact_v3.py`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving/s4r_gpu/warp_pair_contact_v3.py)（V3）或 [`oracle_v4.py`](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/blob/91cbaf2672a6f587075ad75ceed770717d5e6e85/Penetration_Solving/s4r_gpu/oracle_v4.py)（V4）。Warp narrow phase 采样 vertices、每条 edge 的 interior points，并有 edge-ray crossing pass；通过 winding-number SDF 的 packed argmin 选 witness / normal。GPU oracle 是 contact generator；`s4r_gpu_native.py` 在 full scale 先做 GPU tail，再默认执行 exact `PrebuiltFCLOracle` full-scale recheck/correction，返回 `fcl_verified`，最后仍由 shared evaluator 给 `pen/status`。

当前 GPU driver 默认参数是 `apgd_max_iter=500`、`apgd_check_every=8`、`apgd_use_graph=False`、`tail_max_iters=50`、`verify_with_fcl=True`；CUDA graph 变体通过 `apgd_use_graph=True` 才启用，图迭代数默认 80。

## 当前代码和论文文字的差异（阅读时要分开）

这些差异是固定提交代码与论文正文/实验描述的对照，不表示论文算法主旨改变：

- **start separation：论文 single sweep，当前代码 iterative。** 论文 Sec. 3.2 明确说起始尺度违规 pair 的 deterministic perturbation “visits offending pairs once rather than iterating the sweep to a fixed point”。当前 `separate_coincident_centroids` 的 docstring 明确要求多轮，因为分开一对可能让另一对重新违规，`for p in range(max_passes)` 默认最多 100 轮；所以实现比论文所述 single pass 更强地修复 admissibility，并把额外位移计入 RMSD。
- **GPU Lipschitz：论文 power iteration，当前代码 certified Gershgorin。** 论文 Sec. 4.1 描述 GPU dual-APGD 用 power-iteration estimate、膨胀 15.5%；当 row 数不超过 4 时才使用 Gershgorin +5% margin。固定提交 `DualAPGD.solve`/`_certified_L` 改为由 actual-normal sparse Gram 的 Gershgorin row-sum（再乘 5%）给上界；没有论文那种 power-iteration 分支。当前 driver 生产默认也不是论文固定 60-iteration CUDA graph，而是 500-iteration 非 graph APGD，除非显式打开 `apgd_use_graph`。
- **sampled oracle 与 full FCL：论文描述 GPU solver 的 Warp oracle 为 FCL-free sampled winding-SDF，并统一用 final evaluator；当前 driver 明确新增/保留 exact FCL end refinement。** `verify_with_fcl=True` 时，在 GPU tail 后以 FCL 精确检测 triangle crossing / containment，并运行 correction iterations；`fcl_verified=True` 才代表 exact recheck 通过。不能把 sampled GPU contact generator 单独当作最终几何证书。
- **当前 broadphase 的 0.9 band。** CPU `PrebuiltFCLOracle._broadphase_pairs` 和 GPU V3/V4 都计算 `s_contact=max(0,(||c_i-c_j||−d_hat)/(E_i+E_j))`，并保留 `s+ds >= 0.9*s_contact`。这个 10% band 是 SOI event cull 的实现护栏；AABB gap 仍按 `(ds*(E_i+E_j)+d_hat+extra_margin)/s` 的保守 margin 过滤。论文 Sec. 3.4.1 同样说明使用 `0.9` 的 event-bound slack，但它是 culling heuristic，不是对线性化 QP 的几何可行性证明。
- **tail/defaults 与报告数字。** 论文正文的 GPU 实验写 fixed 60-iteration CUDA-graph schedule、tail `K_tail=20`；当前提交 API 默认 GPU `apgd_use_graph=False`, `apgd_max_iter=500`, `tail_max_iters=50`，CPU tail 默认 20。README 的作者 benchmark 表则报告 CPU N=40/100/500/1000/2000/3000 solve `0.35/0.98/6.0/15.7/37.2/79.1 s`、RMSD `0.0400/0.0366/0.0359/0.0365/0.0380/0.0388`，GPU 对应 `6.7/7.1/10.7/17.0/31.2/54.4 s`、RMSD `0.0372/0.0356/0.0346/0.0346/0.0358/0.0362`，硬件注明 Xeon E5-2680 v4 + RTX 2080 Ti。这些是 README/论文作者数字，不是本机复测。

## 测试、运行边界和未验证内容

仓库声明的 CPU 回归命令是：

```bash
cd Penetration_Solving
python -m unittest discover tests
```

GPU 回归命令是：

```bash
python -m unittest tests.test_gpu_contracts -v
```

本次工作只做了固定 commit 的远程源码读取和静态调用关系核对，**没有在上游完整 checkout 上实际运行 `run_kubric.py` 或完整测试套件**，因此不把作者 README 的 timing/pen/RMSD 作为本机验收。曾将少量源码拼接到临时目录作环境探针，CPU contract 中需要 `rtree` 的 containment/closest-point 用例因临时环境缺包报错，GPU 用例因临时目录没有 CUDA/Warp import 关系而跳过；这不是上游 checkout 的验证结果。要做真正复现，应保留完整 Git checkout、安装上述依赖（尤其 `python-fcl`, `osqp`, `rtree`），再按 README 命令运行，并记录本机 GPU、driver、CUDA、commit 和数据集版本。

## 与 “To Scale or Not to Scale” 的阅读连接

博客把 S4R 放在“scaling for resolution”一节：整场景统一缩放不会改变交叠关系，而围绕固定 reference centers 缩小每个 body 才会制造 clearance；这和 Coverage Axis 中用于 shape abstraction 的 ball-radius dilation / set-cover 是不同的 tolerance 语义。S4R 的 scale 是求解路径参数，最终尺寸被规定为 1；恢复 full scale 后，局部 correction 历史仍会留下 layout bias。

博客还强调三点，代码中都有对应证据：

1. scale continuation 把一个 deep-overlap 全局线性化换成多个 shallow-contact QP，但并不让完整几何问题变凸；feature switch、cavity、nested body 和 active-set conditioning 仍可能使局部模型失效。
2. 每一步 minimum-norm 不等于最终 displacement minimum；早期选择会改变后续 contacts，故 `rmsd` 是路径结果。
3. 几何修复、关系约束和物理稳定不是同一个成功标志。`SimReady_Gate` 在同一仓库上层把语言约束编译为 scale-space placement/rows，再做 mesh verifier 与 MuJoCo settle；修复通过不保证后续重力/摩擦/姿态行为都通过。
