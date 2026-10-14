# 代码对照：仓库当前实现（固定 SHA）

本节对照 `https://github.com/ChengzhuUwU/Untangling26/tree/d361b71e4ae97b2604c3b704fe31299e3b5dfb2f` 阅读。文件中的行号会随上游变化，因此以函数/结构体名为主。

## 入口和数据流

| 论文概念 | 当前代码位置 | 观察 |
|---|---|---|
| EF → contour | `Solver/CollisionDetector/intersection_resolver2.cpp`：`host_resolve_intersections_PRP`、`intersection_contour_construction_from_ext_adjacent` | 从 `narrow_phase_list_ef` 读 EF pair，用扩展 adjacency flood-fill 成 contour。 |
| runtime | `intersection_resolver2.h/.cpp`：`PrpStandardContourRuntime`、`build_standard_runtime` | 保存 object/role、boundary anchors、V/E/F candidates、hop/geodesic 距离、directions 和 combo 状态。 |
| 方向 seed | `get_raycasting_direction` | 当前代码正常输入下生成 eigen ±6、两条 contour-center/role-center 方向、global centroid radial ±2；退化时回退 proxy normal。论文正文的 `±(c1-c0)` 与当前实现并非完全相同。 |
| adaptive k-ring | `build_standard_candidates_at_level`、`evaluate_current_runtime_directions_template` | \(k_0=\max(3,\min(\lceil\sqrt{|B|}
ceil,D))\)，frontier 未收敛时倍增。 |
| VF/EE/FV | `intersection_resolver2.cpp` 的 batched evaluator；`intersection_resolver_gpu.cpp` 的 device kernels | CPU 用 cached BVH；FV 复用 reverse-VF；GPU 走 LuisaCompute batch kernels。 |
| cluster culling | `find_adjacent_hits`、`is_hit_adj_contour` 及 flood-fill 段落 | 命中图按 source/destination primitive adjacency 连接，只保留接触 contour boundary 的 component。 |
| CCR | `prp_filter_standard_candidates_by_boundary_normals`、intrinsic/current-distance helpers | 支持当前 Euclidean、rest geodesic、AND/blend 等工程模式；论文只给 current-space \(\phi_s\)。 |
| 方向细化 | `prp_optimize_runtime_directions` | trust radius 初值 0.3，接受 ratio>0.75 时放大（上限 1），拒绝时减半（下限 0.01）；方向优化 CLI 默认 0 次。 |
| response pair | `make_response_from_min_hitinfo`、`make_response_pair_from_raycasting_info` | VF/FV/EE hit 规范化成 `CollisionPairTemplate`，注入 `sa_cgB`/`sa_cgA_diag`。 |
| solver 装配 | `host_apply_untangling_constraint`、`intersection_resolver.cpp` backend dispatch | PRP 与普通 contact/elastic 能量进入同一个 Newton/PCG 系统；`use_gpu` 和 `use_gpu_untangling` 是独立开关。 |

## 代码与论文的三个重要差异

1. **比较器**：当前 `prp_is_better_response_hit_info` 先比较 `penetration_objective / contour_coverage²`，再比较 penetration depth、displacement cost、hit count 和 combo index。论文 Eq. (2) 只显式写了 `D_C² × area_sum`。coverage normalization 会改变候选偏好。
2. **响应深度**：代码使用 `prp_response_depth_offset = 3e-3` 和全局 response-depth 截断构造 `C=max(-depth-0.003,-response_depth)`，因此不能把实现简单等同于正文 Eq. (25) 的无限制 contour-wide target。
3. **默认路径**：`--prp_direction_optimization_iterations` 默认为 0；`--backend cuda` 只选择 Luisa device，不能推出 PRP contour evaluation 一定使用 GPU。要测试 GPU PRP 需显式检查/设置 `--use_gpu_untangling 1`。

## Python 入口

- `PythonBindings/tests/shared_args.py`：统一 `PRP`、`ICM`、`GIA` 方法开关；非 PRP 会关闭 GPU untangling。
- `PythonBindings/tests/demo_unit.py`：七个 canonical OBJ，支持 `--headless`、`--scene_id`、`--method`、`--export_debug`、CCR 和 direction optimization 参数。
- `PythonBindings/tests/demo_synthetic.py`：synthetic fold；README 给出 `--use_ccd_linesearch 0`、`--untangling_response_depth 0.005` 的复现实例。
- `PythonBindings/tests/test_batch_method_comparison_enhanced.py`：Jang benchmark runner；完整 `Dataset_distributed` 不随仓库分发。

## 建议的代码阅读顺序

```text
shared_args.py
  → demo_unit.py / demo_synthetic.py
    → host_resolve_intersections_PRP
      → build_standard_runtime
        → evaluate_current_runtime_directions_template
          → batched VF/EE/reverse-FV + cluster culling
            → prp_optimize_runtime_directions (optional)
              → make_response_from_min_hitinfo
                → host_apply_untangling_constraint
```

仓库 README 明确声明复杂 self-intersection 可能筛不出 restorative hits，刚体空腔/凹陷可能卡住；它是研究代码的边界，而不是异常日志可以忽略的细节。
