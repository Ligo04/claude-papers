# 反思与可检验的延伸

## 如果继续做研究

1. **全局位移基线**：把 S4R 输出作为 warm start，再用全局 contact QP 或小规模 SQP 以原始位姿为目标重优化。报告 S4R、全局解、multi-start 的最终 RMSD、Pen、wall time 和路径失败率，直接测量“局部增量最小 ≠ 全局最终最小”。
2. **非单调尺度路径**：在 continuation 卡住的尺度区间保存 active graph 和可行性诊断，允许 per-body scale trajectory 或离散跨越一个有限 infeasible interval；full-size evaluator 仍是唯一几何验收。
3. **物理一致的 repair**：在 QP 中加入质量权重、摩擦锥近似、速度/冲量或支撑面约束，再用 MuJoCo/PhysX settle 测量初始冲量峰值、沉降距离和任务成功率。这样才能回答“更适合仿真”的动态含义。
4. **接触查询替代方案**：论文消融显示 detection + tail 占总耗时的 73–97%，QP ≤1%。应优先比较 BVH/FCL、GPU winding/SDF、batched exact oracle 和 cache refresh policy，而不是只换 QP 后端。
5. **不确定性与证书**：对 witness feature switch、normal drift 和近重合中心做扰动实验，统计 false-clear / false-penetration；把 heuristic cache 与最终 exact audit 分成可审计的证书阶段。
6. **更丰富的几何尺度**：对薄长件尝试 per-axis/ellipsoid dilation，但必须明确“同样增加三个半轴”不是 Euclidean offset，并用 full mesh 做 clearance 和 containment 验证。

## 脆弱假设

- `s_min` 的包围球条件只是充分的粗筛，不保证任意非凸 mesh 的所有细节都分开。
- 每个中间尺度存在一条可行路径；拥挤容器在 packing fraction 约 `0.18` 附近会出现论文报告的失败趋势，nested cavity 也可能无法退出。
- frozen witness、固定法向和有限 tail budget 不提供每步可行性或最终全局最优保证。
- 近重合参考中心会让分离方向病态；feature switch 会破坏局部二阶误差假设。
- 6-DOF 小角变体限制 `||ω||∞≤0.1 rad`，不能把它当作任意大姿态重排器。
- 几何 Pen=0 不代表物理 settle、upright、摩擦接触或语言关系全部满足。博客记录的 SimReady Gate 案例中，几何证书通过后，圆形水果的 MuJoCo upright 行为仍可能失败。

## 最小验证矩阵

| 维度 | 必测项 | 通过条件 |
|---|---|---|
| 几何 | full-size triangle crossing + containment | `Pen=0`，没有漏掉 nested watertight pair |
| 路径 | `s_min→1`、QP retry、tail stop reason | `continuation_complete=true`，失败有明确 reason |
| 数值 | active set、cache interval、near-coincident centers | 扰动后不出现 silent false-clear |
| 约束 | walls/joints/upright | 约束 evaluator 与 mesh evaluator 同时通过 |
| 动力学 | MuJoCo/PhysX settle | 峰值冲量、沉降、任务成功率达应用阈值 |
| 性能 | detection、QP、tail 分解 | 在目标 N 和目标硬件上分别报告，不混用 tier |

这些是后续实验建议，不是论文已经完成的结果。
