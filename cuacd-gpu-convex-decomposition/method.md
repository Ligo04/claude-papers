# 方法拆解

## 输入与输出

输入为三角网格 V、T 与凹度阈值 τ；输出是若干部件及其凸包。学习时把“切出来的网格片段”和“交给碰撞引擎的凸包”分开：后者一般填入少量非实体区域，误差由凹度控制。公开 API 返回 part_vertices、part_triangles、hull_vertices、hull_triangles。

```text
输入网格 → 连通分量拆分 → 构建 hull / 测凹度
                              ↓ 超过 τ
                    轴向候选 + 凹边候选
                              ↓
               候选切割 → 两侧凸包 → 体积残差
                              ↓
               向更差子块 look-ahead，选择首刀
                              ↓
                    应用首刀 → 再拆连通分量
                              ↺
              达标部件 → 合并可合并凸包 → 输出
```

这是论文层面的算法流程。公开实现还由主机协调迭代和 kernel launch，详见 source-study.md。

## 1. 凹度：搜索代理与终止指标

论文 Eq. (1)：

```text
cost(P) = max(k R_v(P), d_H(P, H(P)))
k = 0.3
```

H(P) 是部件凸包；R_v 是凸包体积与部件体积之差所对应球的半径。由球体积公式可展开为：

```text
ΔV = V(H(P)) − V(P)
R_v = (3 ΔV / (4π))^(1/3)
```

这个展开是对定义的推导，默认网格闭合、体积方向正确且 ΔV 非负。数值实现应处理小负误差，不能忽略大负值暴露的拓扑问题。

双向 Hausdorff 概念为：

```text
d_H(A,B) = max(sup[a∈A] inf[b∈B] ||a−b||,
               sup[b∈B] inf[a∈A] ||a−b||)
```

实现均匀采样表面点，用 LBVH 求点到另一表面三角形的最近距离，再取最大值，因此是估计而非连续曲面上的精确 supremum。搜索用便宜的 kR_v；终止检查加入 Hausdorff，避免仅体积小就接受细长、接触重要的偏差。

## 2. 候选与前瞻

默认轴向候选总预算 W_top=30，凹边采样预算 N_ce=16，每条凹边最多四个平面：两个二面角平分方向的正负偏移面，加两个邻接面的对齐平面。根预算最多 94。论文部分文字“each principal axis at W_top positions”与后文公式易造成误读；这里以 Table 3、§5.3、Appendix B 的总预算口径为准，公开 CLI width 也是总轴向预算。

树向更差的子部件继续展开，默认 D=2、W_sub=5。比较路径上最差子部件代价的平均值，选出最佳路径对应的第一刀。它不是穷举无限深的全局最优分解，也不是简单选当前一步最小凹度。

```python
# 解释性伪代码；省略内存、异常、终止预算。
work = connected_components(input_mesh)
while any(part.cost > tau for part in work):
    for part in over_threshold_parts(work):  # 部件与候选都提供并行度
        candidates = axis_planes(part, 30) + concave_edge_planes(part, 16)
        scores = [lookahead_worst_child(part, plane, depth=2, width=5)
                  for plane in candidates]
        children = cut_and_cap(part, candidates[argmin(scores)])
        replace(part, connected_components(children))
return merge_under_threshold(work, tau)
```

## 3. Warp 原语

- 扫描：lane ℓ 处理 ℓ、ℓ+32、ℓ+64……，最后一批必须屏蔽无效 lane。
- 过滤：ballot 把保留条件收集为 32-bit mask；lane ℓ 的目标位置是 `popcount(mask & ((1<<ℓ)-1))`，再加前面各批保留数。
- 排序：warp 私有待处理段栈，三路 quicksort；32 个等距样本选 pivot，短段用 bitonic。重复元素进入等值区，避免无效递归。
- 对称工作：16 个双 lane 组处理相关两侧，例如凸包桥边两端。
- 分治：为子问题安排 lane/sub-warp；分支不规则仍会降低利用率。

教学 demo 用 CPU 模拟 mask/位置分配和排序语义，短段调用 Python sorted；不模拟 CUDA 内存模型或真实性能。

## 4. 设备端 heap

warp b 路由到 arena b mod 64。每个 arena 的 32 个二次幂大小区间各拆成两半，共 64 个 sub-bin；64-bit bitmap 用于找非空列表。lane 0 取得 arena 锁、分割/分配块、释放时通过 boundary tags 合并相邻空闲块。

合并相邻空闲块（coalescing）不等于搬移活跃分配（compaction）。跨 arena 碎片、sticky pool 不归还上层 allocator 是论文列出的限制。论文/README 写默认预留 70% 空闲显存，所查 csrc/heap.c 实际为 80%；部署应显式配置。锁竞争、容量不足和寄存器压力也不会因 warp-centric 自动消失。

## 5. 切割与封口

对顶点判断平面两侧 → 找跨平面边 → 对无向边 key 排序去重 → 生成交点 → 分裂三角形 → 再排序边以找只出现一次的边界 → 拼边界环 → 区分内外环并桥接孔洞 → ear-clipping 三角化 → 压紧索引写回。

内部流形边出现两次，边界边出现一次，是恢复边界的前提；非流形、开口、共面交点和近退化面都需要专门处理。论文在标准比较中先 PaMO，不能跳过此前提。

## 6. 凸包与安全预筛选

D&C 采用 Bullet 的整数坐标表述和整数行列式谓词；按最长轴排序，16 个双 lane 组构建子凸包，再经四轮成对合并。整数化的尺度与范围决定其数值边界。

大输入先沿 40 个方向扫描，最多留下 80 个正/负极值见证点。其凸包 P0 包含在真实凸包中：严格落在 P0 内的点不可能扩张凸包，因此删去，再对幸存者求完整 hull。n<1024 时预筛成本不划算，直接 D&C。Table 2 中的 Hybrid 指此几何混合算法，和前面学习的模型 Hybrid 控制没有关系。

## 7. 参数与复现风险

| 参数 | 论文默认 | 影响 |
|---|---:|---|
| τ | 0.05 | 更小通常部件更多、几何更精细 |
| k | 0.3 | 体积残差与表面误差的相对权重 |
| ε_axis | min(τ/4, 0.015) | 候选离边界的余量 |
| ε_conc | 0.005 | 凹边候选平移 |
| W_top / W_sub | 30 / 5 | 根部覆盖与深层成本 |
| D | 2 | 前瞻深度 |
| N_ce / I_ce | 16 / 10 | 凹边候选数量与启用轮数 |

公开 Python 默认 width=60、n_concave_edges=32、merge_hulls=False，不能直接代表论文默认；CLI 是 30/16/True。API 要显式归一化到最大包围盒边长 2，CLI 会处理。τ=0.05 是归一化坐标中的长度量，不是普遍的 5 cm，也不是简单 5% 体积误差。

复现时应固定输入网格、PaMO、尺度、seed/重复次数、搜索参数、merge/CC 开关、设备、驱动、冷启动与稳态计时。另设采样密度更高的独立几何评估，并检查 downstream 接触任务；仅部件数接近不足以确认同等质量。
