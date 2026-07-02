"""
quotient_graph_speedup_demo.py
==============================
教学 demo：论文的核心提速手段——把网格图压成"商图（quotient graph）"，
在小得多的商图上做分隔集计算，从而把 NP-complete 子问题的搜索空间骤减。

本 demo 展示三件事（对应论文 §4 Our Algorithm）：
  1. patch 划分：把网格顶点打包成连通的 patch（这里用简单的块划分模拟 Lloyd）。
  2. 构建商图：patch 作节点、patch 间邻接作边——节点数从 n 降到 ~n/patch_size。
  3. 量化搜索空间缩减：分隔集计算代价对图规模高度敏感（这里用一个朴素的
     边界扫描代价模型 + 实际计时来直观展示"在小图上做便宜得多"）。

我们还演示论文的关键优化【商图建一次、跨递归复用】：递归二分时不重建商图，
只"擦掉"已选为分隔集的 patch 节点，复用同一张图。

依赖：仅 numpy。运行：python quotient_graph_speedup_demo.py
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# 1. 构造一个二维网格作为"矩阵图 G"（mesh 邻接图）
# ---------------------------------------------------------------------------
def grid_graph(r, c):
    def idx(i, j):
        return i * c + j
    adj = {idx(i, j): set() for i in range(r) for j in range(c)}
    for i in range(r):
        for j in range(c):
            if i + 1 < r:
                adj[idx(i, j)].add(idx(i + 1, j)); adj[idx(i + 1, j)].add(idx(i, j))
            if j + 1 < c:
                adj[idx(i, j)].add(idx(i, j + 1)); adj[idx(i, j + 1)].add(idx(i, j))
    return adj, idx


# ---------------------------------------------------------------------------
# 2. Patch 划分（gmap）：把网格切成 ps×ps 的方块 patch。
#    论文在 GPU 上用 Lloyd 算法得到 ~256 顶点的连通 patch；这里用块划分模拟，
#    每个块都是连通的，满足论文"patch 须诱导连通子图"的要求。
# ---------------------------------------------------------------------------
def block_patches(r, c, ps):
    """返回 gmap: 顶点 -> patch_id，以及 patch 数量。"""
    gmap = {}
    n_block_c = (c + ps - 1) // ps
    pid_of = {}
    next_pid = 0
    for i in range(r):
        for j in range(c):
            bi, bj = i // ps, j // ps
            key = (bi, bj)
            if key not in pid_of:
                pid_of[key] = next_pid
                next_pid += 1
            gmap[i * c + j] = pid_of[key]
    return gmap, next_pid


# ---------------------------------------------------------------------------
# 3. 构建商图：patch 作节点，patch 间若有原图边相连则连边。
# ---------------------------------------------------------------------------
def build_quotient_graph(adj, gmap):
    q = {}
    for v, nbrs in adj.items():
        pv = gmap[v]
        q.setdefault(pv, set())
        for u in nbrs:
            pu = gmap[u]
            if pu != pv:
                q[pv].add(pu)
                q.setdefault(pu, set()).add(pv)
    return q


# ---------------------------------------------------------------------------
# 4. 一个"分隔集二分"的代价模型：
#    在图上做一次平衡二分 + 找顶点分隔集，朴素实现的代价大致随【边数 × 顶点数】
#    增长（细化/平衡迭代）。我们用一个真实计时的朴素 BFS 二分来体现这种敏感性，
#    分别在【全矩阵图】和【商图】上跑，对比耗时。
# ---------------------------------------------------------------------------
def naive_bisection_cost(graph):
    """
    朴素平衡二分：从一个种子做 BFS，标记前一半为 part0，其余 part1，
    然后扫描所有边统计跨界顶点作为分隔集超集。
    故意写成 O(V + E) 的多趟扫描来模拟真实划分器的规模敏感性，并真实计时。
    返回 (分隔集大小估计, 耗时秒)。
    """
    t0 = time.perf_counter()
    verts = list(graph.keys())
    n = len(verts)
    if n == 0:
        return 0, 0.0
    order = {v: k for k, v in enumerate(verts)}
    # BFS 从第一个顶点扩展，得到一个连续的访问序，前一半 part0
    from collections import deque
    seen = set()
    bfs = []
    dq = deque([verts[0]]); seen.add(verts[0])
    while dq:
        v = dq.popleft(); bfs.append(v)
        for u in graph[v]:
            if u not in seen:
                seen.add(u); dq.append(u)
    # 把未访问到的（不连通部分）补在后面
    for v in verts:
        if v not in seen:
            bfs.append(v)
    half = n // 2
    part = {v: (0 if k < half else 1) for k, v in enumerate(bfs)}
    # 多趟"细化"扫描（模拟划分器的迭代细化，放大规模敏感性）
    REFINE_PASSES = 5
    sep = set()
    for _ in range(REFINE_PASSES):
        sep = set()
        for v in graph:
            for u in graph[v]:
                if part[v] != part[u]:
                    sep.add(v)
                    break
    t1 = time.perf_counter()
    return len(sep), (t1 - t0)


# ---------------------------------------------------------------------------
# 5. 演示"商图建一次、跨递归复用"：递归二分时擦除已选分隔 patch，复用同图。
# ---------------------------------------------------------------------------
def reuse_quotient_recursion(q, depth):
    """
    模拟论文的复用策略：不重建商图，每层只从节点集合里移除上一层选出的分隔节点，
    复用同一张 q。返回总共"擦除"的节点数，体现无需重新扫描矩阵。
    """
    active = set(q.keys())
    total_rebuilds_avoided = 0
    frontier = [active]
    for _ in range(depth):
        new_frontier = []
        for nodes in frontier:
            if len(nodes) <= 2:
                continue
            # 在子集上找一个分隔节点（取度最高的，朴素），擦除它 —— 复用 q，不重建
            sub_deg = {v: len(q[v] & nodes) for v in nodes}
            sep = max(sub_deg, key=sub_deg.get)
            rest = nodes - {sep}
            # 简单按 BFS 连通性二分 rest（教学省略，直接对半）
            rest_list = list(rest)
            half = len(rest_list) // 2
            left = set(rest_list[:half]); right = set(rest_list[half:])
            total_rebuilds_avoided += 1   # 每层每子图本来都要重扫矩阵，现在省了
            new_frontier += [left, right]
        frontier = new_frontier
    return total_rebuilds_avoided


# ---------------------------------------------------------------------------
# 6. 主程序
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("商图压缩如何加速分隔集计算 —— 论文核心提速手段")
    print("=" * 70)

    for (r, c) in [(60, 60), (120, 120), (200, 200)]:
        adj, _ = grid_graph(r, c)
        n = r * c
        full_edges = sum(len(x) for x in adj.values()) // 2

        # patch size 类比论文：这里块边长 8 → 每 patch 64 顶点（演示用，比论文256小）
        ps = 8
        gmap, n_patch = block_patches(r, c, ps)
        q = build_quotient_graph(adj, gmap)
        q_edges = sum(len(x) for x in q.values()) // 2

        # 在全图和商图上各做一次"分隔集二分"，真实计时
        sep_full, t_full = naive_bisection_cost(adj)
        sep_quot, t_quot = naive_bisection_cost(q)

        print(f"\n--- 网格 {r}x{c}  (n = {n} 顶点) ---")
        print(f"  全矩阵图 G : {n:6d} 节点, {full_edges:6d} 边")
        print(f"  商图     q : {n_patch:6d} 节点, {q_edges:6d} 边   "
              f"(压缩 {n / n_patch:.1f}× 节点)")
        print(f"  分隔集二分耗时 : 全图 {t_full*1e3:7.2f} ms  vs  商图 {t_quot*1e3:7.2f} ms"
              f"   → 商图快 {t_full / max(t_quot,1e-9):.1f}×")

        # 复用演示
        depth = 6
        avoided = reuse_quotient_recursion(q, depth)
        print(f"  跨递归复用 : 深度 {depth} 的嵌套剖分中，复用商图避免了约 "
              f"{avoided} 次全矩阵重扫（METIS 每层都要重新粗化）")

    print("\n" + "=" * 70)
    print("结论：")
    print("  • 商图把节点数压缩了一个数量级，分隔集计算（NP-complete）的搜索")
    print("    空间随之骤减，单次二分就快好几倍。")
    print("  • 关键不仅是压缩，还有【建一次、跨递归复用】：每层递归只擦除已选")
    print("    分隔节点，避免 METIS 式的逐层重新粗化整张矩阵。")
    print("  • 代价：分隔集必须沿 patch 边界走，质量略降（fill-in +6%~15%），")
    print("    但换来数量级的重排提速——这是论文的核心赌注。")
    print("=" * 70)
