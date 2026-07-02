"""
nested_dissection_demo.py
=========================
教学 demo：嵌套剖分（nested dissection）如何通过"先排子图、后排分隔集"
来减少 Cholesky 分解的 fill-in（填充）。

这对应论文 §2 Background 和 Figure 3 的核心机制——也是整篇论文要"加速"的
那个重排策略。本 demo 不做任何 patch/商图压缩（那是论文的提速手段），而是
聚焦于**为什么嵌套剖分能减少 fill-in**，让你先理解被加速的对象本身。

我们用一个一维链 + 一个二维网格做对比，展示：
  1. 自然顺序（natural order）的 fill-in
  2. 嵌套剖分顺序（separator 排最后）的 fill-in

依赖：仅 numpy（symbolic Cholesky 我们自己写，不需要 scipy）。
运行：python nested_dissection_demo.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. 符号 Cholesky：只追踪"哪些位置变成非零"，不算数值。
#    消去顶点 v 时，v 的所有未消去邻居两两相连（形成 clique）——这就是 fill-in。
# ---------------------------------------------------------------------------
def symbolic_cholesky_fill(adj, order):
    """
    adj   : dict[int -> set[int]]  无向图邻接表（矩阵 A 的图，不含对角）
    order : list[int]              消元顺序（order[0] 最先被消去）
    返回  : fill-in 边的数量（新产生的非零，不含原有边）
    """
    # 拷贝一份，消元过程中会往里加 fill 边
    g = {v: set(nbrs) for v, nbrs in adj.items()}
    original_edges = sum(len(n) for n in adj.values()) // 2
    fill_edges = 0

    eliminated = set()
    for v in order:
        # v 的当前未消去邻居
        nbrs = [u for u in g[v] if u not in eliminated]
        # 把这些邻居两两连起来（消去 v 在因子里造成的填充）
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b = nbrs[i], nbrs[j]
                if b not in g[a]:
                    g[a].add(b)
                    g[b].add(a)
                    fill_edges += 1
        eliminated.add(v)

    total_nnz_L = original_edges + fill_edges  # L 的非对角非零数（下三角）
    return fill_edges, total_nnz_L


# ---------------------------------------------------------------------------
# 2. 构造测试图
# ---------------------------------------------------------------------------
def path_graph(n):
    """一维链 0-1-2-...-(n-1)"""
    adj = {i: set() for i in range(n)}
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    return adj


def grid_graph(r, c):
    """r×c 二维网格（4-邻接），模拟典型的 mesh/FEM 稀疏模式"""
    def idx(i, j):
        return i * c + j
    adj = {idx(i, j): set() for i in range(r) for j in range(c)}
    for i in range(r):
        for j in range(c):
            if i + 1 < r:
                adj[idx(i, j)].add(idx(i + 1, j)); adj[idx(i + 1, j)].add(idx(i, j))
            if j + 1 < c:
                adj[idx(i, j)].add(idx(i, j + 1)); adj[idx(i, j + 1)].add(idx(i, j))
    return adj


# ---------------------------------------------------------------------------
# 3. 嵌套剖分排序：递归找分隔集，把分隔集排在两个子图之后。
#    这里用最朴素的"几何中线"做分隔集（教学用），论文则在商图上用 METIS。
# ---------------------------------------------------------------------------
def nd_order_path(vertices):
    """对一段连续顶点做嵌套剖分：取中点为分隔，左右递归，分隔点排最后。"""
    if len(vertices) <= 2:
        return list(vertices)
    mid = len(vertices) // 2
    sep = [vertices[mid]]
    left = nd_order_path(vertices[:mid])
    right = nd_order_path(vertices[mid + 1:])
    return left + right + sep   # 关键：separator 排在最后被消去


def nd_order_grid(r, c):
    """
    对网格做嵌套剖分：每次用一整列（或一整行）作分隔集，把网格切成左右两半，
    分隔列排在两个子块之后。递归地切。这正是 Figure 4 的"绿点分隔集"思想。
    """
    def idx(i, j):
        return i * c + j

    def recurse(rows, cols):
        # rows, cols 是当前子块覆盖的行/列索引列表
        if len(rows) <= 1 or len(cols) <= 1:
            return [idx(i, j) for i in rows for j in cols]
        # 沿较长的一维切，分隔集取中间那一线
        if len(cols) >= len(rows):
            m = len(cols) // 2
            sep_col = cols[m]
            sep = [idx(i, sep_col) for i in rows]
            left = recurse(rows, cols[:m])
            right = recurse(rows, cols[m + 1:])
        else:
            m = len(rows) // 2
            sep_row = rows[m]
            sep = [idx(sep_row, j) for j in cols]
            left = recurse(rows[:m], cols)
            right = recurse(rows[m + 1:], cols)
        return left + right + sep   # separator 排最后

    return recurse(list(range(r)), list(range(c)))


# ---------------------------------------------------------------------------
# 4. 跑对比
# ---------------------------------------------------------------------------
def report(name, adj, natural, nd):
    f_nat, nnz_nat = symbolic_cholesky_fill(adj, natural)
    f_nd, nnz_nd = symbolic_cholesky_fill(adj, nd)
    n = len(adj)
    orig = sum(len(x) for x in adj.values()) // 2
    print(f"\n=== {name}  (顶点 {n}, 原始非零边 {orig}) ===")
    print(f"  自然顺序     : fill-in = {f_nat:6d}   nnz(L) = {nnz_nat:6d}")
    print(f"  嵌套剖分顺序 : fill-in = {f_nd:6d}   nnz(L) = {nnz_nd:6d}")
    if f_nd < f_nat:
        ratio = f_nat / max(f_nd, 1)
        print(f"  → 嵌套剖分把 fill-in 降到自然顺序的 {f_nd/f_nat:.1%}，"
              f"约 {ratio:.2f}× 更少填充（越少越好）")
    elif f_nd > f_nat:
        print(f"  → 此结构上自然顺序已更优，嵌套剖分反而多了 {f_nd - f_nat} 条 fill 边")


if __name__ == "__main__":
    print("=" * 64)
    print("嵌套剖分如何减少 Cholesky fill-in —— 论文要加速的核心对象")
    print("=" * 64)

    # 一维链：自然顺序就已经最优（链是树，无 fill），用来当对照
    p = path_graph(31)
    report("一维链 path(31)", p,
           natural=list(range(31)),
           nd=nd_order_path(list(range(31))))
    print("  注：链是树状结构，自然顺序本就无填充，嵌套剖分无额外收益。")

    # 二维网格：这才是 mesh/FEM 的典型，自然顺序 fill-in 很大
    for (r, c) in [(15, 15), (31, 31)]:
        g = grid_graph(r, c)
        nat = list(range(r * c))           # 行优先自然顺序
        nd = nd_order_grid(r, c)
        report(f"二维网格 grid({r}x{c})", g, nat, nd)

    print("\n" + "=" * 64)
    print("结论：")
    print("  • 网格（mesh）上自然顺序的 fill-in 随规模快速膨胀。")
    print("  • 嵌套剖分把分隔集排在最后，切断两侧子图间的填充，显著减少 nnz(L)。")
    print("  • 论文的工作不是改这个策略，而是【极大加速】找分隔集这一步——")
    print("    通过把网格压成商图（patch），并跨递归复用，详见 quotient_graph_speedup_demo.py")
    print("=" * 64)
