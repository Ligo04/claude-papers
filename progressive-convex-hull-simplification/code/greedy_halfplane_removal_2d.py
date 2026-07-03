# -*- coding: utf-8 -*-
"""
2D 版《Progressive Convex Hull Simplification》核心算法演示
============================================================

论文(arXiv 2604.14468)在 3D 中做"贪心半空间删除":每次删掉
"删除后新增体积最小"的那个面,用优先队列驱动,直到剩下目标面数。

本演示把同一思想降到 2D,便于完整可视化:
  - 3D 半空间   → 2D 半平面(凸多边形的一条边)
  - 3D 凸"帽"   → 2D 三角形"帽"(相邻两条边的直线延伸相交补洞)
  - 体积代价     → 面积代价
  - 一环重三角化 → 平凡(2D 删边后邻居直接相交,无拓扑选择)

同时演示论文的两条关键性质:
  1. 【保守性】删半平面 = 外近似,每一步都严格包含原多边形;
     对比:删顶点(V-表示)= 内近似,会"漏出"原形状(碰撞假阴性)。
  2. 【无穷面积检测】删除"封口"边时邻居直线发散,代价为 +∞,
     对应论文 §4.3 的对偶原点出界测试。

运行:python3 greedy_halfplane_removal_2d.py
输出:greedy_simplification_2d.png(渐进简化 + 内外近似对比 + 对偶空间视图)
依赖:numpy, matplotlib
"""

import heapq

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能出图
import matplotlib.pyplot as plt
import numpy as np

# 中文字体兜底:找不到中文字体时退回英文标签也不报错
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# 基础几何工具
# ---------------------------------------------------------------------------

def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew 单调链求 2D 凸包,返回逆时针顶点数组。

    对应论文中的 CGAL convex_hull_3:我们同样"先建精确凸包,再简化"。
    """
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def half(iterable):
        chain: list[np.ndarray] = []
        for p in iterable:
            # 叉积 <= 0 说明拐向不满足严格左转,弹出(保持严格凸)
            while len(chain) >= 2 and np.cross(chain[-1] - chain[-2], p - chain[-2]) <= 0:
                chain.pop()
            chain.append(p)
        return chain

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def edges_to_halfplanes(verts: np.ndarray):
    """逆时针多边形顶点 → 每条边的半平面 (n, b),约定内部满足 n·x + b <= 0。

    这就是论文 §2 的 H-表示:凸集 = 所有半平面的交集。
    """
    normals, offsets = [], []
    m = len(verts)
    for i in range(m):
        d = verts[(i + 1) % m] - verts[i]
        n = np.array([d[1], -d[0]])          # 逆时针多边形的外法向
        n = n / np.linalg.norm(n)
        normals.append(n)
        offsets.append(-n @ verts[i])        # b = -n·(边上任一点)
    return np.array(normals), np.array(offsets)


def line_intersect(n1, b1, n2, b2):
    """两条直线 n·x + b = 0 的交点;近平行返回 None(→ 无穷代价)。"""
    A = np.array([n1, n2])
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return None
    return np.linalg.solve(A, -np.array([b1, b2]))


def polygon_area(verts: np.ndarray) -> float:
    """鞋带公式。对应论文 3D 里的高斯公式(Σ 面法向·质心)。"""
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# ---------------------------------------------------------------------------
# 核心:贪心半平面删除(论文 §4 的 2D 版)
# ---------------------------------------------------------------------------

class GreedyHalfplaneSimplifier:
    """维护"存活半平面"的循环双向链表 + 惰性最小堆。

    与论文的对应关系:
      cap_cost()     ↔ §4.2 几何变化(帽体积,这里是帽三角形面积)
      cost = +inf    ↔ §4.3 无穷体积(对偶原点出界 / 邻线发散)
      惰性堆 + 版本号 ↔ §4.4 lazy min-heap,pop 时校验过期
      prev/next 链表  ↔ 3D 里的半边结构(2D 拓扑退化为环)
    """

    def __init__(self, normals, offsets):
        self.n, self.b = normals.copy(), offsets.copy()
        m = len(normals)
        self.prev = [(i - 1) % m for i in range(m)]
        self.next = [(i + 1) % m for i in range(m)]
        self.alive = [True] * m
        self.version = [0] * m           # 邻居变化后 +1,用于堆项过期判断
        self.count = m
        self.heap: list[tuple[float, int, int]] = []
        for i in range(m):
            self._push(i)

    def _vertex(self, i, j):
        """半平面 i、j 的边界线交点 = 简化多边形的一个顶点(H→V 转换)。"""
        return line_intersect(self.n[i], self.b[i], self.n[j], self.b[j])

    def cap_cost(self, i) -> float:
        """删除半平面 i 新增的面积:三角形 (v_prev, v_new, v_next)。

        v_prev = 边(prev,i) 交点, v_next = 边(i,next) 交点,
        v_new  = 邻居 prev、next 直线延伸的交点(补洞产生的新顶点)。
        新顶点必须落在被删边的外侧(n_i·v + b_i > 0),否则删除
        会让交集无界 —— 论文 §4.3 无穷体积情形,代价记 +∞。
        """
        p, q = self.prev[i], self.next[i]
        v_prev, v_next = self._vertex(p, i), self._vertex(i, q)
        v_new = self._vertex(p, q)
        if v_prev is None or v_next is None or v_new is None:
            return np.inf                                    # 平行 → 无穷
        if self.n[i] @ v_new + self.b[i] <= 1e-12:
            return np.inf                                    # 交点在内侧/边上 → 无界
        return polygon_area(np.array([v_prev, v_new, v_next]))

    def _push(self, i):
        heapq.heappush(self.heap, (self.cap_cost(i), self.version[i], i))

    def simplify_to(self, target: int):
        """弹堆删除直到只剩 target 条半平面(论文 §4.4 主循环)。"""
        while self.count > target and self.heap:
            cost, ver, i = heapq.heappop(self.heap)
            if not self.alive[i] or ver != self.version[i]:
                continue                                     # 过期堆项,跳过
            if not np.isfinite(cost):
                print(f"  剩 {self.count} 条边时全部候选代价为 +∞,提前停止"
                      f"(对应论文 Fig.17 的'菱面体死路')")
                break
            # 从链表摘除 i,邻居直接相连(2D 的"一环重三角化"就这么简单)
            p, q = self.prev[i], self.next[i]
            self.alive[i] = False
            self.next[p], self.prev[q] = q, p
            self.count -= 1
            # 只有两个邻居的局部代价失效 —— 论文局部性的来源
            for j in (p, q):
                self.version[j] += 1
                self._push(j)

    def polygon(self) -> np.ndarray:
        """存活半平面 → 顶点序列(H-表示转回 V-表示用于画图)。"""
        start = self.alive.index(True)
        verts, i = [], start
        while True:
            verts.append(self._vertex(i, self.next[i]))
            i = self.next[i]
            if i == start:
                break
        return np.array(verts)


# ---------------------------------------------------------------------------
# 对照组:V-表示删顶点(内近似,论文 Fig.2 里 CoACD 默认后处理的做法)
# ---------------------------------------------------------------------------

def vertex_dropping(verts: np.ndarray, target: int) -> np.ndarray:
    """每次删掉"删除后损失面积最小"的顶点 —— 同样贪心,但方向相反:
    结果永远被原多边形包含,原形状会露出来(碰撞检测假阴性)。"""
    v = list(map(np.asarray, verts))
    while len(v) > target:
        m = len(v)
        # 删顶点 i 损失的面积 = 三角形 (v[i-1], v[i], v[i+1])
        losses = [polygon_area(np.array([v[i - 1], v[i], v[(i + 1) % m]])) for i in range(m)]
        v.pop(int(np.argmin(losses)))
    return np.array(v)


# ---------------------------------------------------------------------------
# 对偶变换可视化(论文 Eq.2 的 2D 版):半平面 → 对偶点
# ---------------------------------------------------------------------------

def dual_points(normals, offsets, center):
    """φ_i = -n_i / (n_i·c + b_i)。论文用 Chebyshev 中心保证数值安全;
    2D 凸多边形质心足够深入内部,演示中直接使用并在图上标注。"""
    denom = normals @ center + offsets       # 内部点处恒为负
    return -normals / denom[:, None]


# ---------------------------------------------------------------------------
# 主流程:生成形状 → 渐进简化 → 画图
# ---------------------------------------------------------------------------

def main():
    # 一个"有性格"的输入:椭圆环带点噪声,凸包约 30 条边
    t = np.sort(rng.uniform(0, 2 * np.pi, 240))
    r = 1.0 + 0.25 * np.cos(3 * t) + rng.normal(0, 0.04, t.shape)
    cloud = np.c_[1.5 * r * np.cos(t), r * np.sin(t)]
    hull = convex_hull(cloud)
    normals, offsets = edges_to_halfplanes(hull)
    area0 = polygon_area(hull)
    m0 = len(hull)
    print(f"输入点 {len(cloud)},精确凸包 {m0} 条边,面积 {area0:.4f}")

    targets = [16, 8, 5, 4]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))

    # ---- 第一行:渐进外近似(本文方法) ----
    for ax, tgt in zip(axes[0], targets):
        simp = GreedyHalfplaneSimplifier(normals, offsets)
        simp.simplify_to(tgt)                # 每格独立从头简化,便于对照
        poly = simp.polygon()
        ratio = polygon_area(poly) / area0
        print(f"  外近似 → {len(poly):2d} 边,面积 {ratio:.4f}×")
        ax.scatter(cloud[:, 0], cloud[:, 1], s=3, c="#bbbbbb", zorder=1)
        ax.fill(poly[:, 0], poly[:, 1], color="#0ea5e9", alpha=0.25, zorder=2)
        ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
                c="#0369a1", lw=2, zorder=3)
        ax.plot(np.append(hull[:, 0], hull[0, 0]), np.append(hull[:, 1], hull[0, 1]),
                c="#111111", lw=0.8, ls="--", zorder=4)
        ax.set_title(f"外近似(删半平面){m0}→{len(poly)} 边\n面积 {ratio:.3f}×,严格包含[安全]")
        ax.set_aspect("equal"); ax.axis("off")

    # ---- 第二行:内近似对比 / 对偶空间 ----
    # (a)(b) 同预算下删顶点:形状会露出来
    for ax, tgt in zip(axes[1][:2], [8, 5]):
        inner = vertex_dropping(hull, tgt)
        ratio = polygon_area(inner) / area0
        leaked = np.sum([np.any(edges_to_halfplanes(inner)[0] @ p +
                                edges_to_halfplanes(inner)[1] > 1e-9)
                         for p in cloud]) / len(cloud)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=3, c="#ef4444", zorder=1)
        ax.fill(inner[:, 0], inner[:, 1], color="#f97316", alpha=0.30, zorder=2)
        ax.plot(np.append(hull[:, 0], hull[0, 0]), np.append(hull[:, 1], hull[0, 1]),
                c="#111111", lw=0.8, ls="--", zorder=3)
        ax.set_title(f"内近似(删顶点){m0}→{tgt} 边\n面积 {ratio:.3f}×,"
                     f"{leaked * 100:.0f}% 输入点泄漏[不安全]")
        ax.set_aspect("equal"); ax.axis("off")

    # (c) 对偶空间:每条边一个对偶点,简化 = 摘掉对偶点
    center = hull.mean(axis=0)
    phi = dual_points(normals, offsets, center)
    simp8 = GreedyHalfplaneSimplifier(normals, offsets)
    simp8.simplify_to(8)
    survived = np.array(simp8.alive)
    ax = axes[1][2]
    ax.scatter(phi[~survived, 0], phi[~survived, 1], s=25, c="#cccccc", label="被删对偶点")
    ax.scatter(phi[survived, 0], phi[survived, 1], s=60, c="#0369a1", label="幸存对偶点")
    ax.scatter([0], [0], marker="+", s=120, c="#ef4444", label="对偶原点")
    ax.set_title("对偶空间视图(Eq.2)\n删半平面 = 在对偶点集上删点")
    ax.legend(loc="upper right", fontsize=8); ax.set_aspect("equal")

    # (d) 面积增长曲线:紧度-预算谱(对应论文 Fig.1 右侧曲线)
    ax = axes[1][3]
    budgets = list(range(m0, 3, -1))
    ratios = []
    for tgt in budgets:
        s = GreedyHalfplaneSimplifier(normals, offsets)
        s.simplify_to(tgt)
        ratios.append(polygon_area(s.polygon()) / area0)
    ax.plot(budgets, ratios, "o-", c="#0369a1", ms=3)
    ax.axhline(1.0, c="#111111", lw=0.8, ls="--")
    ax.invert_xaxis(); ax.set_xscale("log")
    ax.set_xlabel("剩余边数(log,右为更简)"); ax.set_ylabel("面积 / 原凸包面积")
    ax.set_title("紧度-预算谱(论文 Fig.1 的 2D 版)\n大部分预算区间增长可忽略")

    fig.suptitle("Progressive Convex Hull Simplification — 2D 贪心半平面删除演示", fontsize=15)
    fig.tight_layout()
    out = "greedy_simplification_2d.png"
    fig.savefig(out, dpi=130)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
