"""
mortar_face_sampling_demo.py
============================
GMCP 面样本（face sample）构造的最小可运行演示（论文 §4.1）。

面样本是 GMCP 三类接触采样里最复杂的一类。论文做法：
    对 slave 三角形 T_s 和 master 三角形 T_m：
      1) 把两三角形投影到 T_s 的切平面
      2) 用 Sutherland-Hodgman 裁剪 [1974] 求重叠多边形
      3) 重叠多边形扇形三角化 (fan-triangulation)
      4) 每个小三角形上用 2D Gauss 求积生成接触采样点
      5) 每个样本算有符号间隙 g_k = n_s·(x_m,k - x_s,k)  (公式 6)

本演示用 2D（切平面内）复现 1)-4)，并验证：
    重叠多边形的『求积权重之和 = 重叠面积』（求积正确性）——
    这正是 mortar 积分权重 w_k 的几何来源。

只依赖 numpy。运行：python mortar_face_sampling_demo.py
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)


# ---------------------------------------------------------------------------
# Sutherland-Hodgman 多边形裁剪 [Sutherland & Hodgman 1974]
# 用「裁剪多边形」(凸，这里是 slave 三角形) 去裁「被裁多边形」(master 三角形)
# ---------------------------------------------------------------------------
def _inside(p, a, b):
    """点 p 是否在有向边 a->b 的左侧（含边上）。用于凸裁剪窗口。"""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12


def _intersect(p1, p2, a, b):
    """线段 p1->p2 与直线 a->b 的交点。"""
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = a;  x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-15:
        return p2
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])


def sutherland_hodgman(subject, clip):
    """subject: 被裁多边形顶点列表 (Nx2)；clip: 凸裁剪窗口顶点 (Mx2, 逆时针)。
    返回裁剪后多边形顶点 (Kx2)。
    """
    output = [np.asarray(p, float) for p in subject]
    m = len(clip)
    for i in range(m):
        a = np.asarray(clip[i], float)
        b = np.asarray(clip[(i + 1) % m], float)
        if not output:
            break
        inp = output
        output = []
        for j in range(len(inp)):
            cur = inp[j]
            prev = inp[j - 1]
            cur_in = _inside(cur, a, b)
            prev_in = _inside(prev, a, b)
            if cur_in:
                if not prev_in:
                    output.append(_intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(_intersect(prev, cur, a, b))
    return np.array(output) if output else np.zeros((0, 2))


# ---------------------------------------------------------------------------
# 多边形面积（鞋带公式）
# ---------------------------------------------------------------------------
def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------------
# 三角形上的 2D Gauss 求积（这里用 3 点二阶精度规则）
# 返回 (采样点 x_k, 权重 w_k)，其中 Σw_k = 三角形面积
# ---------------------------------------------------------------------------
def triangle_gauss_3pt(v0, v1, v2):
    """3 点 Gauss 规则（重心坐标 (2/3,1/6,1/6) 的三个轮换），权重各 1/3。"""
    area = polygon_area(np.array([v0, v1, v2]))
    bary = [(2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)]
    pts, ws = [], []
    for (a, b, c) in bary:
        pts.append(a * v0 + b * v1 + c * v2)
        ws.append(area / 3.0)     # 权重和 = area
    return np.array(pts), np.array(ws)


def fan_triangulate_and_quadrature(poly):
    """对凸多边形扇形三角化，逐三角形 Gauss 求积，汇总所有采样点和权重。
    对应论文：『clipped polygon is fan-triangulated and integrated with 2D Gauss quadrature』。
    """
    all_pts, all_ws = [], []
    if len(poly) < 3:
        return np.zeros((0, 2)), np.zeros(0)
    v0 = poly[0]
    for i in range(1, len(poly) - 1):
        pts, ws = triangle_gauss_3pt(v0, poly[i], poly[i + 1])
        all_pts.append(pts); all_ws.append(ws)
    return np.vstack(all_pts), np.concatenate(all_ws)


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
def demo_face_sampling():
    print("=" * 70)
    print("GMCP 面样本：Sutherland-Hodgman 裁剪 + 扇形三角化 + 2D Gauss 求积")
    print("=" * 70)

    # slave 三角形 T_s（切平面内，作为凸裁剪窗口，逆时针）
    Ts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # master 三角形 T_m（与 T_s 部分重叠、非匹配）
    Tm = np.array([[0.3, -0.2], [0.9, 0.5], [0.1, 0.6]])

    print(f"slave 三角形 T_s 顶点:\n{Ts}")
    print(f"master 三角形 T_m 顶点:\n{Tm}\n")

    # 1) 裁剪求重叠多边形
    overlap = sutherland_hodgman(Tm, Ts)
    print(f"裁剪得到的重叠多边形 ({len(overlap)} 个顶点):\n{overlap}")
    area_overlap = polygon_area(overlap)
    print(f"重叠多边形面积（鞋带公式）= {area_overlap:.6f}\n")

    # 2) 扇形三角化 + Gauss 求积
    pts, ws = fan_triangulate_and_quadrature(overlap)
    print(f"生成 {len(pts)} 个接触采样点 (face samples)")
    print(f"{'样本k':>6} | {'位置 x_s,k (切平面)':>26} | {'求积权重 w_k':>14}")
    print("-" * 56)
    for k, (p, w) in enumerate(zip(pts, ws)):
        print(f"{k:>6} | ({p[0]:>9.5f}, {p[1]:>9.5f})       | {w:>14.6f}")

    # 3) 验证求积正确性：Σ w_k 应等于重叠面积
    w_sum = float(np.sum(ws))
    print(f"\n求积权重之和 Σw_k = {w_sum:.6f}")
    print(f"重叠多边形面积    = {area_overlap:.6f}")
    print(f"相对误差          = {abs(w_sum-area_overlap)/area_overlap:.2e}")
    assert abs(w_sum - area_overlap) < 1e-10, "求积权重和应等于重叠面积"
    print("\n✓ 验证通过：Σw_k = 重叠面积。")
    print("  这就是 GMCP barrier 势(公式10)里 mortar 积分权重 w_k 的几何来源——")
    print("  每个面样本的权重 = 它代表的接触面积份额，于是 barrier 势在接触面上")
    print("  正确积分，接触压力分布得以准确传递（非匹配网格也成立）。\n")


def demo_signed_gap():
    print("=" * 70)
    print("有符号间隙 g_k = n_s·(x_m,k - x_s,k)  (公式 6)")
    print("=" * 70)
    print("把 2D 切平面采样点抬回 3D：slave 在 z=0 平面，master 在其上方一点。\n")
    n_s = np.array([0.0, 0.0, 1.0])     # slave 法向（指向 master 一侧）
    # 取两个采样点，master 侧高度（间隙）不同
    examples = [
        ("接近接触", np.array([0.4, 0.2, 0.0]), np.array([0.4, 0.2, 0.0008])),
        ("较大间隙", np.array([0.5, 0.3, 0.0]), np.array([0.5, 0.3, 0.0050])),
    ]
    print(f"{'样本':>10} | {'x_s,k':>22} | {'x_m,k':>22} | {'g_k':>10}")
    print("-" * 74)
    for name, xs, xm in examples:
        g = float(np.dot(n_s, xm - xs))
        print(f"{name:>10} | ({xs[0]:.2f},{xs[1]:.2f},{xs[2]:.4f}) | "
              f"({xm[0]:.2f},{xm[1]:.2f},{xm[2]:.4f}) | {g:>10.5f}")
    print("\n解读：g_k>0 表示未穿透；g_k 越小，barrier B(g_k,ε_k) 越大，排斥力越强。")
    print("      slave 始终是触觉面，压力在 slave 法向上评估，物理意义清晰。\n")


if __name__ == "__main__":
    demo_face_sampling()
    demo_signed_gap()
    print("=" * 70)
    print("要点回顾（GMCP 面样本流水线）：")
    print("  投影到 slave 切平面 -> Sutherland-Hodgman 裁剪重叠多边形")
    print("  -> 扇形三角化 -> 2D Gauss 求积 -> 每个样本(x_s,k, x_m,k, w_k, g_k)")
    print("  w_k 之和 = 重叠面积 => mortar 积分权重，保证接触压力正确传递")
    print("=" * 70)
