"""
dual_mesh_mapper_demo.py
========================
IsaacIPC 双网格映射器（Dual-Mesh Mapper）的最小可运行演示。

论文核心思想（§3.2）：
    维护两套网格——仿真网格 M_sim（粗、用于接触/变形）和视觉网格 M_vis（精、用于渲染）。
    把每个视觉顶点用「重心坐标 + 法向偏移」一次性绑定到某个仿真三角形上（预计算），
    之后每帧用绑定关系把仿真变形「翻译」回视觉网格，从而物理算粗的、渲染看精的。

本演示用纯 numpy 复现：
    1. 预计算：最近点投影 + 解 2x2 Gram 系统求重心坐标 (公式 1-3)
    2. 每帧更新：p'_i = u·v0' + v·v1' + w·v2' + δ·n'  (公式 4)
    3. 验证刚性不变性（rigid-invariant）和 O(h^2) 误差随网格细化的下降

只依赖 numpy。运行：python dual_mesh_mapper_demo.py
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)


# ---------------------------------------------------------------------------
# 1. 预计算：把一个视觉顶点 p 嵌入到仿真三角形 (v0,v1,v2)
# ---------------------------------------------------------------------------
def embed_vertex(p, v0, v1, v2):
    """对应论文公式 (1)-(3)。

    返回 (u, v, w, delta)：
        (u, v, w) 是 p 在三角形上的投影点的重心坐标，u+v+w=1
        delta     是 p 沿三角形法向到三角形平面的有符号偏移
    """
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)          # 单位法向 n_i

    d = p - v0                          # d = p - v0
    delta = np.dot(d, n)                # δ_i = d · n_i  (有符号偏移)

    # 公式 (3)：把 (2) 分别点乘 e1, e2，δ·n 项消失（n⊥e1, n⊥e2）
    #   [ e1·e1  e1·e2 ] [ v ]   [ d·e1 ]
    #   [ e1·e2  e2·e2 ] [ w ] = [ d·e2 ]
    G = np.array([[np.dot(e1, e1), np.dot(e1, e2)],
                  [np.dot(e1, e2), np.dot(e2, e2)]])   # Gram 矩阵（第一基本形式）
    rhs = np.array([np.dot(d, e1), np.dot(d, e2)])
    v_bary, w_bary = np.linalg.solve(G, rhs)
    u_bary = 1.0 - v_bary - w_bary
    return u_bary, v_bary, w_bary, delta


# ---------------------------------------------------------------------------
# 2. 每帧更新：用变形后的三角形恢复视觉顶点位置（公式 4）
# ---------------------------------------------------------------------------
def reconstruct_vertex(u, v, w, delta, v0p, v1p, v2p):
    """对应论文公式 (4):
        p'_i = u·v0' + v·v1' + w·v2' + δ·n'_i
    其中 (u,v,w,δ) 是预计算的常数，(v0',v1',v2') 是变形后三角形顶点。
    """
    e1 = v1p - v0p
    e2 = v2p - v0p
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)           # 变形后三角形的单位法向 n'_i
    return u * v0p + v * v1p + w * v2p + delta * n


# ---------------------------------------------------------------------------
# 演示 1：基本正确性 + 刚性不变性
# ---------------------------------------------------------------------------
def demo_rigid_invariance():
    print("=" * 70)
    print("演示 1：预计算 / 重建 的正确性 与 刚性不变性")
    print("=" * 70)

    # 一个仿真三角形
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])

    # 一个视觉顶点：在三角形上方 0.2，且 xy 落在三角形内部
    p = np.array([0.3, 0.3, 0.2])

    u, v, w, delta = embed_vertex(p, v0, v1, v2)
    print(f"预计算结果: u={u:.4f}, v={v:.4f}, w={w:.4f}, δ={delta:.4f}")
    print(f"  (u+v+w={u+v+w:.4f}, 应为 1.0；δ 应等于离面高度 0.2)")

    # 未变形时重建，应精确还原 p
    p_rec = reconstruct_vertex(u, v, w, delta, v0, v1, v2)
    print(f"未变形重建: {p_rec}  (原始 p={p})")
    assert np.allclose(p_rec, p), "未变形重建应精确还原"

    # 施加一个刚体变换（旋转 + 平移），重建点应跟随同样的刚体变换
    theta = np.radians(37.0)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])
    t = np.array([2.0, -1.0, 0.5])

    v0p, v1p, v2p = (R @ v0 + t), (R @ v1 + t), (R @ v2 + t)
    p_rigid_expected = R @ p + t
    p_rigid_rec = reconstruct_vertex(u, v, w, delta, v0p, v1p, v2p)

    err = np.linalg.norm(p_rigid_rec - p_rigid_expected)
    print(f"刚体变换后:")
    print(f"  期望 (R·p+t):   {p_rigid_expected}")
    print(f"  重建结果:        {p_rigid_rec}")
    print(f"  误差: {err:.2e}  --> 刚性不变性成立 (误差≈0)\n")
    assert err < 1e-12, "映射应是刚性不变的"


# ---------------------------------------------------------------------------
# 演示 2：O(h^2) 误差随网格细化二次下降
# ---------------------------------------------------------------------------
def true_deformation(x):
    """一个已知的非线性（弯曲）变形场，用作 ground truth。
    把平面 z=0 上的点弯成一个抛物面，模拟可变形体表面的弯曲。

    注：本收敛性演示让视觉顶点贴在 sim 表面 (z=0, 故 δ=0)，
    从而误差纯粹来自『面内分段线性插值』，干净地体现 O(h^2)。
    （带厚度偏移 δ 时，δ·n' 的法向旋转会叠加一个与 h 无关的常数误差，
    会掩盖 O(h^2) 阶——这本身也说明：尖锐弯曲处粗网格的厚度恢复是误差源，
    呼应论文『尖锐特征处映射误差常数大』的局限。）
    """
    X, Y, Z = x
    dz = 0.4 * (X**2 + Y**2)        # 抛物弯曲（二阶导非零 -> O(h^2) 误差可见）
    return np.array([X, Y, Z + dz])


def sample_visual_points(n_pts=400, seed=0):
    """在单位正方形内部采样视觉顶点，贴在 sim 表面 (z=0)。"""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.06, 0.94, size=(n_pts, 2))
    z = np.zeros((n_pts, 1))       # 贴面 (δ=0)，隔离出纯面内插值误差
    return np.hstack([xy, z])


def build_grid_triangles(h):
    """在 [0,1]^2 上构建边长约为 h 的规则三角网格（仿真网格 M_sim）。
    返回三角形顶点索引和顶点坐标（z=0 的参考构型）。
    """
    n = max(1, int(round(1.0 / h)))
    xs = np.linspace(0, 1, n + 1)
    ys = np.linspace(0, 1, n + 1)
    verts = []
    idx = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            idx[(i, j)] = len(verts)
            verts.append([x, y, 0.0])
    verts = np.array(verts)
    tris = []
    for j in range(n):
        for i in range(n):
            a = idx[(i, j)]; b = idx[(i + 1, j)]
            c = idx[(i, j + 1)]; d = idx[(i + 1, j + 1)]
            tris.append((a, b, d))
            tris.append((a, d, c))
    return np.array(tris), verts


def nearest_triangle(p, tris, verts):
    """找包含 p 的 xy 投影的仿真三角形（参考构型在 z=0 平面）。
    论文用最近点投影；因参考网格平铺在 z=0，这里用『点在三角形内』判定，
    既正确又避免边界外插。若都不含（数值边界），回退到质心最近。
    """
    px, py = p[0], p[1]
    for ti, (a, b, c) in enumerate(tris):
        x0, y0 = verts[a, 0], verts[a, 1]
        x1, y1 = verts[b, 0], verts[b, 1]
        x2, y2 = verts[c, 0], verts[c, 1]
        # 重心坐标符号判定（点在三角形内 <=> 三个重心坐标同号非负）
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-15:
            continue
        l1 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det
        l2 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det
        l3 = 1.0 - l1 - l2
        if l1 >= -1e-9 and l2 >= -1e-9 and l3 >= -1e-9:
            return ti
    centroids = verts[tris].mean(axis=1)
    d = np.linalg.norm(centroids[:, :2] - p[:2], axis=1)
    return int(np.argmin(d))


def demo_convergence():
    print("=" * 70)
    print("演示 2：双网格映射误差随仿真网格细化的收敛（论文称 O(h^2)）")
    print("=" * 70)
    print("做法：用粗 sim 网格做重心嵌入，对比「映射重建」与「真实变形」的误差，")
    print("      把 sim 网格边长 h 逐步减半，观察误差是否约降为 1/4。\n")

    visual_pts = sample_visual_points(n_pts=600)

    hs = [0.5, 0.25, 0.125, 0.0625]
    prev_err = None
    print(f"{'h (sim网格边长)':>16} | {'平均映射误差':>14} | {'误差比(上一行/本行)':>18}")
    print("-" * 60)
    for h in hs:
        tris, verts_ref = build_grid_triangles(h)
        # 变形后的仿真顶点（把已知变形场作用到参考顶点）
        verts_def = np.array([true_deformation(v) for v in verts_ref])

        errs = []
        for p in visual_pts:
            ti = nearest_triangle(p, tris, verts_ref)
            a, b, c = tris[ti]
            # 预计算：在参考构型上嵌入
            u, v, w, delta = embed_vertex(p, verts_ref[a], verts_ref[b], verts_ref[c])
            # 每帧重建：用变形后的三角形
            p_mapped = reconstruct_vertex(u, v, w, delta,
                                          verts_def[a], verts_def[b], verts_def[c])
            # ground truth：把真实变形场直接作用到视觉顶点
            p_true = true_deformation(p)
            errs.append(np.linalg.norm(p_mapped - p_true))
        mean_err = float(np.mean(errs))
        ratio = (prev_err / mean_err) if prev_err else float("nan")
        print(f"{h:>16.4f} | {mean_err:>14.3e} | {ratio:>18.2f}")
        prev_err = mean_err

    print("\n解读：h 每减半，误差比 ≈ 4，正是 O(h^2) 的特征")
    print("      （误差 ∝ h^2，h→h/2 则误差→1/4）。")
    print("      => 渲染保真度随仿真网格细化『二次』提升，与论文 §3.2 一致。\n")


if __name__ == "__main__":
    demo_rigid_invariance()
    demo_convergence()
    print("=" * 70)
    print("要点回顾：")
    print(" - 预计算存 (T_i, u, v, w, δ)，每帧只算公式(4)，M_vis 拓扑/纹理/材质不变")
    print(" - 重心坐标 + 法向偏移 => 刚性不变（整体刚体运动下零误差）")
    print(" - 分段线性近似 => O(h^2) 误差，加密 sim 网格可二次提升渲染保真度")
    print("=" * 70)
