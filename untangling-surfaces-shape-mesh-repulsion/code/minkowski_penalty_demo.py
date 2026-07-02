"""
Minkowski 惩罚演示

演示核心概念：
1. 两个三角形的 Minkowski 差
2. 半空间表示和有符号距离
3. 穿透深度作为优化信号
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

np.random.seed(42)


def triangle_edge_normals_2d(tri):
    """计算 2D 三角形各边的外法线"""
    normals = []
    for i in range(3):
        e = tri[(i + 1) % 3] - tri[i]
        n = np.array([e[1], -e[0]])
        n = n / (np.linalg.norm(n) + 1e-10)
        normals.append(n)
    return normals


def compute_penetration_depth_2d(tri_a, tri_b):
    """
    使用半空间方法计算 2D 穿透深度
    ϕ(0, A-B) = max_{n∈N} (-h_n)
    """
    normals_a = triangle_edge_normals_2d(tri_a)
    normals_b = triangle_edge_normals_2d(tri_b)
    candidate_normals = normals_a + normals_b

    max_phi = -np.inf
    best_normal = None

    for n in candidate_normals:
        h_n = -np.inf
        for a in tri_a:
            for b in tri_b:
                val = np.dot(n, a - b)
                h_n = max(h_n, val)
        phi = -h_n
        if phi > max_phi:
            max_phi = phi
            best_normal = n

    return max_phi, best_normal


def convex_hull_2d(points):
    """简单的 2D 凸包（Graham scan）"""
    points = np.array(points)
    n = len(points)
    if n < 3:
        return points

    # 找最低点
    start = np.argmin(points[:, 1])
    pts = np.roll(points, -start, axis=0)
    pivot = pts[0]

    # 按极角排序
    angles = np.arctan2(pts[1:, 1] - pivot[1], pts[1:, 0] - pivot[0])
    order = np.argsort(angles)
    pts = np.vstack([pivot, pts[1:][order]])

    # Graham scan
    hull = [0, 1]
    for i in range(2, n):
        while len(hull) > 1:
            o, a, b = pts[hull[-2]], pts[hull[-1]], pts[i]
            cross = (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            if cross > 0:
                break
            hull.pop()
        hull.append(i)
    return pts[hull]


def visualize_minkowski():
    """可视化 Minkowski 差和穿透深度"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    configs = [
        {
            'name': 'Intersecting (ϕ < 0)',
            'tri_a': np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]]),
            'tri_b': np.array([[0.3, 0.1], [1.3, 0.1], [0.8, 0.9]]),
            'color': '#f56565'
        },
        {
            'name': 'Tangent (ϕ ≈ 0)',
            'tri_a': np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]]),
            'tri_b': np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 0.8]]),
            'color': '#ed8936'
        },
        {
            'name': 'Separated (ϕ > 0)',
            'tri_a': np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]]),
            'tri_b': np.array([[1.5, 0.0], [2.5, 0.0], [2.0, 0.8]]),
            'color': '#48bb78'
        }
    ]

    for col, cfg in enumerate(configs):
        tri_a = cfg['tri_a']
        tri_b = cfg['tri_b']

        # 上排：三角形配置
        ax = axes[0, col]
        poly_a = Polygon(tri_a, closed=True, fill=True, facecolor='#667eea', alpha=0.4, edgecolor='#667eea', linewidth=2)
        poly_b = Polygon(tri_b, closed=True, fill=True, facecolor='#f56565', alpha=0.4, edgecolor='#f56565', linewidth=2)
        ax.add_patch(poly_a)
        ax.add_patch(poly_b)
        ax.set_xlim(-0.5, 3.0)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(cfg['name'], fontsize=11, color=cfg['color'], fontweight='bold')
        ax.text(0.3, -0.3, 'A', fontsize=12, color='#667eea', fontweight='bold')
        ax.text(tri_b[0, 0] + 0.3, -0.3, 'B', fontsize=12, color='#f56565', fontweight='bold')
        ax.grid(True, alpha=0.2)

        # 下排：Minkowski 差
        ax = axes[1, col]

        # 计算 Minkowski 差顶点
        all_verts = []
        for a in tri_a:
            for b in tri_b:
                all_verts.append(a - b)
        all_verts = np.array(all_verts)

        # 凸包
        hull = convex_hull_2d(all_verts)
        poly_mink = Polygon(hull, closed=True, fill=True, facecolor='#2d3748', alpha=0.5,
                           edgecolor=cfg['color'], linewidth=2)
        ax.add_patch(poly_mink)

        # 标记原点
        ax.scatter([0], [0], c='white', s=150, zorder=5, edgecolors='black', linewidths=2)

        # 计算穿透深度
        phi, normal = compute_penetration_depth_2d(tri_a, tri_b)

        ax.scatter(all_verts[:, 0], all_verts[:, 1], c='gray', s=20, alpha=0.5, zorder=3)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#0d1117')
        ax.set_title(f'Minkowski Diff (A-B)\nϕ(0, A-B) = {phi:.3f}', fontsize=10)
        ax.grid(True, alpha=0.1, color='gray')

    plt.suptitle('Minkowski Penalty: Triangle Intersection via Origin-in-Polytope Test\n'
                 'ϕ < 0 → intersecting, ϕ = 0 → tangent, ϕ > 0 → separated',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ligo/claude-papers/papers/untangling-surfaces-shape-mesh-repulsion/code/minkowski_penalty_demo.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Minkowski penalty demo saved.")


def visualize_separation():
    """演示通过 Minkowski 惩罚梯度驱动三角形分离"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    tri_a = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    tri_b_init = np.array([[0.3, 0.1], [1.3, 0.1], [0.8, 0.9]])

    tri_b = tri_b_init.copy()
    lr = 0.3
    all_states = [tri_b.copy()]

    for _ in range(20):
        phi, normal = compute_penetration_depth_2d(tri_a, tri_b)
        if phi >= 0:
            break
        tri_b = tri_b + lr * normal * abs(phi)
        all_states.append(tri_b.copy())

    display_steps = [0, min(2, len(all_states)-1), min(4, len(all_states)-1), len(all_states)-1]

    for idx, s in enumerate(display_steps):
        ax = axes[idx]
        current_b = all_states[s]
        poly_a = Polygon(tri_a, closed=True, fill=True, facecolor='#667eea', alpha=0.5, edgecolor='#667eea', linewidth=2)
        poly_b = Polygon(current_b, closed=True, fill=True, facecolor='#f56565', alpha=0.5, edgecolor='#f56565', linewidth=2)
        ax.add_patch(poly_a)
        ax.add_patch(poly_b)

        phi, _ = compute_penetration_depth_2d(tri_a, current_b)
        status = "Separated ✓" if phi >= 0 else f"ϕ = {phi:.3f}"
        color = '#48bb78' if phi >= 0 else '#f56565'
        ax.set_title(f'Step {s}: {status}', fontsize=10, color=color)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Minkowski Penalty Gradient Drives Triangle Separation',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ligo/claude-papers/papers/untangling-surfaces-shape-mesh-repulsion/code/minkowski_separation_demo.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Separation optimization demo saved.")


if __name__ == "__main__":
    visualize_minkowski()
    visualize_separation()
    print("\nKey takeaways:")
    print("1. Minkowski difference converts intersection test to origin-containment test")
    print("2. Signed distance ϕ provides smooth penetration depth measure")
    print("3. Gradient from active half-space drives efficient separation")
