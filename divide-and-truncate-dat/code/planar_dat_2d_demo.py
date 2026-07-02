"""
Planar-DAT 核心算法演示：2D 顶点-三角形碰撞截断

演示 Isotropic-DAT vs Planar-DAT 在 2D 简化场景中的行为差异。
可视化：位移保留率、人工阻尼、死锁现象。

运行: python3 planar_dat_2d_demo.py
依赖: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


def isotropic_truncate(vertex_pos, displacement, closest_dist):
    """各向同性截断：限制位移不超过到最近面距离的一半"""
    max_disp = 0.5 * closest_dist
    disp_mag = np.linalg.norm(displacement)
    if disp_mag > max_disp:
        return displacement * (max_disp / disp_mag)
    return displacement.copy()


def planar_truncate(vertex_pos, displacement, triangle_center, tri_displacement, gamma_r=0.9):
    """
    Planar-DAT 截断：仅限制朝向碰撞面的法向分量

    参数:
        vertex_pos: 顶点位置
        displacement: 顶点位移
        triangle_center: 三角形最近点
        tri_displacement: 三角形位移（取最大法向分量）
        gamma_r: 松弛系数
    """
    normal = vertex_pos - triangle_center
    dist = np.linalg.norm(normal)
    if dist < 1e-10:
        return np.zeros_like(displacement)
    normal = normal / dist

    delta_v = max(-np.dot(displacement, normal), 0)
    delta_t = max(np.dot(tri_displacement, normal), 0)

    if delta_v == 0 and delta_t == 0:
        lam = 0.5
    else:
        lam = delta_t / (delta_t + delta_v)

    p = vertex_pos - lam * (vertex_pos - triangle_center)

    signed_dist_to_plane = np.dot(displacement, -normal)
    available = np.dot(vertex_pos - p, normal)

    if signed_dist_to_plane <= 0:
        return displacement.copy()

    t_i = available / signed_dist_to_plane
    if t_i < 0 or t_i >= 1.0 / gamma_r:
        return displacement.copy()

    t_v = gamma_r * t_i
    truncated = displacement * min(t_v, 1.0)
    return truncated


def demo_comparison():
    """对比演示：切向运动场景"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scenarios = [
        {
            "title": "场景1: 切向运动\n(Tangential Motion)",
            "vertex": np.array([0.0, 1.0]),
            "displacement": np.array([2.0, 0.0]),
            "tri_center": np.array([0.0, 0.0]),
            "tri_disp": np.array([0.0, 0.0]),
        },
        {
            "title": "场景2: 远离运动\n(Moving Away)",
            "vertex": np.array([0.0, 1.0]),
            "displacement": np.array([0.5, 1.5]),
            "tri_center": np.array([0.0, 0.0]),
            "tri_disp": np.array([0.0, 0.0]),
        },
        {
            "title": "场景3: 接近运动\n(Approaching)",
            "vertex": np.array([0.0, 1.0]),
            "displacement": np.array([0.3, -0.8]),
            "tri_center": np.array([0.0, 0.0]),
            "tri_disp": np.array([0.0, 0.3]),
        },
    ]

    for ax, scenario in zip(axes, scenarios):
        v = scenario["vertex"]
        d = scenario["displacement"]
        tc = scenario["tri_center"]
        td = scenario["tri_disp"]
        dist = np.linalg.norm(v - tc)

        iso_result = isotropic_truncate(v, d, dist)
        planar_result = planar_truncate(v, d, tc, td)

        tri_verts = np.array([[-1.5, 0], [1.5, 0], [0, -0.3]])
        triangle = Polygon(tri_verts, closed=True, alpha=0.3, color='gray')
        ax.add_patch(triangle)

        ax.plot(*v, 'ko', markersize=10, zorder=5)
        ax.annotate('v', v + np.array([0.05, 0.05]), fontsize=12, fontweight='bold')

        ax.annotate('', xy=v + d, xytext=v,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, linestyle='--'))
        ax.annotate('', xy=v + iso_result, xytext=v,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
        ax.annotate('', xy=v + planar_result, xytext=v,
                    arrowprops=dict(arrowstyle='->', color='green', lw=2.5))

        circle = Circle(v, dist * 0.5, fill=False, linestyle=':', color='red', alpha=0.5)
        ax.add_patch(circle)

        normal = (v - tc) / dist
        lam = 0.5
        delta_v = max(-np.dot(d, normal), 0)
        delta_t = max(np.dot(td, normal), 0)
        if delta_v + delta_t > 0:
            lam = delta_t / (delta_t + delta_v)
        p = v - lam * (v - tc)
        perp = np.array([-normal[1], normal[0]])
        ax.plot([p[0] - 2 * perp[0], p[0] + 2 * perp[0]],
                [p[1] - 2 * perp[1], p[1] + 2 * perp[1]],
                'g--', alpha=0.5, lw=1.5)

        iso_ratio = np.linalg.norm(iso_result) / max(np.linalg.norm(d), 1e-10)
        planar_ratio = np.linalg.norm(planar_result) / max(np.linalg.norm(d), 1e-10)

        ax.set_title(scenario["title"], fontsize=11)
        ax.text(0.02, 0.98, f'Iso: {iso_ratio:.1%}\nPlanar: {planar_ratio:.1%}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlim(-2, 3)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    legend_elements = [
        mpatches.FancyArrowPatch((0, 0), (1, 0), arrowstyle='->', color='blue', linestyle='--'),
    ]
    fig.legend(
        [plt.Line2D([0], [0], color='blue', linestyle='--', lw=1.5),
         plt.Line2D([0], [0], color='red', lw=2.5),
         plt.Line2D([0], [0], color='green', lw=2.5),
         plt.Line2D([0], [0], color='red', linestyle=':', alpha=0.5),
         plt.Line2D([0], [0], color='green', linestyle='--', alpha=0.5)],
        ['原始位移 (Original)', 'Isotropic-DAT', 'Planar-DAT',
         '各向同性区域 (Isotropic Region)', '划分平面 (Division Plane)'],
        loc='lower center', ncol=5, fontsize=9
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('planar_dat_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: planar_dat_comparison.png")


def demo_deadlock():
    """演示死锁现象：密集接触时 Isotropic-DAT 的退化"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    np.random.seed(42)
    center = np.array([0.0, 0.0])
    n_neighbors = 8
    radius = 0.5

    angles = np.linspace(0, 2 * np.pi, n_neighbors, endpoint=False)
    neighbors = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])

    displacement = np.array([0.3, 0.2])

    for ax_idx, (ax, title) in enumerate(zip(axes, ['Isotropic-DAT (死锁)', 'Planar-DAT (自由运动)'])):
        ax.plot(*center, 'ko', markersize=12, zorder=5)

        for n in neighbors:
            ax.plot(*n, 'rs', markersize=8, zorder=4)
            ax.plot([center[0], n[0]], [center[1], n[1]], 'gray', alpha=0.3, lw=0.5)

        if ax_idx == 0:
            min_dist = radius
            result = isotropic_truncate(center, displacement, min_dist)
            circle = Circle(center, min_dist * 0.5, fill=False, linestyle=':', color='red', alpha=0.7)
            ax.add_patch(circle)
        else:
            result = displacement.copy()
            for n in neighbors:
                result = planar_truncate(center, result, n, np.zeros(2))

            for n in neighbors:
                normal = center - n
                dist = np.linalg.norm(normal)
                normal = normal / dist
                p = center - 0.5 * (center - n)
                perp = np.array([-normal[1], normal[0]])
                ax.plot([p[0] - 0.4 * perp[0], p[0] + 0.4 * perp[0]],
                        [p[1] - 0.4 * perp[1], p[1] + 0.4 * perp[1]],
                        'g-', alpha=0.4, lw=1)

        ax.annotate('', xy=center + displacement, xytext=center,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, linestyle='--'))
        ax.annotate('', xy=center + result, xytext=center,
                    arrowprops=dict(arrowstyle='->', color='green' if ax_idx == 1 else 'red', lw=2.5))

        ratio = np.linalg.norm(result) / np.linalg.norm(displacement)
        ax.set_title(f'{title}\n位移保留: {ratio:.1%}', fontsize=12)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('planar_dat_deadlock.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: planar_dat_deadlock.png")


def demo_statistics():
    """统计对比：随机位移下的保留率分布"""
    np.random.seed(123)
    n_trials = 5000

    iso_ratios = []
    planar_ratios = []

    for _ in range(n_trials):
        vertex = np.array([0.0, np.random.uniform(0.3, 2.0)])
        displacement = np.random.randn(2) * 0.5
        tri_center = np.array([np.random.uniform(-0.5, 0.5), 0.0])
        tri_disp = np.random.randn(2) * 0.1
        dist = np.linalg.norm(vertex - tri_center)

        iso = isotropic_truncate(vertex, displacement, dist)
        planar = planar_truncate(vertex, displacement, tri_center, tri_disp)

        orig_mag = np.linalg.norm(displacement)
        if orig_mag > 1e-10:
            iso_ratios.append(np.linalg.norm(iso) / orig_mag)
            planar_ratios.append(np.linalg.norm(planar) / orig_mag)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(iso_ratios, bins=bins, alpha=0.6, color='red', label=f'Isotropic-DAT (均值: {np.mean(iso_ratios):.3f})')
    ax.hist(planar_ratios, bins=bins, alpha=0.6, color='green', label=f'Planar-DAT (均值: {np.mean(planar_ratios):.3f})')
    ax.axvline(np.mean(iso_ratios), color='red', linestyle='--', lw=2)
    ax.axvline(np.mean(planar_ratios), color='green', linestyle='--', lw=2)
    ax.set_xlabel('位移保留率 (Displacement Preservation Ratio)', fontsize=11)
    ax.set_ylabel('频次 (Count)', fontsize=11)
    ax.set_title(f'随机场景下的位移保留率分布 (N={n_trials})\n'
                 f'Planar/Isotropic 改进比: {np.mean(planar_ratios)/np.mean(iso_ratios):.2f}×', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('planar_dat_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: planar_dat_statistics.png")
    print(f"\n统计结果:")
    print(f"  Isotropic-DAT 平均保留率: {np.mean(iso_ratios):.3f}")
    print(f"  Planar-DAT 平均保留率:    {np.mean(planar_ratios):.3f}")
    print(f"  改进比: {np.mean(planar_ratios)/np.mean(iso_ratios):.2f}×")


if __name__ == "__main__":
    print("=" * 60)
    print("Planar-DAT 核心算法 2D 演示")
    print("=" * 60)
    print()

    demo_comparison()
    demo_deadlock()
    demo_statistics()

    print("\n所有图表已生成完毕。")
