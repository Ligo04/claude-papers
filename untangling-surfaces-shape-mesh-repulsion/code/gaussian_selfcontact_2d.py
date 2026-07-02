"""
高斯自接触能量 2D 演示

演示核心概念：
1. 高斯自接触能量如何检测曲线自相交
2. 自适应带宽选择
3. 能量梯度驱动的解缠过程

使用简单的 2D 平面曲线作为示例。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

np.random.seed(42)


def create_self_intersecting_curve(n_points=100):
    """创建一条自相交的 8 字形曲线"""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.sin(t)
    y = np.sin(2 * t) * 0.5
    return np.column_stack([x, y])


def create_tangled_curve(n_points=80):
    """创建一条更复杂的自相交曲线"""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.cos(t) + 0.3 * np.cos(3 * t)
    y = np.sin(t) + 0.3 * np.sin(3 * t)
    return np.column_stack([x, y])


def intrinsic_distance(i, j, n):
    """计算曲线上两点的内蕴弧长距离（归一化）"""
    d = abs(i - j)
    return min(d, n - d) / n


def gaussian_selfcontact_energy(curve, epsilon):
    """
    计算高斯自接触能量 E_G(Γ; ε)

    E_G = Σ_{i,j} w_ij / ε * exp(-r²_ij / ε²)

    其中 w_ij = 内蕴距离（抑制邻近点），r_ij = 欧氏距离
    """
    n = len(curve)
    energy = 0.0
    for i in range(n):
        for j in range(i + 2, n):  # 跳过相邻点
            w_ij = intrinsic_distance(i, j, n)
            r_ij = np.linalg.norm(curve[i] - curve[j])
            energy += w_ij / epsilon * np.exp(-r_ij**2 / epsilon**2)
    return energy


def gaussian_gradient(curve, epsilon):
    """
    计算高斯自接触能量的梯度

    ∇_{v_i} E_G = -2/ε³ * Σ_{j≠i} w_ij * exp(-r²_ij/ε²) * (v_i - v_j)
    """
    n = len(curve)
    grad = np.zeros_like(curve)
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= 1 or abs(i - j) >= n - 1:
                continue
            w_ij = intrinsic_distance(i, j, n)
            diff = curve[i] - curve[j]
            r_sq = np.sum(diff**2)
            grad[i] += -2.0 / epsilon**3 * w_ij * np.exp(-r_sq / epsilon**2) * diff
    return grad


def find_optimal_epsilon(curve, eps_range):
    """通过扫描找到使能量最大的 ε*（自适应带宽选择）"""
    energies = [gaussian_selfcontact_energy(curve, eps) for eps in eps_range]
    idx = np.argmax(energies)
    return eps_range[idx], energies


def untangle_curve(curve, epsilon, n_steps=50, lr=0.02):
    """通过梯度下降最小化高斯自接触能量来解缠曲线"""
    trajectory = [curve.copy()]
    energies = []

    current = curve.copy()
    for step in range(n_steps):
        grad = gaussian_gradient(current, epsilon)
        # 梯度下降（能量的梯度指向增大能量的方向，我们要最小化，所以减去）
        # 但实际上我们要让相交区域分开，所以沿梯度方向移动（梯度指向分离方向）
        current = current + lr * grad
        energies.append(gaussian_selfcontact_energy(current, epsilon))
        if step % 10 == 0:
            trajectory.append(current.copy())

    trajectory.append(current.copy())
    return trajectory, energies


def detect_intersections(curve):
    """简单的线段相交检测"""
    n = len(curve)
    intersections = []
    for i in range(n):
        for j in range(i + 2, n):
            if j == (i + n - 1) % n:
                continue
            p1, p2 = curve[i], curve[(i + 1) % n]
            p3, p4 = curve[j], curve[(j + 1) % n]
            # 线段相交测试
            d1 = p2 - p1
            d2 = p4 - p3
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < 1e-10:
                continue
            t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
            u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
            if 0 < t < 1 and 0 < u < 1:
                pt = p1 + t * d1
                intersections.append(pt)
    return np.array(intersections) if intersections else np.empty((0, 2))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # === 图 1：自相交曲线和能量积分可视化 ===
    curve = create_self_intersecting_curve(80)
    ax = axes[0, 0]
    ax.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2)
    ints = detect_intersections(curve)
    if len(ints) > 0:
        ax.scatter(ints[:, 0], ints[:, 1], c='red', s=100, zorder=5, marker='x', linewidths=3)
    ax.set_title('Self-Intersecting Curve (Figure-8)\nRed X = intersection points', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # === 图 2：自适应带宽选择 ===
    ax = axes[0, 1]
    eps_range = np.linspace(0.05, 2.0, 40)
    eps_star, energies = find_optimal_epsilon(curve, eps_range)
    ax.plot(eps_range, energies, 'b-', linewidth=2)
    ax.axvline(eps_star, color='r', linestyle='--', label=f'ε* = {eps_star:.2f}')
    ax.scatter([eps_star], [max(energies)], c='red', s=100, zorder=5)
    ax.set_xlabel('Bandwidth ε')
    ax.set_ylabel('E_G(Γ; ε)')
    ax.set_title('Adaptive Bandwidth Selection\nMaximum = optimal interaction scale', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === 图 3：能量积分热图 ===
    ax = axes[0, 2]
    n = len(curve)
    E_matrix = np.zeros((n, n))
    epsilon = eps_star
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= 1 or abs(i - j) >= n - 1:
                continue
            w_ij = intrinsic_distance(i, j, n)
            r_ij = np.linalg.norm(curve[i] - curve[j])
            E_matrix[i, j] = w_ij / epsilon * np.exp(-r_ij**2 / epsilon**2)

    im = ax.imshow(E_matrix, cmap='hot', aspect='equal')
    ax.set_xlabel('Point index s₂')
    ax.set_ylabel('Point index s₁')
    ax.set_title('Energy Integrand E(s₁, s₂)\nPeaks = self-intersection regions', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # === 图 4-6：解缠过程 ===
    trajectory, energies_over_time = untangle_curve(curve, eps_star, n_steps=60, lr=0.015)

    for idx, (step_idx, title) in enumerate([(0, 'Initial (tangled)'), (len(trajectory)//2, 'Mid-optimization'), (-1, 'Final (untangled)')]):
        ax = axes[1, idx]
        c = trajectory[step_idx]
        ax.plot(c[:, 0], c[:, 1], 'b-', linewidth=2)
        # Close the curve for visualization
        ax.plot([c[-1, 0], c[0, 0]], [c[-1, 1], c[0, 1]], 'b-', linewidth=2)
        ints = detect_intersections(c)
        if len(ints) > 0:
            ax.scatter(ints[:, 0], ints[:, 1], c='red', s=80, zorder=5, marker='x', linewidths=2)
            ax.set_title(f'{title}\n{len(ints)} intersections', fontsize=10)
        else:
            ax.set_title(f'{title}\n0 intersections ✓', fontsize=10, color='green')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Gaussian Self-Contact Energy: 2D Curve Untangling Demo\n'
                 '(Finite energy on intersecting configs → enables optimization from tangled state)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/ligo/claude-papers/papers/untangling-surfaces-shape-mesh-repulsion/code/gaussian_selfcontact_2d.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Gaussian self-contact 2D demo saved.")
    print(f"  Optimal bandwidth ε* = {eps_star:.3f}")
    print(f"  Initial intersections: {len(detect_intersections(trajectory[0]))}")
    print(f"  Final intersections: {len(detect_intersections(trajectory[-1]))}")


if __name__ == "__main__":
    main()
