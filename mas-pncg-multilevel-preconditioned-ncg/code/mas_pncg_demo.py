"""
MAS-PNCG 核心概念演示：Sparse-Input Woodbury 更新 + 2D 子空间最小化

演示在一个 2D 弹簧-接触系统上，MAS-PNCG 的三个核心机制：
1. MAS 预条件器（简化为两层：细粒度子域 + 全局粗网格）
2. Woodbury 增量更新处理接触变化
3. 2D 子空间最小化搜索方向

依赖：numpy, scipy, matplotlib
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import NamedTuple

# ============================================================
# 1. 场景设置：2D 弹簧网格 + 地面接触
# ============================================================

def create_spring_mesh(nx: int = 8, ny: int = 8, spacing: float = 0.1):
    """创建 2D 弹簧网格，返回顶点位置和弹簧连接。"""
    n = nx * ny
    pos = np.zeros((n, 2))
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            pos[idx] = [i * spacing, j * spacing + 0.3]  # 悬在地面上方

    springs = []
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            if i + 1 < nx:
                springs.append((idx, (i + 1) * ny + j))
            if j + 1 < ny:
                springs.append((idx, i * ny + j + 1))
    return pos, np.array(springs)


class ContactInfo(NamedTuple):
    vertex: int
    distance: float
    stiffness: float
    normal: np.ndarray


# ============================================================
# 2. 能量和梯度
# ============================================================

def compute_elastic_energy_grad(x, x_rest, springs, k_spring=100.0):
    """弹簧弹性能量和梯度。"""
    n = len(x)
    grad = np.zeros(n * 2)
    for s in springs:
        i, j = s
        dx = x[i] - x[j]
        dx0 = x_rest[i] - x_rest[j]
        l = np.linalg.norm(dx)
        l0 = np.linalg.norm(dx0)
        if l < 1e-10:
            continue
        f = k_spring * (l - l0) * dx / l
        grad[2 * i:2 * i + 2] += f
        grad[2 * j:2 * j + 2] -= f
    return grad


def compute_barrier_contacts(x, d_hat=0.05, kappa=1000.0):
    """地面(y=0)接触barrier检测。"""
    contacts = []
    for i in range(len(x)):
        y = x[i, 1]
        if y < d_hat:
            d = max(y, 1e-8)
            k = kappa * (-np.log(d / d_hat) / d + 1.0 / d_hat) / d
            contacts.append(ContactInfo(
                vertex=i, distance=d, stiffness=k,
                normal=np.array([0.0, 1.0])
            ))
    return contacts


def compute_barrier_grad(x, contacts):
    """Barrier 梯度。"""
    n = len(x)
    grad = np.zeros(n * 2)
    d_hat = 0.05
    kappa = 1000.0
    for c in contacts:
        d = max(x[c.vertex, 1], 1e-8)
        dE_dd = kappa * (-np.log(d / d_hat) / d + 1.0 / d_hat)
        grad[2 * c.vertex + 1] -= dE_dd  # -dE/dy (barrier pushes up)
    return grad


def compute_gravity_grad(x, mass=1.0):
    """重力梯度。"""
    n = len(x)
    grad = np.zeros(n * 2)
    for i in range(n):
        grad[2 * i + 1] += mass * 9.8  # gravity pulls down
    return grad


def total_gradient(x, x_rest, springs):
    g = compute_elastic_energy_grad(x, x_rest, springs)
    contacts = compute_barrier_contacts(x)
    g += compute_barrier_grad(x, contacts)
    g += compute_gravity_grad(x)
    return g, contacts


# ============================================================
# 3. MAS 预条件器（简化两层实现）
# ============================================================

def build_elastic_hessian_diag(x, x_rest, springs, k_spring=100.0):
    """弹性 Hessian 对角块（用于 Level-0 基础逆）。"""
    n = len(x)
    diag = np.ones(n * 2) * 1e-4  # regularization
    for s in springs:
        i, j = s
        dx = x[i] - x[j]
        l = np.linalg.norm(dx)
        if l < 1e-10:
            continue
        stiff = k_spring  # 简化：用标量近似
        diag[2 * i:2 * i + 2] += stiff
        diag[2 * j:2 * j + 2] += stiff
    return diag


class MASPreconditioner:
    """简化的两层 MAS 预条件器。"""

    def __init__(self, n_dof, n_subdomains=4):
        self.n_dof = n_dof
        self.n_sub = n_subdomains
        self.sub_size = n_dof // n_subdomains

        # Level-0: 子域逆（对角近似）
        self.base_inv = [None] * n_subdomains  # 基础逆
        self.woodbury_U = [None] * n_subdomains  # 更新矩阵
        self.woodbury_active = [False] * n_subdomains

        # Level-1: 粗网格校正（简化为全局对角）
        self.coarse_inv = None

    def build(self, diag_hessian):
        """完全重建（全局重启时调用）。"""
        for d in range(self.n_sub):
            s = d * self.sub_size
            e = s + self.sub_size
            self.base_inv[d] = 1.0 / diag_hessian[s:e]
            self.woodbury_U[d] = None
            self.woodbury_active[d] = False

        self.coarse_inv = 1.0 / np.mean(diag_hessian.reshape(-1, self.sub_size), axis=1)

    def woodbury_update(self, contacts, x):
        """Sparse-Input Woodbury 增量更新 Level-0 子域。"""
        for d in range(self.n_sub):
            s = d * self.sub_size
            e = s + self.sub_size

            # 收集子域内的接触更新向量
            updates = []
            for c in contacts:
                local_idx = 2 * c.vertex + 1 - s  # y 分量
                if 0 <= local_idx < self.sub_size:
                    u = np.zeros(self.sub_size)
                    u[local_idx] = np.sqrt(c.stiffness)
                    updates.append(u)

            if updates:
                # Top-K 截断 (K=4 for demo)
                K = min(4, len(updates))
                self.woodbury_U[d] = np.column_stack(updates[:K])
                self.woodbury_active[d] = True
            else:
                self.woodbury_active[d] = False

    def apply(self, g):
        """应用预条件器：z = P * g。"""
        z = np.zeros_like(g)

        # Level-0: 子域贡献（含 Woodbury 更新）
        for d in range(self.n_sub):
            s = d * self.sub_size
            e = s + self.sub_size
            g_local = g[s:e]

            # Base solve
            z_base = self.base_inv[d] * g_local

            if self.woodbury_active[d] and self.woodbury_U[d] is not None:
                U = self.woodbury_U[d]
                B_inv = self.base_inv[d]

                # Woodbury: B_new^{-1} = B^{-1} - B^{-1}U(I+U^T B^{-1} U)^{-1} U^T B^{-1}
                BU = B_inv[:, None] * U  # B^{-1} U
                cap = np.eye(U.shape[1]) + U.T @ BU  # Capacitance matrix
                r = U.T @ z_base
                lam = np.linalg.solve(cap, r)
                z_local = z_base - BU @ lam
            else:
                z_local = z_base

            z[s:e] = z_local

        # Level-1: 粗网格校正
        for d in range(self.n_sub):
            s = d * self.sub_size
            e = s + self.sub_size
            g_mean = np.mean(g[s:e])
            z[s:e] += self.coarse_inv[d] * g_mean * 0.1  # 弱粗网格校正

        return z


# ============================================================
# 4. 2D 子空间最小化
# ============================================================

def solve_2d_subspace(z, p_prev, g, H_approx_diag):
    """在 {-mu*z + nu*p_prev} 子空间内求最优系数。"""
    Hz = H_approx_diag * z
    Hp = H_approx_diag * p_prev

    A = np.array([
        [z @ Hz, -(z @ Hp)],
        [-(p_prev @ Hz), p_prev @ Hp]
    ])
    b = np.array([z @ g, -(p_prev @ g)])

    # 正则化防止奇异
    A += np.eye(2) * 1e-8

    try:
        coeffs = np.linalg.solve(A, b)
        mu, nu = coeffs[0], coeffs[1]
    except np.linalg.LinAlgError:
        mu, nu = 1.0, 0.0

    return mu, nu


# ============================================================
# 5. MAS-PNCG 主循环
# ============================================================

def mas_pncg_solve(x_init, x_rest, springs, max_iter=200, tol=1e-4, restart_threshold=0.3):
    """MAS-PNCG 求解器。"""
    n = len(x_init)
    ndof = n * 2
    x = x_init.copy()

    precond = MASPreconditioner(ndof, n_subdomains=4)

    history = {
        "grad_norm": [],
        "restart_iters": [],
        "woodbury_iters": [],
        "positions": [x.copy()],
    }

    p_prev = np.zeros(ndof)
    z_prev = np.zeros(ndof)
    restart = True

    for k in range(max_iter):
        g, contacts = total_gradient(x, x_rest, springs)
        g_norm = np.linalg.norm(g)
        history["grad_norm"].append(g_norm)

        if g_norm < tol and restart:
            print(f"  Converged at iter {k}, |g| = {g_norm:.2e}")
            break

        if restart:
            # 全局重启：完全重建 MAS
            diag_H = build_elastic_hessian_diag(x, x_rest, springs)
            # 加入接触刚度
            for c in contacts:
                diag_H[2 * c.vertex + 1] += c.stiffness
            precond.build(diag_H)
            history["restart_iters"].append(k)
        else:
            # Woodbury 增量更新
            precond.woodbury_update(contacts, x)
            history["woodbury_iters"].append(k)

        # 预条件梯度
        z = precond.apply(g)

        if restart:
            # 初始步长 μ = z^T g / z^T H z
            Hz = build_elastic_hessian_diag(x, x_rest, springs) * z
            mu = (z @ g) / (z @ Hz + 1e-10)
            p = -mu * z
        else:
            # 2D 子空间最小化
            H_diag = build_elastic_hessian_diag(x, x_rest, springs)
            for c in contacts:
                H_diag[2 * c.vertex + 1] += c.stiffness
            mu, nu = solve_2d_subspace(z, p_prev, g, H_diag)
            p = -mu * z + nu * p_prev

        # 简化线搜索（步长限制）
        alpha = 1.0
        for c in contacts:
            py = p[2 * c.vertex + 1]
            if py < 0:  # 向地面移动
                max_step = max(0.5 * x[c.vertex, 1] / (-py), 0.01)
                alpha = min(alpha, max_step)

        x_flat = x.flatten() + alpha * p
        x = x_flat.reshape(-1, 2)

        # 地面约束
        x[:, 1] = np.maximum(x[:, 1], 1e-6)

        history["positions"].append(x.copy())

        # 正交性检查 → 重启判断
        if np.abs(z_prev @ z_prev) > 1e-10:
            r_k = abs(g @ z_prev) / (g @ z + 1e-10)
            restart = r_k > restart_threshold
        else:
            restart = False

        p_prev = p.copy()
        z_prev = z.copy()

    return x, history


# ============================================================
# 6. 对比：Jacobi-PNCG（无 MAS，仅对角预条件）
# ============================================================

def jacobi_pncg_solve(x_init, x_rest, springs, max_iter=200, tol=1e-4):
    """Jacobi-PNCG 基线。"""
    n = len(x_init)
    x = x_init.copy()
    history = {"grad_norm": []}

    p_prev = np.zeros(n * 2)
    g_prev = np.zeros(n * 2)

    for k in range(max_iter):
        g, contacts = total_gradient(x, x_rest, springs)
        g_norm = np.linalg.norm(g)
        history["grad_norm"].append(g_norm)

        if g_norm < tol:
            break

        # Jacobi 预条件：对角 Hessian 逆
        diag_H = build_elastic_hessian_diag(x, x_rest, springs)
        for c in contacts:
            diag_H[2 * c.vertex + 1] += c.stiffness
        z = g / diag_H

        # Polak-Ribiere beta
        if k == 0 or np.dot(g_prev, g_prev) < 1e-10:
            beta = 0.0
        else:
            beta = max(0, np.dot(g, z - z) / np.dot(g_prev, g_prev / diag_H))

        p = -z + beta * p_prev

        # 线搜索
        alpha = 0.01
        x_flat = x.flatten() + alpha * p
        x = x_flat.reshape(-1, 2)
        x[:, 1] = np.maximum(x[:, 1], 1e-6)

        p_prev = p.copy()
        g_prev = g.copy()

    return x, history


# ============================================================
# 7. 可视化
# ============================================================

def visualize_results(x_init, x_final_mas, x_final_jacobi, history_mas, history_jacobi, springs):
    """绘制对比图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) 初始 vs MAS-PNCG 最终位置
    ax = axes[0, 0]
    ax.set_title("MAS-PNCG: 初始 → 最终", fontsize=12)
    for s in springs:
        ax.plot([x_init[s[0], 0], x_init[s[1], 0]],
                [x_init[s[0], 1], x_init[s[1], 1]], 'b-', alpha=0.2, linewidth=0.5)
        ax.plot([x_final_mas[s[0], 0], x_final_mas[s[1], 0]],
                [x_final_mas[s[0], 1], x_final_mas[s[1], 1]], 'r-', alpha=0.5, linewidth=1)
    ax.scatter(x_init[:, 0], x_init[:, 1], c='blue', s=15, alpha=0.3, label='初始')
    ax.scatter(x_final_mas[:, 0], x_final_mas[:, 1], c='red', s=15, label='MAS-PNCG')
    ax.axhline(y=0, color='brown', linewidth=2, label='地面')
    ax.set_xlim(-0.1, 0.9)
    ax.set_ylim(-0.05, 0.8)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')

    # (0,1) 收敛曲线对比
    ax = axes[0, 1]
    ax.set_title("收敛曲线对比", fontsize=12)
    ax.semilogy(history_mas["grad_norm"], 'r-', linewidth=2, label='MAS-PNCG')
    ax.semilogy(history_jacobi["grad_norm"], 'b--', linewidth=2, label='Jacobi-PNCG')
    # 标记重启点
    for ri in history_mas["restart_iters"]:
        ax.axvline(x=ri, color='orange', alpha=0.3, linewidth=1)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("梯度范数")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0) Woodbury vs Restart 迭代分布
    ax = axes[1, 0]
    ax.set_title("迭代类型分布 (MAS-PNCG)", fontsize=12)
    n_restart = len(history_mas["restart_iters"])
    n_woodbury = len(history_mas["woodbury_iters"])
    n_total = n_restart + n_woodbury
    if n_total > 0:
        bars = ax.bar(["全局重启\n(MAS 重建)", "Woodbury 更新\n(增量维护)"],
                      [n_restart, n_woodbury],
                      color=['orange', 'green'], alpha=0.7)
        ax.set_ylabel("迭代次数")
        for bar, count in zip(bars, [n_restart, n_woodbury]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{count} ({100 * count / n_total:.0f}%)',
                    ha='center', fontsize=10)

    # (1,1) 演化过程（中间帧）
    ax = axes[1, 1]
    ax.set_title("MAS-PNCG 优化轨迹", fontsize=12)
    positions = history_mas["positions"]
    n_frames = len(positions)
    frame_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(frame_indices)))
    for fi, color in zip(frame_indices, colors):
        if fi < n_frames:
            pos = positions[fi]
            ax.scatter(pos[:, 0], pos[:, 1], c=[color], s=8, alpha=0.6,
                       label=f'iter {fi}')
    ax.axhline(y=0, color='brown', linewidth=2)
    ax.set_xlim(-0.1, 0.9)
    ax.set_ylim(-0.05, 0.8)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("mas_pncg_demo_results.png", dpi=150, bbox_inches='tight')
    print("  Results saved to mas_pncg_demo_results.png")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("MAS-PNCG 核心概念演示")
    print("场景：2D 弹簧网格在重力下落向地面（barrier 接触）")
    print("=" * 60)

    x_init, springs = create_spring_mesh(nx=8, ny=8, spacing=0.1)
    x_rest = x_init.copy()

    print("\n[1] 运行 MAS-PNCG...")
    x_mas, hist_mas = mas_pncg_solve(x_init, x_rest, springs, max_iter=150)
    print(f"  总迭代: {len(hist_mas['grad_norm'])}")
    print(f"  全局重启: {len(hist_mas['restart_iters'])} 次")
    print(f"  Woodbury 更新: {len(hist_mas['woodbury_iters'])} 次")

    print("\n[2] 运行 Jacobi-PNCG（基线）...")
    x_jac, hist_jac = jacobi_pncg_solve(x_init, x_rest, springs, max_iter=150)
    print(f"  总迭代: {len(hist_jac['grad_norm'])}")

    print("\n[3] 生成可视化...")
    visualize_results(x_init, x_mas, x_jac, hist_mas, hist_jac, springs)

    # 打印对比
    final_mas = hist_mas["grad_norm"][-1] if hist_mas["grad_norm"] else float('inf')
    final_jac = hist_jac["grad_norm"][-1] if hist_jac["grad_norm"] else float('inf')
    print(f"\n[结果对比]")
    print(f"  MAS-PNCG  最终梯度范数: {final_mas:.2e}")
    print(f"  Jacobi    最终梯度范数: {final_jac:.2e}")
    if final_jac > 0:
        print(f"  MAS-PNCG 收敛更快 {final_jac / max(final_mas, 1e-10):.1f}×")


if __name__ == "__main__":
    main()
