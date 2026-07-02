"""
AGIPC 代数粗化核心概念演示

演示内容：
1. 基于应变增量的边标记
2. 并行聚合（connected components）
3. Galerkin 投影组装粗网格系统
4. 延拓 + 后精炼

使用 2D 弹簧网格作为简化模型，展示代数粗化的完整流程。
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import NamedTuple


class Mesh2D(NamedTuple):
    vertices: np.ndarray   # (N, 2) 节点位置
    edges: np.ndarray      # (E, 2) 边索引
    fixed: np.ndarray      # (F,) 固定节点索引


def create_grid_mesh(nx: int = 10, ny: int = 10) -> Mesh2D:
    """创建 2D 弹簧网格。"""
    verts = np.array([(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)], dtype=np.float64)
    edges = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            idx = j * (nx + 1) + i
            if i < nx:
                edges.append((idx, idx + 1))
            if j < ny:
                edges.append((idx, idx + (nx + 1)))
    edges = np.array(edges, dtype=np.int32)
    fixed = np.array([j * (nx + 1) for j in range(ny + 1)], dtype=np.int32)  # 左边界固定
    return Mesh2D(verts, edges, fixed)


def compute_spring_energy_grad_hess(
    x: np.ndarray, rest_lengths: np.ndarray, edges: np.ndarray, k: float = 100.0
) -> tuple[float, np.ndarray, sparse.csr_matrix]:
    """计算弹簧系统的能量、梯度、Hessian。"""
    n = len(x)
    grad = np.zeros(n * 2)
    rows, cols, vals = [], [], []
    energy = 0.0

    for ei, (i, j) in enumerate(edges):
        dx = x[j] - x[i]
        l = np.linalg.norm(dx)
        l0 = rest_lengths[ei]
        if l < 1e-12:
            continue

        # 能量: k/2 * (l - l0)^2
        energy += 0.5 * k * (l - l0) ** 2

        # 梯度
        f = k * (l - l0) / l * dx
        for d in range(2):
            grad[i * 2 + d] -= f[d]
            grad[j * 2 + d] += f[d]

        # Hessian (SPD 近似)
        n_hat = dx / l
        H_block = k * np.outer(n_hat, n_hat)
        for di in range(2):
            for dj in range(2):
                for si, sj, sign in [(i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)]:
                    rows.append(si * 2 + di)
                    cols.append(sj * 2 + dj)
                    vals.append(sign * H_block[di, dj])

    H = sparse.coo_matrix((vals, (rows, cols)), shape=(n * 2, n * 2)).tocsr()
    return energy, grad, H


def compute_green_strain_per_edge(x: np.ndarray, edges: np.ndarray, rest_lengths: np.ndarray) -> np.ndarray:
    """简化版 Green 应变：每条边的 (l/l0)^2 - 1。"""
    strains = np.zeros(len(edges))
    for ei, (i, j) in enumerate(edges):
        dx = x[j] - x[i]
        l = np.linalg.norm(dx)
        l0 = rest_lengths[ei]
        strains[ei] = (l / l0) ** 2 - 1.0
    return strains


def tag_edges(
    strain_curr: np.ndarray, strain_prev: np.ndarray | None, tau: float = 5e-3
) -> np.ndarray:
    """标记可坍缩边。True = collapsible, False = protected。"""
    if strain_prev is None:
        return np.zeros(len(strain_curr), dtype=bool)
    delta = np.abs(strain_curr - strain_prev)
    return delta < tau


def aggregate_nodes(edges: np.ndarray, tags: np.ndarray, n_verts: int) -> np.ndarray:
    """将 collapsible 边的端点合并为 supernode。返回 fine→coarse 映射。"""
    parent = np.arange(n_verts)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ei, (i, j) in enumerate(edges):
        if tags[ei]:
            union(i, j)

    # 重新编号
    roots = {}
    mapping = np.zeros(n_verts, dtype=np.int32)
    for v in range(n_verts):
        r = find(v)
        if r not in roots:
            roots[r] = len(roots)
        mapping[v] = roots[r]

    return mapping


def build_prolongation(mapping: np.ndarray, n_coarse: int) -> sparse.csr_matrix:
    """构建 prolongation 矩阵 U (2D)。"""
    n_fine = len(mapping)
    rows, cols, vals = [], [], []
    # 计算每个 coarse 节点的权重 (均匀)
    counts = np.zeros(n_coarse)
    for v in range(n_fine):
        counts[mapping[v]] += 1

    for v in range(n_fine):
        c = mapping[v]
        for d in range(2):
            rows.append(v * 2 + d)
            cols.append(c * 2 + d)
            vals.append(1.0)  # 简单映射，不加权

    U = sparse.coo_matrix((vals, (rows, cols)), shape=(n_fine * 2, n_coarse * 2)).tocsr()
    return U


def galerkin_project(H: sparse.csr_matrix, g: np.ndarray, U: sparse.csr_matrix):
    """Galerkin 投影：Hc = Uᵀ H U, gc = Uᵀ g。"""
    Hc = U.T @ H @ U
    gc = U.T @ g
    return Hc, gc


def apply_fixed_dofs(H, g, fixed_nodes):
    """应用固定边界条件。"""
    H = H.tolil()
    for node in fixed_nodes:
        for d in range(2):
            dof = node * 2 + d
            if dof < H.shape[0]:
                H[dof, :] = 0
                H[:, dof] = 0
                H[dof, dof] = 1.0
                g[dof] = 0.0
    return H.tocsr(), g


def solve_one_newton_step(
    x: np.ndarray,
    mesh: Mesh2D,
    rest_lengths: np.ndarray,
    strain_prev: np.ndarray | None,
    tau: float = 5e-3,
    use_coarsening: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """执行一次 Newton 步（可选代数粗化）。"""
    n = len(x)
    energy, grad, H = compute_spring_energy_grad_hess(x, rest_lengths, mesh.edges)

    # 加入重力
    for i in range(n):
        grad[i * 2 + 1] += 9.8 * 0.01  # 质量 × 重力

    strain_curr = compute_green_strain_per_edge(x, mesh.edges, rest_lengths)
    info = {"energy": energy, "strain": strain_curr, "n_fine": n}

    if use_coarsening:
        tags = tag_edges(strain_curr, strain_prev, tau)
        mapping = aggregate_nodes(mesh.edges, tags, n)
        n_coarse = mapping.max() + 1
        info["n_coarse"] = n_coarse
        info["tags"] = tags
        info["mapping"] = mapping
        info["ratio"] = n_coarse / n

        if n_coarse < n * 0.95:  # 只在粗化有效时使用
            U = build_prolongation(mapping, n_coarse)
            Hc, gc = galerkin_project(H, grad, U)
            Hc_bc, gc_bc = apply_fixed_dofs(Hc.copy(), gc.copy(), [mapping[f] for f in mesh.fixed])
            Hc_bc += sparse.eye(Hc_bc.shape[0]) * 1e-6

            dx_coarse, _ = cg(Hc_bc, -gc_bc, maxiter=100)
            dx_fine = U @ dx_coarse

            # 后精炼 (≤10 CG 迭代)
            H_bc, g_bc = apply_fixed_dofs(H.copy(), grad.copy(), mesh.fixed)
            H_bc += sparse.eye(H_bc.shape[0]) * 1e-6
            residual = g_bc + H_bc @ dx_fine
            dx_refine, _ = cg(H_bc, -residual, x0=np.zeros_like(dx_fine), maxiter=10)
            dx_fine += dx_refine
            info["method"] = "coarsened"
        else:
            H_bc, g_bc = apply_fixed_dofs(H.copy(), grad.copy(), mesh.fixed)
            H_bc += sparse.eye(H_bc.shape[0]) * 1e-6
            dx_fine, _ = cg(H_bc, -g_bc, maxiter=200)
            info["method"] = "fine (coarsening skipped)"
    else:
        H_bc, g_bc = apply_fixed_dofs(H.copy(), grad.copy(), mesh.fixed)
        H_bc += sparse.eye(H_bc.shape[0]) * 1e-6
        dx_fine, _ = cg(H_bc, -g_bc, maxiter=200)
        info["method"] = "fine"
        info["n_coarse"] = n
        info["ratio"] = 1.0

    # 线搜索
    alpha = 1.0
    dx_2d = dx_fine.reshape(-1, 2)
    for _ in range(20):
        x_new = x + alpha * dx_2d
        e_new = compute_spring_energy_grad_hess(x_new, rest_lengths, mesh.edges)[0]
        for i in range(len(x_new)):
            e_new += 9.8 * 0.01 * x_new[i, 1]
        if e_new < energy + 1e-4 * alpha * np.dot(grad, dx_fine):
            break
        alpha *= 0.5

    x_new = x + alpha * dx_2d
    # 固定节点恢复
    for f in mesh.fixed:
        x_new[f] = x[f]

    info["alpha"] = alpha
    return x_new, strain_curr, info


def visualize(mesh: Mesh2D, x: np.ndarray, info: dict, step: int, ax):
    """可视化网格和粗化信息。"""
    ax.clear()

    # 画边，collapsible 用绿色，protected 用红色
    if "tags" in info:
        colors = ["#2ecc71" if t else "#e74c3c" for t in info["tags"]]
    else:
        colors = ["#3498db"] * len(mesh.edges)

    segments = []
    for i, j in mesh.edges:
        segments.append([x[i], x[j]])

    lc = LineCollection(segments, colors=colors, linewidths=1.5)
    ax.add_collection(lc)

    # 画节点
    ax.scatter(x[:, 0], x[:, 1], c="#2c3e50", s=10, zorder=5)
    ax.scatter(x[mesh.fixed, 0], x[mesh.fixed, 1], c="#e74c3c", s=30, zorder=6, marker="s")

    method = info.get("method", "unknown")
    ratio = info.get("ratio", 1.0)
    n_fine = info.get("n_fine", 0)
    n_coarse = info.get("n_coarse", 0)
    ax.set_title(
        f"Step {step} | {method}\n"
        f"Fine: {n_fine} → Coarse: {n_coarse} ({ratio:.1%}) | α={info.get('alpha', 0):.3f}",
        fontsize=10,
    )
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.5, 1.1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def main():
    print("=" * 60)
    print("AGIPC 代数粗化概念演示")
    print("=" * 60)
    print()
    print("创建 10×10 弹簧网格，左边界固定，施加重力...")

    mesh = create_grid_mesh(10, 10)
    x = mesh.vertices.copy()
    rest_lengths = np.array([np.linalg.norm(x[j] - x[i]) for i, j in mesh.edges])

    # 添加初始扰动
    x[50:, 1] -= 0.05

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    strain_prev = None
    steps_to_show = [0, 2, 5, 10, 15, 20]
    plot_idx = 0

    print(f"\n{'Step':>5} | {'Method':<30} | {'Fine':>5} → {'Coarse':>6} | {'Ratio':>7} | {'α':>6}")
    print("-" * 80)

    for step in range(25):
        x, strain_prev, info = solve_one_newton_step(
            x, mesh, rest_lengths, strain_prev, tau=5e-3, use_coarsening=True
        )

        print(
            f"{step:5d} | {info['method']:<30} | {info['n_fine']:5d} → {info['n_coarse']:6d} "
            f"| {info['ratio']:6.1%} | {info.get('alpha', 0):6.3f}"
        )

        if step in steps_to_show and plot_idx < len(axes):
            visualize(mesh, x, info, step, axes[plot_idx])
            plot_idx += 1

    fig.suptitle("AGIPC 代数粗化演示\n绿色=collapsible 边, 红色=protected 边, 红方=固定节点", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(__file__).replace(".py", "_result.png"), dpi=150, bbox_inches="tight")
    print(f"\n结果已保存到 {__file__.replace('.py', '_result.png')}")
    plt.show()


if __name__ == "__main__":
    main()
