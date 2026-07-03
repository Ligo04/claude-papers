"""
Affine Body Dynamics (ABD, SIGGRAPH 2022) —— 2D 最小实现:方块落地与堆叠
====================================================================

论文原文(见 fulltext.txt 第3、4节, arXiv 2201.10022)核心公式,这里做 3D→2D 类比:

  仿射坐标  q = [p, a1, a2, a3] ∈ R^12  (3D)  →  q = [p, a1, a2] ∈ R^6  (2D)
  世界坐标  x = A x̄ + p = J(x̄) q,   J(x̄) = [I, I⊗x̄^T]  是【常数】矩阵 (Eq.2)
  广义质量  M = ∫ρ J(x̄)^T J(x̄) dΩ   是【常数】矩阵     (Eq.4)
  正交势    V⊥(q) = κν‖AA^T − I‖²_F                    (Eq.6, 2D 展开见 Eq.7)
  增量势    E_b(q) = ½‖q−q̃‖²_M + Δt² V⊥(q)             (Eq.9)
            q̃ = q^t + Δt q̇^t + Δt² M⁻¹ f
  接触势    V_C = κ_c Σ B(d, d̂),  B(d,d̂) = −(d−d̂)² ln(d/d̂), 0<d<d̂   (Eq.10-11)

教学重点:
  1) J(x̄) 只依赖物质点 x̄，与 A、p 无关 ⇒ M 是【常数矩阵】，运动方程里
     没有刚体动力学那种依赖姿态的 Coriolis/陀螺项——这正是 ABD 比 SE(3)
     刚体表示更适合直接套用标准隐式积分器的原因。
  2) V⊥ 只惩罚 A 偏离正交(AA^T=I)，κ 越大惩罚越陡、A 越接近纯旋转——
     直观上"刚体感"越强，但本质仍是**弹性体**，只是刚度极高，因此不需要
     专门的刚体约束求解器或极分解(polar decomposition)。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(0)

# 1. 仿射坐标核心:常数 Jacobian J(x̄)，世界坐标 x = J(x̄) @ q
def J_mat(xbar):
    """2D 类比论文 Eq.2 的 J(x̄) = [I, I⊗x̄^T]，是只依赖物质点坐标 x̄ 的常数矩阵。"""
    x1, x2 = xbar
    return np.array([
        [1.0, 0.0, x1, x2, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, x1, x2],
    ])

def world_point(q, xbar):
    return J_mat(xbar) @ q

# 2. 正交势 V⊥ 及其解析梯度 (2D 展开，对应论文 Eq.7-8)
#    V⊥ = κν[(a1·a1−1)² + (a2·a2−1)² + 2(a1·a2)²]
#    (2x2 对称矩阵 AA^T−I 的 Frobenius 范数平方，非对角项 (1,2)/(2,1) 各算一次)
def V_perp(q, kappa, nu):
    a1, a2 = q[2:4], q[4:6]
    return kappa * nu * ((a1 @ a1 - 1) ** 2 + (a2 @ a2 - 1) ** 2 + 2 * (a1 @ a2) ** 2)

def grad_V_perp(q, kappa, nu):
    a1, a2 = q[2:4], q[4:6]
    g = np.zeros(6)
    g[2:4] = kappa * nu * 4 * ((a1 @ a1 - 1) * a1 + (a1 @ a2) * a2)
    g[4:6] = kappa * nu * 4 * ((a2 @ a2 - 1) * a2 + (a1 @ a2) * a1)
    return g

# 3. 刚体(方块)定义：常数质量矩阵由多边形顶点采样求和得到 (Eq.4 的离散近似)
class Box:
    def __init__(self, hx, hy, density, x0, y0, vy0=0.0):
        self.hx, self.hy = hx, hy
        self.local_verts = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        self.area = 4 * hx * hy          # 2D 里的"体积"就是面积 ν
        self.mass_total = density * self.area
        self.q = np.array([x0, y0, 1.0, 0.0, 0.0, 1.0])   # 初始 A=I（未形变）
        self.qdot = np.array([0.0, vy0, 0.0, 0.0, 0.0, 0.0])
        self.M = self._build_mass_matrix()
    def _build_mass_matrix(self):
        # 论文 M = ∫ρ JᵀJ dΩ 是常数矩阵；这里用 4 个顶点采样近似积分（教学简化）。
        m_v = self.mass_total / 4.0
        M = np.zeros((6, 6))
        for v in self.local_verts:
            M += m_v * (J_mat(v).T @ J_mat(v))
        return M
    def world_vertices(self, q=None):
        q = self.q if q is None else q
        return np.array([world_point(q, v) for v in self.local_verts])

# 4. 接触势：点-线段距离及其解析梯度（包络定理），用于顶点-地面/顶点-边 barrier

def point_segment(p, a, b):
    """返回 (距离 d, ∂d/∂p, ∂d/∂a, ∂d/∂b)；由包络定理，最近点参数 t* 在最优时
    ∂d/∂t=0，故对端点 a,b 求导可把 t 当常数处理，无需展开 t(a,b) 的链式法则。"""
    ab = b - a
    denom = ab @ ab
    t = 0.0 if denom < 1e-12 else np.clip((p - a) @ ab / denom, 0.0, 1.0)
    closest = a + t * ab
    diff = p - closest
    d = np.linalg.norm(diff)
    unit = diff / d if d > 1e-12 else np.zeros(2)
    return d, unit, -(1 - t) * unit, -t * unit

def barrier(d, dhat):
    """IPC 风格平滑截断对数 barrier，论文 Eq.11。"""
    if d >= dhat:
        return 0.0, 0.0
    d = max(d, 1e-9)
    B = -(d - dhat) ** 2 * np.log(d / dhat)
    dB = -2 * (d - dhat) * np.log(d / dhat) - (d - dhat) ** 2 / d
    return B, dB

# 5. 全局增量势的能量与梯度（多刚体 + 地面 + 体间接触）

GROUND_A = np.array([-100.0, 0.0])
GROUND_B = np.array([100.0, 0.0])

def gather_contact_pairs(boxes, dhat):
    """收集候选接触对：每个体的顶点 vs 地面边 + 其它体的 4 条边。体数很少，
    直接全对枚举 O(n^2) 即可，不追求论文里给百万三角形场景用的 i-AABB 加速结构。"""
    pairs = []  # (vert_owner, local_vert_idx, edge_owner or -1, local_edge_idx)
    n = len(boxes)
    for i in range(n):
        for k in range(4):
            pairs.append((i, k, -1, -1))          # 顶点 i,k vs 地面
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(4):                    # 顶点 i,k vs 体 j 的 4 条边
                for e in range(4):
                    pairs.append((i, k, j, e))
    return pairs

def edge_world(boxes, owner, e, qs):
    if owner == -1:
        return GROUND_A, GROUND_B
    wv = np.array([world_point(qs[owner], v) for v in boxes[owner].local_verts])
    return wv[e], wv[(e + 1) % 4]

def pair_contrib(qs, boxes, pair, dhat, body_idx):
    """接触对(顶点 vs 边)对 body_idx 一方的势能贡献与梯度；B(d,d̂) 只算一次，
    梯度按 body_idx 是顶点方还是边方分别投影。"""
    vi, vk, ej, ee = pair
    p = world_point(qs[vi], boxes[vi].local_verts[vk])
    a, bb = edge_world(boxes, ej, ee, qs)
    d, gp, ga, gb = point_segment(p, a, bb)
    if d >= dhat:
        return 0.0, np.zeros(6)
    Bv, dB = barrier(d, dhat)
    if body_idx == vi:
        grad = dB * (J_mat(boxes[vi].local_verts[vk]).T @ gp)
    else:
        eb = boxes[ej]
        va, vb = eb.local_verts[ee], eb.local_verts[(ee + 1) % 4]
        grad = dB * (J_mat(va).T @ ga + J_mat(vb).T @ gb)
    return Bv, grad

def body_energy_and_grad(i, qs, boxes, qtilde, dt, kappa, kappa_c, dhat, pairs_i):
    """只算第 i 体自身 E_b(q_i)（Eq.9）+ 相关接触 barrier，其它体位置固定
    ——这是块坐标下降（block Gauss-Seidel）的核心。"""
    b = boxes[i]
    dq = qs[i] - qtilde[i]
    E = 0.5 * dq @ (b.M @ dq) + dt ** 2 * V_perp(qs[i], kappa, b.area)
    grad = b.M @ dq + dt ** 2 * grad_V_perp(qs[i], kappa, b.area)
    for pair in pairs_i:
        Bv, g = pair_contrib(qs, boxes, pair, dhat, i)
        E += kappa_c * Bv
        grad += kappa_c * g
    return E, grad

def pair_distance_single_move(qs, boxes, pair, body_idx, direction, alpha):
    """只移动 body_idx（沿 direction 走 alpha 步），其余体固定，求该接触对的距离。"""
    q_try = list(qs)
    q_try[body_idx] = qs[body_idx] + alpha * direction
    vi, vk, ej, ee = pair
    p = world_point(q_try[vi], boxes[vi].local_verts[vk])
    a, bb = edge_world(boxes, ej, ee, q_try)
    d, *_ = point_segment(p, a, bb)
    return d

# 6. 每步隐式欧拉求解：块坐标下降（Gauss-Seidel）+ 质量预条件下降方向
#    + IPC 风格可行域步长过滤。论文用全局耦合 Newton；这里逐体轮流求解。
#
#    关键教学点：步长上限和收敛判据必须【逐体】计算，不能用一个全局标量
#    同时缩放所有体——否则贴地 box 因 barrier 曲率陡而需要的小步长，会
#    连带拖慢完全无接触、本该自由下落的其它体，把"接触是局部耦合"错误地
#    当成了"全局耦合"处理。块坐标下降天然避免这一点：只有共享同一接触对
#    的两个体才会相互影响。

def solve_step(boxes, dt, kappa, kappa_c, dhat, g=9.8, max_iter=25, tol=1e-4):
    n = len(boxes)
    qtilde = []
    for b in boxes:
        f = np.zeros(6)
        f[1] = -b.mass_total * g          # 重力通过质心(x̄=0)作用，只贡献平动分量
        qtilde.append(b.q + dt * b.qdot + dt ** 2 * np.linalg.solve(b.M, f))
    qs = [b.q.copy() for b in boxes]
    pairs = gather_contact_pairs(boxes, dhat)
    pairs_of_body = [[] for _ in range(n)]
    for pair in pairs:
        vi, vk, ej, ee = pair
        pairs_of_body[vi].append(pair)
        if ej != -1:
            pairs_of_body[ej].append(pair)
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        max_gnorm = 0.0
        for i in range(n):
            Ei, gi = body_energy_and_grad(i, qs, boxes, qtilde, dt, kappa, kappa_c, dhat, pairs_of_body[i])
            gnorm = np.linalg.norm(gi)
            max_gnorm = max(max_gnorm, gnorm)
            if gnorm < tol:
                continue
            di = -np.linalg.solve(boxes[i].M, gi)   # 质量预条件下降方向（对二次项即 Newton 方向）
            # 可行步长上限：只看与 body i 相关的接触对，二分求最大无穿透步长
            # （IPC 风格线性步长过滤，见 FACTS.md「Affine CCD」：位移轨迹
            # ΔA x̄+Δp 是线性的，无需处理论文里刚体那种 curved CCD）。
            cap = 1.0
            for pair in pairs_of_body[i]:
                if pair_distance_single_move(qs, boxes, pair, i, di, 1.0) > 1e-9:
                    continue
                lo, hi = 0.0, 1.0
                for _ in range(25):
                    mid = 0.5 * (lo + hi)
                    if pair_distance_single_move(qs, boxes, pair, i, di, mid) > 1e-9:
                        lo = mid
                    else:
                        hi = mid
                cap = min(cap, 0.9 * lo)
            beta = cap    # Armijo 回溯：在可行上限内收缩，保证该体自身能量下降
            for _ls in range(30):
                q_try = qs[i] + beta * di
                qs_try = qs[:i] + [q_try] + qs[i + 1:]
                E_try, _ = body_energy_and_grad(i, qs_try, boxes, qtilde, dt, kappa, kappa_c, dhat, pairs_of_body[i])
                if E_try <= Ei + 1e-4 * beta * (gi @ di):
                    break
                beta *= 0.5
            qs[i] = qs[i] + beta * di
        if max_gnorm < tol:
            break
    for i, b in enumerate(boxes):
        b.qdot = (qs[i] - b.q) / dt
        b.q = qs[i]
    return n_iter

# 7. 主仿真循环

def main():
    dt = 1.0 / 100
    kappa = 2e4          # 正交刚度（教学用相对尺度，非论文中 >100GPa 的物理量纲）
    kappa_c = 1e3         # 接触 barrier 刚度
    dhat = 0.02
    n_steps = 260
    boxes = [
        Box(hx=0.5, hy=0.5, density=1.0, x0=0.0, y0=1.0),   # 先落地
        Box(hx=0.5, hy=0.5, density=1.0, x0=0.05, y0=2.4),  # 略偏移，落在第一个上面
        Box(hx=0.35, hy=0.35, density=1.0, x0=-1.5, y0=1.6),  # 旁边独立落地
    ]
    history = {"t": [], "orth_err": [[] for _ in boxes]}
    snapshot_steps = set(np.linspace(0, n_steps - 1, 8).astype(int))
    snapshots = []
    for step in range(n_steps):
        n_iter = solve_step(boxes, dt, kappa, kappa_c, dhat)
        t = step * dt
        history["t"].append(t)
        max_orth_err = 0.0
        for i, b in enumerate(boxes):
            A = np.array([b.q[2:4], b.q[4:6]])
            orth_err = np.linalg.norm(A @ A.T - np.eye(2))
            history["orth_err"][i].append(orth_err)
            max_orth_err = max(max_orth_err, orth_err)
        if step % 40 == 0 or step == n_steps - 1:
            print(f"[step {step:4d}] t={t:5.3f}s  Newton迭代={n_iter:2d}  "
                  f"最大‖AAᵀ-I‖={max_orth_err:.2e}  "
                  f"各体 p_y={[round(b.q[1], 3) for b in boxes]}")
        if step in snapshot_steps:
            snapshots.append([b.world_vertices().copy() for b in boxes])
    # ---------------- 输出图像 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(snapshots)))
    for verts_list, c in zip(snapshots, colors):
        for verts in verts_list:
            poly = patches.Polygon(verts, closed=True, fill=False,
                                    edgecolor=c, linewidth=1.5, alpha=0.85)
            ax.add_patch(poly)
    ax.axhline(0, color="k", linewidth=2)
    ax.set_xlim(-2.5, 2.0)
    ax.set_ylim(-0.3, 3.0)
    ax.set_aspect("equal")
    ax.set_title("2D ABD 落体轨迹快照（颜色越亮越接近最终帧）")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax2 = axes[1]
    for i in range(len(boxes)):
        ax2.plot(history["t"], history["orth_err"][i], label=f"body {i}  ||AA^T - I||")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("orthogonality error")
    ax2.set_title("正交误差随时间变化（κ 越大越接近 0，验证近刚性）")
    ax2.legend()
    ax2.set_yscale("log")
    fig.tight_layout()
    out_path = "abd_2d_falling_boxes_trajectory.png"
    fig.savefig(out_path, dpi=130)
    print(f"\n已保存图像: {out_path}")
    final_max_err = max(history["orth_err"][i][-1] for i in range(len(boxes)))
    print(f"仿真结束。最终最大正交误差 = {final_max_err:.3e}（应远小于 1，说明保持近刚性）")
if __name__ == "__main__":
    main()
