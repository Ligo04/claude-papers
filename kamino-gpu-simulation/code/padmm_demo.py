"""
Proximal-ADMM 求解器演示 — Kamino 核心算法的简化实现

演示一个 2D 两刚体系统在重力和地面接触下的约束动力学求解。
包含：双边关节约束 + 摩擦接触的 PADMM 统一求解。

运行: python padmm_demo.py
依赖: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass


# ============================================================
# 1. 系统定义（极大坐标）
# ============================================================

@dataclass
class RigidBody2D:
    """2D 刚体（简化版，使用 xy 位置 + 角度）"""
    mass: float
    inertia: float
    pos: np.ndarray      # [x, y]
    angle: float          # rad
    vel: np.ndarray       # [vx, vy]
    omega: float          # rad/s

    @property
    def state(self):
        return np.array([*self.pos, self.angle])

    @property
    def twist(self):
        return np.array([*self.vel, self.omega])

    @property
    def mass_matrix(self):
        """3x3 块对角质量矩阵 (2D)"""
        return np.diag([self.mass, self.mass, self.inertia])


def build_system_mass_matrix(bodies):
    """组装块对角系统质量矩阵 M = diag(M1, M2, ...)"""
    n = len(bodies) * 3
    M = np.zeros((n, n))
    for i, body in enumerate(bodies):
        M[3*i:3*i+3, 3*i:3*i+3] = body.mass_matrix
    return M


# ============================================================
# 2. 约束定义
# ============================================================

def revolute_joint_constraint(body_a, body_b, anchor_a, anchor_b):
    """
    旋转关节约束: 两刚体在各自局部坐标系中的锚点必须重合。
    f(q) = pos_a + R(θ_a) @ anchor_a - pos_b - R(θ_b) @ anchor_b = 0
    返回: (约束值 c, Jacobian J_a, J_b)
    """
    def rot2d(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    R_a = rot2d(body_a.angle)
    R_b = rot2d(body_b.angle)

    # 世界坐标中的锚点位置
    world_anchor_a = body_a.pos + R_a @ anchor_a
    world_anchor_b = body_b.pos + R_b @ anchor_b

    # 约束违反量 (2D: 2 个标量)
    c = world_anchor_a - world_anchor_b

    # Jacobian: dc/d(pos_a, θ_a) 和 dc/d(pos_b, θ_b)
    # d(R@p)/dθ = [-sin(θ)*px - cos(θ)*py, cos(θ)*px - sin(θ)*py]
    dRa_dtheta = np.array([[-np.sin(body_a.angle), -np.cos(body_a.angle)],
                            [np.cos(body_a.angle), -np.sin(body_a.angle)]]) @ anchor_a

    dRb_dtheta = np.array([[-np.sin(body_b.angle), -np.cos(body_b.angle)],
                            [np.cos(body_b.angle), -np.sin(body_b.angle)]]) @ anchor_b

    J_a = np.column_stack([np.eye(2), dRa_dtheta])   # 2x3
    J_b = np.column_stack([-np.eye(2), -dRb_dtheta])  # 2x3

    return c, J_a, J_b


def ground_contact_constraint(body, contact_point_local, ground_y=0.0):
    """
    地面接触约束 (单边 + 摩擦)
    法向: φ = contact_point_world_y - ground_y >= 0
    返回: (间隙 gap, 法向 Jacobian J_n, 切向 Jacobian J_t)
    """
    def rot2d(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    R = rot2d(body.angle)
    world_point = body.pos + R @ contact_point_local

    gap = world_point[1] - ground_y

    # 法向 Jacobian (沿 y 方向)
    dR_dtheta = np.array([[-np.sin(body.angle), -np.cos(body.angle)],
                           [np.cos(body.angle), -np.sin(body.angle)]]) @ contact_point_local
    J_n = np.array([0, 1, dR_dtheta[1]])  # 1x3

    # 切向 Jacobian (沿 x 方向)
    J_t = np.array([1, 0, dR_dtheta[0]])  # 1x3

    return gap, J_n, J_t


# ============================================================
# 3. Proximal-ADMM 求解器
# ============================================================

def coulomb_cone_projection(x_n, x_t, mu):
    """
    投影到 2D 库仑摩擦锥: {(λ_n, λ_t) : |λ_t| <= μ·λ_n, λ_n >= 0}
    """
    if x_n <= 0 and abs(x_t) <= -mu * x_n:
        # 在对偶锥内部 → 投影到原点
        return 0.0, 0.0
    elif abs(x_t) <= mu * x_n:
        # 已在锥内
        return x_n, x_t
    else:
        # 投影到锥的边界
        sign_t = np.sign(x_t)
        # 投影到边界线 |λ_t| = μ·λ_n
        proj_n = (x_n + mu * abs(x_t)) / (1 + mu**2)
        proj_n = max(0.0, proj_n)
        proj_t = sign_t * mu * proj_n
        return proj_n, proj_t


def padmm_solve(D, v_free, n_bilateral, n_contacts, mu, eta=1e-3, rho=1.0,
                max_iter=30, tol=1e-6, warm_start=None):
    """
    Proximal-ADMM 求解器。

    参数:
        D: Delassus 矩阵
        v_free: 自由速度向量
        n_bilateral: 双边约束数
        n_contacts: 接触约束数 (每个接触 = 1 法向 + 1 切向)
        mu: 摩擦系数
        eta, rho: PADMM 参数
        max_iter: 最大迭代数
        tol: 收敛容差
        warm_start: 上一步的收敛解
    """
    n = len(v_free)
    assert n == n_bilateral + 2 * n_contacts

    # 构建 D_{η,ρ}（在整个求解过程中保持不变！）
    D_reg = D + (eta + rho) * np.eye(n)

    # Cholesky 分解（每时间步一次）
    L = np.linalg.cholesky(D_reg)

    # 初始化
    if warm_start is not None:
        x = warm_start.copy()
    else:
        x = np.zeros(n)
    y = x.copy()
    z = np.zeros(n)

    # Nesterov 加速变量
    a = 1.0
    y_hat, z_hat = y.copy(), z.copy()

    residuals = []

    for iteration in range(max_iter):
        # Step 4-5: 线性求解 (使用缓存的 Cholesky 分解)
        rhs = -(v_free - eta * x - rho * y_hat - z_hat)
        x_new = np.linalg.solve(L @ L.T, rhs)

        # Step 6: 锥投影
        y_new = np.zeros(n)

        # 双边约束: 无约束 (自由空间)
        y_new[:n_bilateral] = x_new[:n_bilateral] - z_hat[:n_bilateral] / rho

        # 接触约束: 投影到库仑摩擦锥
        for k in range(n_contacts):
            idx_n = n_bilateral + 2 * k
            idx_t = n_bilateral + 2 * k + 1
            proj_input_n = x_new[idx_n] - z_hat[idx_n] / rho
            proj_input_t = x_new[idx_t] - z_hat[idx_t] / rho
            y_new[idx_n], y_new[idx_t] = coulomb_cone_projection(
                proj_input_n, proj_input_t, mu
            )

        # Step 7: 对偶更新
        z_new = z_hat - rho * (x_new - y_new)

        # 计算残差
        r_primal = np.linalg.norm(x_new - y_new, np.inf)
        r_dual = np.linalg.norm(rho * (y_new - y), np.inf)
        residuals.append((r_primal, r_dual))

        # Nesterov 加速 + 自适应重启
        a_new = (1 + np.sqrt(1 + 4 * a**2)) / 2
        beta = (a - 1) / a_new

        # 检查是否需要重启
        combined = r_primal + r_dual
        if iteration > 0 and combined > residuals[-2][0] + residuals[-2][1]:
            a_new = 1.0
            beta = 0.0

        y_hat = y_new + beta * (y_new - y)
        z_hat = z_new + beta * (z_new - z)
        a = a_new

        x, y, z = x_new, y_new, z_new

        # 收敛检查
        if r_primal < tol and r_dual < tol:
            break

    return y, residuals  # y 是锥投影后的解 = λ*


# ============================================================
# 4. 仿真循环
# ============================================================

def simulate(n_steps=500, dt=1.0/240, gravity=-9.81, mu=0.5):
    """
    仿真一个两连杆摆系统落到地面上的过程。

    系统: 刚体 A (上) 通过旋转关节连接到刚体 B (下)
    刚体 A 顶部固定在世界 (pin joint)
    刚体 B 底部可能与地面接触
    """
    # 连杆参数
    link_length = 0.5
    link_mass = 1.0
    link_inertia = link_mass * link_length**2 / 12

    # 初始化两个刚体（极大坐标，各自独立位姿）
    body_a = RigidBody2D(
        mass=link_mass, inertia=link_inertia,
        pos=np.array([0.0, 1.5]),  # 质心位置
        angle=0.3,  # 初始倾斜
        vel=np.array([0.0, 0.0]),
        omega=0.0
    )
    body_b = RigidBody2D(
        mass=link_mass, inertia=link_inertia,
        pos=np.array([0.0, 0.8]),
        angle=-0.2,
        vel=np.array([0.0, 0.0]),
        omega=0.0
    )

    # 记录轨迹
    trajectory_a = []
    trajectory_b = []
    all_residuals = []
    warm_lambda = None

    for step in range(n_steps):
        bodies = [body_a, body_b]

        # -- 组装系统 --
        M = build_system_mass_matrix(bodies)  # 6x6 块对角
        M_inv = np.linalg.inv(M)

        # 外力（重力）
        h = np.array([0, gravity * body_a.mass, 0,
                       0, gravity * body_b.mass, 0])

        # 当前速度
        u = np.concatenate([body_a.twist, body_b.twist])

        # -- 约束 --
        # 约束 1: body_a 顶端固定到世界 (0, 1.75)
        # 等效于 body_a 的 anchor_a 处的世界位置 = (0, 1.75)
        anchor_a_top = np.array([0.0, link_length / 2])  # 局部坐标: 顶部
        c1, J1_a, _ = revolute_joint_constraint(
            body_a,
            RigidBody2D(1, 1, np.array([0.0, 1.75]), 0.0, np.zeros(2), 0.0),
            anchor_a_top, np.array([0.0, 0.0])
        )
        # 固定到世界: 只有 J_a 部分
        J_world = np.zeros((2, 6))
        J_world[:, 0:3] = J1_a

        # 约束 2: body_a 底端 ↔ body_b 顶端 (旋转关节)
        anchor_a_bottom = np.array([0.0, -link_length / 2])
        anchor_b_top = np.array([0.0, link_length / 2])
        c2, J2_a, J2_b = revolute_joint_constraint(body_a, body_b, anchor_a_bottom, anchor_b_top)
        J_joint = np.zeros((2, 6))
        J_joint[:, 0:3] = J2_a
        J_joint[:, 3:6] = J2_b

        # 约束 3: body_b 底端地面接触
        contact_point = np.array([0.0, -link_length / 2])
        gap, J_n, J_t = ground_contact_constraint(body_b, contact_point)

        # 组装约束系统
        bilateral_constraints = []
        bilateral_J = []

        # Baumgarte 稳定化
        beta_baumgarte = 10.0
        bilateral_constraints.append(c1)
        bilateral_J.append(J_world)
        bilateral_constraints.append(c2)
        bilateral_J.append(J_joint)

        n_bilateral = sum(len(c) for c in bilateral_constraints)

        # 接触约束（仅在间隙 < 阈值时激活）
        contacts_active = gap < 0.01
        n_contacts = 1 if contacts_active else 0

        # 构建完整 Jacobian
        J_rows = []
        v_star_list = []

        for c, J in zip(bilateral_constraints, bilateral_J):
            J_rows.append(J)
            v_star_list.append(-beta_baumgarte * c)

        if contacts_active:
            J_contact_n = np.zeros((1, 6))
            J_contact_n[0, 3:6] = J_n
            J_contact_t = np.zeros((1, 6))
            J_contact_t[0, 3:6] = J_t
            J_rows.append(J_contact_n)
            J_rows.append(J_contact_t)
            v_star_list.append(np.array([-beta_baumgarte * min(gap, 0)]))
            v_star_list.append(np.array([0.0]))

        if len(J_rows) == 0:
            # 无约束，纯自由落体
            u_new = u + dt * M_inv @ h
        else:
            J = np.vstack(J_rows)
            v_star = np.concatenate(v_star_list)

            # Delassus 矩阵
            R = np.zeros((J.shape[0], J.shape[0]))  # 无正则化
            D = J @ M_inv @ J.T + R

            # 自由速度
            v_free = J @ (u + dt * M_inv @ h) + v_star

            # PADMM 求解!
            lambda_star, residuals = padmm_solve(
                D, v_free, n_bilateral, n_contacts, mu,
                eta=1e-4, rho=10.0, max_iter=30, tol=1e-6,
                warm_start=warm_lambda if warm_lambda is not None and len(warm_lambda) == len(v_free) else None
            )
            warm_lambda = lambda_star
            all_residuals.append(residuals)

            # 更新速度
            u_new = u + dt * M_inv @ (h + J.T @ lambda_star)

        # 积分位姿（显式 Euler）
        body_a.pos = body_a.pos + dt * u_new[0:2]
        body_a.angle = body_a.angle + dt * u_new[2]
        body_a.vel = u_new[0:2]
        body_a.omega = u_new[2]

        body_b.pos = body_b.pos + dt * u_new[3:5]
        body_b.angle = body_b.angle + dt * u_new[5]
        body_b.vel = u_new[3:5]
        body_b.omega = u_new[5]

        trajectory_a.append(body_a.pos.copy())
        trajectory_b.append(body_b.pos.copy())

    return np.array(trajectory_a), np.array(trajectory_b), all_residuals


# ============================================================
# 5. 可视化
# ============================================================

def visualize(traj_a, traj_b, residuals):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 轨迹图
    ax = axes[0]
    ax.plot(traj_a[:, 0], traj_a[:, 1], 'b-', alpha=0.5, label='Body A (upper)')
    ax.plot(traj_b[:, 0], traj_b[:, 1], 'r-', alpha=0.5, label='Body B (lower)')
    ax.axhline(y=0, color='brown', linestyle='--', label='Ground')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Rigid Body Trajectories (Maximal Coordinates)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 高度随时间变化
    ax = axes[1]
    t = np.arange(len(traj_a)) / 240.0
    ax.plot(t, traj_a[:, 1], 'b-', label='Body A y')
    ax.plot(t, traj_b[:, 1], 'r-', label='Body B y')
    ax.axhline(y=0, color='brown', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Height vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PADMM 收敛
    ax = axes[2]
    if residuals:
        # 绘制几个代表性时间步的收敛曲线
        steps_to_show = [0, len(residuals)//4, len(residuals)//2, -1]
        for idx in steps_to_show:
            res = residuals[idx]
            primal = [r[0] for r in res]
            ax.semilogy(primal, label=f'Step {idx}')
    ax.set_xlabel('PADMM Iteration')
    ax.set_ylabel('Primal Residual')
    ax.set_title('PADMM Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('padmm_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved to padmm_simulation.png")


if __name__ == '__main__':
    print("=" * 60)
    print("Proximal-ADMM Demo — Kamino 核心算法简化实现")
    print("=" * 60)
    print()
    print("系统: 两连杆摆 (极大坐标表述)")
    print("  - Body A: 固定在世界顶端，通过旋转关节连接 Body B")
    print("  - Body B: 底端可能与地面发生摩擦接触")
    print("  - 所有约束通过 PADMM 统一求解")
    print()

    traj_a, traj_b, residuals = simulate(n_steps=600, dt=1.0/240)

    print(f"仿真完成: {len(traj_a)} 步")
    print(f"Body A 最终位置: ({traj_a[-1, 0]:.3f}, {traj_a[-1, 1]:.3f})")
    print(f"Body B 最终位置: ({traj_b[-1, 0]:.3f}, {traj_b[-1, 1]:.3f})")
    print(f"PADMM 平均迭代数: {np.mean([len(r) for r in residuals]):.1f}")

    visualize(traj_a, traj_b, residuals)
