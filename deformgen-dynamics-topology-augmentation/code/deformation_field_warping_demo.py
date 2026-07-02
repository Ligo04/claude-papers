"""
DeformGen 核心算法演示：形变场扭曲（Deformation-Field Warping）
=================================================================

这个脚本独立实现论文 Sec. 3.2 的轨迹增强核心：
把"源点云 -> 形变后点云"的逐粒子位移，升维成一个连续的空间形变场 D(x)，
再用它同时扭曲末端执行器的【位置】（公式 2-3）和【姿态】（公式 4-7）。

为什么这么做（核心 why）：
  可变形物体变形后，材料点不再刚性同步运动（非等变）。直接对整条示教轨迹
  套一个全局刚性变换会让抓取位姿与新几何错位。形变场扭曲让轨迹上每一点
  按其【局部邻域】的真实位移来调整，从而与变形后的物体保持对齐。

只依赖 numpy。运行：
    python deformation_field_warping_demo.py
"""

import numpy as np


# ----------------------------------------------------------------------
# 1) 位置扭曲：KNN + 反距离加权（IDW） + 时间衰减   —— 论文公式 (2)(3)
# ----------------------------------------------------------------------
def warp_position(x_t, p_orig, delta, K, alpha_t, eps=1e-6):
    """把一个末端位置 x_t 按形变场扭曲。

    x_t      : (3,)        当前末端执行器位置
    p_orig   : (N, 3)      源点云
    delta    : (N, 3)      逐粒子位移 = p_def - p_orig
    K        : int         最近邻数量（grasp 用小 K；manipulation 用 N）
    alpha_t  : float       时间衰减 decay(t) ∈ [0,1]
    返回      : (3,) 扭曲后的位置 x_warp, 以及聚合位移 d
    """
    # 找 x_t 在源点云中的 K 个最近邻
    dists = np.linalg.norm(p_orig - x_t, axis=1)          # (N,)
    nn = np.argsort(dists)[:K]                             # K 个最近邻索引

    # 反距离加权（距离越近权重越大），公式 (2)
    w = 1.0 / (dists[nn] + eps)
    w = w / w.sum()                                       # 归一化 w̃

    # 聚合这些粒子的位移得到该点的形变场取值 d(x_t)
    d = (w[:, None] * delta[nn]).sum(axis=0)              # (3,)

    # 时间衰减：初期跟随局部形变，后期回归原路径，公式 (3)
    x_warp = x_t + alpha_t * d
    return x_warp, d


# ----------------------------------------------------------------------
# 2) 姿态适配：局部雅可比（最小二乘）-> SO(3) 投影 -> SLERP  —— 公式 (4)-(7)
# ----------------------------------------------------------------------
def estimate_local_jacobian(x_t, p_orig, delta, K, eps=1e-6):
    """用 KNN 邻域内"局部相对向量怎么变"来最小二乘拟合局部雅可比 J_t。公式 (5)(6)"""
    dists = np.linalg.norm(p_orig - x_t, axis=1)
    nn = np.argsort(dists)[:K]

    # 局部相对坐标：源 ℓ_orig 和形变后 ℓ_def，公式 (4)
    l_orig = (p_orig[nn] - x_t).T                         # (3, K)
    l_def = (p_orig[nn] + delta[nn] - x_t).T             # (3, K)

    # 闭式最小二乘解 J = X_def X_orig^T (X_orig X_orig^T)^+ ，公式 (6)
    XoXoT = l_orig @ l_orig.T                             # (3,3)
    J = l_def @ l_orig.T @ np.linalg.pinv(XoXoT)         # Moore-Penrose 伪逆
    return J


def project_to_SO3(M):
    """把任意 3x3 矩阵投影到最近的旋转矩阵（SVD）。J_t 一般含剪切/缩放，必须投影。"""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:          # 保证是旋转而非反射
        U[:, -1] *= -1
        R = U @ Vt
    return R


def slerp(R0, R1, t):
    """两个旋转矩阵之间的球面线性插值（用轴角实现），公式 (7)。"""
    R_rel = R0.T @ R1
    # 旋转矩阵 -> 轴角
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return R0.copy()
    axis = np.array([R_rel[2, 1] - R_rel[1, 2],
                     R_rel[0, 2] - R_rel[2, 0],
                     R_rel[1, 0] - R_rel[0, 1]]) / (2 * np.sin(angle))
    a = angle * t
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R_step = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)  # Rodrigues
    return R0 @ R_step


def warp_orientation(x_t, R_t, p_orig, delta, K, alpha_t):
    """扭曲末端姿态：估雅可比 -> 投影 SO(3) -> SLERP。

    完整版（论文公式 5-7）。注意：当局部邻域共线（如 1D 绳子）时，
    雅可比是秩亏的，闭式解会退化。论文针对桌面场景把旋转约束到 Z 轴，
    见下面的 warp_orientation_zaxis —— 那是更鲁棒、也更贴近论文实现的版本。
    """
    J = estimate_local_jacobian(x_t, p_orig, delta, K)
    R_prime = project_to_SO3(J @ R_t)            # 诱导旋转
    R_warp = slerp(R_t, R_prime, alpha_t)        # 按时间衰减插值
    return R_warp


def local_tangent_angle(points):
    """用 PCA 主方向估计一组点的局部切线在 xy 平面内的朝向角(弧度)。"""
    c = points - points.mean(0)
    # xy 平面内的主方向
    cov = c[:, :2].T @ c[:, :2]
    w, v = np.linalg.eigh(cov)
    main = v[:, np.argmax(w)]                     # 最大特征值对应主方向
    if main[0] < 0:                               # 消除 PCA 主方向的 ±180° 符号歧义
        main = -main                              # 约定切线指向 +x（绳子走向）
    return np.arctan2(main[1], main[0])


def warp_orientation_zaxis(x_t, R_t, p_orig, delta, K, alpha_t):
    """论文做法：桌面场景把旋转约束到垂直桌面的 Z 轴。

    通过比较局部邻域形变前/后的【切线朝向】得到 Z 轴旋转角 dθ，
    再按时间衰减 α_t 缩放后绕 Z 轴旋转 R_t。对 1D 物体鲁棒、有意义。
    """
    dists = np.linalg.norm(p_orig - x_t, axis=1)
    nn = np.argsort(dists)[:max(K, 3)]
    ang_orig = local_tangent_angle(p_orig[nn])
    ang_def = local_tangent_angle(p_orig[nn] + delta[nn])
    dtheta = np.arctan2(np.sin(ang_def - ang_orig),
                        np.cos(ang_def - ang_orig))   # 归一化到 [-pi,pi]
    dtheta *= alpha_t                                  # 时间衰减
    c, s = np.cos(dtheta), np.sin(dtheta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ R_t, np.degrees(dtheta)


# ----------------------------------------------------------------------
# 3) 演示：构造一个"弯曲"的形变，对比刚性迁移 vs 形变场扭曲
# ----------------------------------------------------------------------
def make_decay(T, start=1.0, end=0.2):
    """简单线性时间衰减 decay(t)：初期 1.0（紧贴形变），后期 0.2（回归原路径）。"""
    return np.linspace(start, end, T)


def main():
    rng = np.random.default_rng(0)

    # 源物体：一根沿 x 轴的"绳子"（20 个粒子）
    N = 20
    p_orig = np.stack([np.linspace(0, 1, N),
                       np.zeros(N),
                       np.zeros(N)], axis=1)            # (N,3)

    # 形变后：绳子被弯成一条抛物线（y 随 x^2 抬起）—— 一个非刚性拓扑变化
    p_def = p_orig.copy()
    p_def[:, 1] = 0.5 * (p_orig[:, 0] ** 2)             # 弯曲
    delta = p_def - p_orig                              # 逐粒子位移

    # 源示教轨迹：末端沿绳子上方 5cm 平移（10 个 waypoint）
    T = 10
    traj = np.stack([np.linspace(0.1, 0.9, T),
                     np.full(T, 0.05),
                     np.zeros(T)], axis=1)              # (T,3)
    R_t = np.eye(3)                                     # 初始姿态都设为单位

    alpha = make_decay(T)
    K_manip = N                                         # manipulation：用全部点

    print("=== 形变场扭曲 vs 刚性迁移 对比 ===\n")
    print("绳子被弯成抛物线。看末端轨迹如何被扭曲以贴合新几何。\n")
    print(f"{'t':>2} | {'源 waypoint':>22} | {'刚性迁移(整体平移)':>26} | {'形变场扭曲(本文)':>26}")
    print("-" * 90)

    # 刚性 baseline：用所有粒子的平均位移作为单一平移施加到整条轨迹
    rigid_shift = delta.mean(axis=0)

    print(f"{'':>2}   {'':>22}   {'':>26}   {'':>26}   {'Z轴旋转':>8}")
    for t in range(T):
        x_t = traj[t]
        x_rigid = x_t + rigid_shift
        x_warp, d = warp_position(x_t, p_orig, delta, K_manip, alpha[t])
        # 姿态用 Z 轴约束版本（论文桌面做法），对 1D 绳子鲁棒
        _, dtheta_deg = warp_orientation_zaxis(x_t, R_t, p_orig, delta,
                                               K=5, alpha_t=alpha[t])

        print(f"{t:>2} | {np.array2string(x_t, precision=2):>22} | "
              f"{np.array2string(x_rigid, precision=2):>26} | "
              f"{np.array2string(x_warp, precision=2):>26} | "
              f"{dtheta_deg:>6.1f}°")

    print("\n关键观察：")
    print("  - 刚性迁移对每个 waypoint 施加【相同】的平移，无法贴合局部弯曲。")
    print("  - 形变场扭曲让靠近大形变区域(绳尾)的 waypoint 抬得更高，")
    print("    贴合抛物线形状；这正是'形变感知'轨迹迁移的意义。")
    print("  - 最右列：末端姿态绕 Z 轴的旋转量(论文桌面场景约束)。绳子弯曲越大的")
    print("    区域、形变早期(α_t 大)，姿态旋转越多，以对齐变形后的局部切线方向。")


if __name__ == "__main__":
    main()
