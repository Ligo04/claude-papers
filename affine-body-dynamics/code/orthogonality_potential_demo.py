"""
Affine Body Dynamics —— 正交势 V⊥ 可视化演示
===========================================

论文 Eq.6 (fulltext.txt 第3.2节): V⊥(q) = κν‖AA^T − I‖²_F，2D 展开为 Eq.7 的
V⊥ = κν[(a1·a1−1)² + (a2·a2−1)² + 2(a1·a2)²]（a1,a2 是 A 的两行）。

本文件把 2D 仿射矩阵参数化为 A = R(θ)·S(s,γ)，R 是纯旋转，S 是"拉伸+剪切"：
    S(s,γ) = [[s, γ], [0, 1]]
用等高线图直接展示 V⊥ 的能量地形，教学重点：
  1) V⊥ 在 s=1,γ=0（S=I，A 为纯旋转）处取零，且沿 θ 方向完全平坦——这就是
     "旋转不变性"：势能只惩罚偏离正交，不惩罚旋转本身，因此纯转动不消耗
     任何弹性能量。
  2) κ 越大，偏离 s=1,γ=0 的代价增长得越陡（等高线越密集），对应论文里
     "κ>100GPa ⇒ 形变被强烈压制"；这里用小 κ vs 大 κ 下同一次"碰撞冲量"
     引发的形变幅度做直接对比，验证形变幅度 ∝ 1/sqrt(κ)。
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def R_mat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def S_mat(s, gamma):
    """拉伸(沿 x 方向因子 s)+ 剪切(系数 γ)，S=I 时对应无形变。"""
    return np.array([[s, gamma], [0.0, 1.0]])


def V_perp(A, kappa, nu):
    """论文 Eq.6：V⊥ = κν‖AA^T − I‖²_F，与 2D 展开 Eq.7 完全等价
    （AA^T 的对角元是 a1·a1, a2·a2，非对角元是 a1·a2，与 Frobenius 范数一致）。"""
    diff = A @ A.T - np.eye(2)
    return kappa * nu * np.sum(diff * diff)


# ------------------------------------------------------------------
# 第一部分：能量地形 —— 等高线图 + 旋转不变性数值验证
# ------------------------------------------------------------------

kappa_demo, nu_demo = 1.0, 1.0
s_grid = np.linspace(0.3, 1.7, 120)
g_grid = np.linspace(-0.8, 0.8, 120)
S_, G_ = np.meshgrid(s_grid, g_grid)

# theta=0 时的能量地形（因为旋转不变性，任何 theta 应给出完全相同的地形）
E0 = np.zeros_like(S_)
for i in range(S_.shape[0]):
    for j in range(S_.shape[1]):
        E0[i, j] = V_perp(S_mat(S_[i, j], G_[i, j]), kappa_demo, nu_demo)

# 用另一个任意 theta 重新计算，验证地形与 theta 无关（旋转不变性，Eq.6 的核心性质）
theta_probe = 1.3
E_theta = np.zeros_like(S_)
for i in range(S_.shape[0]):
    for j in range(S_.shape[1]):
        A = R_mat(theta_probe) @ S_mat(S_[i, j], G_[i, j])
        E_theta[i, j] = V_perp(A, kappa_demo, nu_demo)
invariance_err = np.max(np.abs(E0 - E_theta))
print(f"[验证1] 旋转不变性：任意 theta 下能量地形与 theta=0 的最大差异 = {invariance_err:.3e}"
      f"（应为机器精度量级，证明 V⊥ 只依赖 S 不依赖 R）")

# 纯旋转轨迹 s=1,γ=0，theta 扫过一圈，能量应恒为 0
thetas = np.linspace(0, 2 * np.pi, 200)
E_pure_rotation = [V_perp(R_mat(th) @ S_mat(1.0, 0.0), kappa_demo, nu_demo) for th in thetas]
print(f"[验证2] 纯旋转能量谷：θ∈[0,2π) 时 V⊥ 最大值 = {max(E_pure_rotation):.3e}"
      f"（应≈0，说明旋转本身不消耗弹性能量）")


# ------------------------------------------------------------------
# 第二部分：小 κ vs 大 κ —— 同一次"碰撞冲量"下的形变对比
#   限定 γ=0,θ=0，A=diag(s,1)，退化为 1D 弹簧：V(s)=κν(s²−1)²
#   （对应 Eq.7 的 (a1·a1−1)²项；a1=(s,0) 时 a1·a1=s²，其余两项为 0）
#   有效刚度 k_eff = V''(1) = 8κν（在 s=1 附近线性化），故形变振幅
#   应近似满足 amplitude ∝ 1/sqrt(κ)（等能量冲量下，振幅 = sqrt(2E/k_eff)）。
# ------------------------------------------------------------------

def V_1d(s, kappa, nu):
    return kappa * nu * (s ** 2 - 1) ** 2


def dV_1d(s, kappa, nu):
    return kappa * nu * 4 * s * (s ** 2 - 1)


def simulate_impact(kappa, nu, v0, mass=1.0, dt=1e-4, n_steps=4000):
    """辛欧拉(半隐式)积分 1D 弹簧-质点：s(0)=1(无形变)，初速度 v0 模拟碰撞冲量。"""
    s, v = 1.0, v0
    traj_s, traj_t = [s], [0.0]
    for k in range(n_steps):
        v += -dV_1d(s, kappa, nu) / mass * dt
        s += v * dt
        traj_s.append(s)
        traj_t.append((k + 1) * dt)
    return np.array(traj_t), np.array(traj_s)


nu_1d = 1.0
kappa_small, kappa_large = 5.0, 500.0
v0 = 2.0   # 统一的"碰撞冲量"（初速度），代表相同大小的撞击

t_small, s_small = simulate_impact(kappa_small, nu_1d, v0)
t_large, s_large = simulate_impact(kappa_large, nu_1d, v0)
amp_small = np.max(np.abs(s_small - 1.0))
amp_large = np.max(np.abs(s_large - 1.0))
predicted_ratio = np.sqrt(kappa_large / kappa_small)
actual_ratio = amp_small / amp_large
print(f"[对比] kappa_small={kappa_small}: 最大形变 |s-1| = {amp_small:.4f}")
print(f"[对比] kappa_large={kappa_large}: 最大形变 |s-1| = {amp_large:.4f}")
print(f"[对比] 形变比 amp_small/amp_large = {actual_ratio:.3f}，"
      f"理论预测 sqrt(kappa_large/kappa_small) = {predicted_ratio:.3f}（量级应吻合）")


# ------------------------------------------------------------------
# 绘图：2x2 —— 能量地形 / 旋转不变性 / 1D 势能曲线 / 冲击形变轨迹
# ------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
cs = ax.contourf(S_, G_, E0, levels=30, cmap="viridis")
fig.colorbar(cs, ax=ax, label="V_perp")
ax.plot(1.0, 0.0, "r*", markersize=16, label="S=I (零能量谷)")
ax.set_xlabel("拉伸 s")
ax.set_ylabel("剪切 γ")
ax.set_title(f"V⊥(s,γ) 能量地形 (κ={kappa_demo}, θ=0，任意 θ 地形相同)")
ax.legend()

ax = axes[0, 1]
ax.plot(thetas, E_pure_rotation)
ax.set_xlabel("旋转角 θ (rad)")
ax.set_ylabel("V_perp")
ax.set_title("纯旋转轨迹 (s=1, γ=0) 的能量 —— 应恒为 0")
ax.set_ylim(-0.01, 0.05)

ax = axes[1, 0]
s_line = np.linspace(0.4, 1.6, 200)
ax.plot(s_line, V_1d(s_line, kappa_small, nu_1d), label=f"κ={kappa_small} (软)")
ax.plot(s_line, V_1d(s_line, kappa_large, nu_1d), label=f"κ={kappa_large} (硬)")
ax.set_xlabel("拉伸 s")
ax.set_ylabel("V(s) = κν(s²−1)²")
ax.set_title("1D 势能曲线：κ 越大势阱越陡")
ax.legend()

ax = axes[1, 1]
ax.plot(t_small, s_small - 1.0, label=f"κ={kappa_small}, 形变幅度={amp_small:.3f}")
ax.plot(t_large, s_large - 1.0, label=f"κ={kappa_large}, 形变幅度={amp_large:.3f}")
ax.set_xlabel("time")
ax.set_ylabel("s − 1 (形变量)")
ax.set_title(f"同一次碰撞冲量(v0={v0})下的形变轨迹对比")
ax.legend()

fig.tight_layout()
out_path = "orthogonality_potential_demo.png"
fig.savefig(out_path, dpi=130)
print(f"\n已保存图像: {out_path}")
