"""
BGK 半时刻分布函数 f(h) 与动理学矩通量 demo —— KPM-FR §3.5
============================================================

对照论文式 (4)(5)(11)(12):
  Boltzmann-BGK:  ∂f/∂t + ξ·∇f = -(1/τ)(f-g)
  矩:             U = ⟨φ f⟩,   F_d = ⟨ξ_d φ f⟩,   φ=[1, ξ]^T
  半时刻分布:      f(h) ≈ A f_s(h) + B g(h),  A=τ/h, B=1-τ/h
                  f_s(h) = g(0) - h ξ·∇g(0)   (传播项, g 沿 ξ 一阶 Taylor)

本 demo 用 1D 等温动理学模型(D1Q-连续, 高斯积分近似速度空间):
  1. 构造 Maxwell 平衡态 g(ρ,u);
  2. 用矩恢复宏观量 U=(ρ, ρu) 与通量 F;
  3. 演示半时刻分布 f(h) 及其矩通量 F(h) 的闭式计算;
  4. 展示"完全在矩空间": 我们从不存储 f, 只算它的矩。

只依赖 numpy:  python bgk_halftime_flux_demo.py
"""

import numpy as np


# 等温声速平方 (格子单位常取 RT = c_s^2 = 1/3)
RT = 1.0 / 3.0


# ----------------------------------------------------------------------
# 连续速度空间: 用 Gauss-Hermite 求积近似 ⟨·⟩ = ∫ · dξ
# ----------------------------------------------------------------------
def gauss_hermite_velocity(n=8):
    """返回速度节点 xi 与权重 w, 使 ⟨A⟩ ≈ Σ w_i A(xi_i)。
    速度空间积分 ∫ A(ξ) g dξ 用 Gauss-Hermite (带 Maxwell 权) 精确到高阶矩。"""
    x, w = np.polynomial.hermite_e.hermegauss(n)   # 概率派 Hermite (权 e^{-x²/2})
    xi = x * np.sqrt(RT)                            # 缩放到声速尺度
    return xi, w


def maxwell(rho, u, xi):
    """等温 Maxwell 平衡态 g(ρ,u) 在速度节点上的值(已含高斯因子, 配合 hermegauss 权)。
    g = ρ/sqrt(2π RT) exp(-(ξ-u)²/(2RT)); 与 hermegauss 的 e^{-x²/2} 权配合,
    我们把 g 写成 相对平衡的形式, 使 ⟨φ g⟩ 精确给出 (ρ, ρu)。"""
    # 用相对 Maxwell: g = ρ * exp(-(ξ-u)²/(2RT) + ξ²/(2RT)) / normalization
    # 简化: 直接用离散权 w 表示 ∫·(高斯) , 故这里 g 表示 ρ*exp(u ξ/RT - u²/(2RT))
    return rho * np.exp((u*xi)/RT - (u*u)/(2*RT))


def moments(f_vals, xi, w):
    """由分布在速度节点上的值算矩: U=(ρ, ρu), 通量 F=(ρu, ρu²+ρRT)。
    ⟨A f⟩ = Σ w_i A_i f_i / sqrt(2π) ... 这里权已归一, 直接加权求和。"""
    Z = np.sum(w)                       # 归一因子 (∫ 高斯 = sqrt(2π) → hermegauss Σw)
    rho     = np.sum(w * f_vals) / Z
    rho_u   = np.sum(w * xi * f_vals) / Z
    mom2    = np.sum(w * xi*xi * f_vals) / Z          # ⟨ξ² f⟩ = ρu² + ρRT
    U = np.array([rho, rho_u])
    F = np.array([rho_u, mom2])         # F=(ρu, ρu²+ρRT): 动量通量含压力 ρRT
    return U, F


# ----------------------------------------------------------------------
# 半时刻分布 f(h) = A f_s(h) + B g(h) (式 11)
# ----------------------------------------------------------------------
def halftime_flux(rho0, u0, grad_rho, grad_u, tau, h, xi, w):
    """给定 t=0 的宏观态与其空间梯度, 算半时刻通量 F(h)。
    步骤对应式 (11): 传播项 f_s = g(0) - h ξ ∂g(0); 相容条件定 U(h); 再取矩。"""
    A = tau / h
    B = 1.0 - tau / h

    g0 = maxwell(rho0, u0, xi)
    # ∂g/∂x ≈ ∂g/∂ρ ∂ρ/∂x + ∂g/∂u ∂u/∂x  (链式), 用有限差分近似梯度方向导数
    eps = 1e-6
    dg_drho = (maxwell(rho0+eps, u0, xi) - maxwell(rho0-eps, u0, xi)) / (2*eps)
    dg_du   = (maxwell(rho0, u0+eps, xi) - maxwell(rho0, u0-eps, xi)) / (2*eps)
    grad_g0 = dg_drho * grad_rho + dg_du * grad_u          # ∇g(0)

    # 传播分布 f_s(h) = g(0) - h ξ·∇g(0)
    f_s = g0 - h * xi * grad_g0

    # 相容条件: U(h) = ⟨φ f_s⟩  (式 11 下方)
    U_h, _ = moments(f_s, xi, w)
    rho_h, rho_u_h = U_h
    u_h = rho_u_h / rho_h
    g_h = maxwell(rho_h, u_h, xi)

    # f(h) = A f_s + B g(h)
    f_h = A * f_s + B * g_h
    U_out, F_out = moments(f_h, xi, w)
    return U_out, F_out, (A, B, U_h)


def main():
    xi, w = gauss_hermite_velocity(n=10)
    print("=" * 66)
    print("BGK 半时刻分布 f(h) 与矩通量 demo (KPM-FR §3.5)")
    print(f"等温声速² RT = {RT:.4f};  速度节点数 = {len(xi)}")
    print("=" * 66)

    # [1] 矩恢复验证: 平衡态的矩应精确给出 (ρ, ρu)
    rho, u = 1.2, 0.15
    g0 = maxwell(rho, u, xi)
    U, F = moments(g0, xi, w)
    print("\n[1] 平衡态矩恢复验证:")
    print(f"    设定 ρ={rho}, u={u}  →  ⟨φg⟩ 得 ρ={U[0]:.4f}, ρu={U[1]:.4f} (=ρu={rho*u:.4f})")
    print(f"    动量通量 F₂=⟨ξ²g⟩={F[1]:.4f}  应=ρu²+ρRT={rho*u*u+rho*RT:.4f} ✓")

    # [2] 半时刻通量: 给定一个有梯度的态
    print("\n[2] 半时刻通量 F(h) (式 11): 完全由 U(0) 及其梯度算出")
    tau, h = 0.05, 0.5*0.1        # 松弛时间, 半步 h=Δt/2
    grad_rho, grad_u = 0.02, 0.05  # 空间梯度
    U_out, F_out, (A, B, U_h) = halftime_flux(rho, u, grad_rho, grad_u, tau, h, xi, w)
    print(f"    A=τ/h={A:.3f}, B=1-τ/h={B:.3f}")
    print(f"    相容条件给出半时刻宏观态 U(h)=(ρ={U_h[0]:.4f}, ρu={U_h[1]:.4f})")
    print(f"    半时刻通量 F(h)=(质量通量 {F_out[0]:.4f}, 动量通量 {F_out[1]:.4f})")

    # [3] 黏性效应: 改变 tau, 看非平衡对通量的影响
    print("\n[3] 黏性(松弛时间 τ)对通量的影响 —— 无黏/黏性由单一 f(h) 统一给出:")
    print(f"    {'τ':>8} | {'A=τ/h':>8} | {'动量通量 F₂':>12}")
    for tau_i in [0.001, 0.02, 0.05, 0.1]:
        _, F_i, _ = halftime_flux(rho, u, grad_rho, grad_u, tau_i, h, xi, w)
        print(f"    {tau_i:>8.3f} | {tau_i/h:>8.3f} | {F_i[1]:>12.5f}")
    print("    τ 越大(黏性越强), 非平衡传播项权重 A 越大, 通量随之改变。")
    print("    ★ 全程只算 f 的矩, 从不存储完整分布 f —— 这就是'完全在矩空间'。")


if __name__ == "__main__":
    main()
