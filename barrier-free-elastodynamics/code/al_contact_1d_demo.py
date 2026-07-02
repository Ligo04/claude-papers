#!/usr/bin/env python3
"""
al_contact_1d_demo.py
=====================
论文核心洞察的最小可运行复现:
"Robust and Efficient Penetration-Free Elastodynamics without Barriers"
(Zheng, Luo, Li — ACM TOG 2026, arXiv:2512.12151)

本 demo 用一个 1D 接触问题,直观对比两种 barrier-free 接触响应:
  (A) 朴素二次罚函数(naive quadratic penalty)
  (B) 增广拉格朗日法(augmented Lagrangian, 本文核心)

复现的核心论点(论文 §6.3 / Fig.19):
  朴素罚即使把刚度 μ 调到极大,仍会残留穿透(残留 ≈ F/μ);
  而增广拉格朗日通过【迭代更新乘子 λ】而非【加大刚度 μ】,
  能用一个【温和的 μ】把约束违反精确驱动到目标偏移 δ,
  既消除残留穿透,又保持系统良态(小 μ → 小 Hessian 条件数)。

问题设定(1D):
  一个质点位置 x,被一根弹簧(刚度 k_elastic,静止长度使其平衡点在 x0>0)
  拉着,同时被一个恒定外力 F_ext 推向位于原点的刚性墙。
  非穿透约束:c(x) = x - δ ≥ 0  (希望质点停在墙外侧 δ 处)。

  弹性势能:    U(x) = 0.5 * k_elastic * (x - x_rest)^2  - F_ext * x
  (− F_ext·x 项把外力并入势能,F_ext>0 表示把质点推向墙/原点)

运行:
  python3 al_contact_1d_demo.py
依赖:仅 numpy(matplotlib 可选,用于出图)
"""

import numpy as np

# ----------------------------- 物理/数值参数 -----------------------------
K_ELASTIC = 1.0e3      # 弹性刚度(类比材料 Young's modulus 的角色)
X_REST    = 2.0        # 弹簧静止平衡位置(无接触时质点会停在这里)
F_EXT     = 5.0e3      # 恒定外力,把质点推向墙(原点)。注意 F_EXT 远大于弹性回复力
DELTA     = 0.05       # 目标接触偏移 δ:希望质点停在 x = δ 处(墙外侧)
MAX_ITERS = 200        # 每个求解器的最大迭代数


def elastic_grad_hess(x):
    """弹性 + 外力部分的梯度和 Hessian(1D 标量)。
    U(x) = 0.5*k*(x - x_rest)^2 + F_ext*x   (+F_ext*x 表示外力把质点推向墙/原点)
    U'(x) = k*(x - x_rest) + F_ext
    U''(x) = k
    无接触约束时平衡点 x = x_rest - F_ext/k < 0,即质点会穿过墙 → 必须靠接触约束顶回。
    """
    g = K_ELASTIC * (x - X_REST) + F_EXT
    h = K_ELASTIC
    return g, h


def constraint(x):
    """有符号约束 c(x) = x - δ。 c >= 0 表示无穿透(在墙外侧 δ 之外)。"""
    return x - DELTA


# ============================ (A) 朴素二次罚 ============================
def solve_naive_penalty(mu, x_init=X_REST, verbose=False):
    """
    朴素二次罚:在 c(x) < 0(穿透)时加罚能量 0.5*mu*c(x)^2。
      罚梯度  = mu * min(c, 0)            (只在穿透时激活)
      罚 Hessian = mu  (穿透时) / 0 (无穿透)
    用 Newton 迭代到收敛(梯度→0)。

    返回 (x_final, 残留穿透 = max(0, δ - x_final))。

    论点:平衡时 弹性力 ≈ 罚力 → F_ext ≈ mu * |c| → 残留穿透 |c| ≈ F_ext/mu。
          要让残留→0,必须 mu→∞,而大 mu 直接放大 Hessian 条件数(病态)。
    """
    x = x_init
    for it in range(MAX_ITERS):
        g, h = elastic_grad_hess(x)
        c = constraint(x)
        if c < 0.0:                 # 穿透,激活罚
            g += mu * c             # c<0,罚梯度把 x 往正方向推
            h += mu
        # Newton 步
        step = -g / h
        x += step
        if abs(step) < 1e-12:
            break
    residual_pen = max(0.0, DELTA - x)   # 仍在墙内侧(x<δ)的深度
    cond = (K_ELASTIC + (mu if constraint(x) < 0 else 0.0)) / K_ELASTIC
    if verbose:
        print(f"   [naive μ={mu:.1e}] x={x:.6f}  c={constraint(x):+.6f}  "
              f"残留穿透={residual_pen:.6f}  Hessian条件数比≈{cond:.1e}")
    return x, residual_pen, cond


# ====================== (B) 增广拉格朗日(本文核心) ======================
def solve_augmented_lagrangian(mu, n_outer=30, x_init=X_REST, verbose=False):
    """
    增广拉格朗日(对应论文 Eq.12 / Algorithm 2 的 1D 简化版):
      L(x) = U(x) + γ[ 0.5*mu*(c - s)^2 - λ*(c - s) ]
    交替:
      1) 松弛变量闭式更新   s = max(0, c - λ/mu)            (论文 Eq.14)
      2) 对 x 做(投影)Newton 内迭代,使 L 下降
      3) 外层乘子更新       λ ← λ - mu*(c - s)              (论文 Eq.13)
    这里 γ(decay) 恒为 1(单约束,不涉及失活淘汰)。

    关键:mu 保持【温和】,靠 λ 的累积把 c 精确驱动到 0(即 x→δ)。
    """
    x = x_init
    lam = 0.0
    history = []
    for outer in range(n_outer):
        # --- 内层:对 x 做 Newton(固定 λ,内含 s 的闭式更新) ---
        for _ in range(50):
            c = constraint(x)
            s = max(0.0, c - lam / mu)          # 论文 Eq.14
            g, h = elastic_grad_hess(x)
            # L 对 x 的梯度/Hessian(dc/dx = 1):
            #   d/dx [0.5*mu*(c-s)^2 - λ(c-s)] = mu*(c - λ/mu - s) ... (论文 Alg.2 line 10)
            g += mu * (c - lam / mu - s)
            h += mu                              # 论文 Alg.2 line 11:mu * (dc/dx)^2 = mu
            step = -g / h
            x += step
            if abs(step) < 1e-12:
                break
        # --- 外层:乘子更新(论文 Eq.13)---
        c = constraint(x)
        s = max(0.0, c - lam / mu)
        lam = lam - mu * (c - s)
        history.append((outer, x, lam, c))
        if verbose:
            print(f"   [AL outer={outer:2d} μ={mu:.1e}] x={x:.6f}  "
                  f"c={c:+.6f}  λ={lam:.3f}")
        if abs(c) < 1e-9 and outer > 0:
            break
    residual_pen = max(0.0, DELTA - x)
    cond = (K_ELASTIC + mu) / K_ELASTIC          # AL 的 Hessian 始终含 mu(温和)
    return x, residual_pen, cond, lam, history


# ================================ 主程序 ================================
def main():
    print("=" * 70)
    print("Barrier-Free Elastodynamics — 1D 接触:增广拉格朗日 vs 朴素罚")
    print("=" * 70)
    print(f"参数: k_elastic={K_ELASTIC:.0e}, x_rest={X_REST}, "
          f"F_ext={F_EXT:.0e}, 目标偏移 δ={DELTA}")
    print(f"无接触平衡: x_rest - F_ext/k = {X_REST - F_EXT/K_ELASTIC:.3f} "
          f"(<0,质点会穿墙,故必须靠接触约束顶在 x=δ 外侧)\n")

    # ---- (A) 朴素罚:扫不同 μ,看残留穿透如何随 μ 缓慢减小 ----
    print("【A. 朴素二次罚】即便 μ 极大,仍残留穿透,且 Hessian 条件数随 μ 暴涨:")
    for mu in [1e3, 1e4, 1e5, 1e6, 1e9]:
        solve_naive_penalty(mu, verbose=True)

    # ---- (B) 增广拉格朗日:用温和 μ,靠乘子 λ 精确停在 δ ----
    print("\n【B. 增广拉格朗日】用温和 μ=1e3,靠迭代更新 λ 精确收敛到 x=δ:")
    mu_al = 1e3
    x_al, res_al, cond_al, lam_al, hist = solve_augmented_lagrangian(
        mu_al, verbose=True)

    # ---- 对比总结 ----
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    x_p, res_p, cond_p = solve_naive_penalty(1e3)      # μ=1e3 的罚
    xp6, res_p6, cond_p6 = solve_naive_penalty(1e6)    # μ=1e6 的罚
    print(f"{'方法':<26}{'最终 x':<12}{'残留穿透':<14}{'Hessian条件数比'}")
    print("-" * 70)
    print(f"{'朴素罚 (μ=1e3)':<22}{x_p:<12.6f}{res_p:<14.6f}{cond_p:.1e}")
    print(f"{'朴素罚 (μ=1e6)':<22}{xp6:<12.6f}{res_p6:<14.6f}{cond_p6:.1e}")
    print(f"{'增广拉格朗日 (μ=1e3)':<18}{x_al:<12.6f}{res_al:<14.6f}{cond_al:.1e}"
          f"   (λ={lam_al:.1f})")
    print("-" * 70)
    print("结论:")
    print(f"  • 朴素罚 μ=1e3:残留穿透 {res_p:.4f},约束严重违反(质点陷在墙内)。")
    print(f"  • 朴素罚 μ=1e6:残留减到 {res_p6:.6f},但 Hessian 条件数比飙到 {cond_p6:.0e}(病态!)。")
    print(f"  • 增广拉格朗日 μ=1e3:残留穿透 {res_al:.2e}(≈0),条件数比仅 {cond_al:.0e}(良态)。")
    print(f"    → 这正是论文用【更新乘子 λ】替代【加大刚度 μ】的核心价值:")
    print(f"      既消除残留穿透(让 CCD 能产生非零 TOI),又保持系统良态(PCG 收敛快)。")

    # ---- 可选出图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mus = np.logspace(3, 9, 25)
        res_naive = [solve_naive_penalty(m)[1] for m in mus]
        cond_naive = [solve_naive_penalty(m)[2] for m in mus]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.loglog(mus, np.maximum(res_naive, 1e-12), 'r-o', ms=3,
                   label='Naive penalty: residual penetration')
        ax1.axhline(res_al if res_al > 0 else 1e-12, color='g', ls='--',
                    label=f'Augmented Lagrangian (mu=1e3): residual~{res_al:.0e}')
        ax1.set_xlabel('penalty stiffness mu'); ax1.set_ylabel('residual penetration', color='r')
        ax1.tick_params(axis='y', labelcolor='r'); ax1.legend(loc='upper right')
        ax2 = ax1.twinx()
        ax2.loglog(mus, cond_naive, 'b-^', ms=3, alpha=0.6)
        ax2.set_ylabel('Hessian condition-number ratio (blue)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        plt.title('Naive penalty: reducing penetration needs mu^, cost is ill-conditioning')
        plt.tight_layout()
        out = 'al_vs_penalty_1d.png'
        plt.savefig(out, dpi=120)
        print(f"\n[已保存图像: {out}]")
    except Exception as e:
        print(f"\n[跳过出图:{e}]")


if __name__ == "__main__":
    main()
