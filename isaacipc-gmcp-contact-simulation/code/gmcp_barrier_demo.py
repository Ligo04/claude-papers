"""
gmcp_barrier_demo.py
====================
GMCP（几何 mortar 接触势）核心组件的最小可运行演示。

聚焦论文 §4.2-§4.5 的四个数值要素，用 1D/简化设置把抽象公式变成可看的曲线/数字：
    1. IPC barrier 函数 B(g, ε)：间隙 g→0 时能量→∞，保证无穿透
    2. C^1 Hermite 阶跃 H(x, δ) = 3t^2 - 2t^3：master 特征权重 γ 的平滑过渡（公式 13）
    3. 自适应支撑半径 ε_k = min(0.9·g_ref, ε_max)：避免静止态伪力（公式 17）
    4. 线性化间隙步长 α* = 0.9·g/(-∇g·Δx)：保证牛顿步无穿透（公式 18-19）

并用一个「1D barrier 弹簧床」演示 mortar 积分权重 w_k 如何让接触压力分布
在非匹配采样下正确传递（呼应 patch test 的均匀压力传递）。

只依赖 numpy。运行：python gmcp_barrier_demo.py
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)


# ---------------------------------------------------------------------------
# 1. IPC barrier 函数 B(g, ε)  [Li et al. 2020]
# ---------------------------------------------------------------------------
def ipc_barrier(g, eps):
    """IPC 的 log-barrier：
        B(g) = -(g - ε)^2 * ln(g / ε),   0 < g < ε
        B(g) = 0,                         g >= ε
    性质：g→0+ 时 B→+∞（不可穿透）；g=ε 时 B 与 B' 都为 0（C^1 接到 0）。
    """
    g = np.asarray(g, dtype=float)
    out = np.zeros_like(g)
    active = (g > 0) & (g < eps)
    ga = g[active]
    out[active] = -(ga - eps) ** 2 * np.log(ga / eps)
    # g<=0 视为穿透 -> 能量无穷大
    out[g <= 0] = np.inf
    return out


def ipc_barrier_grad(g, eps):
    """B'(g) = dB/dg，用于接触力 = -B'(g)。"""
    g = np.asarray(g, dtype=float)
    out = np.zeros_like(g)
    active = (g > 0) & (g < eps)
    ga = g[active]
    # d/dg [ -(g-ε)^2 ln(g/ε) ]
    #   = -2(g-ε)ln(g/ε) - (g-ε)^2 / g
    out[active] = -2 * (ga - eps) * np.log(ga / eps) - (ga - eps) ** 2 / ga
    return out


def demo_barrier():
    print("=" * 70)
    print("1) IPC barrier 函数 B(g, ε)：无穿透的能量势垒")
    print("=" * 70)
    eps = 1e-2
    gaps = np.array([1.0, 0.5, 0.1, 0.05, 0.02, 0.011, 0.005, 0.001, 1e-4])
    print(f"支撑半径 ε = {eps}")
    print(f"{'间隙 g':>10} | {'B(g)':>14} | {'接触力 -dB/dg':>16} | 状态")
    print("-" * 64)
    for g in gaps:
        B = float(ipc_barrier(np.array([g]), eps)[0])
        F = float(-ipc_barrier_grad(np.array([g]), eps)[0])
        state = "barrier 未激活 (g≥ε)" if g >= eps else "barrier 激活，排斥力↑"
        print(f"{g:>10.4f} | {B:>14.4e} | {F:>16.4e} | {state}")
    print("\n解读：g≥ε 时能量和力都为 0；g 越接近 0，能量和排斥力急剧增大 →∞，")
    print("      于是优化器永远不会让 g 穿过 0，实现『可行迭代(feasible)』无穿透。\n")


# ---------------------------------------------------------------------------
# 2. C^1 Hermite 阶跃 H(x, δ)  (公式 13) —— master 特征权重 γ 的核心
# ---------------------------------------------------------------------------
def hermite_step(x, delta):
    """公式 (13):
        H(x,δ) = 0,            x<=0
                 3t^2-2t^3,    0<x<δ, t=x/δ
                 1,            x>=δ
    C^1 连续：H(0)=H'(0)=0, H(δ)=1, H'(δ)=0。
    """
    x = np.asarray(x, dtype=float)
    t = np.clip(x / delta, 0.0, 1.0)
    return 3 * t ** 2 - 2 * t ** 3


def hermite_step_deriv(x, delta):
    """H'(x,δ) = (6t - 6t^2)/δ，验证端点导数为 0（C^1）。"""
    x = np.asarray(x, dtype=float)
    t = np.clip(x / delta, 0.0, 1.0)
    d = (6 * t - 6 * t ** 2) / delta
    d[(x <= 0) | (x >= delta)] = 0.0
    return d


def demo_hermite():
    print("=" * 70)
    print("2) C^1 Hermite 阶跃 H(x,δ)：master 特征权重 γ 的平滑过渡 (公式13)")
    print("=" * 70)
    delta = 0.2
    xs = np.linspace(-0.05, 0.25, 13)
    print(f"过渡宽度 δ = {delta}")
    print(f"{'x':>8} | {'H(x,δ)':>10} | {'dH/dx':>10}")
    print("-" * 34)
    for x in xs:
        H = float(hermite_step(np.array([x]), delta)[0])
        Hp = float(hermite_step_deriv(np.array([x]), delta)[0])
        print(f"{x:>8.3f} | {H:>10.4f} | {Hp:>10.4f}")
    # 验证 C^1：端点处导数应为 0
    print("\nC^1 验证：H'(0)={:.3e}, H'(δ)={:.3e}  (都应≈0)".format(
        float(hermite_step_deriv(np.array([0.0]), delta)[0]),
        float(hermite_step_deriv(np.array([delta]), delta)[0])))
    print("意义：face 样本靠近主三角形边缘时 γ→0、edge 样本接管；C^1 保证 Hessian")
    print("      平滑，牛顿法不抖动，且避免 face/edge/point 之间重复计数。\n")


# ---------------------------------------------------------------------------
# 3. 自适应支撑半径 ε_k = min(0.9·g_ref, ε_max)  (公式 17)
# ---------------------------------------------------------------------------
def adaptive_eps(g_ref, eps_max):
    """公式 (17)。保证 g_ref > ε_k，于是静止态 barrier 不激活 -> 无伪力。"""
    return min(0.9 * g_ref, eps_max)


def demo_adaptive_eps():
    print("=" * 70)
    print("3) 自适应支撑半径 ε_k = min(0.9·g_ref, ε_max)：消除静止态伪力 (公式17)")
    print("=" * 70)
    eps_max = 1e-3
    print(f"ε_max = {eps_max}")
    print(f"{'参考间隙 g_ref':>16} | {'ε_k':>12} | {'g_ref>ε_k?':>12} | {'静止态 B':>12}")
    print("-" * 60)
    for g_ref in [5e-2, 1e-2, 2e-3, 1e-3, 5e-4]:
        eps_k = adaptive_eps(g_ref, eps_max)
        safe = g_ref > eps_k
        B_rest = float(ipc_barrier(np.array([g_ref]), eps_k)[0])
        print(f"{g_ref:>16.1e} | {eps_k:>12.3e} | {str(safe):>12} | {B_rest:>12.3e}")
    print("\n解读：因 0.9<1，恒有 g_ref>ε_k，故在参考(静止)间隙处 barrier=0，")
    print("      不产生本不该有的力（伪力）。这是 patch test 伪应力低至 2e-4 的关键。\n")


# ---------------------------------------------------------------------------
# 4. 线性化间隙步长 α* = 0.9·g/(-∇g·Δx)  (公式 18-19)
# ---------------------------------------------------------------------------
def linearized_step_size(gaps, grad_dot_dx):
    """公式 (18)-(19)。
    gaps:        各样本当前间隙 g_k (>0)
    grad_dot_dx: 各样本的 ∇g_k·Δx（<0 表示该样本间隙在减小，有穿透风险）
    返回全局安全步长 α = min_k α*_k。
    """
    gaps = np.asarray(gaps, float)
    gdx = np.asarray(grad_dot_dx, float)
    closing = gdx < 0                       # 只看间隙在减小的样本
    if not np.any(closing):
        return 1.0                          # 没有样本在闭合 -> 整步可走
    alpha_k = 0.9 * gaps[closing] / (-gdx[closing])   # 公式(18)
    return float(np.min(alpha_k))           # 公式(19) 取全局最小


def demo_step_size():
    print("=" * 70)
    print("4) 线性化间隙步长 α=min_k 0.9·g_k/(-∇g_k·Δx)：牛顿步无穿透 (公式18-19)")
    print("=" * 70)
    # 三个接触样本：g 和 ∇g·Δx
    gaps = np.array([0.10, 0.04, 0.20])
    gdx = np.array([-0.50, -0.80, +0.10])   # 第三个在张开(>0)，不限制步长
    print(f"{'样本':>6} | {'间隙 g_k':>10} | {'∇g_k·Δx':>10} | {'α*_k':>10}")
    print("-" * 46)
    for i, (g, d) in enumerate(zip(gaps, gdx)):
        if d < 0:
            a = 0.9 * g / (-d)
            print(f"{i:>6} | {g:>10.3f} | {d:>10.3f} | {a:>10.4f}")
        else:
            print(f"{i:>6} | {g:>10.3f} | {d:>10.3f} | {'(张开,不限)':>10}")
    alpha = linearized_step_size(gaps, gdx)
    print(f"\n全局步长 α = min = {alpha:.4f}")
    # 验证：走完 α 步后，最危险样本的间隙仍 > 0
    g_after = gaps + alpha * gdx
    print(f"走 α 步后各间隙: {g_after}  (均应 > 0，保留 ~10% 安全裕度)")
    print("解读：取全局最小，保证最危险的接触样本也不穿透；0.9 留安全裕度。\n")


# ---------------------------------------------------------------------------
# 5. mortar 积分权重 w_k：让接触压力在非匹配采样下正确传递
# ---------------------------------------------------------------------------
def demo_mortar_pressure_transfer():
    print("=" * 70)
    print("5) mortar 积分权重 w_k：非匹配/非均匀采样下接触压力正确传递")
    print("=" * 70)
    print("设定：长度 L=1 的 1D 接触界面，真实压力『线性变化』p(x)=2+16x。")
    print("      真实总合力 = ∫_0^1 p(x)dx = 2 + 8 = 10。")
    print("      采样点疏密不均（左密右疏，模拟非匹配网格裁剪结果），")
    print("      看『点对点等权』vs『mortar 积分权重』谁能正确还原合力。\n")

    L = 1.0
    p_fn = lambda x: 2.0 + 16.0 * x          # 线性压力分布
    total_true = 2.0 + 8.0                    # ∫_0^1 (2+16x) dx = 10

    # 非均匀采样点（左密右疏）—— 关键：等权会过度代表密集区
    samples = np.array([0.05, 0.12, 0.20, 0.55, 0.62, 0.70, 0.95])
    n = len(samples)
    p_vals = p_fn(samples)

    # 方法 A：点对点等权（每个样本权重 = L/n）—— 朴素「每点一票」
    w_equal = np.full(n, L / n)
    force_equal = float(np.sum(w_equal * p_vals))

    # 方法 B：mortar 积分权重（样本覆盖的区间长度，1D 求积权重）
    edges = np.concatenate([[0.0], (samples[:-1] + samples[1:]) / 2, [L]])
    w_mortar = np.diff(edges)                 # 每个样本控制的区间长度
    force_mortar = float(np.sum(w_mortar * p_vals))

    print(f"采样点位置: {samples}")
    print(f"对应压力 p(x_k): {p_vals}")
    print(f"真实总合力 ∫p dx = {total_true:.4f}\n")
    print(f"{'方法':>22} | {'Σw_k':>8} | {'还原合力':>10} | {'相对误差':>10}")
    print("-" * 60)
    print(f"{'A. 点对点等权(L/n)':>22} | {np.sum(w_equal):>8.4f} | "
          f"{force_equal:>10.4f} | {abs(force_equal-total_true)/total_true:>10.2%}")
    print(f"{'B. mortar 积分权重':>22} | {np.sum(w_mortar):>8.4f} | "
          f"{force_mortar:>10.4f} | {abs(force_mortar-total_true)/total_true:>10.2%}")
    print("\n解读：两种方法权重和都=L=1，但在『非均匀压力 + 非均匀采样』下，")
    print("      等权『每点一票』过度代表了左侧密集区的低压、低估了右侧高压，")
    print("      合力偏小；mortar 权重让每个样本按它『覆盖的接触面积/长度』计票，")
    print("      对线性分布近乎精确。这正是 GMCP 用 w_k(公式10)的价值——")
    print("      非匹配网格上接触压力分布也能被正确积分与传递。\n")


if __name__ == "__main__":
    demo_barrier()
    demo_hermite()
    demo_adaptive_eps()
    demo_step_size()
    demo_mortar_pressure_transfer()
    print("=" * 70)
    print("GMCP 势 (公式10): Ψ = Σ_q Σ_k  κ_q · w_k · γ_k · B(g_k, ε_k)")
    print("                            └刚度┘ └mortar┘└特征┘ └─IPC barrier─┘")
    print("  本演示逐一拆解了 B(barrier)、γ(Hermite权重)、ε(自适应半径)、")
    print("  α(步长) 和 w_k(mortar积分权重) 五个要素。")
    print("=" * 70)
