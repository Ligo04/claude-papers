"""
flow_steering_demo.py
=====================
SCORE 机制核心 demo：flow steering 如何"免费"实现支撑集约束。

我们训练一个真正的（迷你）1D 条件 flow matching 策略 π_base(a|o)，
让它学到一个"双模态"行为分布（一个快模式、一个慢模式）。
然后演示 SCORE 的关键操作：

    冻结 flow 网络，只在隐噪声 z 上"steer"，
    -> 无论怎么调 z，输出动作永远落在基座能生成的集合 A_base(o) 内（支撑集约束）
    -> steering 学会永远挑'快模式'（把好模式的概率拉满）

对比：如果我们像'无约束 RL'那样直接优化动作 a，就会跑到支撑集之外的
'漏洞动作'上（此处真机回报为 0）。

运行：
    python flow_steering_demo.py

依赖：numpy。全自包含，CPU 秒级可跑。用 numpy 手写 flow matching 训练，
不依赖 torch，方便任何环境直接运行、逐行读懂。
"""

from __future__ import annotations
import numpy as np

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. 目标行为分布：给定观测 o（这里固定 o=0 简化），动作是双模态：
#    快模式 a≈+1.0（真机回报高），慢模式 a≈-1.0（真机回报低）。
#    基座策略要学会复现这个双模态分布（模仿学习）。
# ---------------------------------------------------------------------------
FAST_MODE, SLOW_MODE = 1.0, -1.0


def sample_demo_actions(n):
    """真实演示数据：50% 快 + 50% 慢（含噪声）。基座会同时学到两个模式。"""
    which = rng.random(n) < 0.5
    a = np.where(which, FAST_MODE, SLOW_MODE) + rng.normal(0, 0.08, n)
    return a


def real_return(a):
    """真机回报地形：快模式高(1.0)，慢模式低(0.4)，其余(如支撑集外)→0。"""
    fast = np.exp(-0.5 * ((a - FAST_MODE) / 0.15) ** 2) * 1.0
    slow = np.exp(-0.5 * ((a - SLOW_MODE) / 0.15) ** 2) * 0.4
    return np.maximum(fast, slow)


# ---------------------------------------------------------------------------
# 2. 迷你 flow matching：学速度场 v_theta(x, tau)。
#    这里 o 固定，所以 v 只是 (x, tau) 的函数。我们用一个小的多项式基 +
#    线性权重来参数化 v，用 flow matching 目标 (Eq.4) 训练。
#    目标插值：x_tau = (1-tau)*x0 + tau*a，回归 (a - x0)。
# ---------------------------------------------------------------------------
def features(x, tau):
    """简单特征基 φ(x, tau)，让 v 有足够表达力拟合双模态输运。"""
    return np.stack([
        np.ones_like(x), x, x ** 2, x ** 3,
        tau, tau * x, tau * x ** 2, (1 - tau) * x,
    ], axis=-1)


def train_flow(n_iters=4000, batch=512, lr=0.05):
    """用最小二乘/梯度下降拟合线性速度场权重 w。"""
    w = np.zeros(features(np.zeros(1), np.zeros(1)).shape[-1])
    for it in range(n_iters):
        a = sample_demo_actions(batch)          # 目标动作
        x0 = rng.normal(0, 1, batch)            # 先验噪声 z~N(0,1)
        tau = rng.random(batch)                 # τ~U[0,1]
        x_tau = (1 - tau) * x0 + tau * a        # 插值点
        target = a - x0                         # flow matching 回归目标
        phi = features(x_tau, tau)              # 特征
        pred = phi @ w
        grad = phi.T @ (pred - target) / batch  # MSE 梯度
        w -= lr * grad
    return w


def flow_forward(w, z, steps=50):
    """
    π_base(o, z)：从 x0=z 积分 ODE dx/dτ = v(x,τ) 到 x1=a。
    这是'冻结的'基座前向过程——SCORE 训练中它权重不变。
    """
    x = np.array(z, dtype=float)
    dt = 1.0 / steps
    for i in range(steps):
        tau = np.full_like(x, i * dt)
        v = features(x, tau) @ w
        x = x + v * dt
    return x  # = 动作 a


# ---------------------------------------------------------------------------
# 3. 演示支撑集：扫大量 z，看看基座能生成的动作集合 A_base(o)。
# ---------------------------------------------------------------------------
def show_support(w):
    z = rng.normal(0, 1, 20000)
    a = flow_forward(w, z)
    print("\n[支撑集 A_base(o)] 基座能生成的动作范围：")
    print(f"    动作范围 [{a.min():.2f}, {a.max():.2f}]，"
          f"落在快模式(≈+1)附近比例 {np.mean(np.abs(a-FAST_MODE)<0.3):.1%}，"
          f"慢模式(≈-1)附近比例 {np.mean(np.abs(a-SLOW_MODE)<0.3):.1%}")
    print("    -> 只要只调 z，动作永远落在这个双模态集合内（这就是支撑集约束）")
    return a


# ---------------------------------------------------------------------------
# 4. SCORE 的 steering：冻结 flow，只在 z 上做'RL'（这里用简单的
#    分布搜索代替 PPO）。目标 = 最大化真机回报。
#    因为动作 = flow_forward(w, z)，steering 只能重选'哪些 z' ->
#    只能在支撑集内重分配概率 -> 学会永远挑快模式。
# ---------------------------------------------------------------------------
def score_steering(w, n=20000):
    """在 z 空间'steer'：偏好那些经 flow 映射后真机回报高的 z。"""
    z = rng.normal(0, 1, n)
    a = flow_forward(w, z)
    r = real_return(a)
    # 用回报做 softmax 重加权（模拟 RL 把概率压到高回报区）
    prob = np.exp(r / 0.05)
    prob /= prob.sum()
    idx = rng.choice(n, size=n, p=prob)
    z_steered, a_steered = z[idx], a[idx]
    return z_steered, a_steered


# ---------------------------------------------------------------------------
# 5. 对比'无约束 RL'：直接在动作空间优化，会跑到支撑集外的漏洞动作。
# ---------------------------------------------------------------------------
A_HACK = 2.5  # 支撑集外的漏洞动作：真机回报=0（real_return 在此≈0）


def unconstrained_rl():
    """直接优化动作 -> 若仿真在 a_hack 虚高，则收敛到 a_hack（真机=0）。"""
    return np.full(20000, A_HACK)


def main():
    print("=" * 70)
    print("Flow Steering demo：冻结 flow 基座，只在噪声 z 上 steer")
    print("=" * 70)

    print("\n[1] 训练迷你 flow matching 基座策略（学双模态：快 +1 / 慢 -1）...")
    w = train_flow()
    support = show_support(w)

    print("\n[2] 基座策略（未 steer）的平均真机回报：")
    base_r = real_return(support).mean()
    print(f"    平均真机回报 {base_r:.3f}（快慢各半，被慢模式拉低）")

    print("\n[3] SCORE steering（冻结 flow，只在 z 上偏好高回报）：")
    z_s, a_s = score_steering(w)
    score_r = real_return(a_s).mean()
    fast_frac = np.mean(np.abs(a_s - FAST_MODE) < 0.3)
    print(f"    平均真机回报 {score_r:.3f}，落在快模式比例 {fast_frac:.1%}")
    print(f"    动作范围 [{a_s.min():.2f}, {a_s.max():.2f}] "
          f"-> 仍在支撑集内（从不越界到 a_hack={A_HACK}）")

    print("\n[4] 对比 无约束 RL（直接优化动作，跑到支撑集外漏洞 a_hack）：")
    a_unc = unconstrained_rl()
    unc_r = real_return(a_unc).mean()
    print(f"    动作恒为 {A_HACK}（支撑集外），真机回报 {unc_r:.3f}（崩溃）")

    print("\n" + "=" * 70)
    print("结论对比：")
    print(f"    基座(未steer)     真机回报 {base_r:.3f}")
    print(f"    SCORE steering    真机回报 {score_r:.3f}  <-- 提升且从不越界")
    print(f"    无约束 RL         真机回报 {unc_r:.3f}  <-- 越界到漏洞，崩溃")
    print("=" * 70)
    print("★ 机制要点：steering 只改'用哪个 z'，动作恒 ∈ A_base(o)（支撑集约束由")
    print("  flow 前向结构自动保证）；RL 的自由度全部用于在支撑集内挑最优模式。")


if __name__ == "__main__":
    main()
