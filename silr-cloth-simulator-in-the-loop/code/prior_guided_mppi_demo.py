"""
先验引导 MPPI 折叠精修 —— 论文核心机制的可运行概念实现
=========================================================

论文: Enabling Robust Cloth Manipulation via Inference-Time
      Simulator-in-the-Loop Refinement (arXiv:2606.24552)
      项目页: https://silr-cloth.github.io/

本 demo 用纯 Python 标准库(无 numpy/torch)复刻论文支柱 (iii)
"先验引导 MPPI"(§3.4,式 10–13)的控制流,并嵌入一个极简的
2D "布料"前向模型代替 FLASH,让你在没有 GPU/仿真器的情况下
直观看到:

  * 名义动作序列(离线策略先验 Ū_t)如何被在线 rollout 精修;
  * 为什么"先验引导"(锚定采样)比"无引导"(大范围盲采)更高效;
  * 滚动时域(receding-horizon)real-to-sim-to-real 闭环的形状。

核心对应关系:
  式10  U^(k) = Ū_t + ε^(k),  ε ~ N(0, Σ)     -> sample_candidates()
  式11  τ^(k) = F_{0:H}(ŝ_t, U^(k))            -> rollout()
  式12  w_k = softmax(-S(τ^(k)) / λ)           -> mppi_weights()
  式13  U*_t = Σ_k w_k U^(k)                    -> 加权合成
  a*_t = U*_t[0],执行首动作后重规划            -> control_loop()

注意: 这是教学骨架。真实系统里 F 是 FLASH 可变形仿真器(三角网格 +
摩擦接触),ŝ_t 来自 RGB-native real-to-sim(DINOv2 + canonical tokens)。
这里用一个"被抓点朝目标移动、布料其余点跟随"的玩具动力学代替,
重点在 MPPI 的采样-rollout-加权-精修控制流。

运行:  python3 prior_guided_mppi_demo.py
"""

import math
import random

RNG = random.Random(20260623)   # 固定种子,可复现


# =============================================================================
# 极简 2D "布料"前向模型 F(代替 FLASH)
# =============================================================================
# 状态: 抓取点位置 (x, y)。动作: 每步的 (dx, dy) 位移增量。
# "布料代价"用抓取点到目标折叠位置的距离 + 简单正则(工作空间/平滑)近似,
# 对应论文 S(τ) = 折叠目标 + 工作空间限制/桌面穿透/动作平滑。

WORKSPACE = (0.0, 10.0)       # 合法工作空间 [0,10]^2
TARGET = (8.0, 7.0)           # 目标折叠位置(布料角应到达处)


def step(state, action):
    """一步前向: F(x_t, a_t)。返回新状态(并夹在工作空间内)。"""
    x, y = state
    dx, dy = action
    nx = min(max(x + dx, WORKSPACE[0]), WORKSPACE[1])
    ny = min(max(y + dy, WORKSPACE[0]), WORKSPACE[1])
    return (nx, ny)


def rollout(state, action_seq):
    """式11: τ = F_{0:H}(s, U)。返回整条状态轨迹。"""
    traj = [state]
    s = state
    for a in action_seq:
        s = step(s, a)
        traj.append(s)
    return traj


def task_cost(traj, action_seq):
    """式11 的 S(τ): 折叠目标 + 正则项(工作空间穿透惩罚 + 动作平滑)。"""
    final = traj[-1]
    # 1) 折叠目标: 末态到目标的距离(主项)
    goal = math.dist(final, TARGET)
    # 2) 工作空间穿透惩罚: 轨迹是否贴边/越界(被夹说明想越界)
    pen = 0.0
    for (x, y) in traj:
        for v in (x, y):
            if v <= WORKSPACE[0] + 1e-6 or v >= WORKSPACE[1] - 1e-6:
                pen += 0.2
    # 3) 动作平滑: 相邻动作变化越大代价越高
    smooth = 0.0
    for i in range(1, len(action_seq)):
        smooth += math.dist(action_seq[i], action_seq[i - 1])
    return 3.0 * goal + pen + 0.1 * smooth


# =============================================================================
# 先验引导 MPPI(§3.4)
# =============================================================================
def sample_candidates(nominal, K, sigma):
    """式10: U^(k) = Ū + ε^(k), ε ~ N(0, σ²I)。锚定在名义序列附近。"""
    candidates = []
    for _ in range(K):
        cand = [(a[0] + RNG.gauss(0, sigma), a[1] + RNG.gauss(0, sigma))
                for a in nominal]
        candidates.append(cand)
    return candidates


def mppi_weights(costs, lam):
    """式12: w_k = exp(-S_k/λ) / Σ exp(-S_j/λ)。数值稳定版(减最小值)。"""
    cmin = min(costs)
    exps = [math.exp(-(c - cmin) / lam) for c in costs]
    z = sum(exps)
    return [e / z for e in exps]


def mppi_refine(state, nominal, K=64, sigma=0.6, lam=1.0):
    """一次 MPPI 精修: 采样 -> rollout -> 加权 -> 合成精修序列 U*。"""
    candidates = sample_candidates(nominal, K, sigma)
    costs = [task_cost(rollout(state, U), U) for U in candidates]   # 式11
    w = mppi_weights(costs, lam)                                    # 式12
    H = len(nominal)
    refined = []
    for h in range(H):                                             # 式13: Σ_k w_k U^(k)
        rx = sum(w[k] * candidates[k][h][0] for k in range(K))
        ry = sum(w[k] * candidates[k][h][1] for k in range(K))
        refined.append((rx, ry))
    best_cost = min(costs)
    return refined, best_cost


# =============================================================================
# 一个"离线策略"先验(刻意不完美,代表 IL/VLA 名义轨迹)
# =============================================================================
def base_policy(state, H):
    """离线蒸馏策略给出的名义序列 Ū_t。

    刻意做成"方向大致对但有系统偏差" —— 朝目标走但每步偏小且带固定偏移,
    模拟真实策略的累积误差。MPPI 的任务就是在线把它精修回来。
    """
    x, y = state
    dx = (TARGET[0] - x) / (H + 4)      # 偏小(走不到位)
    dy = (TARGET[1] - y) / (H + 4) - 0.15   # 带向下系统偏移(错误)
    return [(dx, dy) for _ in range(H)]


# =============================================================================
# 滚动时域闭环(real-to-sim-to-real 的骨架)
# =============================================================================
def control_loop(use_mppi=True, prior_guided=True, steps=12, H=6,
                 K=64, sigma=0.6, lam=1.0):
    """模拟逐步闭环。每步: 估计状态 -> 取先验 -> (MPPI 精修) -> 执行首动作。

    use_mppi=False     : 直接执行基础策略(对应 Table 3 的 'Base-policy' 消融)
    prior_guided=False : MPPI 不锚定先验(从零均值大范围盲采),对比采样效率
    """
    state = (1.0, 1.0)        # 布料角初始位置
    history = [state]
    for t in range(steps):
        # (ii) real-to-sim: 这里直接拿到真值状态(真实系统是 ŝ_t=f_θ(I_t))
        s_hat = state
        # 先验: 离线策略名义序列
        nominal = base_policy(s_hat, H)
        if use_mppi:
            anchor = nominal if prior_guided else [(0.0, 0.0)] * H
            refined, _ = mppi_refine(s_hat, anchor, K=K, sigma=sigma, lam=lam)
            action = refined[0]              # a*_t = U*_t[0]
        else:
            action = nominal[0]              # 直接执行基础策略首动作
        state = step(state, action)          # 真机执行(此处即仿真推进)
        history.append(state)
    final_err = math.dist(state, TARGET)
    return history, final_err


# =============================================================================
# 主程序: 三种配置对比
# =============================================================================
def ascii_plot(history, label):
    """把 2D 轨迹画成 ASCII 网格,T=目标, *=轨迹, S=起点。"""
    W = 11
    grid = [[' '] * W for _ in range(W)]
    def put(x, y, ch):
        gx = min(max(int(round(x)), 0), W - 1)
        gy = min(max(int(round(y)), 0), W - 1)
        grid[W - 1 - gy][gx] = ch   # y 向上
    for (x, y) in history:
        put(x, y, '*')
    put(*TARGET, 'T')
    put(history[0][0], history[0][1], 'S')
    print(f"  {label}")
    for row in grid:
        print("    |" + ''.join(row) + "|")
    print("    +" + "-" * W + "+   (S=起点 T=目标 *=轨迹)")


def main():
    print("=" * 70)
    print("先验引导 MPPI 折叠精修 demo —— SILR-Cloth (arXiv:2606.24552)")
    print("=" * 70)
    print(f"目标折叠位置 TARGET = {TARGET},  起点 = (1.0, 1.0)\n")

    # 配置 1: 仅基础策略(无精修)—— 对应 Table 3 'Base-policy' 消融
    h1, e1 = control_loop(use_mppi=False)
    # 配置 2: 先验引导 MPPI(完整方法)
    h2, e2 = control_loop(use_mppi=True, prior_guided=True)
    # 配置 3: 无引导 MPPI(不锚定先验,大范围盲采,同样预算 K)
    h3, e3 = control_loop(use_mppi=True, prior_guided=False, sigma=2.0)

    print("-" * 70)
    print("配置 1 · 仅基础策略(开环,不纠错)")
    ascii_plot(h1, f"末端到目标误差 = {e1:.3f}")
    print()
    print("-" * 70)
    print("配置 2 · 先验引导 MPPI(完整方法:锚定名义轨迹 + 物理 rollout 精修)")
    ascii_plot(h2, f"末端到目标误差 = {e2:.3f}")
    print()
    print("-" * 70)
    print("配置 3 · 无引导 MPPI(同样 K=64,但从零均值大范围盲采 σ=2.0)")
    ascii_plot(h3, f"末端到目标误差 = {e3:.3f}")

    print("\n" + "=" * 70)
    print("结果对比(末端到目标误差,越小越好):")
    print(f"  仅基础策略      : {e1:.3f}   <- 系统偏差未被纠正(开环累积误差)")
    print(f"  先验引导 MPPI   : {e2:.3f}   <- 锚定先验 + 物理精修,最优")
    print(f"  无引导 MPPI     : {e3:.3f}   <- 同预算盲采,高维下采样低效")
    print("=" * 70)
    print("要点:")
    print("  1. 基础策略(IL/VLA 先验)有系统偏差,开环执行无法自我纠错。")
    print("  2. 先验引导 MPPI 把策略输出当锚点,在其邻域内用物理 rollout 精修,")
    print("     以有限采样预算高效收敛 —— 对应论文式 10-13。")
    print("  3. 同样的采样预算下,不锚定先验的盲采在高维动作空间里效率低,")
    print("     这正是论文强调'先验引导'对布料高维非凸目标的必要性。")
    print("  4. 真实系统: F=FLASH 可变形仿真, 状态来自单 RGB 的 real-to-sim。")
    print("=" * 70)


if __name__ == "__main__":
    main()
