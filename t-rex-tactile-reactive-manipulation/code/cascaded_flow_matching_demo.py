# -*- coding: utf-8 -*-
"""
T-Rex —— 异步触觉反应式级联流匹配（最小 Demo）
===============================================

这是一个独立、仅依赖 NumPy 的教学版重实现，对应 T-Rex 的核心机制（4.2 节）：
单条流匹配去噪轨迹在固定时刻 tau_split 处“分裂”。一个“慢”动作专家每个 chunk
去噪上段（tau: 1 -> tau_split）一次；一个“快”触觉专家在 chunk 内多次去噪下段
（tau_split -> 0），每次都以新鲜触觉信号为条件。

这个 Demo 想说明什么
--------------------
论文的贡献不是新损失——而是*在哪里分裂*去噪、以及*每一段看到什么*。为了不用机器人、
也不用 7B VLA 就把这点讲清楚，我们用一个二维玩具“动作”（想象成一个平面末端执行器增量）。
动作专家被告知（略带噪声的）目标；触觉专家被给予一个实时接触约束（一堵动作不可越过的墙），
而这堵墙只有通过触觉才能知道。

你应当“看到”的要点：
  * 单独的慢规划朝目标走，但*越过*了接触墙。
  * 快触觉精修——只作用于去噪的*最后一段*——把动作轻推到尊重墙，*而不重新规划*目标。
  * 两个专家回归*同一个*流目标 v* = eps - x0（式 3），所以触觉专家是延续轨迹而非与之打架。

运行：
    python cascaded_flow_matching_demo.py
"""

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 流匹配基础（论文式 1 与式 3）
# ---------------------------------------------------------------------------
# 约定（与论文一致）：tau = 1 是纯噪声，tau = 0 是干净动作。
#   x_tau = (1 - tau) * x0 + tau * eps          （线性插值）
#   v*    = eps - x0                            （常速度目标）
# 去噪 = 积分 dx/d(tau) = -v*（我们把 tau 向 0 递减），即
#   x(tau - dtau) = x(tau) - v_pred * dtau
#
# 这里的“专家”只是一个在给定 (x_tau, tau, 上下文) 下对 v* 的估计器。
# 真实的 T-Rex 专家是 Transformer；我们的是该玩具任务的*解析*最优场，
# 这样 Demo 既无依赖、行为又正确。


def slow_action_expert(x_tau, tau, goal, noisy_eps):
    """低频专家：知道目标（粗规划），但不知道接触墙。

    它回归 v* = eps - x0，其中 x0 是它想要的动作（朝目标走）。
    """
    x0_plan = goal                      # 慢专家承诺的动作目标
    return noisy_eps - x0_plan          # 朝规划目标的 v*


def fast_tactile_expert(x_tau, tau, goal, noisy_eps, wall_x):
    """高频专家：额外“感觉”到一堵位于 x = wall_x 的接触墙。

    它精修*同一条*轨迹，但朝向一个被投影到尊重墙的目标（动作的 x 坐标不得超过 wall_x）。
    这就是“残差精修”——它只在响应慢专家从未拥有的信号（触觉）时，最小幅度地改变目标。
    """
    x0_refined = goal.copy()
    if x0_refined[0] > wall_x:           # 运行时通过触觉发现的接触约束
        x0_refined[0] = wall_x           # 投影到接触面上
    return noisy_eps - x0_refined


def euler_denoise(v_fn, x_start, tau_hi, tau_lo, k_steps):
    """把速度场从 tau_hi 积分到 tau_lo，分 k_steps 步。"""
    x = x_start.copy()
    taus = np.linspace(tau_hi, tau_lo, k_steps + 1)
    for i in range(k_steps):
        tau = taus[i]
        dtau = taus[i + 1] - taus[i]     # 负值（tau 递减）
        v = v_fn(x, tau)
        x = x + v * dtau                 # x(tau+dtau) = x + v*dtau，dtau<0
    return x


# ---------------------------------------------------------------------------
# 级联推理（式 4 与式 5）
# ---------------------------------------------------------------------------
def cascaded_inference(goal, wall_x, tau_split=0.4, k_slow=6, k_fast=4,
                       use_tactile=True, seed=None):
    """一个动作 chunk 的级联去噪。返回 (慢流中间态, 最终动作)。"""
    r = np.random.default_rng(seed)
    eps = r.standard_normal(2)           # x_1 = eps（纯噪声起点）
    x1 = eps

    # ---- 慢流：动作专家，tau 1 -> tau_split，跑一次 ----------
    slow_v = lambda x, tau: slow_action_expert(x, tau, goal, eps)
    x_split = euler_denoise(slow_v, x1, 1.0, tau_split, k_slow)
    # >>> 此处 T-Rex 缓存 KV_{tau_split}；我们的“缓存”就是 (eps, goal) <<<

    # ---- 快流：触觉专家，tau_split -> 0，每个子 chunk 都跑 --
    if use_tactile:
        fast_v = lambda x, tau: fast_tactile_expert(x, tau, goal, eps, wall_x)
    else:
        fast_v = slow_v                  # 消融：“去掉触觉”-> 动作专家自己收尾
    x_final = euler_denoise(fast_v, x_split, tau_split, 0.0, k_fast)
    return x_split, x_final


def main():
    print(__doc__)
    goal = np.array([1.40, 0.80])        # 视觉*认为*动作该落在哪里
    wall_x = 1.00                        # 只有触觉能感知的接触面

    print(f"目标（视觉规划）           : {goal}")
    print(f"接触墙位于 x =               {wall_x}（动作 x 不得超过它）")
    print(f"tau_split = 0.4 | K_slow = 6 | K_fast = 4（论文默认）\n")

    # 在多个噪声种子上平均，说明不是偶然。
    viol_no_tac, viol_tac, err_tac = [], [], []
    for s in range(2000):
        _, final_no_tac = cascaded_inference(goal, wall_x, use_tactile=False, seed=s)
        x_split, final_tac = cascaded_inference(goal, wall_x, use_tactile=True, seed=s)
        viol_no_tac.append(max(0.0, final_no_tac[0] - wall_x))
        viol_tac.append(max(0.0, final_tac[0] - wall_x))
        err_tac.append(abs(final_tac[1] - goal[1]))   # 离轴(y)仍跟踪目标

    print("单个 chunk 示例（seed=0）：")
    xs, fnt = cascaded_inference(goal, wall_x, use_tactile=False, seed=0)
    _,  ft  = cascaded_inference(goal, wall_x, use_tactile=True,  seed=0)
    print(f"  中间态 x_split（慢流后）        : {xs.round(3)}")
    print(f"  最终（无触觉，仅动作专家）       : {fnt.round(3)}  "
          f"-> 越墙 {max(0.0, fnt[0]-wall_x):.3f}")
    print(f"  最终（带触觉精修）               : {ft.round(3)}  "
          f"-> 越墙 {max(0.0, ft[0]-wall_x):.3f}\n")

    mv_no, mv_yes = np.mean(viol_no_tac), np.mean(viol_tac)
    print("在 2000 个噪声种子上平均：")
    print(f"  平均越墙  无触觉 : {mv_no:.3f}")
    print(f"  平均越墙  有触觉 : {mv_yes:.3f}   "
          f"<-- 触觉精修使越墙减少了 {(1 - mv_yes/mv_no)*100:.0f}%")
    print(f"  有触觉时的离轴(y)误差: {np.mean(err_tac):.3f}   "
          "<-- 目标仍被跟踪：触觉是*精修*，不是重新规划")

    print("\n注意这是*部分*修正：触觉专家只拥有较低的 tau 段 [0, 0.4]，")
    print("所以它最多把动作朝尊重接触的目标混合 40% —— 因此越墙从 0.40 降到")
    print("0.24，而*非*降到 0。这正是论文消融的 tau_split 权衡（图 4）：")
    print("tau_split 太*小*则触觉专家发挥空间太小；太*大*则视觉运动规划被饿死。")
    print("\n要点：触觉专家只作用于去噪的最后一段（tau: 0.4 -> 0）。它轻推违反")
    print("接触的那个维度，同时保持其余已规划动作不变 —— 这正是 T-Rex 的")
    print("“触觉是对已定规划的残差精修”（4.2 节）。")


if __name__ == "__main__":
    main()
