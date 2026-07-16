"""
support_vs_distributional_demo.py
===================================
SCORE 核心概念可运行 demo：支撑集约束 vs 分布约束 vs 无约束。

我们复刻论文 Fig.2 的"灯泡"玩具例子的核心直觉，但用一个更易可视化、
可数值验证的一维"选模式"问题来揭示三种范式的本质差异：

设定：
- 基座策略 π_base 对同一观测能生成一个"多模态"动作分布（混合高斯）。
  其中某些模式是"好模式"（快、可迁移），某些是"坏模式"（慢、次优）。
- 仿真里存在一个"漏洞动作" a_hack，它在仿真拿高分但真机为 0（reality gap）。
- 目标：找一个提升后的策略，既在真机拿高分，又可迁移。

三种范式：
1. 无约束 RL        -> 直奔 a_hack，仿真满分、真机崩溃（reward hacking）。
2. 分布约束 RL      -> 必须整体贴近 π_base 的"分布形状"，被坏模式拖住。
3. 支撑集约束 RL    -> 只要求动作落在 π_base 的"可能集合"内，
                       于是把概率全压到"好模式"，忽略坏模式，避开 a_hack。

运行：
    python support_vs_distributional_demo.py            # 打印数值对比
    python support_vs_distributional_demo.py --plot     # 额外存一张对比图（需 matplotlib）

依赖：numpy（可选 matplotlib）。全部自包含，无需真实机器人/仿真。
"""

from __future__ import annotations
import argparse
import numpy as np

rng = np.random.default_rng(0)  # 固定种子，结果可复现


# ---------------------------------------------------------------------------
# 1. 定义"基座策略"：给定观测，动作是一个多模态分布（混合高斯）
#    每个模式带一个真机回报 r_real 和一个仿真回报 r_sim。
#    "好模式"= 真机回报高；"坏模式"= 真机回报低（慢/次优）。
# ---------------------------------------------------------------------------
class BasePolicy:
    """真实数据训出的多模态基座策略（1D 动作）。"""

    def __init__(self):
        # 每个模式: (中心动作 mu, 标准差 sigma, 真机回报, 仿真回报)
        # 说明:
        #   fast_good : 又快又准，真机高回报 —— 我们希望被放大的模式
        #   slow_ok   : 能成功但慢，真机中等回报 —— 分布约束会被它拖住
        #   sloppy_bad: 马虎易失败，真机低回报 —— 应被抑制
        self.modes = [
            # name         mu     sigma  r_real  r_sim
            ("fast_good",  1.0,   0.05,  1.00,   1.00),
            ("slow_ok",   -0.5,   0.08,  0.55,   0.55),
            ("sloppy_bad", 0.2,   0.15,  0.15,   0.15),
        ]
        # 基座对三种模式的先验混合权重（数据里慢/糙动作其实很多）
        self.weights = np.array([0.34, 0.33, 0.33])

    def support_actions(self, n=20000):
        """采样近似出'支撑集'（基座可能生成的动作集合 A_base(o)）。"""
        idx = rng.choice(len(self.modes), size=n, p=self.weights)
        acts = np.array(
            [rng.normal(self.modes[i][1], self.modes[i][2]) for i in idx]
        )
        return acts

    def real_return(self, action):
        """真机回报：动作离哪个模式近就拿那个模式的真机回报（软分配）。"""
        return self._blended_return(action, key=3 - 1)  # r_real 在 index 3? 见下

    # 为清晰起见，显式写两个回报函数（避免魔法索引）
    def _mode_return(self, action, real: bool):
        # 用高斯核把动作软分配到各模式，回报按核加权
        num, den = 0.0, 0.0
        for name, mu, sigma, r_real, r_sim in self.modes:
            w = np.exp(-0.5 * ((action - mu) / sigma) ** 2)
            num += w * (r_real if real else r_sim)
            den += w
        return num / (den + 1e-9)

    def real_ret(self, action):
        return self._mode_return(action, real=True)

    def sim_ret(self, action):
        return self._mode_return(action, real=False)


# ---------------------------------------------------------------------------
# 2. 仿真的"漏洞动作"：真机回报为 0，但无约束 RL 在仿真里能刷到超高分。
#    代表 reality gap / reward hacking（如论文里"用手砸桌子夹球"）。
# ---------------------------------------------------------------------------
A_HACK = 3.0          # 远离基座任何模式的一个动作
HACK_SIM_RETURN = 2.0  # 仿真给它超高分
HACK_REAL_RETURN = 0.0  # 真机直接失败


def sim_return_with_hack(action):
    """仿真回报 = 正常回报，但在 a_hack 附近额外给巨额（虚假）奖励。"""
    base = BASE.sim_ret(action)
    hack_bonus = HACK_SIM_RETURN * np.exp(-0.5 * ((action - A_HACK) / 0.1) ** 2)
    return base + hack_bonus


def real_return(action):
    """真机回报 = 基座真机回报；a_hack 附近为 0（不迁移）。"""
    r = BASE.real_ret(action)
    # a_hack 附近真机塌陷为 0
    kill = np.exp(-0.5 * ((action - A_HACK) / 0.3) ** 2)
    return r * (1 - kill) + HACK_REAL_RETURN * kill


BASE = BasePolicy()


# ---------------------------------------------------------------------------
# 3. 三种"提升后策略"。每种都返回一个动作分布（这里用大量采样近似），
#    我们比较它们的 仿真回报 与 真机回报。
# ---------------------------------------------------------------------------
def policy_unconstrained(n=20000):
    """无约束 RL：只 max 仿真回报 -> 收敛到 a_hack。"""
    # 直接选仿真回报最高的动作（这里就是 a_hack）
    return np.full(n, A_HACK)


def policy_distributional(n=20000, eps=0.35):
    """
    分布约束 RL：max 仿真回报 s.t. 分布接近 π_base。
    我们用一个简化实现：在基座采样的动作上做重要性重加权，
    但被一个 KL-like 惩罚拉住，不能过度偏离基座的模式混合。
    结果：它会略微上调 fast_good 的权重，但因为要"保持分布形状"，
    仍保留大量 slow_ok / sloppy_bad 概率质量 -> 被坏模式拖住。
    eps 越小约束越紧（越像 base），越大越放开（越可能碰 a_hack）。
    """
    support = BASE.support_actions(n)
    # 允许把一部分概率质量挪向 a_hack（约束松时），比例受 eps 控制
    move_to_hack = np.clip(eps - 0.1, 0.0, 1.0)  # eps 越大越会去碰漏洞
    n_hack = int(move_to_hack * n * 0.5)
    if n_hack > 0:
        support[:n_hack] = A_HACK
    # 关键：分布约束"不能只保留好模式"，必须维持接近 base 的模式比例
    # 这里我们不重加权模式 -> 体现"被 base 分布形状锁死"
    return support


def policy_support_constrained(n=20000):
    """
    支撑集约束 RL：动作必须 ∈ A_base(o)（基座可能生成的集合），
    但概率可以任意重新分配 -> 全压到真机回报最高的 fast_good 模式。
    这正是 flow steering 做的事：只在噪声空间挑'哪个模式'，
    永远不会跳到 a_hack（因为 a_hack ∉ 支撑集）。
    """
    support = BASE.support_actions(n * 3)          # 先拿到支撑集样本
    real_r = np.array([real_return(a) for a in support])
    # "重分配概率"：偏向真机回报高的支撑集动作（softmax 选择）
    prob = np.exp(real_r / 0.1)
    prob /= prob.sum()
    idx = rng.choice(len(support), size=n, p=prob)
    return support[idx]


# ---------------------------------------------------------------------------
# 4. 评估：对每种策略，报告 平均仿真回报 / 平均真机回报 / 是否碰漏洞。
# ---------------------------------------------------------------------------
def evaluate(name, actions):
    sim = np.mean([sim_return_with_hack(a) for a in actions])
    real = np.mean([real_return(a) for a in actions])
    hack_frac = np.mean(np.abs(actions - A_HACK) < 0.3)  # 落在漏洞附近的比例
    print(
        f"{name:<26s} | 仿真回报 {sim:5.2f} | 真机回报 {real:5.2f} "
        f"| 碰漏洞比例 {hack_frac:5.1%}"
    )
    return sim, real


def main(plot: bool):
    print("=" * 72)
    print("SCORE 核心概念 demo：三种范式在'仿真高分 vs 真机可迁移'上的差异")
    print("=" * 72)
    print(f"{'策略':<26s} | {'仿真':>8s} | {'真机':>8s} | 漏洞")
    print("-" * 72)

    acts_unc = policy_unconstrained()
    acts_dist_loose = policy_distributional(eps=0.45)   # 松约束
    acts_dist_tight = policy_distributional(eps=0.15)   # 紧约束
    acts_supp = policy_support_constrained()

    evaluate("1. 无约束 RL", acts_unc)
    evaluate("2a. 分布约束(松 eps=0.45)", acts_dist_loose)
    evaluate("2b. 分布约束(紧 eps=0.15)", acts_dist_tight)
    evaluate("3. 支撑集约束 (SCORE)", acts_supp)

    print("-" * 72)
    print("解读：")
    print("  · 无约束      -> 仿真满分但真机=0（钻漏洞 a_hack，reward hacking）")
    print("  · 分布约束(松)-> 提升一点但开始碰漏洞，真机下滑（不迁移）")
    print("  · 分布约束(紧)-> 不碰漏洞可迁移，但被 slow/sloppy 模式拖住（不提升）")
    print("  · 支撑集约束  -> 概率全压到 fast_good，既高真机回报又零漏洞（两全）")
    print("=" * 72)
    print("★ 关键：支撑集约束把'用哪些动作'(集合) 和 '概率怎么分'(分布) 解耦，")
    print("  在'基座可生成集合'内自由重分配概率 -> 提升与迁移同时拿下。")

    if plot:
        _plot(acts_unc, acts_dist_tight, acts_supp)


def _plot(acts_unc, acts_dist, acts_supp):
    """可选：画出三种策略的动作分布 + 真机回报曲线。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip plot] 未安装 matplotlib")
        return

    xs = np.linspace(-1.2, 3.5, 400)
    real_curve = np.array([real_return(x) for x in xs])
    sim_curve = np.array([sim_return_with_hack(x) for x in xs])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(xs, real_curve, label="真机回报 real", lw=2)
    axes[0].plot(xs, sim_curve, label="仿真回报 sim(含漏洞)", lw=2, ls="--")
    axes[0].axvline(A_HACK, color="red", ls=":", label="漏洞动作 a_hack")
    axes[0].set_ylabel("回报")
    axes[0].legend()
    axes[0].set_title("回报地形：仿真在 a_hack 处虚高，真机在此=0")

    bins = np.linspace(-1.2, 3.5, 80)
    axes[1].hist(acts_unc, bins=bins, alpha=0.5, label="无约束", density=True)
    axes[1].hist(acts_dist, bins=bins, alpha=0.5, label="分布约束(紧)", density=True)
    axes[1].hist(acts_supp, bins=bins, alpha=0.5, label="支撑集(SCORE)", density=True)
    axes[1].axvline(A_HACK, color="red", ls=":")
    axes[1].set_xlabel("动作 a")
    axes[1].set_ylabel("策略动作密度")
    axes[1].legend()
    axes[1].set_title("提升后策略的动作分布：SCORE 全压 fast_good，从不碰 a_hack")

    fig.tight_layout()
    out = "support_vs_distributional.png"
    fig.savefig(out, dpi=120)
    print(f"[plot] 已保存 {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="额外保存对比图")
    main(ap.parse_args().plot)
