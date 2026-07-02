#!/usr/bin/env python3
"""
multitask_balanced_rl_demo.py
================================================================================
Embodied-R1.5 的核心技术贡献之一：**多任务平衡 RL 配方**的优势归一化（论文 §4.2.1）。

本 demo 直观对比三种优势(advantage)估计，重现论文的两个设计决策：

  (A) Vanilla GRPO   Â = (R - μ_group) / (σ_group + ε)
  (B) 朴素全局归一化  Â = (R - μ_batch) / (σ_batch + ε)
  (C) 本文配方 ★      Â = (R - μ_group) / (σ_batch + ε)        ← 论文式

论文原话：
  "The group-level mean preserves intra-group relative ordering (which rollout
   is better), while the batch-level std unifies gradient magnitudes across
   tasks, without requiring task labels or historical state."

两个关键问题，本 demo 各用一段验证：

  问题①（分母：为何用 σ_batch 而非 σ_group？）
    Vanilla GRPO 每组都除以"自己组的 σ_group"，于是**任何组都被强制拉成单位方差** ——
    无论该组是"中等难度(高方差=学习信号最强)"还是"太易/太难(低方差=信号很弱)"，
    都贡献**等量**梯度。结果：中等难度样本被**相对欠优化**。
    本文改用**单一全局 σ_batch** → 各组梯度量级 ∝ 其真实奖励方差 →
    中等难度(高方差)组自然主导梯度，太易/太难/退化组被自然降权。

  问题②（分子：为何用 μ_group 而非 μ_batch？）
    用 μ_group 把每个 query 的 rollout **组内零中心化** → 只奖励"相对同组更好的"rollout。
    若用 μ_batch(朴素全局)，来自"容易 query"的 rollout 会因绝对奖励高而**整组拿正优势**，
    哪怕它在组内其实更差 —— 这会把策略推向"挑容易的 query"而非"给更好的回答"。

依赖：仅 Python 标准库。运行：python multitask_balanced_rl_demo.py
================================================================================
"""

import random
import statistics
from dataclasses import dataclass, field

random.seed(42)  # 固定种子，输出可复现
EPS = 1e-6
N_ROLLOUTS = 8   # 论文设置：每个 prompt 采 n=8 个并行 rollout


# ---------------------------------------------------------------------------
# 1. 构造批次：按"难度"分类的 group。难度 ≈ 组内奖励方差。
#    中等难度 = 高方差(有成有败) = 学习信号最强；太易/太难 = 低方差 = 信号弱。
# ---------------------------------------------------------------------------
@dataclass
class DiffClass:
    name: str
    reward_mean: float
    reward_std: float       # ← 组内方差，决定"学习信号"强弱
    n_queries: int
    desc: str


DIFFICULTIES = [
    DiffClass("易(low-var)",  0.90, 0.04, 3, "rollout 几乎都成功 → 方差小 → 学习信号弱"),
    DiffClass("中(high-var)", 0.50, 0.27, 4, "有成有败 → 方差大 → 学习信号最强 ★"),
    DiffClass("难(low-var)",  0.12, 0.04, 3, "rollout 几乎都失败 → 方差小 → 学习信号弱"),
]


@dataclass
class Group:
    diff: str
    rewards: list
    advantages: dict = field(default_factory=dict)


def sample_group(d: DiffClass) -> Group:
    rs = [min(1.0, max(0.0, random.gauss(d.reward_mean, d.reward_std)))
          for _ in range(N_ROLLOUTS)]
    return Group(diff=d.name, rewards=rs)


def build_batch():
    batch = [sample_group(d) for d in DIFFICULTIES for _ in range(d.n_queries)]
    # 注入一个"退化组"：rollout 奖励完全相同 → 零梯度（演示动态过滤）
    batch.append(Group(diff="退化(degenerate)", rewards=[0.7] * N_ROLLOUTS))
    return batch


# ---------------------------------------------------------------------------
# 2. 三种优势估计
# ---------------------------------------------------------------------------
def pstd(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def compute_all(batch):
    all_r = [r for g in batch for r in g.rewards]
    mu_batch, sigma_batch = statistics.fmean(all_r), pstd(all_r)
    for g in batch:
        mu_g, sigma_g = statistics.fmean(g.rewards), pstd(g.rewards)
        g.advantages["A_vanilla"] = [(r - mu_g) / (sigma_g + EPS) for r in g.rewards]
        g.advantages["B_naive"]   = [(r - mu_batch) / (sigma_batch + EPS) for r in g.rewards]
        g.advantages["C_paper"]   = [(r - mu_g) / (sigma_batch + EPS) for r in g.rewards]
    return mu_batch, sigma_batch


# ---------------------------------------------------------------------------
# 3. 分析工具
# ---------------------------------------------------------------------------
def mean_abs(xs):
    return statistics.fmean(abs(x) for x in xs)


def is_degenerate(g):
    return pstd(g.rewards) < 1e-9


def grad_magnitude_by_difficulty(batch, method):
    """按难度聚合 平均|advantage| ——代理'该难度对梯度的贡献量级'。"""
    agg = {}
    for g in batch:
        if is_degenerate(g):
            continue
        agg.setdefault(g.diff, []).append(mean_abs(g.advantages[method]))
    return {k: statistics.fmean(v) for k, v in agg.items()}


def within_group_centered(batch, method):
    """组内是否零中心化（advantage 组内均值≈0）。这是 GRPO 正确 baseline 的标志。"""
    for g in batch:
        if is_degenerate(g):
            continue
        if abs(statistics.fmean(g.advantages[method])) > 1e-6:
            return False
    return True


def bar(v, scale, width=34):
    n = int(min(width, (v / scale) * width)) if scale > 0 else 0
    return "█" * n + "·" * (width - n)


# ---------------------------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------------------------
def main():
    batch = build_batch()
    mu_batch, sigma_batch = compute_all(batch)

    print("=" * 80)
    print(" Embodied-R1.5 · 多任务平衡 RL 配方：优势归一化 demo")
    print("=" * 80)
    print(f" 批次：{len(batch)} 组（每组 n={N_ROLLOUTS}），按难度分类 + 1 退化组")
    print(f" 全局批统计： μ_batch={mu_batch:.3f}   σ_batch={sigma_batch:.3f}")
    for d in DIFFICULTIES:
        print(f"   - 难度[{d.name:<11}] 奖励≈{d.reward_mean:.2f}±{d.reward_std:.2f} | {d.desc}")

    methods = [
        ("A_vanilla", "Vanilla GRPO    Â=(R-μ_group)/(σ_group+ε)"),
        ("B_naive",   "朴素全局        Â=(R-μ_batch)/(σ_batch+ε)"),
        ("C_paper",   "本文配方 ★      Â=(R-μ_group)/(σ_batch+ε)"),
    ]

    # ---- 问题①：分母 σ_group vs σ_batch ----
    print("\n" + "-" * 80)
    print(" 问题①  分母：各难度对梯度的贡献量级（平均 |advantage|）")
    print("   期望：中等难度(高方差,信号最强)应当**主导**梯度，太易/太难应被降权")
    print("-" * 80)
    for key, label in methods:
        if key == "B_naive":
            continue  # 分母问题只对比 A 与 C
        mag = grad_magnitude_by_difficulty(batch, key)
        scale = max(mag.values())
        print(f"\n ▶ {label}")
        for d in DIFFICULTIES:
            v = mag[d.name]
            print(f"     难度[{d.name:<11}] |{bar(v, scale)}| {v:5.2f}")
        ratio = mag["中(high-var)"] / mag["易(low-var)"]
        print(f"     → 中/易 梯度量级比 = {ratio:4.2f}  "
              f"({'✅ 中等难度主导(信号proportional)' if ratio > 1.5 else '❌ 被拉平,中等难度未被突出'})")

    print("\n   解读：Vanilla GRPO 把**每组**都除以自己的 σ_group → 强制单位方差 →")
    print("        易/中/难贡献被**拉平**(比≈1)，中等难度(信号最强)没被突出。")
    print("        本文用单一 σ_batch → 量级 ∝ 真实方差 → 中等难度自然主导。✅")

    # ---- 问题②：分子 μ_group vs μ_batch（组内中心化）----
    print("\n" + "-" * 80)
    print(" 问题②  分子：组内是否零中心化（正确的 per-query baseline）")
    print("   期望：组内 advantage 均值≈0 → 只奖励'相对同组更好'的 rollout")
    print("-" * 80)
    for key, label in methods:
        centered = within_group_centered(batch, key)
        # 找一个"易"组展示其组内 advantage 均值，揭示 B 的偏置
        easy = next(g for g in batch if g.diff == "易(low-var)")
        m = statistics.fmean(easy.advantages[key])
        print(f"   {label[:16]:<16} 组内零中心: {'✅' if centered else '❌'}  "
              f"| 某'易'组的组内 adv 均值={m:+.3f}")
    print("\n   解读：朴素全局(B)用 μ_batch 做 baseline → '易' query 的 rollout 整组拿**正**优势")
    print("        (均值>0)，哪怕组内更差也被推高 → 策略被推向'挑容易的 query'。❌")
    print("        本文用 μ_group → 组内零中心 → 只奖励组内相对更优者。✅")

    # ---- 动态过滤 ----
    print("\n" + "-" * 80)
    print(" 附：动态过滤（退化组：组内奖励全相同 → 零梯度 → 应屏蔽）")
    print("-" * 80)
    for i, g in enumerate(batch):
        if is_degenerate(g):
            adv = g.advantages["C_paper"]
            print(f"   group #{i} [{g.diff}] 奖励全={g.rewards[0]} → "
                  f"advantage={[round(a,3) for a in adv[:3]]}... 全0 → 本文动态过滤屏蔽该组")

    print("\n" + "=" * 80)
    print(" 总结：本文配方 = μ_group(组内零中心,保 GRPO 核心) + σ_batch(单一全局尺度,")
    print("       让梯度量级∝真实学习信号) + 难度筛选 + 动态过滤。无需任务标签/历史状态。")
    print("=" * 80)


if __name__ == "__main__":
    main()
