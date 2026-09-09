#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capability_dimension_diagnosis_demo.py
=========================================

【教学演示，非官方代码】
本脚本演示 RoboDojo 论文 (arXiv:2607.04434v3) 的评测哲学——
「能力维度诊断 (capability dimension diagnosis)」——而不是官方评测代码。

论文原文事实（引用自 FACTS.md，真实榜单数字，不得编造）：
    - 仿真侧 42 任务分成 5 个能力维度：
        Generalization 泛化 (12 任务)
        Memory 记忆 (6 任务)
        Long-Horizon 长程 (8 任务)
        Precision 精度 (8 任务)
        Open 开放语义 (8 任务)
    - 总分 = 5 个维度分数的【均值】，而不是 42 个任务的加权平均，
      目的是防止任务数最多的维度（Generalization, 12 个任务）主导总分。
    - Table 1 Average 榜单前 13 名 + Human Expert (Teleop) 参考。
    - 各维度榜首（策略、score/SR）+ Human 在该维度的 score/SR：
        Generalization: Spatial Forcing 14.12/9.33%  (Human 90.05/87.83%)
        Precision:      X-VLA          18.32/12.00%  (Human 68.06/64.00%)
        Long-Horizon:   Hy-Embodied    25.74/14.92%  (Human 83.63/74.25%)
        Memory:         Hy-Embodied    13.37/12.11%  (Human 75.25/74.33%)
        Open:           π0.5            1.98/1.67%   (Human 85.13/79.75%)

本脚本的「为什么」：
    1) 「5 维度均值」与「42 任务加权平均」是两种不同的聚合方式，
       在任务数不均（12/6/8/8/8）时会给出不同排名结论。本脚本用一个
       明确标注为【教学构造示例】的假设策略，量化展示两者的差异，
       以及为什么 RoboDojo 选择"均值"而不是"加权平均"。
       （注意：这一节的策略分数是为了讲清楚数学差异而构造的示例数字，
       不是真实榜单数据；真实榜单数据只标注在下面第 2)、3) 部分。）
    2) 用真实的「各维度榜首 vs Human」数据画图，可以看到：
       - Average 总榜榜首 (Hy-Embodied) 其实只在 5 个维度里的 2 个
         (Long-Horizon、Memory) 拿到榜首，在另外 3 个维度上并不是最强
         —— 这正是"总分掩盖维度差异"的证据，也是维度诊断的价值所在。
       - Open 维度榜首成绩 (1.98) 与 Human (85.13) 差距最悬殊，
         印证论文 Finding F6："开放语义操作基本未解"。
       - Precision 维度即使是榜首 (18.32) 也远低于 Human (68.06)，
         印证论文 Finding F4："精度是主要瓶颈"。

运行：
    python3 capability_dimension_diagnosis_demo.py
"""

from __future__ import annotations

import logging

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _HAS_MPL = True
except ImportError:  # pragma: no cover - 环境降级路径
    _HAS_MPL = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("capability_dimension_diagnosis_demo")

# ---------------------------------------------------------------------------
# 真实数据区（来自 FACTS.md，2026-07-03 快照，禁止修改数值）
# ---------------------------------------------------------------------------

# 5 个能力维度的真实任务数（来自论文 3.1 节），总和为 42
DIMENSION_TASK_COUNTS: dict[str, int] = {
    "Generalization": 12,
    "Memory": 6,
    "Long-Horizon": 8,
    "Precision": 8,
    "Open": 8,
}
assert sum(DIMENSION_TASK_COUNTS.values()) == 42, "任务数总和必须等于论文的 42 个任务"

# Table 1：Average 榜单（score / success rate%），真实数字，含 Human 参考
TABLE1_AVERAGE_LEADERBOARD: list[tuple[str, float, float]] = [
    ("Hy-Embodied-0.5-VLA", 13.07, 8.80),
    ("Spatial Forcing", 12.38, 8.04),
    ("π0.5", 11.41, 6.91),
    ("X-VLA", 10.13, 6.52),
    ("X-WAM", 7.69, 3.83),
    ("Xiaomi-Robotics-0", 6.93, 4.18),
    ("StarVLA-α", 6.40, 3.24),
    ("GigaWorld-Policy", 6.20, 3.27),
    ("GalaxeaVLA (G0)", 5.82, 2.96),
    ("LingBot-VLA", 5.50, 2.96),
    ("EventVLA", 4.97, 2.81),
    ("AHA-WAM", 4.82, 2.39),
    ("ABot-M0", 3.67, 1.73),
]
HUMAN_AVERAGE = ("Human Expert (Teleop)", 80.42, 76.03)

# 各维度榜首（策略名, score, SR%, Human score, Human SR%），真实数字
DIMENSION_CHAMPIONS: dict[str, dict] = {
    "Generalization": dict(policy="Spatial Forcing", score=14.12, sr=9.33, human_score=90.05, human_sr=87.83),
    "Precision": dict(policy="X-VLA", score=18.32, sr=12.00, human_score=68.06, human_sr=64.00),
    "Long-Horizon": dict(policy="Hy-Embodied-0.5-VLA", score=25.74, sr=14.92, human_score=83.63, human_sr=74.25),
    "Memory": dict(policy="Hy-Embodied-0.5-VLA", score=13.37, sr=12.11, human_score=75.25, human_sr=74.33),
    "Open": dict(policy="π0.5", score=1.98, sr=1.67, human_score=85.13, human_sr=79.75),
}

DIMENSION_ORDER = ["Generalization", "Memory", "Long-Horizon", "Precision", "Open"]


# ---------------------------------------------------------------------------
# 第 1 部分：均值 vs 加权平均 —— 用【教学构造示例】说明聚合方式的差异
# ---------------------------------------------------------------------------

def demo_mean_vs_task_weighted_average() -> None:
    """构造一个假设策略的 5 维度分数，量化对比两种聚合方式的差异。

    真实榜单里我们只公开了「各维度榜首」这一份数据，并没有同一个策略
    在全部 5 个维度上的完整分数矩阵，所以这里无法用全部真实数字复现
    "总分计算过程"；但论文里明确写了计算规则："总分 = 5 个维度的均值，
    不是全任务平均，防止任务多的维度主导"。为了把这句话的数学含义讲清楚，
    这里构造一个【教学示例】：假设某策略在任务多的 Generalization 维度
    （12 个任务）上表现出色，而在任务少的 Memory 维度（6 个任务）上表现
    很差，看看两种聚合方式会给出多不同的总分。
    """
    print("\n" + "=" * 78)
    print("第 1 部分：5 维度均值 vs 42 任务加权平均（教学构造示例，非真实榜单数据）")
    print("=" * 78)

    # 教学构造的示例分数：刻意让「任务多的维度」分数很高、「任务少的维度」分数很低，
    # 以放大两种聚合方式的差异，从而看清楚"总分被任务数主导"的风险
    illustrative_scores = {
        "Generalization": 20.0,   # 12 个任务，示例中表现"好"
        "Memory": 2.0,             # 6 个任务，示例中表现"差"
        "Long-Horizon": 10.0,      # 8 个任务
        "Precision": 8.0,          # 8 个任务
        "Open": 6.0,               # 8 个任务
    }
    print("教学构造的策略『示例策略 P』各维度分数（非真实数据，仅用于演示计算差异）：")
    for dim in DIMENSION_ORDER:
        print(f"  {dim:<16} 任务数={DIMENSION_TASK_COUNTS[dim]:<3} 示例分数={illustrative_scores[dim]}")

    dim_mean = sum(illustrative_scores[d] for d in DIMENSION_ORDER) / len(DIMENSION_ORDER)

    total_tasks = sum(DIMENSION_TASK_COUNTS.values())
    task_weighted_avg = sum(
        illustrative_scores[d] * DIMENSION_TASK_COUNTS[d] for d in DIMENSION_ORDER
    ) / total_tasks

    print(f"\n方式 A —— RoboDojo 实际采用：5 维度均值 = {dim_mean:.2f}")
    print(f"方式 B —— 若改用 42 任务加权平均 = {task_weighted_avg:.2f}")
    print(
        f"\n差异 = {task_weighted_avg - dim_mean:+.2f}"
        "（正值说明加权平均会被任务数最多的 Generalization 维度的高分拉高，"
        "掩盖了该策略在 Memory 维度上的严重短板）"
    )
    print(
        "\n结论：如果用「全任务加权平均」，一个策略只要在任务多的维度里刷分，\n"
        "就能获得较高总分，即使它在任务少但同样重要的维度（如 Memory）上几乎失败。\n"
        "RoboDojo 选择「5 维度均值」正是为了让每个能力维度的权重相等，\n"
        "避免总分被任务数分布不均所左右——这就是论文强调的诊断哲学。"
    )


# ---------------------------------------------------------------------------
# 第 2 部分：真实数据 —— Average 总榜 vs Human，展示整体差距（Finding F1）
# ---------------------------------------------------------------------------

def print_average_leaderboard_gap() -> None:
    print("\n" + "=" * 78)
    print("第 2 部分：Table 1 Average 总榜（真实数据）—— 顶尖策略 vs Human 的差距")
    print("=" * 78)
    print(f"{'策略':<26}{'score':>10}{'SR%':>10}")
    print("-" * 78)
    for name, score, sr in TABLE1_AVERAGE_LEADERBOARD:
        print(f"{name:<26}{score:>10.2f}{sr:>10.2f}")
    print("-" * 78)
    name, score, sr = HUMAN_AVERAGE
    print(f"{name:<26}{score:>10.2f}{sr:>10.2f}   <-- 人类专家参考")

    top_name, top_score, _ = TABLE1_AVERAGE_LEADERBOARD[0]
    gap = HUMAN_AVERAGE[1] - top_score
    print(
        f"\nFinding F1: 榜首策略 {top_name} 的 Average score 仅 {top_score:.2f}，\n"
        f"与 Human ({HUMAN_AVERAGE[1]:.2f}) 相差 {gap:.2f} 分，说明离「均衡的通用操作」\n"
        "仍有巨大差距，且顶尖策略之间的分数也高度聚集在低区间。"
    )


# ---------------------------------------------------------------------------
# 第 3 部分：真实数据 —— 各维度榜首 vs Human，诊断差异化短板
# ---------------------------------------------------------------------------

def print_dimension_diagnosis() -> None:
    print("\n" + "=" * 78)
    print("第 3 部分：各维度榜首 vs Human（真实数据）—— 能力维度诊断")
    print("=" * 78)
    print(f"{'维度':<16}{'榜首策略':<24}{'score':>8}{'SR%':>8}{'Human score':>14}{'gap':>8}")
    print("-" * 78)
    for dim in DIMENSION_ORDER:
        c = DIMENSION_CHAMPIONS[dim]
        gap = c["human_score"] - c["score"]
        print(
            f"{dim:<16}{c['policy']:<24}{c['score']:>8.2f}{c['sr']:>8.2f}"
            f"{c['human_score']:>14.2f}{gap:>8.2f}"
        )

    # 统计 Average 总榜榜首在 5 个维度里到底拿了几个"第一"
    average_champion = TABLE1_AVERAGE_LEADERBOARD[0][0]
    dims_won_by_average_champion = [
        dim for dim in DIMENSION_ORDER if DIMENSION_CHAMPIONS[dim]["policy"] == average_champion
    ]
    print(
        f"\nAverage 总榜榜首『{average_champion}』只在 5 个维度中的 "
        f"{len(dims_won_by_average_champion)} 个（{', '.join(dims_won_by_average_champion)}）"
        "拿到该维度榜首，其余维度另有更强的策略。"
    )
    print(
        "这正是维度诊断相对于单一总分排行榜的价值：总分第一不等于每个能力\n"
        "维度都最强，聚合成一个数字会掩盖策略之间差异化的强弱项。"
    )

    open_gap = DIMENSION_CHAMPIONS["Open"]["human_score"] - DIMENSION_CHAMPIONS["Open"]["score"]
    precision_gap = DIMENSION_CHAMPIONS["Precision"]["human_score"] - DIMENSION_CHAMPIONS["Precision"]["score"]
    print(
        f"\nFinding F6（Open 几近全崩）: Open 维度即使是榜首 π0.5 也只有 "
        f"{DIMENSION_CHAMPIONS['Open']['score']:.2f} 分，与 Human "
        f"({DIMENSION_CHAMPIONS['Open']['human_score']:.2f}) 相差 {open_gap:.2f} 分，"
        "是 5 个维度里差距最悬殊的，说明开放语义操作基本未解决。"
    )
    print(
        f"Finding F4（Precision 是主要瓶颈）: Precision 维度榜首 X-VLA 有 "
        f"{DIMENSION_CHAMPIONS['Precision']['score']:.2f} 分，与 Human "
        f"({DIMENSION_CHAMPIONS['Precision']['human_score']:.2f}) 相差 {precision_gap:.2f} 分，"
        "反映接触感知控制、off-trajectory 恢复能力仍是瓶颈。"
    )


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_dimension_radar(out_path: str) -> None:
    """雷达图：5 个维度上，「维度榜首」 vs 「Human」 的分数对比。"""
    if not _HAS_MPL:
        log.warning("matplotlib 不可用，退化为纯文本柱状图。")
        _print_ascii_dimension_bars()
        return

    labels = DIMENSION_ORDER
    champion_scores = [DIMENSION_CHAMPIONS[d]["score"] for d in labels]
    human_scores = [DIMENSION_CHAMPIONS[d]["human_score"] for d in labels]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    champion_scores_closed = champion_scores + champion_scores[:1]
    human_scores_closed = human_scores + human_scores[:1]
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1, polar=True)

    ax1.plot(angles_closed, human_scores_closed, color="#2f7ed8", linewidth=2, label="Human Expert")
    ax1.fill(angles_closed, human_scores_closed, color="#2f7ed8", alpha=0.15)
    ax1.plot(angles_closed, champion_scores_closed, color="#d62728", linewidth=2, label="各维度榜首策略")
    ax1.fill(angles_closed, champion_scores_closed, color="#d62728", alpha=0.2)

    ax1.set_xticks(angles)
    ax1.set_xticklabels(labels)
    ax1.set_title("各维度榜首 vs Human（score）")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    # 右图：柱状图展示 gap，突出 Open 崩溃 / Precision 瓶颈
    ax2 = fig.add_subplot(1, 2, 2)
    gaps = [DIMENSION_CHAMPIONS[d]["human_score"] - DIMENSION_CHAMPIONS[d]["score"] for d in labels]
    colors = ["#d62728" if d == "Open" else ("#ff7f0e" if d == "Precision" else "#8c8c8c") for d in labels]
    bars = ax2.bar(labels, gaps, color=colors)
    ax2.set_ylabel("与 Human 的 score 差距 (gap)")
    ax2.set_title("各维度榜首与 Human 的差距（越高越难）")
    for rect, g in zip(bars, gaps):
        ax2.text(rect.get_x() + rect.get_width() / 2, g + 1, f"{g:.1f}", ha="center")
    ax2.tick_params(axis="x", rotation=15)

    fig.suptitle("RoboDojo 能力维度诊断：总分之外，各维度差异化短板一目了然（教学演示）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info("图表已保存到 %s", out_path)
    plt.close(fig)


def _print_ascii_dimension_bars() -> None:
    """matplotlib 缺失时的纯文本降级可视化：各维度 gap 柱状图。"""
    print("\n各维度榜首与 Human 的 score 差距（gap，文本柱状图）:")
    max_gap = max(
        DIMENSION_CHAMPIONS[d]["human_score"] - DIMENSION_CHAMPIONS[d]["score"] for d in DIMENSION_ORDER
    )
    for d in DIMENSION_ORDER:
        gap = DIMENSION_CHAMPIONS[d]["human_score"] - DIMENSION_CHAMPIONS[d]["score"]
        width = int(40 * gap / max_gap) if max_gap > 0 else 0
        print(f"  {d:<16}|{'#' * width} {gap:.1f}")


def main() -> None:
    log.info("开始运行能力维度诊断演示 ...")
    demo_mean_vs_task_weighted_average()
    print_average_leaderboard_gap()
    print_dimension_diagnosis()

    out_path = __file__.replace(".py", "_result.png")
    plot_dimension_radar(out_path)
    log.info("演示运行结束。")


if __name__ == "__main__":
    main()
