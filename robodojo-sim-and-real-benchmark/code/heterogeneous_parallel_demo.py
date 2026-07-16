#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heterogeneous_parallel_demo.py
================================

【教学演示，非官方代码】
本脚本是对 RoboDojo 论文 (arXiv:2607.04434) 最核心系统贡献
——「异构并行仿真 (Heterogeneous Parallelism)」——的简化教学复现，
不是论文作者的官方实现，也不调用真实的 Isaac Sim/Isaac Lab。

论文原文事实（引用自 FACTS.md，不得编造）：
    - 标准 vectorized 仿真采用「同构克隆环境」：所有并行 env 共享同一
      场景模板，仅位姿/随机种子不同。
    - RoboDojo 提出「异构并行」：多个 env 共享同一个 vectorized 接口
      一起 step，但每个 env 独立采样自己的场景配置（不同物体类别数/
      几何/杂物数 up to 25/铰接结构/任务布局）。
    - 效率数据（8×RTX 4090，每卡 1 进程 10 并行环境，zero-action rollout）：
        异构并行 77.4 interactions/s  vs  非异构 40.0 interactions/s
        → 1.94× 加速。
    - 既提速又保留场景多样性；支持多 GPU sharding。

本脚本的「为什么」：
    同构方案若想覆盖论文级别的场景多样性（杂物数 0~25、物体类别数、
    铰接结构、任务布局等），只能不断把整批 env 的场景模板整体替换掉
    （相当于让所有并行进程停下来重新加载资产），每次替换都要付出一次
    昂贵的 reload 开销 —— 这正是同构方案吞吐更低的根因。
    异构并行把「采样多样性」下放到每个 env 自己的 reset 逻辑里，
    每个 env 各管各的，不需要让整批环境停下来等待资产重载，
    因此吞吐更高；同时因为每个 env 都在独立采样，多样性覆盖反而更好。
    这就是「异构为何能兼顾提速与多样性」的核心机制。

本脚本用一个简化的离散时间模型（不是真实物理仿真）来复现这个机制的
量级关系，让读者能直观看到 "throughput" 与 "scene diversity coverage"
两个指标之间的权衡是如何被异构并行同时打破的。

运行：
    python3 heterogeneous_parallel_demo.py
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

# matplotlib 是可选依赖：没有时优雅降级为纯文本柱状图
try:
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境下也能保存 png
    import matplotlib.pyplot as plt
    # 指定支持中文的字体，避免图中中文渲染成方块（系统若无该字体会自动忽略）
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _HAS_MPL = True
except ImportError:  # pragma: no cover - 环境降级路径
    _HAS_MPL = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger("heterogeneous_parallel_demo")

# ---------------------------------------------------------------------------
# 场景配置空间定义（简化版，用来量化"场景多样性"）
# 真实论文里的自由度包括：物体类别/几何、杂物数(up to 25)、铰接结构、
# 任务布局等；这里各用一个离散取值范围来代表，凑出一个可枚举的组合空间，
# 只是为了能算出一个"覆盖了组合空间的百分之多少"这样的可比较指标。
# ---------------------------------------------------------------------------
OBJECT_CATEGORY_CHOICES = list(range(1, 7))     # 1~6 种物体类别
CLUTTER_COUNT_CHOICES = list(range(0, 26))       # 0~25 个杂物（论文原话：最多 25 个）
ARTICULATION_CHOICES = [False, True]             # 是否含铰接结构（柜子/抽屉等）
TASK_LAYOUT_CHOICES = list(range(1, 6))          # 5 种任务/工作空间布局变体

COMBO_SPACE_SIZE = (
    len(OBJECT_CATEGORY_CHOICES)
    * len(CLUTTER_COUNT_CHOICES)
    * len(ARTICULATION_CHOICES)
    * len(TASK_LAYOUT_CHOICES)
)  # = 6 * 26 * 2 * 5 = 1560 种可枚举场景配置组合

# ---------------------------------------------------------------------------
# 仿真吞吐模型参数
# 数值经过校准，使「异构」吞吐落在论文 77.4 interactions/s 附近、
# 「同构」吞吐落在 40.0 interactions/s 附近（量级/比例对齐，而非声称精确复现）。
# ---------------------------------------------------------------------------
NUM_ENVS = 80                 # 论文设置：8 张 RTX 4090 × 每卡 10 并行 env
BASE_TICK_S = 0.013            # 单步物理/渲染计算耗时（异构、同构共享，代表底层仿真本身的成本）
EPISODE_STEPS = 50              # 每个 episode 的 step 数，到期就要 reset 场景
RELOAD_OVERHEAD_S = 0.6         # 同构方案「整批切换场景模板」时的资产重载停顿（异构没有这个开销）
WALL_BUDGET_S = 120.0           # 两种方案公平对比用的仿真墙钟时间预算（秒）

# 论文里的真实参考数值，仅用于打印对比，不参与本脚本的计算
PAPER_REFERENCE = {
    "heterogeneous_interactions_per_s": 77.4,
    "homogeneous_interactions_per_s": 40.0,
    "speedup": 1.94,
}


def sample_scene_config(rng: random.Random) -> tuple[int, int, bool, int]:
    """独立采样一个场景配置组合（物体类别数, 杂物数, 是否铰接, 任务布局）。"""
    return (
        rng.choice(OBJECT_CATEGORY_CHOICES),
        rng.choice(CLUTTER_COUNT_CHOICES),
        rng.choice(ARTICULATION_CHOICES),
        rng.choice(TASK_LAYOUT_CHOICES),
    )


@dataclass
class SimResult:
    mode: str
    total_steps: int
    elapsed_s: float
    throughput_interactions_per_s: float
    episodes: int
    unique_configs: int
    coverage_pct: float


def simulate_homogeneous(seed: int = 0) -> SimResult:
    """同构克隆环境：整批 env 共享同一场景模板，只有切模板时才换新配置。

    为了也能覆盖多样场景，同构方案只能定期把整批 env 的场景模板整体
    替换掉（等价于让所有并行进程一起停下来重新加载资产），每次替换
    都会产生一次 RELOAD_OVERHEAD_S 的停顿——这正是它吞吐更低、
    多样性积累更慢的根本原因。
    """
    rng = random.Random(seed)
    t = 0.0
    total_steps = 0
    episodes = 0
    unique_configs: set[tuple[int, int, bool, int]] = set()

    while True:
        step_time = EPISODE_STEPS * BASE_TICK_S
        if t + step_time > WALL_BUDGET_S:
            remaining = WALL_BUDGET_S - t
            steps_possible = int(remaining / BASE_TICK_S)
            total_steps += steps_possible
            t += steps_possible * BASE_TICK_S
            break

        # 整批 env 在当前共享模板下并行 step 完一个 episode
        t += step_time
        total_steps += EPISODE_STEPS
        episodes += 1

        # episode 结束，想要新的场景多样性 -> 整批切换模板（只采样 1 个新配置，
        # 因为同构方案里所有 env 共享同一个模板，只有位姿/种子不同）
        config = sample_scene_config(rng)
        unique_configs.add(config)

        if t >= WALL_BUDGET_S:
            break
        t += RELOAD_OVERHEAD_S  # 资产重载停顿：整批 env 一起等待

    throughput = total_steps / t if t > 0 else 0.0
    coverage = len(unique_configs) / COMBO_SPACE_SIZE * 100.0
    return SimResult(
        mode="同构 (homogeneous clones)",
        total_steps=total_steps,
        elapsed_s=t,
        throughput_interactions_per_s=throughput,
        episodes=episodes,
        unique_configs=len(unique_configs),
        coverage_pct=coverage,
    )


def simulate_heterogeneous(seed: int = 0) -> SimResult:
    """异构并行：同一个 vectorized 接口一起 step，但每个 env 独立采样自己的场景配置。

    因为多样性被下放到了每个 env 自己的 reset 逻辑里，不需要让整批
    env 停下来重载资产，所以没有 RELOAD_OVERHEAD_S 这一项停顿，
    吞吐更高；同时因为每个 episode 里 NUM_ENVS 个 env 各自独立采样，
    单位时间内积累的独立场景配置数量远高于同构方案。
    """
    rng = random.Random(seed)
    t = 0.0
    total_steps = 0
    episodes = 0
    unique_configs: set[tuple[int, int, bool, int]] = set()

    while True:
        step_time = EPISODE_STEPS * BASE_TICK_S
        if t + step_time > WALL_BUDGET_S:
            remaining = WALL_BUDGET_S - t
            steps_possible = int(remaining / BASE_TICK_S)
            total_steps += steps_possible
            t += steps_possible * BASE_TICK_S
            break

        t += step_time
        total_steps += EPISODE_STEPS
        episodes += 1

        # 异构并行的关键：批内每个 env 各自独立采样自己的场景配置，
        # 不需要整体停顿，也不需要所有 env 保持一致
        for _ in range(NUM_ENVS):
            unique_configs.add(sample_scene_config(rng))

        # 没有 reload 停顿：t 不再额外增加

    throughput = total_steps / t if t > 0 else 0.0
    coverage = min(len(unique_configs) / COMBO_SPACE_SIZE * 100.0, 100.0)
    return SimResult(
        mode="异构并行 (heterogeneous parallel)",
        total_steps=total_steps,
        elapsed_s=t,
        throughput_interactions_per_s=throughput,
        episodes=episodes,
        unique_configs=len(unique_configs),
        coverage_pct=coverage,
    )


def print_comparison_table(homo: SimResult, hetero: SimResult) -> None:
    speedup = hetero.throughput_interactions_per_s / homo.throughput_interactions_per_s

    print("\n" + "=" * 78)
    print("RoboDojo 异构并行仿真 — 简化教学演示结果")
    print("=" * 78)
    header = f"{'指标':<28}{'同构 (homogeneous)':>22}{'异构 (heterogeneous)':>26}"
    print(header)
    print("-" * 78)
    print(f"{'总 step 数':<28}{homo.total_steps:>22,}{hetero.total_steps:>26,}")
    print(f"{'实际耗时 (s)':<28}{homo.elapsed_s:>22.2f}{hetero.elapsed_s:>26.2f}")
    print(f"{'episode 数':<28}{homo.episodes:>22,}{hetero.episodes:>26,}")
    print(
        f"{'吞吐 throughput (interactions/s)':<28}"
        f"{homo.throughput_interactions_per_s:>22.2f}"
        f"{hetero.throughput_interactions_per_s:>26.2f}"
    )
    print(
        f"{'唯一场景配置数':<28}{homo.unique_configs:>22,}{hetero.unique_configs:>26,}"
        f"   (组合空间共 {COMBO_SPACE_SIZE} 种)"
    )
    print(
        f"{'场景多样性覆盖度 (%)':<28}{homo.coverage_pct:>22.2f}{hetero.coverage_pct:>26.2f}"
    )
    print("-" * 78)
    print(f"本演示模拟得到的加速比: {speedup:.2f}×")
    print(
        "论文真实参考数值 (Table/效率章节): "
        f"异构 {PAPER_REFERENCE['heterogeneous_interactions_per_s']} interactions/s "
        f"vs 同构 {PAPER_REFERENCE['homogeneous_interactions_per_s']} interactions/s "
        f"→ {PAPER_REFERENCE['speedup']}× 加速"
    )
    print("=" * 78)
    print(
        "\n结论：异构并行没有让每个 env 变得更快（单步物理计算成本 BASE_TICK_S 相同），\n"
        "而是消灭了同构方案里「整批停下来重载资产才能获得新场景」的固定开销；\n"
        "多样性因此从「必须靠切换模板才能获得」变成了「每个 env 与生俱来」，\n"
        "所以吞吐和多样性覆盖度可以同时提升，而不是此消彼长的权衡关系。\n"
    )


def plot_results(homo: SimResult, hetero: SimResult, out_path: str) -> None:
    """如果 matplotlib 可用，画柱状图对比吞吐与多样性覆盖度；否则退化为文本柱状图。"""
    if not _HAS_MPL:
        log.warning("matplotlib 不可用，退化为纯文本柱状图。")
        _print_ascii_bars(homo, hetero)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    modes = ["同构\n(homogeneous)", "异构\n(heterogeneous)"]
    colors = ["#8c8c8c", "#2f7ed8"]

    throughputs = [homo.throughput_interactions_per_s, hetero.throughput_interactions_per_s]
    ax1.bar(modes, throughputs, color=colors)
    ax1.set_ylabel("interactions / s")
    ax1.set_title("吞吐 Throughput (zero-action rollout)")
    for i, v in enumerate(throughputs):
        ax1.text(i, v + max(throughputs) * 0.02, f"{v:.1f}", ha="center")
    ax1.axhline(
        PAPER_REFERENCE["homogeneous_interactions_per_s"],
        color="#8c8c8c", linestyle="--", linewidth=1, alpha=0.6,
    )
    ax1.axhline(
        PAPER_REFERENCE["heterogeneous_interactions_per_s"],
        color="#2f7ed8", linestyle="--", linewidth=1, alpha=0.6,
    )

    coverages = [homo.coverage_pct, hetero.coverage_pct]
    ax2.bar(modes, coverages, color=colors)
    ax2.set_ylabel("场景多样性覆盖度 (%)")
    ax2.set_title(f"Scene Diversity Coverage (共 {COMBO_SPACE_SIZE} 种组合)")
    ax2.set_ylim(0, 105)
    for i, v in enumerate(coverages):
        ax2.text(i, v + 2, f"{v:.1f}%", ha="center")

    fig.suptitle("RoboDojo 异构并行 vs 同构克隆：吞吐与多样性同时提升（教学演示）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info("图表已保存到 %s", out_path)
    plt.close(fig)


def _print_ascii_bars(homo: SimResult, hetero: SimResult) -> None:
    """matplotlib 缺失时的纯文本降级可视化。"""
    def bar(value: float, max_value: float, width: int = 40) -> str:
        n = int(width * value / max_value) if max_value > 0 else 0
        return "#" * n

    max_tp = max(homo.throughput_interactions_per_s, hetero.throughput_interactions_per_s)
    print("\n吞吐 (interactions/s):")
    print(f"  同构   |{bar(homo.throughput_interactions_per_s, max_tp)} {homo.throughput_interactions_per_s:.1f}")
    print(f"  异构   |{bar(hetero.throughput_interactions_per_s, max_tp)} {hetero.throughput_interactions_per_s:.1f}")

    print("\n场景多样性覆盖度 (%):")
    print(f"  同构   |{bar(homo.coverage_pct, 100)} {homo.coverage_pct:.1f}%")
    print(f"  异构   |{bar(hetero.coverage_pct, 100)} {hetero.coverage_pct:.1f}%")


def main() -> None:
    log.info("开始模拟同构 (homogeneous) 克隆环境仿真 ...")
    homo = simulate_homogeneous(seed=42)
    log.info("同构模拟完成：%d steps, throughput=%.2f interactions/s", homo.total_steps, homo.throughput_interactions_per_s)

    log.info("开始模拟异构 (heterogeneous) 并行仿真 ...")
    hetero = simulate_heterogeneous(seed=42)
    log.info("异构模拟完成：%d steps, throughput=%.2f interactions/s", hetero.total_steps, hetero.throughput_interactions_per_s)

    print_comparison_table(homo, hetero)

    out_path = __file__.replace(".py", "_result.png")
    plot_results(homo, hetero, out_path)


if __name__ == "__main__":
    main()
