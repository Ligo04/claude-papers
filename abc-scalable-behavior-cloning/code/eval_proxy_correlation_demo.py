#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_proxy_correlation_demo.py
============================================================
教学演示：ABC 论文「三级评测代理」的相关性哲学
（论文 3.4 节 + 第 4 节 Sim-to-real correlation，事实来源见 FACTS_ABC.md）

论文背景（精简复述，非官方代码，仅用于教学理解）：
    做机器人 behavior cloning 研究时，最终目标是「真机成功率」，
    但真机评测非常昂贵（每个 checkpoint 每任务要跑 50 trials，
    ABC 论文的 sim-to-real 相关性实验总共只做了 12 个 checkpoint x 3 任务 x 50 trials）。
    如果每改一个小设计决策（比如 connector 用 AdaLN 还是 cross-attn）都要上真机，
    研究效率会很低。于是 ABC 提出一套「三级评测代理」金字塔：

        Level 1（最廉价）：离线标量指标 —— 训练 loss / 验证 loss / 验证 action error。
            优点：几乎零成本，每个 checkpoint 都能算。
            缺点（论文 3.4 节实测结论）：
                - 验证 action error 与真机表现相关性最强：
                  strict 上 Pearson r=-0.89 (p=3.8e-6), Spearman ρ=-0.93
                - 训练 loss 也显著相关：r=-0.84
                - 验证 loss 几乎不相关：r=-0.04 (p=0.886) —— 反直觉的对照组，
                  提醒我们不是所有「看起来合理」的离线指标都管用。
            结论：适合「廉价筛选」小的设计决策，但不能可靠预测最终成功率或
                  揭示失败模式。

        Level 2（中等成本）：仿真评测（ABC-Sim, MuJoCo + VR 遥操作数据）。
            论文用 12 个 checkpoint（ABC-DiT/ABC-VLA 在 ABC-130K 上各 4 个 +
            4 个内部 7K+仿真混合数据的 ABC-DiT），每个在 3 个真机任务上各测 50 trials，
            同时也在对应仿真任务里评测，比较 sim 表现与 real 表现的相关性：
                - Strict success: Pearson r=0.85 (p=4.2e-4, n=12)
                - Task progress: Pearson r=0.91 (p=5.0e-5, n=12)
            这说明仿真是真机表现的「有意义预测器」，可以在真机评测前，
            用仿真来 debug/筛选设计决策，大幅降低真机评测的样本需求。

        Level 3（最昂贵，但最终唯一可信）：真机评测（ABC-Eval）。
            大规模真机 rollout（>100 小时）+ 评分 rubric，是最终验证的黄金标准。

    这套「金字塔」的核心逻辑：用便宜的代理做大范围粗筛（离线指标一天能筛几十个变体），
    用中等成本的代理做进一步验证（仿真几十个 checkpoint），
    最后只用最贵的真机评测做「最终考试」（少量 checkpoint，高置信度）。
    相关系数越高，说明这一级代理越值得信赖、越能替代下一级的评测。

本脚本做两件事：
    1. 打印论文 Table 1（架构 ablation：6 个变体的真机 mean strict/progress）
       以及第 3.4 节 + 第 4 节的相关系数汇总表 —— 这些都是论文里的真实数字。
    2. 用「给定 Pearson r 生成对应散点」的标准方法（先生成两个独立标准正态序列，
       再用 y = r*x + sqrt(1-r^2)*z 的方式构造具有目标相关系数的合成点对），
       可视化三条关系：
         (a) val action error vs real progress —— 强负相关 (r=-0.89 的近似，
             这里用 strict 的 r=-0.89 生成，因为论文该系数是在 strict 上报告的)
         (b) val loss vs real progress —— 几乎无相关的对照组 (r=-0.04)
         (c) sim progress vs real progress —— 强正相关 (r=0.91)
       注意：散点的具体坐标是合成的（不是论文真实的逐 checkpoint 数据点，
       论文没有公开原始逐点数据），但相关系数 r 严格取自论文报告的数字，
       用来做「相关性强弱」的直觉可视化，而不是复现论文原始图表。

依赖：仅标准库 + numpy；若无 numpy 会退化为纯 Python 实现。
若有 matplotlib 则画三张子图对比相关性强弱，否则优雅降级为文本输出。
"""

from __future__ import annotations

import math
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_proxy_correlation_demo")

try:
    import numpy as np  # noqa: F401
    HAS_NUMPY = True
    log.info("numpy 可用，使用向量化实现")
except ImportError:
    HAS_NUMPY = False
    log.info("numpy 不可用，退化为纯 Python 实现（速度较慢但结果等价）")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "AR PL UMing CN", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
    log.info("matplotlib 可用，将生成示意图")
except ImportError:
    HAS_MPL = False
    log.info("matplotlib 不可用，跳过绘图，仅打印文本结果")


# ============================================================
# 第一部分：论文真实数据（严格来自 FACTS_ABC.md，不得编造）
# ============================================================

# Table 1：架构 ablation（真机评测，200k steps，内部 7000h 语料）
# 6 个变体：VLA 三种 connector + DiT 三种视觉表示，聚合 mean strict / progress / latency
TABLE1_ARCH_ABLATION = [
    # (模型名, mean_strict_%, mean_progress_%, latency_ms)
    ("VLA Pooled adaLN",     32.8, 61.4, 17.24),
    ("VLA FAST+X-Attn",       3.6, 32.6, 19.24),
    ("VLA X-Attn",            0.0, 11.7, 19.24),
    ("DiT DINOv3-xattn",     32.9, 67.5, 37.4),
    ("DiT CLIP-adaln",       13.4, 47.3, 27.5),
    ("DiT CLIP-xattn",       24.5, 58.8, 37.5),
]

# 第 3.4 节：离线指标与真机表现（strict success）的相关系数
# val action error: 固定 10 diffusion steps、无 action prefix 下的动作 L2 误差
OFFLINE_METRIC_CORR = [
    # (指标名, pearson_r, 备注)
    ("验证 action error (val action error)", -0.89, "相关最强；p=3.8e-6, Spearman rho=-0.93"),
    ("训练 loss (train loss)",               -0.84, "显著负相关"),
    ("验证 loss (val loss)",                 -0.04, "几乎不相关；p=0.886，反直觉对照组"),
]

# 第 4 节：sim-to-real 相关性实验（12 个 checkpoint，3 个真机任务各 50 trials）
SIM_REAL_CORR = [
    # (指标名, pearson_r, p_value, n)
    ("Strict success (sim vs real)", 0.85, 4.2e-4, 12),
    ("Task progress (sim vs real)",  0.91, 5.0e-5, 12),
]


def print_table1():
    print("\n" + "=" * 78)
    print("表 A：论文 Table 1 —— 架构 ablation（真机评测，200k steps，内部 7000h 语料）")
    print("=" * 78)
    print(f"{'模型':<20} | {'Mean strict':>12} | {'Mean progress':>14} | {'Latency(ms)':>12}")
    print("-" * 78)
    for name, strict, progress, latency in TABLE1_ARCH_ABLATION:
        print(f"{name:<20} | {strict:>11.1f}% | {progress:>13.1f}% | {latency:>12.2f}")
    print("(真机 3 任务：Bottles / Dishrack / Mugs，每任务 50 trials；数据来自论文 Table 1)")


def print_correlation_tables():
    print("\n" + "=" * 78)
    print("表 B：离线指标 vs 真机表现（strict success）相关系数（论文 3.4 节）")
    print("=" * 78)
    print(f"{'离线指标':<40} | {'Pearson r':>10} | 备注")
    print("-" * 78)
    for name, r, note in OFFLINE_METRIC_CORR:
        print(f"{name:<40} | {r:>10.2f} | {note}")

    print("\n" + "=" * 78)
    print("表 C：Sim-to-real 相关性（论文第 4 节，n=12 checkpoint）")
    print("=" * 78)
    print(f"{'指标':<32} | {'Pearson r':>10} | {'p-value':>10} | {'n':>4}")
    print("-" * 78)
    for name, r, p, n in SIM_REAL_CORR:
        print(f"{name:<32} | {r:>10.2f} | {p:>10.1e} | {n:>4d}")


# ============================================================
# 第二部分：给定 Pearson r，合成具有该相关系数的散点对
# ------------------------------------------------------------
# 原理：设 x, z 为两个独立标准正态序列，令 y = r*x + sqrt(1-r^2)*z，
# 则 (x, y) 的总体 Pearson 相关系数恰为 r（标准的双变量正态构造法）。
# 样本相关系数会在 r 附近有小幅抖动（n 越大越接近 r），这里用较大的 n
# 让抖动可控，从而清楚地展示"强相关 vs 弱相关 vs 对照组"的视觉差异。
# ============================================================

def make_correlated_pair(r, n, rng, seed_offset=0):
    """
    生成 n 对 (x, y)，使其**样本** Pearson 相关系数精确等于 r（而非仅总体期望上等于 r）。

    为什么不用最朴素的 y = r*x + sqrt(1-r^2)*z：
    那种构造只保证「总体」相关系数是 r，但小样本下抽样噪声很大
    （比如 n=30 时，目标 r=0 附近的样本 r 标准差约 1/sqrt(n-3) ≈ 0.19，
    实测中曾出现目标 r=-0.04 却抽到样本 r=+0.42 的情况，会误导「几乎无相关」
    这个教学结论）。

    因此改用 Gram-Schmidt 正交化的精确构造法：
        1. 采样 x, z 两组独立标准正态，各自去均值；
        2. 把 z 相对 x 做 Gram-Schmidt 正交化，得到与 x 完全正交的 z_orth；
        3. 将 x、z_orth 都归一化为单位范数；
        4. y = r * x_unit + sqrt(1-r^2) * z_orth_unit。
    这样构造出的 (x, y) 样本相关系数**精确**等于 r（浮点误差范围内），
    与样本量 n 无关，避免小样本噪声掩盖我们想展示的相关性强弱对比。
    """
    if HAS_NUMPY:
        x = rng.normal(0, 1, size=n)
        z = rng.normal(0, 1, size=n)
        x = x - x.mean()
        z = z - z.mean()
        # Gram-Schmidt: 去掉 z 中与 x 共线的分量，得到与 x 正交的 z_orth
        z_orth = z - (np.dot(x, z) / np.dot(x, x)) * x
        x_unit = x / np.linalg.norm(x)
        z_unit = z_orth / np.linalg.norm(z_orth)
        y = r * x_unit + math.sqrt(max(0.0, 1 - r * r)) * z_unit
        return x_unit, y
    else:
        n_f = float(n)
        x = [rng.gauss(0, 1) for _ in range(n)]
        z = [rng.gauss(0, 1) for _ in range(n)]
        mx = sum(x) / n_f
        mz = sum(z) / n_f
        x = [xi - mx for xi in x]
        z = [zi - mz for zi in z]
        xz = sum(xi * zi for xi, zi in zip(x, z))
        xx = sum(xi * xi for xi in x)
        z_orth = [zi - (xz / xx) * xi for xi, zi in zip(x, z)]
        x_norm = math.sqrt(sum(xi * xi for xi in x))
        z_norm = math.sqrt(sum(zi * zi for zi in z_orth))
        x_unit = [xi / x_norm for xi in x]
        z_unit = [zi / z_norm for zi in z_orth]
        y = [r * xi + math.sqrt(max(0.0, 1 - r * r)) * zi for xi, zi in zip(x_unit, z_unit)]
        return x_unit, y


def pearson_r(x, y):
    """计算样本 Pearson 相关系数，用于核对合成数据确实落在目标 r 附近。"""
    if HAS_NUMPY:
        return float(np.corrcoef(x, y)[0, 1])
    else:
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        vx = sum((xi - mx) ** 2 for xi in x)
        vy = sum((yi - my) ** 2 for yi in y)
        return cov / math.sqrt(vx * vy)


def main():
    log.info("=" * 60)
    log.info("ABC 论文「三级评测代理」相关性哲学 —— 数据展示与可视化")
    log.info("论文数据来源: arXiv 2606.27375v1, 3.4 节 + 第 4 节")
    log.info("=" * 60)

    print_table1()
    print_correlation_tables()

    # ------------------------------------------------------------
    # 合成散点：用论文真实 r 值构造直觉可视化数据
    # ------------------------------------------------------------
    rng = np.random.default_rng(2026) if HAS_NUMPY else random.Random(2026)
    n_offline = 30   # 离线指标场景：模拟 30 个训练 checkpoint/变体（论文未给出确切 n，仅为可视化选取合理规模）
    n_simreal = SIM_REAL_CORR[0][3]  # sim-real 场景严格用论文的 n=12

    log.info("合成散点：val action error vs real progress (目标 r=-0.89)")
    x_err, y_err = make_correlated_pair(-0.89, n_offline, rng)
    log.info("合成散点：val loss vs real progress (目标 r=-0.04，对照组)")
    x_vl, y_vl = make_correlated_pair(-0.04, n_offline, rng)
    log.info("合成散点：sim progress vs real progress (目标 r=0.91, n=12)")
    x_sim, y_sim = make_correlated_pair(0.91, n_simreal, rng)

    print("\n" + "=" * 78)
    print("表 D：合成散点的实测相关系数核对（应与目标 r 接近，验证生成方法正确）")
    print("=" * 78)
    for label, (x, y), target_r in [
        ("val action error vs real progress", (x_err, y_err), -0.89),
        ("val loss vs real progress (对照组)", (x_vl, y_vl), -0.04),
        ("sim progress vs real progress", (x_sim, y_sim), 0.91),
    ]:
        emp_r = pearson_r(x, y)
        print(f"  {label:<38} 目标 r={target_r:+.2f}   实测 r={emp_r:+.3f}")

    print("\n结论：")
    print("  1) 验证 action error 与真机 progress 强负相关（r≈-0.89），是最值得信赖的")
    print("     离线代理——可用于每天筛选几十个训练变体，快速淘汰明显更差的设计。")
    print("  2) 验证 loss 与真机表现几乎无相关（r≈-0.04），提醒我们「loss 更低」不等于")
    print("     「策略更好」——扩散模型的验证 loss 在训练后期会回升，但真机表现仍在改善，")
    print("     这是论文特意点出的反直觉对照组。")
    print("  3) 仿真 progress 与真机 progress 强正相关（r=0.91，n=12，论文真实实验），")
    print("     说明可以在没有真机、或真机评测昂贵时，先用仿真验证设计决策，")
    print("     只在少量高置信度的 checkpoint 上才动用最贵的真机评测。")
    print("  4) 三级代理金字塔：离线指标(近乎免费，筛大量小决策) → 仿真(中等成本，")
    print("     r≈0.9 高保真验证) → 真机(最贵，最终唯一可信的裁判)。")

    # ------------------------------------------------------------
    # 可选绘图：三个子图分别展示强负相关/无相关(对照)/强正相关
    # ------------------------------------------------------------
    if HAS_MPL:
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

            def scatter_with_fit(ax, x, y, title, color, r_label):
                ax.scatter(x, y, color=color, alpha=0.75, edgecolor="white", s=55)
                if HAS_NUMPY:
                    coeffs = np.polyfit(x, y, 1)
                    xs = np.linspace(min(x), max(x), 50)
                    ax.plot(xs, coeffs[0] * xs + coeffs[1], "--", color="gray", linewidth=1.5)
                ax.set_title(f"{title}\n(r={r_label:+.2f})")
                ax.set_xlabel("指标 (标准化)")
                ax.set_ylabel("真机 progress (标准化)")
                ax.grid(True, alpha=0.3)

            scatter_with_fit(
                axes[0], x_err, y_err,
                "val action error vs 真机 progress\n(强负相关，最可信离线代理)",
                "#C0392B", -0.89,
            )
            scatter_with_fit(
                axes[1], x_vl, y_vl,
                "val loss vs 真机 progress\n(几乎无相关，对照组)",
                "#7F8C8D", -0.04,
            )
            scatter_with_fit(
                axes[2], x_sim, y_sim,
                "仿真 progress vs 真机 progress\n(强正相关，n=12，论文真实实验)",
                "#2E86AB", 0.91,
            )

            fig.suptitle("ABC 三级评测代理相关性直觉可视化（散点为合成数据，r 值取自论文）", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            out_path = "eval_proxy_correlation_demo.png"
            fig.savefig(out_path, dpi=130)
            log.info("图已保存到: %s", out_path)
        except Exception as exc:
            log.warning("绘图过程出现异常，已优雅降级为纯文本输出: %s", exc)
    else:
        log.info("matplotlib 不可用，已跳过绘图部分。")

    print("\n[提示] 本脚本中的相关系数 r 均严格取自论文报告数字（见 FACTS_ABC.md），")
    print("       但散点坐标为按目标 r 合成的示意数据，并非论文原始逐点数据")
    print("       （论文未公开每个 checkpoint 的具体 x/y 值）。仅用于教学直觉展示。")


if __name__ == "__main__":
    main()
