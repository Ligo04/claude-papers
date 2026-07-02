# -*- coding: utf-8 -*-
"""
WALL-WM —— Staircase 并行潜 CoT 解码：计算成本模型（最小 Demo）
=================================================================

WALL-WM 用 Staircase 解码把“串行潜思维链(CoT)”变成“深度并行潜 rollout”
（论文 §3.4、Fig.6）。核心机制：在中继深度 Nr 处切分 Transformer——
  • 只有【第一个】潜位置走完全部低层（0..Nr），产出一个被所有推理位置复用的
    “共享中继表示”；
  • 其余 Kc-1 个潜状态在高层（Nr..L）【并行】生成，不再重复低层计算。

本仅依赖 NumPy 的 Demo 用一个【层前向次数】成本模型，对比三种 CoT 调度的成本，
复现论文 Fig.6 的定性结论：Staircase 在保持潜推理表达力的同时大幅降低层成本。

成本单位 = “一次 Transformer 层前向”。设总层数 L、潜状态数 Kc、中继深度 Nr：
  (a) 传统/串行 CoT（离散或潜）：每个潜状态都要走完整 L 层，且串行依赖 →
        层前向次数 = Kc * L（串行，无法并行）。
  (b) Staircase（本文）：
        - 低层只算一次（第一个潜位置）：Nr 层；
        - 高层每个潜状态各算一次：Kc * (L - Nr) 层；
        - 总层前向 = Nr + Kc*(L - Nr)，且高层部分可【并行】。
  我们同时报告“总层前向次数”（算力/能耗）与“关键路径深度”（延迟）。

运行：
    python staircase_latent_cot_demo.py
"""


def serial_cot_cost(L, Kc):
    total = Kc * L          # 每个潜状态走完整 L 层
    critical = Kc * L       # 串行依赖：关键路径 = 全部叠加
    return total, critical


def staircase_cost(L, Kc, Nr):
    total = Nr + Kc * (L - Nr)        # 低层算一次 + 高层每状态一次
    # 关键路径：先走低层 Nr，再走一支高层 (L-Nr)（高层各状态并行）
    critical = Nr + (L - Nr)          # = L
    return total, critical


def bar(x, xmax, width=30):
    return "█" * int(round(x / xmax * width))


def main():
    print(__doc__)
    L = 36           # Transformer 层数（示意，量级接近 Qwen3.5-9B）
    print(f"设定：Transformer 总层数 L = {L}\n")

    print("=" * 72)
    print("总层前向次数（算力/能耗，越低越好）")
    print("-" * 72)
    print(f"  {'Kc':>3} {'Nr':>3} | {'串行CoT':>8} | {'Staircase':>10} | {'省下':>6} | 图示(Staircase)")
    configs = [(4, 24), (8, 24), (8, 30), (16, 30)]
    serial_max = max(serial_cot_cost(L, kc)[0] for kc, _ in configs)
    for Kc, Nr in configs:
        s_total, _ = serial_cot_cost(L, Kc)
        t_total, _ = staircase_cost(L, Kc, Nr)
        save = (1 - t_total / s_total) * 100
        print(f"  {Kc:>3} {Nr:>3} | {s_total:>8} | {t_total:>10} | {save:>5.0f}% | {bar(t_total, serial_max)}")

    print()
    print("=" * 72)
    print("关键路径深度（延迟，越低越好）—— 串行 vs 并行")
    print("-" * 72)
    print(f"  {'Kc':>3} {'Nr':>3} | {'串行CoT关键路径':>14} | {'Staircase关键路径':>16} | {'加速':>6}")
    for Kc, Nr in configs:
        _, s_crit = serial_cot_cost(L, Kc)
        _, t_crit = staircase_cost(L, Kc, Nr)
        speed = s_crit / t_crit
        print(f"  {Kc:>3} {Nr:>3} | {s_crit:>14} | {t_crit:>16} | {speed:>5.1f}×")

    print()
    print("解读：")
    print("  • 串行 CoT 的关键路径 = Kc×L：潜状态越多、延迟线性暴涨（串行依赖瓶颈）。")
    print("  • Staircase 的关键路径恒为 L（先走低层 Nr，再并行走高层 L−Nr），与 Kc 无关")
    print("    → 推理链越长，相对加速越大；同时低层只算一次，省下大量重复计算。")
    print("  • 这就是论文 Fig.6 的要点：跨层中继 + 深度并行，让长链潜推理既省算力又降延迟，")
    print("    且潜状态可微、直接注入 WAM 跨注意力（无需离散 token 采样）。")
    print()
    print("  注：这是【层前向次数】的成本模型，用于揭示标度关系；不是端到端墙钟计时。")


if __name__ == "__main__":
    main()
