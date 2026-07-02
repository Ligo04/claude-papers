#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dream_budget_simulator.py
=========================
对综述标题论点的一个微型、零依赖模拟：
            "少做梦，多行动（Dream Less, Act More）。"

论文反复出现的论断（第 3、5、7.1 节）是：WAM 设计受
            *保真度 <-> 控制回路成本* 权衡 统辖：

  * 渲染更多未来（"做梦"）改善动作头可用的基底，但拉长了观测->控制的路径。
  * 行动者很少消费完整渲染的未来，所以超过某点后，额外保真度不再帮助任务成功，
    却仍花费延迟。
  * 运行时的最佳策略是按 *预期动作价值* 花费生成算力：接触/不确定处多做梦，
    常规移动时少做梦。

本脚本建模一个控制回路，扫过抽象阶梯，然后展示一个 *自适应* 做梦预算
（只在该步困难时花算力）胜过"总是多做梦"和"总是少做梦"。无真机、无学习——
只呈现综述所描述权衡的定性形状。

运行：  python3 dream_budget_simulator.py
"""

import math

# ---- 抽象阶梯（第 5.4 节），每一级 = 一个基底选择。 ----
# 每级有：它能承载多少控制相关信息（cap），以及在该级每"单位"做梦每控制步的延迟。
LADDER = [
    # 名称,                信息容量, 每单位延迟(ms)
    ("完整 RGB",            1.00,     22.0),   # Render-and-Decode，解码像素
    ("RGB-D / 法向",        0.92,     16.0),
    ("光流 / 3D 光流",      0.80,      7.0),
    ("语义掩码",            0.70,      4.5),
    ("隐 / 特征",           0.62,      2.0),   # Latent-Only / Video-Generation-Free
]

CONTROL_PERIOD_MS = 100.0   # 10 Hz 控制回路；超过它则动作迟到。


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def fidelity_from_budget(info_cap, dream_units):
    """收益递减曲线：更多去噪/rollout（'dream_units'）把达成的保真度推向基底的
    信息容量，然后饱和。这就是"行动者很少消费完整渲染未来"的效应。"""
    return info_cap * (1.0 - math.exp(-0.6 * dream_units))


def step_success_prob(fidelity, difficulty):
    """一步以随达成保真度上升的概率成功，但仅升到该步实际 *需要* 的程度。常规步
    需要很少保真度；接触丰富步需要很多。建模为围绕该步保真度需求的 sigmoid。"""
    demand = difficulty            # difficulty ∈ [0,1] == 该步所需保真度
    margin = fidelity - demand
    return sigmoid(8.0 * margin)


def latency_ms(rung_latency, dream_units):
    return rung_latency * dream_units


def run_loop(policy_fn, episode, label):
    """在一个回合（逐步难度列表）下运行一个策略，该策略逐步返回 (基底索引, 做梦单位)。
    报告平均成功率、平均延迟，以及超出控制周期的步骤比例（因果/延迟失效 -> 动作迟到）。"""
    succ, lat, late = [], [], 0
    for difficulty in episode:
        ri, du = policy_fn(difficulty)
        name, cap, lpu = LADDER[ri]
        fid = fidelity_from_budget(cap, du)
        ms = latency_ms(lpu, du)
        p = step_success_prob(fid, difficulty)
        # 若错过控制周期，该步的有效成功率崩塌（未来越过了它必须指导的时刻——第 5.2 节）。
        if ms > CONTROL_PERIOD_MS:
            late += 1
            p *= 0.3
        succ.append(p)
        lat.append(ms)
    n = len(episode)
    print(f"  {label:<30} 成功率={sum(succ)/n:0.3f}  "
          f"延迟={sum(lat)/n:5.1f}ms  迟到步={late}/{n}")
    return sum(succ) / n


# ---- 三种策略 ----
def always_dream_big(difficulty):
    """Render-and-Decode 极大主义：完整 RGB，每步大量去噪单位。"""
    return (0, 9.0)


def always_dream_small(difficulty):
    """激进 Latent-Only：最便宜基底，每步极少算力。"""
    return (4, 2.0)


def adaptive(difficulty):
    """'按预期动作价值花算力'（第 7.1 节）。
    简单步 -> 便宜的隐基底，少做梦。
    困难/接触步 -> 攀升阶梯向几何靠拢并多做梦，但绝不多到吹爆控制周期。"""
    if difficulty < 0.45:
        return (4, 2.0)            # 常规移动：隐，最少
    elif difficulty < 0.70:
        return (2, 4.0)            # 中等：光流，适度做梦
    else:
        return (1, 6.0)            # 接触丰富：RGB-D，多做梦（仍 < 周期）


def main():
    print("=" * 70)
    print("做梦预算模拟器  —  《少做梦，多行动》(arXiv 2606.20781)")
    print("=" * 70)

    print("\n[A] 抽象阶梯（第 5.4 节）：信息容量 vs 每单位延迟\n")
    print(f"    {'级别':<18}{'信息容量':<12}{'ms/单位'}")
    for name, cap, lpu in LADDER:
        print(f"    {name:<18}{cap:<12.2f}{lpu:.1f}")

    print("\n[B] 在完整 RGB 级别上'做梦'的收益递减")
    print("    （达成保真度饱和——额外去噪买不到多少）：\n")
    cap = LADDER[0][1]
    for du in (1, 2, 4, 6, 9, 14):
        f = fidelity_from_budget(cap, du)
        bar = "#" * int(f * 40)
        print(f"    做梦单位={du:>2}  保真度={f:0.3f}  {bar}")

    # 一个回合：大多为常规步，夹几个接触丰富的尖峰。
    episode = [0.2, 0.3, 0.25, 0.8, 0.85, 0.3, 0.2, 0.75, 0.35, 0.9,
               0.25, 0.3, 0.82, 0.2, 0.28]
    print(f"\n[C] 一个 {len(episode)} 步的回合"
          f"（难度 = 该步真正需要的保真度），")
    print(f"    控制周期 = {CONTROL_PERIOD_MS:.0f} ms。三种策略：\n")
    run_loop(always_dream_big,   episode, "总是多做梦（完整 RGB）")
    run_loop(always_dream_small, episode, "总是少做梦（隐）")
    run_loop(adaptive,           episode, "自适应（按动作价值花算力）")

    print("\n[D] 结果解读（吻合综述论点）：")
    print("    * '总是多'在困难步上赢，却处处付延迟，并有吹爆控制周期的风险")
    print("      （一个因果/延迟失效）。")
    print("    * '总是少'快，却饿死了接触丰富的步骤。")
    print("    * 自适应只在动作价值证明合理时做梦 -> 以零头延迟拿到最佳成功率。")
    print("      少做梦，多行动。\n")


if __name__ == "__main__":
    main()
