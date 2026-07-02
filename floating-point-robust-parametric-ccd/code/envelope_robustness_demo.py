#!/usr/bin/env python3
"""
envelope_robustness_demo.py
===========================================================================
演示论文的核心现象：在构造分段线性包络时，一次翻转的浮点比较不仅仅是把一个
数值轻微挪动——它能破坏包络的“拓扑”（哪些线被保留），产生几何上错误的结果。
这正是经典的“按误差界放宽阈值”技巧不够用的原因，也是 TDIBM-E 测试
`Phi < -eps` 而非 `Phi < 0` 的原因。

设置：三条近乎共点的线，其上方（“max”）包络含一个由中间线贡献的微小平顶帽：

        L_left :  a=-1, b=0     ->  值 -t
        L_mid  :  a= 0, b=h     ->  值  h   （小 h > 0）
        L_right:  a=+1, b=0     ->  值  t

t 在 [-1, 1] 上的真实上方包络：
    -t  当 t < -h ，   h  当 -h <= t <= h ，   t  当 t > h
两条外线交于 (t=0, 值=0)；中间线恰好在该交点上方 h 处，故它确实是包络的一部分。
对 L_mid 的“保留/丢弃”决策取决于 Phi = 2h 的符号——任意接近 0，故舍入误差可翻转它。

  * 丢弃 L_mid（拓扑破坏）-> 包络在 t=0 跌到 0，即真实 max h 之下。对 max-包络而言
    这会“缩小”碰撞窗 -> 漏报 -> 不安全。
  * 论文的安全措施 `Phi < -eps` 让“弹出”更难发生，故即便在同样的舍入噪声下 L_mid
    仍被保留 -> 包络始终 >= 真实 max -> 安全。

运行:  python3 envelope_robustness_demo.py     （无第三方依赖）
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Line:
    a: float  # 斜率
    b: float  # 截距

    def __call__(self, t: float) -> float:
        return self.a * t + self.b


def upper_envelope(lines: List[Line], drop_eps: float = 0.0,
                   phi_noise: float = 0.0) -> List[Line]:
    """
    用按斜率排序的栈机构造上方（max）包络（步骤 II）。

    drop_eps   : 论文的安全措施。仅当占优测试超过 0 达至少 drop_eps（即
                 `Phi < -drop_eps`）时才弹出一条线。drop_eps = 0 复现易错的
                 朴素测试 `Phi < 0`。
    phi_noise  : 一个加到 Phi 上的合成扰动，用以模拟舍入误差翻转比较
                 （代表真实算法中的浮点不精确）。
    """
    # 按斜率升序排序；斜率相同时保留截距更大的（去重）
    pts = sorted(lines, key=lambda L: (L.a, -L.b))
    dedup: List[Line] = []
    for L in pts:
        if dedup and abs(dedup[-1].a - L.a) < 1e-18:
            continue
        dedup.append(L)

    stack: List[Line] = []
    for L in dedup:
        while len(stack) >= 2:
            g1, g0 = stack[-1], stack[-2]            # 栈顶两条：gamma, gamma-1
            # Phi > 0 => 保留 g1 ; Phi < 0 => 弹出 g1（它已冗余）。式(5a)
            Phi = (L.a - g0.a) * (g1.b - g0.b) - (g1.a - g0.a) * (L.b - g0.b)
            Phi_observed = Phi + phi_noise            # 算法“看到”的值
            if Phi_observed < -drop_eps:
                stack.pop()
            else:
                break
        stack.append(L)
    return stack


def envelope_value(env: List[Line], t: float) -> float:
    return max(L(t) for L in env)


def true_max(lines: List[Line], t: float) -> float:
    return max(L(t) for L in lines)


def evaluate(env, all_lines, tL, tU, n=4001):
    """返回 (最大 |偏差|, t_at, 最坏点处的带符号偏差)。带符号>=0 表示安全。"""
    worst, worst_t, signed = 0.0, tL, 0.0
    for k in range(n):
        t = tL + (tU - tL) * k / (n - 1)
        d = envelope_value(env, t) - true_max(all_lines, t)
        if abs(d) > worst:
            worst, worst_t, signed = abs(d), t, d
    return worst, worst_t, signed


def report(name, env, lines, tL, tU):
    segs = [(round(L.a, 3), round(L.b, 6)) for L in env]
    err, t_at, signed = evaluate(env, lines, tL, tU)
    print(f"  活跃线段       : {segs}")
    print(f"  最大偏差       : {err:.3e} 于 t={t_at:+.3f}")
    if signed < -1e-18:
        verdict = "包络低于真实 max -> 窗被缩小 -> 漏报（不安全）"
    elif signed > 1e-18:
        verdict = "包络高于真实 max -> 窗被扩大 -> 误报（安全）"
    else:
        verdict = "包络 == 真实 max（精确）"
    print(f"  带符号偏差     : {signed:+.3e}  ->  {verdict}")
    return signed


def main():
    tL, tU = -1.0, 1.0
    h = 1e-13                       # 微小帽高；Phi = 2h ~ 2e-13
    lines = [Line(-1.0, 0.0), Line(0.0, h), Line(1.0, 0.0)]

    print("=" * 72)
    print(f"三条近乎共点的线；中间线给出高 h={h:g} 的帽")
    print("=" * 72)

    print("\n[1] 精确（Phi < 0，无舍入噪声）—— 中间线被正确保留")
    env = upper_envelope(lines, drop_eps=0.0, phi_noise=0.0)
    report("exact", env, lines, tL, tU)

    print("\n[2] 被破坏：一次舍入误差翻转了对 L_mid 的保留/丢弃测试")
    # 注入使观测到的 Phi = 2h + 噪声 变为负的噪声。
    env_bad = upper_envelope(lines, drop_eps=0.0, phi_noise=-3 * h)
    report("corrupted", env_bad, lines, tL, tU)
    print("  -> L_mid 被丢弃：线段“集合”改变了（拓扑错误），")
    print("     而非只是一个数值，且误差朝向不安全方向。")

    print("\n[3] 安全措施（论文）：测试 Phi < -eps，eps >= 最坏情况 Phi 误差")
    eps = 4 * h                     # 推导出的界超过噪声大小
    env_safe = upper_envelope(lines, drop_eps=eps, phi_noise=-3 * h)
    report("safe", env_safe, lines, tL, tU)
    print("  -> 保守测试拒绝边缘性的丢弃；L_mid 被保留，故包络从不跌破真值。")
    print("     最坏情况只是无害的误报。")

    print("\n" + "=" * 72)
    print("要点：该算法的鲁棒性关乎保持包络的“组合结构”。一次翻转的比较改变")
    print("哪些线存活；把每个分支偏向“保留”（Phi < -eps）使每个不确定决策都变成")
    print("误报、而非漏报——这正是 TDIBM-E 所证明的保证。")


if __name__ == "__main__":
    main()
