#!/usr/bin/env python3
"""
inverse_construction_demo.py
===========================================================================
演示论文的数据集思想：与其正向计算 CCD 真值（不可行 / 需符号求解器），不如
“反演”流程——规定我们想要的碰撞，再“反向”求解出能产生它的曲面配置——并用
“精确有理算术”进行，同时强制“二进分数（dyadic）性质”，使最终输入能无损地
用 IEEE-754 double 表示。

它还复现论文的核心微妙之处（§5.4）：求解约束系统会“破坏”二进分数性质，因为
除数是接触参数 `s`。我们展示：

  * 尝试 A（朴素）：接触参数 s = 3/4 在分母引入因子 3 -> 求解出的控制点“不是”
    二进分数 -> 往返检查（有理->double->有理）失败 -> 该用例会被“丢弃”。
  * 尝试 B（谨慎）：选 s 为 2 的幂（s = 1/2）使每个值保持二进分数 -> 往返检查
    通过 -> 一个有效、无损的测试用例。

为保持自包含，我们用一对简化的 1 参数“曲面”、控制点线性运动（式 2 精神），
但图 4 的每个概念步骤都在。仅用 Python 内置 `fractions.Fraction`（精确有理数）。

运行:  python3 inverse_construction_demo.py
"""

from fractions import Fraction as F


# --- 二进分数辅助：二进分数（p/2^k, |p|<2^53）恰好是 double
def is_dyadic(x: F) -> bool:
    d = x.denominator
    return (d & (d - 1)) == 0                       # 分母是 2 的幂


def round_trip_ok(x: F) -> bool:
    """论文的最后兜底：有理 -> double -> 有理 必须恒等。"""
    return F(float(x)) == x and abs(x.numerator) < (1 << 53)


def dyadic(numerator: int, k: int) -> F:
    return F(numerator, 1 << k)


def curve_point(a, va, b, vb, s: F, t: F) -> F:
    """P(s,t) = (1-s)(a + t*va) + s(b + t*vb)，控制点线性运动。"""
    return (1 - s) * (a + t * va) + s * (b + t * vb)


def solve_b(a, va, vb, s, t, x):
    """步骤 II：求解控制点 b 使 P(s,t)=x。除数是 s。"""
    return (x - (1 - s) * (a + t * va)) / s - t * vb


def construct(s1, s2, label):
    print("=" * 72)
    print(f"{label}：接触参数  s1 = {s1},  s2 = {s2}")
    print("=" * 72)

    # 步骤 I：规定碰撞（全为二进分数）
    t_star = dyadic(1, 1)        # t* = 1/2
    x_star = dyadic(3, 2)        # 接触坐标 = 3/4
    print(f"步骤 I  规定:  t* = {t_star}, 接触 x* = {x_star}")

    # 自由自由度（二进分数）。速度使两曲线相向（步骤 IV）。
    a1, va1, vb1 = dyadic(1, 0), dyadic(1, 1), dyadic(-1, 1)
    a2, va2, vb2 = dyadic(2, 0), dyadic(-1, 1), dyadic(1, 2)

    # 步骤 II：反向求解（精确有理）
    b1 = solve_b(a1, va1, vb1, s1, t_star, x_star)
    b2 = solve_b(a2, va2, vb2, s2, t_star, x_star)
    print(f"步骤 II 求解:   b1 = {b1},  b2 = {b2}")

    # 验证规定的接触“精确”成立
    p1 = curve_point(a1, va1, b1, vb1, s1, t_star)
    p2 = curve_point(a2, va2, b2, vb2, s2, t_star)
    assert p1 == x_star == p2, "构造未能命中规定的接触"
    print(f"验证  P1={p1}, P2={p2}  （精确接触确认）")

    # 二进分数 + 往返检查（步骤 VI 兜底）
    vals = [t_star, x_star, s1, s2, a1, b1, va1, vb1, a2, b2, va2, vb2, p1, p2]
    dyad = all(is_dyadic(v) for v in vals)
    rt = all(round_trip_ok(v) for v in vals)
    print(f"二进分数性质?   {dyad}")
    print(f"往返检查?       {rt}")
    if dyad and rt:
        print("结果：接受 —— 可无损地用 float64 表示为输入。\n")
    else:
        bad = [str(v) for v in vals if not is_dyadic(v)]
        print(f"结果：丢弃 —— 存在非二进分数值: {bad}")
        print("        （把它们转成 double 会把截断误差注入“输入”，\n"
              "         污染鲁棒性度量。）\n")
    return dyad and rt, (t_star, x_star, a1, va1, b1, vb1, a2, va2, b2, vb2, s1, s2)


def main():
    # 尝试 A：朴素接触参数 3/4 -> 破坏二进分数性
    okA, _ = construct(s1=dyadic(3, 2), s2=dyadic(3, 2), label="尝试 A（朴素）")

    # 尝试 B：谨慎的 2 的幂接触参数 1/2 -> 保持二进分数
    okB, cfg = construct(s1=dyadic(1, 1), s2=dyadic(1, 1), label="尝试 B（谨慎）")

    assert not okA and okB, "demo 预期不符"

    # 对被接受的用例做步骤 VI：构造 GT=1 与匹配的 GT=0 用例
    (t_star, x_star, a1, va1, b1, vb1, a2, va2, b2, vb2, s1, s2) = cfg
    print("=" * 72)
    print("步骤 VI  真值标签（对“被接受”的用例）")
    print("=" * 72)
    print(f"  GT = 1（碰撞）:    用完整步长查询  DT = {t_star}")
    eps_T = dyadic(1, 6)         # 1/64
    dt_neg = t_star - eps_T
    g1 = curve_point(a1, va1, b1, vb1, s1, dt_neg)
    g2 = curve_point(a2, va2, b2, vb2, s2, dt_neg)
    print(f"  GT = 0（无碰撞）:  把步长缩为 DT = {dt_neg}（= t* - {eps_T}）")
    print(f"     在 DT={dt_neg}: P1={g1}, P2={g2}  ->  重合? {g1 == g2}")

    print("\n要点：反演 CCD 把不可行的正向真值变成“精确线性求解”。但求解的除数")
    print("（接触参数 s）会破坏二进分数性——故输入必须选得使结果保持 p/2^k，")
    print("并用往返检查兜底（论文报告 0% 丢弃率）。")


if __name__ == "__main__":
    main()
