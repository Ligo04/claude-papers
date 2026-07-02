#!/usr/bin/env python3
"""
floating_point_error_bound_demo.py
===========================================================================
在一个具体多项式表达式上演示论文的“误差界”机制（§4.3、定理 4.1、附录 B），
并展示两个关键实现安全措施：

  (1) 定理 4.1 的界：对由 +,-,* 构成的 f：
          |f - f_hat| <= B_f * [ (1+eps)^k_f - 1 ]
      其 (B, k) 更新规则：
          o = l +/- r  ->  B_o = B_l + B_r,   k_o = max(k_l,k_r) + 1
          o = l * r    ->  B_o = B_l * B_r,   k_o = k_l + k_r + 1

  (2) 界的安全计算（附录 B）：界本身也在浮点下算，故保守地上估为
      eps' = 2*B*k*eps（把指数项线性化、×2 安全），保证 eps' >= 真界。

  (3) 用 nextafter 构造可证明保守的数 x + eps（把最后的组合朝安全方向舍入）。

我们通过在大量随机输入上把 float64 求值与“精确”有理（fractions.Fraction）求值
对比，经验性地验证界成立。

运行:  python3 floating_point_error_bound_demo.py     （仅用标准库）
"""

import math
from fractions import Fraction as F


EPS = 2.0 ** -53          # IEEE-754 双精度机器 epsilon（舍入到最近）
# 注：论文正文为双精度印 2**-52、单精度印 2**-17；标准的舍入到最近值是 2**-53
# 与 2**-24。此处用标准双精度值。界的“形式”不受影响。


class Tracked:
    """携带其定理-4.1 误差界状态 (B, k) 的浮点值。"""
    __slots__ = ("v", "B", "k")

    def __init__(self, v, B=None, k=0):
        self.v = float(v)
        self.B = float(abs(v)) if B is None else float(B)  # 叶子：B = |v|, k = 0
        self.k = k

    # o = l + r
    def __add__(self, o):
        return Tracked(self.v + o.v, self.B + o.B, max(self.k, o.k) + 1)

    # o = l - r
    def __sub__(self, o):
        return Tracked(self.v - o.v, self.B + o.B, max(self.k, o.k) + 1)

    # o = l * r
    def __mul__(self, o):
        return Tracked(self.v * o.v, self.B * o.B, self.k + o.k + 1)

    def true_bound_exact(self):
        """定理 4.1 的界 B * [ (1+eps)^k - 1 ]，用“精确”有理数计算
        （这是真正的数学界，不含其自身的舍入）。"""
        eps_f = F(1, 1 << 53)                       # eps = 2^-53 精确
        return F(self.B) * ((1 + eps_f) ** self.k - 1)

    def true_bound_naive_float(self):
        """同一个界用 float64 朴素计算——这正是附录 B 警告的：对小 k，
        (1.0+EPS) 会舍回 1.0，故整个界“坍缩为 0”、不再安全。"""
        return self.B * ((1.0 + EPS) ** self.k - 1.0)

    def safe_bound(self):
        """附录 B 的保守线性化上估 2*B*k*eps。
        绝不在浮点下形成 (1+eps)^k，从而避免上面的坍缩。"""
        return 2.0 * self.B * self.k * EPS


def conservative_add(x: float, eps: float) -> float:
    """构造 x + eps 使结果可证明 >= 精确的 x+eps（附录 B）。"""
    return math.nextafter(x + eps, math.inf)


def conservative_sub(x: float, eps: float) -> float:
    """构造 x - eps 使结果可证明 <= 精确的 x-eps（附录 B）。"""
    return math.nextafter(x - eps, -math.inf)


def expr_tracked(a, b, c):
    """一个几何风格的示例表达式，类似 Phi/Psi 的交叉项：
           f = (a - b) * (a - c) - (b * c)
       仅由 +, -, * 构成，故定理 4.1 适用。"""
    A = Tracked(a)
    B = Tracked(b)
    C = Tracked(c)
    return (A - B) * (A - C) - (B * C)


def expr_exact(a, b, c):
    fa, fb, fc = F(a), F(b), F(c)
    return (fa - fb) * (fa - fc) - (fb * fc)


def random_input(rng):
    """一个宽范围内的全精度 double。每个本身可表示，但两个 53 位值的“乘积”
    需要多达 106 位，故表达式在 float64 下确实会舍入——这正是界必须压住的。"""
    return rng.uniform(-1000.0, 1000.0)


def main():
    import random
    rng = random.Random(12345)   # 固定种子：确定性、无 Math.random 问题

    print("=" * 72)
    print("定理 4.1 误差界 vs.“精确”（有理）真值")
    print("  表达式:  f = (a-b)*(a-c) - (b*c)   [仅 +, -, *]")
    print("=" * 72)

    N = 200_000
    exact_violations = 0       # 精确定理-4.1 界 vs 实际误差
    naive_violations = 0       # 朴素浮点计算的界 vs 实际误差
    safe_violations = 0        # 附录-B 线性化界 vs 实际误差
    max_ratio_safe = 0.0
    worst = None

    for _ in range(N):
        a = random_input(rng)
        b = random_input(rng)
        c = random_input(rng)

        tr = expr_tracked(a, b, c)
        exact = expr_exact(a, b, c)              # ℚ 中的真值
        actual_err = F(tr.v) - exact            # 带符号；下面取绝对值
        ae = abs(actual_err)

        tb = tr.true_bound_exact()              # Fraction
        nb = tr.true_bound_naive_float()        # float（可能坍缩为 0）
        sb = tr.safe_bound()                    # float

        # 每个界都必须“压住”实际误差（这就是全部保证）。
        if ae > tb:
            exact_violations += 1
        if F(ae) > F(nb):
            naive_violations += 1
        if F(ae) > F(sb):
            safe_violations += 1

        if sb > 0:
            r = float(ae) / sb
            if r > max_ratio_safe:
                max_ratio_safe = r
                worst = (a, b, c, float(ae), float(tb), nb, sb)

    print(f"样本数: {N}")
    print(f"[精确定理 4.1 界]     违反: {exact_violations:>6}  （应为 0）  <- 真数学界成立")
    print(f"[朴素浮点 (1+eps)^k-1] 违反: {naive_violations:>6}  （不安全!）  <- 坍缩为 0，附录 B 的警告")
    print(f"[附录 B 安全 2*B*k*eps] 违反: {safe_violations:>6}  （应为 0）  <- 修法；始终安全")
    print(f"安全界的紧度（实际/安全）最大比值: {max_ratio_safe:.3e}  （<=1）")
    if worst:
        a, b, c, ae, tb, nb, sb = worst
        print("\n安全界最紧（最坏）的用例:")
        print(f"  a={a:.6f} b={b:.6f} c={c:.6f}")
        print(f"  实际误差          = {ae:.3e}")
        print(f"  精确定理-4.1 界   = {tb:.3e}   （成立）")
        print(f"  朴素浮点界        = {nb:.3e}   （坍缩 -> 会不安全）")
        print(f"  附录 B 安全界     = {sb:.3e}   （保守、安全）")

    print("\n" + "=" * 72)
    print("用 nextafter 构造保守数（附录 B）")
    print("=" * 72)
    x = 0.1 + 0.2          # 经典的非精确 double（0.30000000000000004）
    eps = 1e-16
    up = conservative_add(x, eps)
    dn = conservative_sub(x, eps)
    print(f"  x            = {x!r}")
    print(f"  x + eps (上) = {up!r}   （>= 精确 x+eps，用于推开包络）")
    print(f"  x - eps (下) = {dn!r}   （<= 精确 x-eps，用于安全 tau_L）")
    print(f"  上 > x ? {up > x}     下 < x ? {dn < x}   （二者都必须为 True）")

    print("\n要点：(B,k) 递推给出可证明的逐表达式误差界；算法随后测试 `theta < -eps`")
    print("并按这些界把几何推开。关键在于“界本身也被安全计算”（2*B*k*eps 上估")
    print("+ nextafter），故没有舍入能从后门把漏报偷渡回来。")


if __name__ == "__main__":
    main()
