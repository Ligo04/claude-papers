#!/usr/bin/env python3
"""
join_union_symbolic_demo.py
===========================================================================
一个自包含、零依赖的概念演示,讲清 YASPS(Tang et al., SIGGRAPH 2026)核心的两个思想:

    1. UNION 把混合参数化的 **组合爆炸** 坍缩成 **单个** 符号节点
       (一个 "kernel" 而非 k^arity 个)。
    2. 符号微分与 JOIN 和 UNION **可交换**,因为两者都是 **线性** 算子
       (收集 / 带标签选择)。所以 JOIN/UNION 的 Hessian 就是子节点 Hessian 的
       JOIN/UNION,混合参数化的能量永远不会分裂成多个 kernel。

我们还用精确数字复现第 10.3 节 / 图 23 的 24x24 -> 15x15 -> 6x6 的 PSD 投影尺寸缩减,
并展示为何投影更小的矩阵才是真正的性能杠杆(EVD 是 O(n^3))。

这 **不是** YASPS。它是一个 ~教学模型~:一个极小的符号图,含三类节点
(自由位置、仿射体位置、一个点-点能量)和一个玩具链式法则,镜像论文的结构。运行:

    python3 join_union_symbolic_demo.py

不需要 numpy(只用标准库),所以到处都能跑。
===========================================================================
"""

from __future__ import annotations
import itertools
import math


# ---------------------------------------------------------------------------
# 第 0 部分:各参数化的 DoF 预算(来自论文)
# ---------------------------------------------------------------------------
# 自由(FREE)顶点直接存其 3D 位置                          -> 3 DoF
# 仿射体(ABD)顶点由 p = A r + t 计算位置,
#   其中 A in R^{3x3},t in R^{3}                          -> 12 DoF
PARAM_DOF = {"free": 3, "abd": 12}


# ===========================================================================
# 第 1 部分:组合爆炸,以及 UNION 如何消除它
# ===========================================================================
def naive_kernel_count(arity: int, parameterizations) -> int:
    """手写 / 按参数化分支的系统,需要为参与的 `arity` 个顶点的 **每种** 参数化组合
    准备 **一个** 专门的导数+装配 kernel。即 k ** arity。"""
    return len(parameterizations) ** arity


def yasps_kernel_count(arity: int, parameterizations) -> int:
    """YASPS 用单个 UNION 节点表示备选项。微分与 UNION 可交换,所以混合能量保持
    **一个** 符号节点,编译成 **一个** kernel,与 arity 或参数化数量无关。"""
    return 1


def demo_explosion():
    print("=" * 75)
    print("第 1 部分  组合爆炸:朴素情况枚举 vs UNION")
    print("=" * 75)
    params = ["free", "abd", "cage"]   # 3 种参数化,如论文所示
    print(f"  可用参数化: {params}  (k = {len(params)})\n")
    print(f"  {'能量 (arity)':<26}{'朴素 kernel 数':>16}{'YASPS kernel 数':>16}")
    print("  " + "-" * 58)
    for name, arity in [("point-point   (2 顶点)", 2),
                        ("point-edge    (3 顶点)", 3),
                        ("point-triangle(4 顶点)", 4),
                        ("edge-edge     (4 顶点)", 4)]:
        naive = naive_kernel_count(arity, params)
        yasps = yasps_kernel_count(arity, params)
        print(f"  {name:<28}{naive:>12}{yasps:>16}")
    print()
    print("  论文头条例子:点-三角,3 种参数化 -> 3^4 = 81")
    print("  朴素系统 81 种情况;UNION 让它恰好为 1。\n")


# ===========================================================================
# 第 2 部分:一个微型符号引擎 —— JOIN / UNION 的微分可交换
# ===========================================================================
# 我们把 "导数形状" 表示为每实例的 (行, 列) 块尺寸。我们想说明的点纯粹是结构性的,
# 所以我们追踪形状/结构而非完整的数值张量。

class Node:
    """符号节点基类。`dof` 是该节点的 Hessian 在 *扩展* DoF 空间中占据的尺寸。"""
    def __init__(self, name, dof):
        self.name = name
        self.dof = dof

    def hessian_dof(self):
        raise NotImplementedError


class FreePosition(Node):
    """直接参数化的顶点位置:p in R^3,3 DoF。线性映射。"""
    def __init__(self, name="free_pos"):
        super().__init__(name, dof=PARAM_DOF["free"])
        self.linear = True

    def hessian_dof(self):
        return self.dof


class AffinePosition(Node):
    """p = A r + t,参数 (A, t) -> 12 DoF。在 (A, t) 上同样是线性映射。"""
    def __init__(self, name="abd_pos"):
        super().__init__(name, dof=PARAM_DOF["abd"])
        self.linear = True

    def hessian_dof(self):
        return self.dof


class Union(Node):
    """同一个 3D 量的可选参数化的 UNION。

    关键事实(论文第 5.2 节):微分与 UNION 可交换。为合并参数块尺寸不同的子节点,
    YASPS 把每个 **填充** 到最大尺寸 -> 符号 Hessian 按 **最坏情况(最大)尺寸** 构建。"""
    def __init__(self, children, name="union_pos"):
        self.children = children
        # 填充到各分支中最大的参数维:
        dof = max(c.dof for c in children)
        super().__init__(name, dof=dof)
        # UNION 是(带标签的)选择 -> 一个线性算子
        self.linear = True

    def hessian_dof(self):
        # 保守 / 最大尺寸块(若不做压缩,这就是被投影的)。例如两个 ABD 点 -> 24x24。
        return self.dof


def differentiate_commutes(node) -> str:
    """符号化地展示规则:UNION 的 d/dx 是子节点导数的 UNION;JOIN 的 d/dx 是子节点
    导数的 JOIN。两者都不分裂成多个图。"""
    if isinstance(node, Union):
        inner = ", ".join(differentiate_commutes(c) for c in node.children)
        return f"UNION(d/dx[{inner}])"          # 一个节点,而非 len(children) 个
    return f"d/dx[{node.name}]"


def demo_commute():
    print("=" * 75)
    print("第 2 部分  微分与 UNION 可交换(无图分裂)")
    print("=" * 75)
    union = Union([FreePosition(), AffinePosition()], name="vertex_position")
    print(f"  前向:           UNION(free_pos, abd_pos)")
    print(f"  求导:           {differentiate_commutes(union)}")
    print("  -> 仍是 **单个** 符号节点。再求导一次它仍是一个节点。")
    print("     这就是为何编译期 kernel 数是 O(1),而非 O(2^k)。\n")


# ===========================================================================
# 第 3 部分:PSD 投影尺寸缩减(第 10.3 节 / 图 23)
# ===========================================================================
# 两个顶点间的点-点势垒,每个顶点要么自由(3)要么 ABD(12)。
# UNION 保守地按最大尺寸(24x24)构建符号 Hessian。
# 两个优化把尺寸收回:
#   (a) 块压缩:运行时一旦知道实际碰撞类型,丢掉失活 UNION 分支的结构零行/列。
#   (b) 降维投影:因为自由和 ABD 都是线性的,二阶链式法则的曲率项消失,所以可以投影
#       小的内层 Hessian(能量依赖两个 3D 点 -> 6x6)。

def projection_sizes(param_a: str, param_b: str):
    """返回 (无优化, 块压缩, 降维) 的矩阵维度。"""
    no_opt = 2 * PARAM_DOF["abd"]                       # 总是最大尺寸:24
    block_compressed = PARAM_DOF[param_a] + PARAM_DOF[param_b]  # 实际 DoF 之和
    reduced_space = 2 * 3                               # 两个 3D 点:6(仅线性时)
    return no_opt, block_compressed, reduced_space


def evd_cost(n: int) -> float:
    """特征分解是 O(n^3)。我们报告相对成本 n^3(无量纲)。"""
    return float(n) ** 3


def demo_projection():
    print("=" * 75)
    print("第 3 部分  PSD 投影尺寸缩减(点-点势垒,自由+ABD)")
    print("=" * 75)
    print("  PSD 投影的 EVD(特征分解)是矩阵尺寸的 O(n^3),")
    print("  所以缩小被投影的矩阵就是 **那个** 性能杠杆。\n")
    print(f"  {'碰撞类型':<14}{'无优化':>8}{'块压缩':>11}{'降维':>9}"
          f"{'  EVD 加速 (n^3)':>20}")
    print("  " + "-" * 64)
    for pa, pb in itertools.product(["free", "abd"], repeat=2):
        no_opt, blk, red = projection_sizes(pa, pb)
        speedup = evd_cost(no_opt) / evd_cost(red)
        print(f"  {pa+'-'+pb:<16}{no_opt:>4}x{no_opt:<1}"
              f"{blk:>8}x{blk:<1}{red:>6}x{red:<1}{speedup:>16.1f}x")
    print()
    print("  最坏情况(无优化 24x24 -> 降维 6x6):(24/6)^3 = 64x")
    print("  更少的 EVD 浮点运算。论文实测端到端约 44x")
    print("  (图 23:各变体归一化 1857x / 449.9x / 42.4x / 1.0x),")
    print("  差异来自 n^3 模型之外的真实硬件效应。\n")


# ===========================================================================
# 第 4 部分:(附加)点-点势垒能量本身,数值求值
# ===========================================================================
def point_point_barrier(p0, p1, d_hat, kappa):
    """E(p0,p1) = kappa (d - d_hat)^2 [log(d / d_hat)]^2,  d = ||p1 - p0||^2
    (论文式 8)。当 d -> 0 时势垒 -> +inf,防止互穿。"""
    d = sum((a - b) ** 2 for a, b in zip(p1, p0))   # 平方距离
    if d >= d_hat:        # 在激活距离之外:无接触能量
        return 0.0
    return kappa * (d - d_hat) ** 2 * (math.log(d / d_hat)) ** 2


def demo_barrier():
    print("=" * 75)
    print("第 4 部分  点-点势垒能量(论文式 8),求值")
    print("=" * 75)
    d_hat, kappa = 1.0e-2, 1.0e3
    print(f"  d_hat(激活距离^2) = {d_hat},kappa(刚度) = {kappa}\n")
    print(f"  {'距离':>10}{'势垒能量':>18}")
    print("  " + "-" * 28)
    for dist in [0.20, 0.12, 0.08, 0.05, 0.02]:
        p0, p1 = (0.0, 0.0, 0.0), (dist, 0.0, 0.0)
        e = point_point_barrier(p0, p1, d_hat, kappa)
        print(f"  {dist:>10.3f}{e:>18.4f}")
    print("\n  能量在两点逼近到 sqrt(d_hat) ~= 0.1 之前为 0,然后")
    print("  急剧上升 -> 这个 '势垒' 让 IPC 不发生互穿。\n")


# ===========================================================================
if __name__ == "__main__":
    print("\nYASPS 概念演示:JOIN/UNION、组合爆炸、以及投影杠杆\n")
    demo_explosion()
    demo_commute()
    demo_projection()
    demo_barrier()
    print("=" * 75)
    print("结论:过早按参数化分支 -> 指数级 kernel。")
    print("UNION 把选择推迟到一个可微节点(编译期 O(1)),")
    print("而同一份结构告诉你如何缩小那个 O(n^3) 的 PSD 投影。")
    print("=" * 75)
