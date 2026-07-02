#!/usr/bin/env python3
"""
pncg_vs_newton_tradeoff.py
===========================================================================
概念演示:为什么 PNCG-IPC "用每次迭代的精度换取迭代吞吐" 在 GPU 上更划算
(论文第 3.1 节)。

核心论点:
  - 牛顿法:每步要组装并【分解 Hessian】(O(n^3)),单步昂贵但精度高,
    迭代次数少。
  - 非线性共轭梯度(NCG / PNCG):不分解 Hessian,只需【梯度 + 对角 Hessian
    + 点积】(O(n) ~ O(n^2)),单步便宜但精度低,需要更多步。
  - 在 GPU 上,NCG 的单步成本极低且高度可并行,所以"很多便宜步"的总成本
    远低于"少数昂贵步",几十次迭代即收敛到【足够精度】(对触觉仿真足够)。

我们在一个 n 维二次能量 E(x) = ½ xᵀ A x − bᵀ x 上对比两者:
  - 收敛所需迭代次数
  - 每步的(模型)浮点成本
  - 总成本(迭代次数 × 单步成本)

这 **不是** PNCG-IPC 的实现,而是一个干净的最小示意,用来感受这个权衡。
只用标准库。运行:
    python3 pncg_vs_newton_tradeoff.py
===========================================================================
"""

from __future__ import annotations
import random
import math

random.seed(0)   # 注意:脚本环境里某些随机受限,这里仅用于构造示例矩阵


# --------------------------------------------------------------------------
# 0. 线性代数小工具(纯 Python,避免 numpy 依赖)
# --------------------------------------------------------------------------
def matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]

def dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def axpy(a, x, y):           # 返回 a*x + y
    return [a * xi + yi for xi, yi in zip(x, y)]

def scale(a, x):
    return [a * xi for xi in x]

def sub(x, y):
    return [xi - yi for xi, yi in zip(x, y)]


def make_spd(n, cond=50.0):
    """构造一个 n x n 的对称正定矩阵(模拟 IPC 能量的 Hessian),
    条件数约为 cond(条件数越大,NCG 越吃力 -> 体现为更多迭代)。"""
    # 用 A = Q diag(eig) Qᵀ 的简化版:对角占优的对称矩阵
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = (math.sin(i * 1.3 + j * 0.7)) * 0.5   # 确定性"伪随机",避免随机受限
            A[i][j] = v
            A[j][i] = v
    # 让它对角占优 + 注入指定条件数的谱
    for i in range(n):
        A[i][i] = cond * (1.0 + i / max(1, n - 1)) + n
    return A


# --------------------------------------------------------------------------
# 1. 牛顿法:每步"分解 Hessian"(这里用高斯消元解 A d = -g)
#    单步成本 ~ O(n^3)(分解),迭代次数少(二次问题 1 步即精确)
# --------------------------------------------------------------------------
def solve_linear(A, b):
    """高斯消元解 A x = b(代表牛顿步里"分解 Hessian"的 O(n^3) 成本)。"""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        # 选主元
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pivot = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pivot
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]

def newton(A, b, tol=1e-8):
    """最小化 ½xᵀAx − bᵀx。梯度 g = Ax − b。牛顿步 d = −A⁻¹g。
    二次问题理论上 1 步精确收敛。返回 (迭代次数, 单步成本模型)。"""
    n = len(A)
    x = [0.0] * n
    iters = 0
    for _ in range(50):
        g = sub(matvec(A, x), b)
        if math.sqrt(dot(g, g)) < tol:
            break
        d = solve_linear(A, scale(-1.0, g))   # 解 A d = -g(O(n^3) 分解)
        x = axpy(1.0, d, x)
        iters += 1
    cost_per_iter = n ** 3                     # 模型:Hessian 分解主导
    return iters, cost_per_iter, x


# --------------------------------------------------------------------------
# 2. 非线性共轭梯度(这里即线性 CG,因为问题是二次的)
#    单步只需 matvec + 点积(O(n^2) / 在 IPC 中可做到 ~O(n)),不分解 Hessian。
#    迭代次数多(取决于条件数),但单步极便宜、高度可并行。
# --------------------------------------------------------------------------
def conjugate_gradient(A, b, tol=1e-6, maxit=2000):
    """解 Ax = b(等价于最小化 ½xᵀAx − bᵀx)。返回 (迭代次数, 单步成本模型)。"""
    n = len(A)
    x = [0.0] * n
    r = sub(b, matvec(A, x))     # 残差 = -梯度
    p = r[:]
    rs = dot(r, r)
    iters = 0
    for _ in range(maxit):
        if math.sqrt(rs) < tol:
            break
        Ap = matvec(A, p)        # 唯一的"重"操作:一次 matvec(GPU 友好)
        alpha = rs / dot(p, Ap)
        x = axpy(alpha, p, x)
        r = axpy(-alpha, Ap, r)
        rs_new = dot(r, r)
        p = axpy(rs_new / rs, p, r)
        rs = rs_new
        iters += 1
    cost_per_iter = n ** 2       # 模型:一次 matvec 主导(IPC 中可稀疏到 ~O(n))
    return iters, cost_per_iter, x


# ==========================================================================
if __name__ == "__main__":
    print("\nPNCG vs Newton:用每步精度换迭代吞吐(论文第 3.1 节)\n")
    print("=" * 74)
    print(f"{'问题维度 n':>10}{'cond':>7} | {'Newton 步':>9}{'步成本~':>10}{'总成本':>12}"
          f" | {'CG 步':>7}{'步成本~':>10}{'总成本':>12}")
    print("-" * 74)

    for n, cond in [(8, 20), (16, 50), (32, 100), (64, 200)]:
        A = make_spd(n, cond)
        b = [math.cos(i * 0.9) for i in range(n)]   # 确定性右端项

        n_it, n_cpi, _ = newton(A, b)
        c_it, c_cpi, _ = conjugate_gradient(A, b)

        n_total = n_it * n_cpi
        c_total = c_it * c_cpi
        print(f"{n:>10}{cond:>7} | {n_it:>9}{n_cpi:>10}{n_total:>12}"
              f" | {c_it:>7}{c_cpi:>10}{c_total:>12}")

    print("=" * 74)
    print("读法:")
    print("  • Newton 迭代次数少(二次问题 1 步),但每步 ~O(n^3)(分解 Hessian)。")
    print("  • CG 迭代次数多(随条件数增长),但每步只是一次 matvec ~O(n^2),")
    print("    且 **只需梯度/对角 Hessian/点积**,无需组装或分解 Hessian。")
    print("  • 在 GPU 上,CG 的单步极便宜且高度可并行,'很多便宜步'的总成本通常")
    print("    远低于 Newton 的'少数昂贵步' —— 这就是 PNCG-IPC 几十次迭代即收敛、")
    print("    并能在多 GPU 上大规模并行的根本原因。对触觉仿真(求'视觉合理+物理")
    print("    一致'而非机器精度)而言,这个权衡极其划算。")
    print("=" * 74)
