"""
高阶 Flux Reconstruction (FR) 1D 微分 + 公共通量校正 demo —— KPM-FR §3.3
========================================================================

对照论文式 (7)(8)(9):
  单元内:   ∂U/∂t = -(2/L) D F,           D_ij = ∂ℓ_j(r_i)/∂r
  加界面:   ∂U/∂t = -(2/L)(Q F + C F*),   Q = D - C R

本 demo 用纯 NumPy:
  1. 在 K 个 Gauss-Lobatto 点上构造 Lagrange 微分矩阵 D;
  2. 构造外插矩阵 R、校正矩阵 C(用 Huynh 的 g2 校正函数), 得 Q = D - C R;
  3. 验证高阶 FR 微分的精度: 对多项式精确、对光滑函数高阶收敛;
  4. 展示公共通量 F* 如何把不连续的单元通量校正为全局连续。

只依赖 numpy:  python fr_1d_differentiation_demo.py
"""

import numpy as np


# ----------------------------------------------------------------------
# Gauss-Lobatto 点 (含端点 ±1) —— KPM-FR 采用
# ----------------------------------------------------------------------
def gauss_lobatto(K):
    """K 个 Gauss-Lobatto-Legendre 点, 落在 [-1,1] 且含 ±1。"""
    if K == 2:
        return np.array([-1.0, 1.0])
    # GLL 内点是 P'_{K-1} 的根; 用 Newton 迭代
    x = np.cos(np.pi * np.arange(K) / (K - 1))          # 初值 (Chebyshev)
    P = np.zeros((K, K))
    xold = 2
    while np.max(np.abs(x - xold)) > 1e-15:
        xold = x.copy()
        P[:, 0] = 1
        P[:, 1] = x
        for k in range(2, K):
            P[:, k] = ((2*k-1)*x*P[:, k-1] - (k-1)*P[:, k-2]) / k
        x = xold - (x*P[:, K-1] - P[:, K-2]) / (K*P[:, K-1])
    return np.sort(x)


# ----------------------------------------------------------------------
# Lagrange 基与微分矩阵 D
# ----------------------------------------------------------------------
def lagrange_diff_matrix(nodes):
    """D_ij = ℓ_j'(r_i), 标准 Lagrange 微分矩阵。"""
    K = len(nodes)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                num = 1.0
                for m in range(K):
                    if m != j and m != i:
                        num *= (nodes[i] - nodes[m])
                den = 1.0
                for m in range(K):
                    if m != j:
                        den *= (nodes[j] - nodes[m])
                D[i, j] = num / den
        # 对角元 = -sum(非对角)  (行和为 0, 因常数导数为 0)
        D[i, i] = -np.sum([D[i, k] for k in range(K) if k != i])
    return D


def lagrange_eval(nodes, j, r):
    """第 j 个 Lagrange 基在 r 处取值 ℓ_j(r)。"""
    K = len(nodes)
    val = 1.0
    for m in range(K):
        if m != j:
            val *= (r - nodes[m]) / (nodes[j] - nodes[m])
    return val


# ----------------------------------------------------------------------
# Huynh g2 校正函数及其导数 (K = P+1)
# ----------------------------------------------------------------------
def g2_derivative_at(nodes):
    """左边界校正函数 g_LB=g2 在各 SP 处的导数。
    对 Gauss-Lobatto, 论文给出稀疏结构:
      g'_LB(r_1) = -K(K-1)/2,  g'_LB(r_i)=0 (i>=2)
    右边界由对称 g_RB(r)=g_LB(-r) => g'_RB(r_K)= K(K-1)/2, 其余 0。
    """
    K = len(nodes)
    gL = np.zeros(K)
    gR = np.zeros(K)
    gL[0] = -K*(K-1)/2.0            # 只有最左 SP 非零
    gR[-1] = K*(K-1)/2.0           # 只有最右 SP 非零 (对称)
    return gL, gR


# ----------------------------------------------------------------------
# 组装 R, C, Q
# ----------------------------------------------------------------------
def build_operators(nodes):
    K = len(nodes)
    D = lagrange_diff_matrix(nodes)
    # R: 2×K 外插到边界 ±1
    R = np.zeros((2, K))
    for j in range(K):
        R[0, j] = lagrange_eval(nodes, j, -1.0)   # 左边界
        R[1, j] = lagrange_eval(nodes, j, +1.0)   # 右边界
    # C: K×2 校正矩阵 (g'_LB, g'_RB 在各 SP)
    gL, gR = g2_derivative_at(nodes)
    C = np.column_stack([gL, gR])
    Q = D - C @ R
    return D, R, C, Q


# ----------------------------------------------------------------------
# 演示
# ----------------------------------------------------------------------
def main():
    print("=" * 68)
    print("高阶 FR 1D 微分 demo (KPM-FR §3.3)")
    print("=" * 68)

    for K in [3, 4, 5]:
        nodes = gauss_lobatto(K)
        D, R, C, Q = build_operators(nodes)
        print(f"\n----- K={K} solution points (度 P={K-1}), Gauss-Lobatto -----")
        print(f"  SP 坐标: {np.round(nodes,4)}")
        print(f"  C 稀疏结构(非零元个数 {np.count_nonzero(C)}/{C.size}): "
              f"g'_LB(r1)={C[0,0]:.1f}, g'_RB(rK)={C[-1,1]:.1f}  (其余 0 ✓)")

        # 精度检验: 微分矩阵对度 <=K-1 多项式精确
        # 取 f(r)=r^(K-1), 精确导数 (K-1) r^(K-2)
        p = K - 1
        f = nodes ** p
        df_exact = p * nodes ** (p-1) if p >= 1 else np.zeros(K)
        df_num = D @ f
        err = np.max(np.abs(df_num - df_exact))
        print(f"  对 r^{p} 求导误差(应≈0, 高阶精确): {err:.2e}")

    # 高阶收敛: 对 sin(pi r) 在单个单元上求导, 看误差随 K 下降
    print("\n----- 高阶收敛: d/dr sin(π r) 在 [-1,1] 单元 -----")
    print(f"  {'K':>3} | {'max 误差':>12}")
    for K in range(3, 9):
        nodes = gauss_lobatto(K)
        D, *_ = build_operators(nodes)
        f = np.sin(np.pi * nodes)
        df_exact = np.pi * np.cos(np.pi * nodes)
        err = np.max(np.abs(D @ f - df_exact))
        print(f"  {K:>3} | {err:>12.2e}")
    print("  —— 误差随 K 快速下降, 体现谱类高阶精度(低数值耗散的根源)。")

    # 公共通量校正演示
    print("\n----- 公共通量 F* 把不连续单元通量校正为全局连续 -----")
    K = 4
    nodes = gauss_lobatto(K)
    D, R, C, Q = build_operators(nodes)
    # 造一个单元通量 (在边界与"公共通量"不一致)
    F = np.array([1.0, 1.2, 1.5, 2.0])          # 单元内 K 个 SP 上的通量
    F_left_bnd, F_right_bnd = R @ F              # 外插到边界
    Fstar = np.array([1.1, 1.9])                # 界面公共通量 (Riemann/迎风给出)
    print(f"  单元通量外插到边界: 左={F_left_bnd:.3f}, 右={F_right_bnd:.3f}")
    print(f"  公共通量 F*:         左={Fstar[0]:.3f}, 右={Fstar[1]:.3f}")
    # 半离散贡献 (式 8): dU/dt = -(2/L)(Q F + C F*)
    L = 1.0
    dUdt = -(2/L) * (Q @ F + C @ Fstar)
    print(f"  ∂U/∂t = -(2/L)(QF + CF*) = {np.round(dUdt,4)}")
    print("  C 只在边界 SP 非零 → 公共通量校正被'注入'到单元边界, 保证全局连续。")


if __name__ == "__main__":
    main()
