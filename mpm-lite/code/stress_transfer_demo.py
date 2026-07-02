#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
MPM Lite 核心贡献教学 demo：应力传输 与 旋转无关拉伸重建
============================================================================

本脚本用纯 numpy 演示 MPM Lite 论文最关键的两个洞见：

(A) 为什么"平均变形梯度 F"是错的，而"重采样广延 Kirchhoff 应力含量 V·tau"是对的。
    - 变形梯度 F 既不是强度量(intensive)也不是广延量(extensive)：它把旋转和拉伸
      纠缠在一起 (F = R S，极分解)。把两个带不同旋转的粒子的 F 平均，得到的矩阵
      其极分解出的拉伸根本不对应任何真实应变 —— 旋转污染了拉伸，应力被破坏。
    - 正确做法：弱形式力项的 Riemann 和需要保持的是 Kirchhoff 应力含量 V·tau，
      它是一个不折不扣的广延量。重采样它满足守恒：sum_c V_c tau_c = sum_p V_p tau_p。

(B) StVK-Hencky 的旋转无关拉伸重建 tau -> sigma_i (论文公式 21-24)。
    - 对各向同性材料，能量/刚度只依赖拉伸 S（F 的奇异值），旋转无关。
    - 所以可以从应力 tau 闭式反演出拉伸 S，使 P(S) S^T = tau。
    - 我们验证它"往返一致"：给定 F -> 算 tau -> 重建 S -> 检查 P(S) S^T == tau，
      并检查重建的 S 与 F 的极分解拉伸一致（差异只来自被丢弃的旋转）。

运行：  python3 stress_transfer_demo.py
依赖：  仅 numpy
============================================================================
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)


# ---------------------------------------------------------------------------
# StVK-Hencky 本构（2D / 3D 通用，基于奇异值）
# 能量： psi = mu * sum_i (log sigma_i)^2 + (lambda/2) * (sum_i log sigma_i)^2   (公式 19)
# 第一 Piola 应力的谱形式（公式 20），Kirchhoff 应力 tau = P(F) F^T
# 主 Kirchhoff 应力（公式 21）： tau_i = 2 mu log sigma_i + lambda * sum_j log sigma_j
# ---------------------------------------------------------------------------
def lame_from_E_nu(E, nu):
    """由杨氏模量 E 与泊松比 nu 得到 Lame 参数 mu, lambda。"""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def kirchhoff_stress_stvk_hencky(F, mu, lam):
    """
    给定变形梯度 F，返回 Kirchhoff 应力 tau = P(F) F^T （对称矩阵）。
    用 SVD：F = U Sigma V^T，主拉伸 sigma_i，对数应变 e_i = log sigma_i。
    P(F) = U (2 mu Sigma^{-1} log Sigma + lambda tr(log Sigma) Sigma^{-1}) V^T  (公式 20)
    则 tau = P F^T = U diag(tau_i) U^T，其中 tau_i = 2 mu log sigma_i + lambda * s  (公式 21)
    """
    U, sig, Vt = np.linalg.svd(F)
    e = np.log(sig)                      # 对数主拉伸 e_i = log sigma_i
    s = np.sum(e)                        # s = sum_j log sigma_j = log J
    tau_principal = 2.0 * mu * e + lam * s   # 主 Kirchhoff 应力（公式 21）
    tau = U @ np.diag(tau_principal) @ U.T   # tau = U diag(tau_i) U^T（对称！）
    return tau


def first_piola_stvk_hencky(F, mu, lam):
    """第一 Piola 应力 P(F)（公式 20），用于验证 P(S) S^T == tau。"""
    U, sig, Vt = np.linalg.svd(F)
    log_sig = np.log(sig)
    s = np.sum(log_sig)
    inv_sig = 1.0 / sig
    diag = 2.0 * mu * inv_sig * log_sig + lam * s * inv_sig
    P = U @ np.diag(diag) @ Vt
    return P


# ---------------------------------------------------------------------------
# (B) 旋转无关拉伸重建： tau -> sigma_i -> F_base = S  （论文公式 21-24）
# ---------------------------------------------------------------------------
def reconstruct_stretch_stvk_hencky(tau, mu, lam):
    """
    给定对称 Kirchhoff 应力 tau，闭式重建旋转无关的拉伸张量 S，
    使得 P(S) S^T = tau。 （StVK-Hencky，公式 22-24）

    步骤：
      1. tau 是对称的 -> 特征分解 tau = U diag(tau_i) U^T，特征向量给主方向 U。
      2. s = tr(tau) / (2 mu + d lambda)             （公式 22）
      3. e_i = (tau_i - lambda s) / (2 mu)            （公式 23）
      4. sigma_i = exp(e_i)                           （公式 24）
      5. S = U diag(sigma_i) U^T   （对称正定，纯拉伸，无旋转）
    """
    d = tau.shape[0]
    # 1. 对称特征分解（eigh 适用于对称矩阵，返回实特征值/正交特征向量）
    tau_i, U = np.linalg.eigh(tau)
    # 2. s = log J
    s = np.trace(tau) / (2.0 * mu + d * lam)          # 公式 22
    # 3-4. 反演对数拉伸再指数化
    e_i = (tau_i - lam * s) / (2.0 * mu)              # 公式 23
    sigma_i = np.exp(e_i)                             # 公式 24（保证 sigma_i > 0）
    # 5. 组装旋转无关拉伸张量 S = U diag(sigma) U^T
    S = U @ np.diag(sigma_i) @ U.T
    return S, sigma_i


# ---------------------------------------------------------------------------
# 工具：构造一个"旋转 theta + 各向拉伸 (a, b)"的变形梯度 F = R(theta) * diag(a,b)
# ---------------------------------------------------------------------------
def make_F_2d(theta, a, b):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    Stretch = np.diag([a, b])
    return R @ Stretch    # F = R S，所以 F 的极分解拉伸恰好是 diag(a, b)


# ===========================================================================
# 演示 A：平均 F 是错的 vs 重采样 V·tau 是对的
# ===========================================================================
def demo_A_stress_transfer():
    print("=" * 76)
    print("演示 A：为什么不能平均变形梯度 F，而要重采样广延 Kirchhoff 应力 V·tau")
    print("=" * 76)

    mu, lam = lame_from_E_nu(E=5000.0, nu=0.4)
    print(f"材料 StVK-Hencky:  E=5000, nu=0.4  ->  mu={mu:.4f}, lambda={lam:.4f}\n")

    # 一个胞内有两个粒子，拉伸状态相同(a=1.3, b=0.8)，但旋转完全不同。
    # 关键设计：两个粒子的"真实应变状态"（拉伸）一样，物理上它们的应力主值一样，
    #           只是应力的主方向被各自的旋转转走了。
    a, b = 1.3, 0.8
    F1 = make_F_2d(theta=np.deg2rad(0.0),  a=a, b=b)   # 粒子1：不旋转
    F2 = make_F_2d(theta=np.deg2rad(85.0), a=a, b=b)   # 粒子2：旋转 85 度

    # 两个粒子的体积（广延量）
    V1, V2 = 1.0, 1.2
    # 线性核权重 w_cp：本 demo 只有一个胞，两个粒子都完全落在该胞内，故 w=1。
    # （多胞情形下 sum_c w_cp = 1，此处单胞即 w_cp = 1。）
    w1, w2 = 1.0, 1.0

    # 各自的真实 Kirchhoff 应力
    tau1 = kirchhoff_stress_stvk_hencky(F1, mu, lam)
    tau2 = kirchhoff_stress_stvk_hencky(F2, mu, lam)

    print("粒子1: F1 = R(0°)  diag(1.3, 0.8)")
    print("粒子2: F2 = R(85°) diag(1.3, 0.8)   <- 拉伸相同，仅旋转不同\n")

    # 真实需要被守恒的量：总 Kirchhoff 应力含量 = sum_p V_p tau_p
    total_Vtau_truth = V1 * tau1 + V2 * tau2
    print("[基准真值] 弱形式力项需要保持的总应力含量  sum_p V_p tau_p =")
    print(total_Vtau_truth, "\n")

    # 胞体积（广延量，两方案共用）： V_c = sum_p w_cp V_p
    V_c = w1 * V1 + w2 * V2

    # ----- 错误做法：体积加权平均变形梯度 F，再从平均 F 算应力 -----
    # F_avg = (sum_p w_cp V_p F_p) / V_c   （体积加权平均，错误！）
    F_avg = (w1 * V1 * F1 + w2 * V2 * F2) / V_c
    tau_from_avgF = kirchhoff_stress_stvk_hencky(F_avg, mu, lam)
    Vtau_avgF = V_c * tau_from_avgF               # 由平均F得到的胞应力含量 V_c·tau

    print("-" * 76)
    print("【错误】方案：平均 F = (sum w V F)/V_c，再算应力 tau(F_avg)")
    print("平均得到的 F_avg =")
    print(F_avg)
    Uavg, sig_avg, _ = np.linalg.svd(F_avg)
    print(f"F_avg 的奇异值（被旋转污染的'拉伸'）= {sig_avg}  <- 不再是 (1.3, 0.8)!")
    print("由平均F重算的胞应力含量 V_c·tau(F_avg) =")
    print(Vtau_avgF)
    err_avgF = np.linalg.norm(Vtau_avgF - total_Vtau_truth)
    print(f">>> 与真值的偏差 ||·|| = {err_avgF:.6f}   (应力含量被破坏，非守恒)\n")

    # ----- 正确做法：重采样广延 Kirchhoff 应力含量 V·tau（公式 6-7）-----
    # V_c = sum_p w_cp V_p ;  tau_c = sum_p w_cp (V_p tau_p) / V_c
    tau_c = (w1 * V1 * tau1 + w2 * V2 * tau2) / V_c
    Vtau_resampled = V_c * tau_c                  # 胞应力含量 = sum_p V_p tau_p（守恒）

    print("-" * 76)
    print("【正确】方案：重采样广延 Kirchhoff 应力含量（公式 6-7）")
    print("  V_c   = sum_p w_cp V_p")
    print("  tau_c = sum_p w_cp (V_p tau_p) / V_c")
    print("胞心应力 tau_c =")
    print(tau_c)
    print("胞应力含量 V_c·tau_c（应等于 sum_p V_p tau_p）=")
    print(Vtau_resampled)
    err_resample = np.linalg.norm(Vtau_resampled - total_Vtau_truth)
    print(f">>> 与真值的偏差 ||·|| = {err_resample:.2e}   (守恒，到机器精度!)\n")

    print("-" * 76)
    print("结论：")
    print(f"  平均 F 的应力偏差    = {err_avgF:.6f}   （大，非物理）")
    print(f"  重采样 V·tau 的偏差  = {err_resample:.2e}  （≈0，守恒）")
    print("  -> 旋转污染了平均 F 的拉伸；而 V·tau 是广延量，重采样天然守恒。")
    print()


# ===========================================================================
# 演示 B：旋转无关拉伸重建的往返验证  F -> tau -> S -> 检查 P(S)S^T == tau
# ===========================================================================
def demo_B_stretch_reconstruction():
    print("=" * 76)
    print("演示 B：StVK-Hencky 旋转无关拉伸重建 tau -> sigma_i 的往返一致性（公式 21-24）")
    print("=" * 76)

    mu, lam = lame_from_E_nu(E=12000.0, nu=0.3)
    print(f"材料 StVK-Hencky:  E=12000, nu=0.3  ->  mu={mu:.4f}, lambda={lam:.4f}\n")

    # ---- 2D 测试 ----
    print("-" * 76)
    print("[2D] 给定带旋转的 F = R(40°) diag(1.25, 0.7)")
    F2 = make_F_2d(theta=np.deg2rad(40.0), a=1.25, b=0.7)
    tau2 = kirchhoff_stress_stvk_hencky(F2, mu, lam)       # 正向：F -> tau
    S2, sigma2 = reconstruct_stretch_stvk_hencky(tau2, mu, lam)  # 反向：tau -> S
    # 验证 1：P(S) S^T == tau  （公式 17 的定义性质）
    P_S2 = first_piola_stvk_hencky(S2, mu, lam)
    residual2 = np.linalg.norm(P_S2 @ S2.T - tau2)
    # 验证 2：重建的拉伸 sigma 应与 F 的极分解拉伸（即 F 的奇异值）一致
    _, sig_true2, _ = np.linalg.svd(F2)
    print(f"  F 的真实奇异值（极分解拉伸）  = {np.sort(sig_true2)}")
    print(f"  从 tau 重建的拉伸 sigma       = {np.sort(sigma2)}")
    print(f"  验证 ||P(S) S^T - tau||       = {residual2:.2e}   (应 ≈ 0)")
    print(f"  拉伸恢复误差                  = {np.linalg.norm(np.sort(sigma2)-np.sort(sig_true2)):.2e}")
    print("  注意：S 丢弃了 F 的旋转 R(40°)，但弹性状态（能量/应力主值）完全保留。\n")

    # ---- 3D 测试 ----
    print("-" * 76)
    print("[3D] 给定带一般旋转的 F = R diag(1.4, 0.9, 0.75)")
    # 构造一个 3D 随机旋转
    rng = np.random.default_rng(7)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    F3 = Q @ np.diag([1.4, 0.9, 0.75])
    tau3 = kirchhoff_stress_stvk_hencky(F3, mu, lam)
    S3, sigma3 = reconstruct_stretch_stvk_hencky(tau3, mu, lam)
    P_S3 = first_piola_stvk_hencky(S3, mu, lam)
    residual3 = np.linalg.norm(P_S3 @ S3.T - tau3)
    _, sig_true3, _ = np.linalg.svd(F3)
    print(f"  F 的真实奇异值（极分解拉伸）  = {np.sort(sig_true3)}")
    print(f"  从 tau 重建的拉伸 sigma       = {np.sort(sigma3)}")
    print(f"  验证 ||P(S) S^T - tau||       = {residual3:.2e}   (应 ≈ 0)")
    print(f"  拉伸恢复误差                  = {np.linalg.norm(np.sort(sigma3)-np.sort(sig_true3)):.2e}")
    print()

    print("-" * 76)
    print("结论：tau -> sigma_i -> F_base = S 的重建严格往返一致（残差到机器精度）。")
    print("      旋转被丢弃，但各向同性材料的弹性能与刚度只依赖拉伸，故一步内无损。")
    print("      （论文：丢弃旋转的速度误差仅 O(dt^2)/步。）")
    print()


if __name__ == "__main__":
    demo_A_stress_transfer()
    demo_B_stretch_reconstruction()
    print("=" * 76)
    print("全部演示完成。核心要点：")
    print("  1) F 不可平均（旋转污染拉伸）；广延量 V·tau 才是正确的重采样对象。")
    print("  2) 各向同性材料可从 tau 闭式重建旋转无关拉伸 S，使 P(S)S^T = tau。")
    print("=" * 76)
