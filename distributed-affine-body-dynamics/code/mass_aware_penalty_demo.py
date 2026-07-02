#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_aware_penalty_demo.py
====================================================================
质量感知惩罚 demo —— 论文《Distributed Affine Body Dynamics with
Adaptive Consensus》(SIGGRAPH 2026) 第 4.3 节。

运行方式：
    python3 mass_aware_penalty_demo.py

目标：用玩具 Hessian 块直观说明**为什么质量感知惩罚优于刚度代理惩罚**。

----------------------------------------------------------------------
论文核心论点（§4.3）
----------------------------------------------------------------------
ADMM 惩罚参数 rho 决定收敛快慢。先前工作（Brown&Narain 2021, Daviet 2020,
Overby 2017, Fang 2019）用**刚度/曲率代理**来标定 rho，即分析局部 Hessian
对角块的特征值：

    H_{i,b} = M_{i,b} + H_ARAP_{i,b} + H_IPC_{i,b}                  (式 11)

但对 ABD 而言，lambda_max(H) **严重高估有效刚度**：
  * ARAP 的最大特征值主要来自三个**缩放模态**，在杨氏模量 1e12 时可达 1e8，
    轻易主导 lambda_max(H)。
  * IPC 是对数障碍：两体非常接近时其特征值**爆炸**，且只惩罚改变距离 d 的
    相对运动方向，曲率极大却高度**瞬态**（障碍对随距离在激活阈值 d_hat 附近
    反复激活/失活）。

因此论文**排除 ARAP 与 IPC**，只用惯性块 M_{i,b}。ABD 中 M_{i,b} 有简单谱界：
    lambda_max(M_{i,b}) <= m_b      (物体质量)
故直接用 m_b 作为干净、稳定的刚度代理：rho_{i,b} = beta * m_b。

本 demo:
  (1) 构造 H = M + H_ARAP + H_IPC，展示 lambda_max(H) 比 m_b 大几个数量级；
  (2) 在共识 ADMM 代理问题上分别用 (a) lambda_max(H) 和 (b) m_b 作为 rho，
      报告收敛迭代数，显示质量感知用远少的迭代收敛。
----------------------------------------------------------------------
"""

import numpy as np

np.random.seed(1)
DIM = 3


# ======================================================================
# 1. 构造玩具 Hessian 块，展示特征值高估
# ======================================================================
def build_hessian_block(m_b, young=1e12, ipc_active=True):
    """
    返回 (M, H_ARAP, H_IPC) 三个 DIM x DIM 对称半正定块。
    * M     : 惯性块，最大特征值 <= m_b（这里构造为 lambda_max = m_b）。
    * H_ARAP: 弹性块，缩放模态特征值 ~ young 的某个比例 (可达 1e8)。
    * H_IPC : 障碍块，接触激活时一个方向的曲率爆炸 (瞬态大特征值)。
    """
    # 惯性块：用一个特征值在 (0, m_b] 的对称矩阵
    Qm, _ = np.linalg.qr(np.random.randn(DIM, DIM))
    eig_M = np.array([m_b, 0.6 * m_b, 0.3 * m_b])
    M = Qm @ np.diag(eig_M) @ Qm.T

    # ARAP：缩放模态特征值 ~ 1e-4 * young （young=1e12 -> 1e8），符合论文
    Qa, _ = np.linalg.qr(np.random.randn(DIM, DIM))
    arap_scale = 1e-4 * young            # = 1e8，对应论文"可达 1e8"
    eig_A = np.array([arap_scale, arap_scale * 0.9, arap_scale * 0.8])
    H_ARAP = Qa @ np.diag(eig_A) @ Qa.T

    # IPC：对数障碍。接近接触时单一相对运动方向曲率爆炸（这里 ~1e10），
    # 其余方向 ~0（障碍只惩罚改变距离 d 的方向）。高度瞬态。
    if ipc_active:
        n = np.random.randn(DIM)
        n /= np.linalg.norm(n)
        ipc_curv = 1e10
        H_IPC = ipc_curv * np.outer(n, n)
    else:
        H_IPC = np.zeros((DIM, DIM))

    return M, H_ARAP, H_IPC


def report_eigen_overestimation():
    print("=" * 72)
    print("[1] 特征值高估演示（论文式 11 + §4.3）")
    print("=" * 72)
    print(f"{'m_b':>10} | {'lam_max(M)':>12} | {'lam_max(H_ARAP)':>16} | "
          f"{'lam_max(H_IPC)':>16} | {'lam_max(H total)':>16}")
    print("-" * 80)
    for m_b in [1.0, 100.0, 1e4]:
        M, HA, HI = build_hessian_block(m_b)
        H = M + HA + HI
        lam_M = np.max(np.linalg.eigvalsh(M))
        lam_A = np.max(np.linalg.eigvalsh(HA))
        lam_I = np.max(np.linalg.eigvalsh(HI))
        lam_H = np.max(np.linalg.eigvalsh(H))
        print(f"{m_b:>10.1f} | {lam_M:>12.3e} | {lam_A:>16.3e} | "
              f"{lam_I:>16.3e} | {lam_H:>16.3e}")
    print("\n观察: lambda_max(H) 比 m_b 大 ~6-10 个数量级 (被 ARAP 缩放模态 1e8 与")
    print("      IPC 障碍 1e10 主导)。用它作 rho 会让共识惩罚过强 -> 局部进展停滞")
    print("      (正是论文图 5 中 Brown&Narain 2021 几乎停滞的原因)。")
    print("      而 lambda_max(M) = m_b 是稳定、干净的代理。\n")


# ======================================================================
# 2. 共识 ADMM 代理问题（同 consensus_admm_demo 的简化结构）
# ======================================================================
def run_consensus(rho_per_body, masses, max_iter=2000, theta=1e-7):
    """
    2 worker、若干共享物体的二次共识 ADMM。
    rho_per_body: dict {b: rho}，由外部按不同策略提供。
    返回收敛所需迭代数。
    """
    SHARED = list(rho_per_body.keys())
    # 每个共享体两个 worker 偏好不同（制造需要共识的冲突）
    a0 = {b: np.random.randn(DIM) + 0.5 for b in SHARED}
    a1 = {b: np.random.randn(DIM) - 0.5 for b in SHARED}
    Wmat = {b: masses[b] * np.eye(DIM) for b in SHARED}

    q0 = {b: np.zeros(DIM) for b in SHARED}
    q1 = {b: np.zeros(DIM) for b in SHARED}
    z = {b: np.zeros(DIM) for b in SHARED}
    u0 = {b: np.zeros(DIM) for b in SHARED}
    u1 = {b: np.zeros(DIM) for b in SHARED}

    for k in range(1, max_iter + 1):
        z_prev = {b: z[b].copy() for b in SHARED}
        # 局部求解（式 5 闭式解）
        for b in SHARED:
            rho = rho_per_body[b]
            A = Wmat[b] + rho * np.eye(DIM)
            q0[b] = np.linalg.solve(A, Wmat[b] @ a0[b] + rho * (z[b] - u0[b]))
            q1[b] = np.linalg.solve(A, Wmat[b] @ a1[b] + rho * (z[b] - u1[b]))
        # 共识更新（式 6，等 rho 下为平均）
        for b in SHARED:
            z[b] = 0.5 * ((q0[b] + u0[b]) + (q1[b] + u1[b]))
        # 对偶更新（式 7）
        for b in SHARED:
            u0[b] += q0[b] - z[b]
            u1[b] += q1[b] - z[b]
        # 残差
        r_inf = max(max(np.max(np.abs(q0[b] - z[b])),
                        np.max(np.abs(q1[b] - z[b]))) for b in SHARED)
        s_inf = max(np.max(np.abs(rho_per_body[b] * (z[b] - z_prev[b])))
                    for b in SHARED)
        if r_inf < theta and s_inf < theta:
            return k
    return max_iter


# ======================================================================
# 3. 比较两种 rho 策略
# ======================================================================
def compare_strategies():
    print("=" * 72)
    print("[2] 共识 ADMM 收敛迭代数: 刚度代理 rho  vs  质量感知 rho")
    print("=" * 72)
    masses = {0: 1.0, 1: 100.0, 2: 1e4}   # 三个异质共享体

    # 策略 A: rho = lambda_max(H)（刚度代理，含 ARAP+IPC）
    rho_stiff = {}
    for b, m in masses.items():
        M, HA, HI = build_hessian_block(m)
        rho_stiff[b] = float(np.max(np.linalg.eigvalsh(M + HA + HI)))

    # 策略 B: rho = m_b（质量感知，beta=1）
    rho_mass = {b: m for b, m in masses.items()}

    print(f"\n刚度代理 rho = {[f'{rho_stiff[b]:.2e}' for b in masses]}")
    print(f"质量感知 rho = {[f'{rho_mass[b]:.2e}' for b in masses]}\n")

    it_stiff = run_consensus(dict(rho_stiff), masses)
    it_mass = run_consensus(dict(rho_mass), masses)

    print(f"  刚度代理 (lambda_max(H)) : 收敛迭代数 = {it_stiff}")
    print(f"  质量感知 (m_b)           : 收敛迭代数 = {it_mass}")
    if it_stiff > it_mass:
        print(f"\n结论: 质量感知用 {it_stiff}/{it_mass} = "
              f"{it_stiff/max(it_mass,1):.0f}x 更少的迭代收敛。")
    print("      刚度代理把 rho 设得过大 -> 局部子问题被二次惩罚主导 ->")
    print("      对原目标进展甚微 (镜像论文图 5 Brown&Narain 停滞)。")
    print("=" * 72)


if __name__ == "__main__":
    report_eigen_overestimation()
    compare_strategies()
