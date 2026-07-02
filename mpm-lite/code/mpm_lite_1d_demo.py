#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
MPM Lite 完整 Unload -> Integrate(显式) -> Load 循环教学 demo（1D）
============================================================================

目的：在一个极小的 1D 网格上，用纯 numpy 跑通 MPM Lite 一个时间步的三阶段，
      直观展示两件事：

  (1) 粒子（particle, p）只是"信使"：它们携带 (x, v, F)，在 Unload 阶段把状态
      卸载到固定的"胞心求积点"（cell center, c）和"网格节点"（grid node, i），
      在 Load 阶段再取回结果。整个 Integrate（力装配 + 时间推进）只在
      胞心与节点之间进行，全程不访问粒子。

  (2) 用的是紧致的"线性核"：
        - 粒子 <-> 胞心：以胞心为中心的帐篷形线性核 N_c(x_p)。
        - 胞心 <-> 节点：常数权重 w_ic = 2^{-d}（1D 即 0.5），纯 gather，无写竞争。

1D 设定（d=1）：
  - 节点 i 位于 x = 0, dx, 2dx, ...
  - 胞心 c 位于相邻两节点中点 x = (i+0.5) dx，每个胞两个角节点 i 和 i+1。
  - 1D 的"变形梯度"F 是标量（拉伸），Kirchhoff 应力 tau 也是标量。

材料：1D StVK-Hencky 的标量版。
  能量      psi(F)   = 0.5 * k * (log F)^2          （k 为 1D 弹性模量）
  Kirchhoff tau(F)   = k * log F                     （= F * d psi/dF）
  反演      F_base   = exp(tau / k)                  （旋转无关，1D 无旋转，直接得 F）

运行：  python3 mpm_lite_1d_demo.py
依赖：  仅 numpy
============================================================================
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)


# ---------------------------------------------------------------------------
# 1D StVK-Hencky 标量本构
# ---------------------------------------------------------------------------
def stress_1d(F, k):
    """1D Kirchhoff 应力 tau = k * log(F)。"""
    return k * np.log(F)


def reconstruct_Fbase_1d(tau, k):
    """从应力反演 1D 旋转无关拉伸参考 F_base = exp(tau / k)，使 tau(F_base)=tau。"""
    return np.exp(tau / k)


# ---------------------------------------------------------------------------
# 线性（帐篷）核：节点 i 在 x_i，宽度 dx。N_i(x) 在 |x - x_i| < dx 内线性，否则 0。
# 返回 (权重, 梯度 dN/dx)。胞心与粒子都用它来与"某个位置"通信。
# ---------------------------------------------------------------------------
def tent(x, x_node, dx):
    r = (x - x_node) / dx
    a = abs(r)
    if a >= 1.0:
        return 0.0, 0.0
    w = 1.0 - a
    dw = -np.sign(r) / dx
    return w, dw


def main():
    # =====================================================================
    # 0. 初始化：一个极小的 1D 网格 + 几个粒子
    # =====================================================================
    dx = 1.0
    n_nodes = 5                                   # 节点 0..4，位置 0,1,2,3,4
    node_x = np.arange(n_nodes) * dx
    n_cells = n_nodes - 1                          # 胞 0..3
    cell_x = (np.arange(n_cells) + 0.5) * dx       # 胞心 0.5,1.5,2.5,3.5
    cell_nodes = [(i, i + 1) for i in range(n_cells)]  # 每胞的两个角节点

    dt = 0.05
    k = 100.0                                      # 1D 弹性模量
    gravity = -2.0                                 # 沿 1D 轴的重力（演示外力）

    # 粒子：放在 [第一个胞心, 最后一个胞心] = [0.5, 3.5] 之间（PPC=3）。
    # 之所以限制在胞心之间，是为了让每个粒子对各胞心的线性核权重之和=1（单位分解），
    # 这样守恒律 sum_c V_c tau_c = sum_p V_p tau_p 才严格成立（否则边界丢权重）。
    # 关键：粒子只存 (x, v, F, mass, vol)。注意没有任何"网格"信息常驻在粒子上。
    ppc = 3
    px = np.linspace(cell_x[0], cell_x[-1], n_cells * ppc)
    n_p = len(px)
    pv = np.zeros(n_p)                              # 初速度 0
    pF = np.ones(n_p)                               # 初始 F=1（无变形）
    # 给一部分粒子一个初始压缩/拉伸，制造非零应力，便于观察力的产生
    pF[:] = 1.0 + 0.1 * np.sin(px)                  # 轻微非均匀变形
    p_mass = np.full(n_p, 1.0)
    p_vol = np.full(n_p, dx / ppc)                  # 每粒子代表的静止体积

    print("=" * 76)
    print("MPM Lite 1D 完整循环 demo：Unload -> Integrate(显式) -> Load")
    print("=" * 76)
    print(f"网格: {n_nodes} 节点, {n_cells} 胞, dx={dx}; 粒子: {n_p} 个 (PPC={ppc}); dt={dt}, k={k}")
    print(f"\n[步首] 粒子状态（仅 x, v, F 常驻在粒子上）:")
    for j in range(n_p):
        print(f"  p{j}: x={px[j]:.3f}  v={pv[j]:.3f}  F={pF[j]:.4f}  tau={stress_1d(pF[j],k):+.3f}")

    # =====================================================================
    # 1. UNLOAD: 粒子 -> 胞心 -> 节点
    # =====================================================================
    # ---- 1a. 粒子 -> 胞心 (公式 1,2,6,7 的 1D 标量版) ----
    m_c = np.zeros(n_cells)
    mv_c = np.zeros(n_cells)
    V_c = np.zeros(n_cells)
    Vtau_c = np.zeros(n_cells)                      # 广延 Kirchhoff 应力含量 V*tau

    for j in range(n_p):
        tau_p = stress_1d(pF[j], k)                 # 就地算粒子应力（不存储在粒子上）
        for c in range(n_cells):
            w, _ = tent(px[j], cell_x[c], dx)       # 线性核 w_cp = N_c(x_p)
            if w == 0.0:
                continue
            m_c[c] += w * p_mass[j]                          # 公式 1
            mv_c[c] += w * p_mass[j] * pv[j]                 # 公式 2（1D 无仿射项简化）
            V_c[c] += w * p_vol[j]                           # 公式 6（广延体积）
            Vtau_c[c] += w * p_vol[j] * tau_p                # 公式 7 分子（广延应力含量）

    active_c = m_c > 1e-12
    v_c = np.zeros(n_cells)
    tau_c = np.zeros(n_cells)
    v_c[active_c] = mv_c[active_c] / m_c[active_c]
    tau_c[active_c] = Vtau_c[active_c] / V_c[active_c]       # 公式 7：tau_c = sum w V tau / V_c

    # 守恒检查：sum_c V_c tau_c 应等于 sum_p V_p tau_p（论文核心守恒律）
    total_p = sum(p_vol[j] * stress_1d(pF[j], k) for j in range(n_p))
    total_c = np.sum(V_c * tau_c)
    print(f"\n[Unload 1a] 粒子 -> 胞心")
    print(f"  胞心应力 tau_c   = {tau_c}")
    print(f"  胞心体积 V_c     = {V_c}")
    print(f"  守恒检查: sum_p V_p tau_p = {total_p:.6f},  sum_c V_c tau_c = {total_c:.6f}")
    print(f"           偏差 = {abs(total_p-total_c):.2e}  (广延量 V*tau 重采样守恒)")

    # ---- 1b. 胞心 -> 节点 (公式 8,9 的 1D 版)，常数权重 w_ic = 2^{-d} = 0.5 ----
    w_ic = 0.5                                       # 1D: 2^{-1}
    m_i = np.zeros(n_nodes)
    mv_i = np.zeros(n_nodes)
    for c in range(n_cells):
        if not active_c[c]:
            continue
        for i in cell_nodes[c]:                      # gather: 每个角节点收集
            m_i[i] += w_ic * m_c[c]                   # 公式 8
            mv_i[i] += w_ic * m_c[c] * v_c[c]         # 公式 9（1D 简化，略去 G_c 项）
    active_i = m_i > 1e-12
    v_i = np.zeros(n_nodes)
    v_i[active_i] = mv_i[active_i] / m_i[active_i]
    print(f"\n[Unload 1b] 胞心 -> 节点 (常数权重 w_ic=0.5, 纯 gather 无写竞争)")
    print(f"  节点质量 m_i = {m_i}")
    print(f"  节点速度 v_i = {v_i}")

    # =====================================================================
    # 2. INTEGRATE（显式）：纯在网格上做，全程不访问粒子！(公式 12)
    #    f_i = - sum_c V_c tau_c * grad(w_ic),  v_i^{n+1} = v_i + dt f_i / m_i
    # =====================================================================
    print(f"\n[Integrate] 显式力装配 —— 注意：此阶段只用胞心(c)和节点(i)，不碰任何粒子！")
    f_i = np.zeros(n_nodes)
    for c in range(n_cells):
        if not active_c[c]:
            continue
        for i in cell_nodes[c]:
            # 1D 线性核在胞心处的梯度：节点 i 对胞 c 的形函数梯度
            # 左节点 grad = -1/dx, 右节点 grad = +1/dx（标准 1D 线性元）
            grad = (cell_x[c] - node_x[i])
            grad = -1.0 / dx if node_x[i] < cell_x[c] else +1.0 / dx
            f_i[i] += -V_c[c] * tau_c[c] * grad      # 公式 12
    # 加上重力外力
    f_i[active_i] += m_i[active_i] * gravity

    v_i_new = v_i.copy()
    v_i_new[active_i] = v_i[active_i] + dt * f_i[active_i] / m_i[active_i]
    print(f"  节点内力 f_i        = {f_i}")
    print(f"  更新后节点速度 v_i' = {v_i_new}")

    # =====================================================================
    # 3. LOAD: 节点 -> 胞心 -> 粒子 (公式 10,11,4,5)
    # =====================================================================
    # ---- 3a. 节点 -> 胞心：速度与速度梯度 G_c ----
    v_c_new = np.zeros(n_cells)
    G_c_new = np.zeros(n_cells)                      # 1D 速度梯度（标量 = dv/dx）
    for c in range(n_cells):
        if not active_c[c]:
            continue
        for i in cell_nodes[c]:
            v_c_new[c] += w_ic * v_i_new[i]          # 公式 10
            grad = -1.0 / dx if node_x[i] < cell_x[c] else +1.0 / dx
            G_c_new[c] += v_i_new[i] * grad          # 公式 11: G_c = sum v_i (x) grad w_ic

    # ---- 3b. 胞心 -> 粒子：插值 v_p, G_p；再平流 x_p、更新 F_p ----
    print(f"\n[Load] 节点 -> 胞心 -> 粒子, 然后平流位置、更新 F")
    for j in range(n_p):
        vp_new = 0.0
        Gp_new = 0.0
        wsum = 0.0
        for c in range(n_cells):
            w, _ = tent(px[j], cell_x[c], dx)
            if w == 0.0:
                continue
            vp_new += w * v_c_new[c]                 # 公式 4
            Gp_new += w * G_c_new[c]                 # 公式 5
            wsum += w
        if wsum > 0:
            vp_new /= wsum                           # 归一（边界胞权重不足时）
            Gp_new /= wsum
        pv[j] = vp_new
        # 更新变形梯度 F_p^{n+1} = (1 + dt G_p) F_p^n（1D 标量版）
        pF[j] = (1.0 + dt * Gp_new) * pF[j]
        # 平流位置 x_p^{n+1} = x_p + dt v_p
        px[j] = px[j] + dt * pv[j]

    print(f"\n[步末] 粒子状态（已通过纯网格积分更新）:")
    for j in range(n_p):
        print(f"  p{j}: x={px[j]:.3f}  v={pv[j]:+.4f}  F={pF[j]:.4f}  tau={stress_1d(pF[j],k):+.3f}")

    print("\n" + "=" * 76)
    print("要点回顾：")
    print("  1) 粒子只在 Unload(卸载)和 Load(装载)时被访问；Integrate 全程只动网格。")
    print("  2) 重采样的是广延量 V*tau（守恒检查偏差≈0），而非平均 F。")
    print("  3) 全部传输用线性核：粒子<->胞心用帐篷核，胞心<->节点用常数权重 0.5。")
    print("  4) 把 PPC 从 3 改大，Integrate 阶段的计算量不变 —— 求解成本与 PPC 解耦。")
    print("=" * 76)


if __name__ == "__main__":
    main()
