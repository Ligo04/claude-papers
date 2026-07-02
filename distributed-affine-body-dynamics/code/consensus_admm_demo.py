#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consensus_admm_demo.py
====================================================================
共识 ADMM 教学 demo —— 论文《Distributed Affine Body Dynamics with
Adaptive Consensus》(SIGGRAPH 2026) 第 4.1 节的最小可运行代理。

运行方式：
    python3 consensus_admm_demo.py

本 demo 用一个**简单的二次代理问题**来镜像论文的结构，
让你在不涉及 ABD/IPC 几何细节的前提下，直观看清共识 ADMM 的三步循环、
原始/对偶残差，以及残差驱动的 rho 自适应（式 12）如何减少迭代数。

----------------------------------------------------------------------
与论文的对应关系（mapping to the paper）
----------------------------------------------------------------------
* 论文：场景被划分给 N 个 worker；每个 worker i 维护局部自由度 q_i，
  其中包含**共享物体**的副本。这里我们用 2 个 worker，每个 worker
  持有若干"物体"（标量/向量自由度），其中有几个物体是**共享的**。

* 论文式 (3) 局部目标 F_i(q_i)：本应是惯性项 + ARAP 弹性 + IPC 障碍。
  在 demo 中简化为每个物体一个二次目标 (1/2)||q - a||^2_W （W 为该物体
  的"质量/刚度"权重，a 为该物体偏好的目标值），这保留了"每个 worker
  想把共享物体拉向自己偏好"的本质冲突 —— 正是需要共识来调和的东西。

* 论文式 (5) 局部求解：min F_i(q_i) + sum_b (rho/2)||q_{i,b}-z_b+u_{i,b}||^2。
  对二次目标，这一步有闭式解（见 local_solve）。

* 论文式 (6) 共识更新 z：去中心化地把各 worker 的 (q_{i,b}+u_{i,b})
  按 rho 加权平均。注意论文强调**无需中央 worker**，每个持有 b 的
  worker 都能算出相同的 z_b。这里 z_b 用加权平均闭式解（式 6 的解）。

* 论文式 (7) 对偶更新：u += q - z。

* 论文式 (12) 残差驱动 rho 自适应：
      若 ||r_b|| > mu ||s_b||  -> rho *= tau   (原始残差大，需更强共识)
      若 ||s_b|| > mu ||r_b||  -> rho /= tau   (对偶残差大，共识太激进)
      否则不变；并钳制到 [sigma_min*rho0, sigma_max*rho0]。

* 单 worker 参考解 q*：把所有物体放在一起解的全局最优。共享物体在
  最优处必然一致。我们逐迭代报告 q^k 与 q* 的 MSE（对应论文图 5）。
----------------------------------------------------------------------
"""

import numpy as np

np.random.seed(0)

# ======================================================================
# 1. 构造代理场景
# ======================================================================
# 设有 N_BODIES 个"物体"，每个物体一个 DIM 维自由度向量（模拟 ABD 的
# 12-DoF，但这里取小维度以便观察）。其中 SHARED 列表里的物体是**共享体**，
# 被两个 worker 同时持有，必须达成共识。
N_BODIES = 8
DIM = 3
SHARED = [3, 4]          # 这两个物体被 worker 0 和 worker 1 共享
W0_BODIES = [0, 1, 2, 3, 4]   # worker 0 持有的物体（含共享体 3,4）
W1_BODIES = [3, 4, 5, 6, 7]   # worker 1 持有的物体（含共享体 3,4）

# 每个物体的"质量/刚度"权重 W（对角，正定）。对应论文里物体质量 m_b 决定
# 局部目标的曲率。这里随机生成不同量级以制造异质性。
masses = np.array([1.0, 2.0, 0.5, 10.0, 4.0, 1.0, 3.0, 0.7])
Wmat = {b: masses[b] * np.eye(DIM) for b in range(N_BODIES)}

# 每个 worker 对每个它持有的物体都有一个"偏好目标" a_{i,b}。对于**共享体**，
# 两个 worker 的偏好不同（制造冲突）—— 共识就是要在它们之间找到加权折中。
targets = {}
for b in range(N_BODIES):
    base = np.random.randn(DIM)
    if b in SHARED:
        # 共享体：两个 worker 偏好不同
        targets[(0, b)] = base + np.array([0.5, 0.0, 0.0])
        targets[(1, b)] = base + np.array([-0.5, 0.2, 0.1])
    else:
        owner = 0 if b in W0_BODIES else 1
        targets[(owner, b)] = base

WORKER_BODIES = {0: W0_BODIES, 1: W1_BODIES}

ADAPT_PERIOD = 5   # 残差驱动自适应的触发周期（迭代）


# ======================================================================
# 2. 单 worker 参考解 q*（对应论文中的非分布式参考）
# ======================================================================
def reference_solution():
    """
    全局最优：对每个物体，把所有持有它的 worker 的目标按权重合并后求极小。
    对于非共享体，只有一个 worker，解就是它的偏好；
    对共享体，全局目标为 sum_i (1/2)||q - a_{i,b}||^2_{W_b}，
    其闭式极小是各 worker 偏好的（等权重 W_b 下）平均。
    """
    q_star = {}
    for b in range(N_BODIES):
        owners = [i for i in (0, 1) if b in WORKER_BODIES[i]]
        # sum_i W_b q = sum_i W_b a_{i,b}  =>  q = mean_i a_{i,b}
        acc = np.zeros(DIM)
        for i in owners:
            acc += targets[(i, b)]
        q_star[b] = acc / len(owners)
    return q_star


Q_STAR = reference_solution()


# ======================================================================
# 3. 共识 ADMM 的三步
# ======================================================================
def local_solve(worker, q_local, z, u, rho):
    """
    论文式 (5) 局部求解（闭式解）。
    对物体 b 的局部目标 (1/2)||q-a||^2_{W_b}，加上共识惩罚
    (rho_b/2)||q - z_b + u_b||^2（仅共享体有惩罚项），
    对 q 求导置零：
        (W_b + rho_b I) q = W_b a + rho_b (z_b - u_b)
    非共享体无惩罚项，直接 q = a。
    """
    new_q = {}
    for b in WORKER_BODIES[worker]:
        a = targets[(worker, b)]
        if b in SHARED:
            A = Wmat[b] + rho[b] * np.eye(DIM)
            rhs = Wmat[b] @ a + rho[b] * (z[b] - u[(worker, b)])
            new_q[b] = np.linalg.solve(A, rhs)
        else:
            new_q[b] = a.copy()  # W_b a 的极小就是 a 本身
    return new_q


def consensus_update(q0, q1, u, rho):
    """
    论文式 (6) 去中心化共识更新。
    z_b = argmin sum_i (rho_b/2)||z - (q_{i,b}+u_{i,b})||^2
        = ( sum_i rho_b (q_{i,b}+u_{i,b}) ) / ( sum_i rho_b )
    这里两个 worker 共用同一 rho[b]，故退化为简单平均；
    若 rho 逐 worker 不同则为加权平均。论文强调每个持有 b 的 worker
    都能本地算出相同的 z_b（无需中央节点）。
    """
    z = {}
    for b in SHARED:
        num = rho[b] * (q0[b] + u[(0, b)]) + rho[b] * (q1[b] + u[(1, b)])
        den = rho[b] + rho[b]
        z[b] = num / den
    return z


def dual_update(q0, q1, z, u):
    """论文式 (7)：u_{i,b} += q_{i,b} - z_b。"""
    for b in SHARED:
        u[(0, b)] = u[(0, b)] + q0[b] - z[b]
        u[(1, b)] = u[(1, b)] + q1[b] - z[b]
    return u


def residuals(q0, q1, z, z_prev, rho):
    """
    原始残差 r_b = q_{i,b} - z_b（共识失配，论文式 9）
    对偶残差 s_b = z_b - z_b^prev（合并状态变化，论文式 10）
    返回逐物体残差与全局 L-inf 范数。
    """
    r_b, s_b = {}, {}
    r_inf, s_inf = 0.0, 0.0
    for b in SHARED:
        r0 = q0[b] - z[b]
        r1 = q1[b] - z[b]
        r_b[b] = max(np.max(np.abs(r0)), np.max(np.abs(r1)))
        s_b[b] = np.max(np.abs(rho[b] * (z[b] - z_prev[b])))  # 缩放对偶残差
        r_inf = max(r_inf, r_b[b])
        s_inf = max(s_inf, s_b[b])
    return r_b, s_b, r_inf, s_inf


def adapt_rho(rho, r_b, s_b, rho0, tau=2.0, mu=5.0,
              sigma_min=0.001, sigma_max=1000.0):
    """论文式 (12) 残差驱动逐物体 rho 自适应 + 钳制。"""
    for b in SHARED:
        if r_b[b] > mu * s_b[b]:
            rho[b] *= tau
        elif s_b[b] > mu * r_b[b]:
            rho[b] /= tau
        # 钳制到初始值附近的有界范围
        rho[b] = float(np.clip(rho[b], sigma_min * rho0[b], sigma_max * rho0[b]))
    return rho


# ======================================================================
# 4. 主循环
# ======================================================================
def mse_to_reference(q0, q1):
    """逐迭代 MSE 到 q*（对应论文图 5 的纵轴）。共享体取两副本均值。"""
    err, cnt = 0.0, 0
    seen = set()
    for i, bodies in WORKER_BODIES.items():
        for b in bodies:
            if b in SHARED:
                continue
            q = q0[b] if i == 0 else q1[b]
            err += np.sum((q - Q_STAR[b]) ** 2)
            cnt += DIM
    for b in SHARED:
        merged = 0.5 * (q0[b] + q1[b])
        err += np.sum((merged - Q_STAR[b]) ** 2)
        cnt += DIM
    return err / cnt


def run_admm(adaptive_rho, beta=1.0, max_iter=300, theta=1e-6, verbose=True,
             uniform_rho=None):
    """
    跑共识 ADMM。adaptive_rho=True 时启用式 (12)；False 为固定 rho。
    beta 对应质量感知初始化 rho0_b = beta * m_b。
    uniform_rho: 若给定一个标量，则所有共享体共用这个固定 rho（模拟"无质量
    感知、用单一全局 rho"的朴素做法）；此时 beta 仅用于自适应的钳制基准。
    返回 (收敛迭代数, 最终 MSE)。
    """
    # 质量感知初始化（论文 §4.3）：rho0_b = beta * m_b
    rho0 = {b: beta * masses[b] for b in SHARED}
    if uniform_rho is not None:
        rho = {b: float(uniform_rho) for b in SHARED}
    else:
        rho = dict(rho0)

    # 初始化 q、z、u
    q0 = {b: np.zeros(DIM) for b in W0_BODIES}
    q1 = {b: np.zeros(DIM) for b in W1_BODIES}
    z = {b: np.zeros(DIM) for b in SHARED}
    u = {(0, b): np.zeros(DIM) for b in SHARED}
    u.update({(1, b): np.zeros(DIM) for b in SHARED})

    converged_iter = max_iter
    for k in range(1, max_iter + 1):
        z_prev = {b: z[b].copy() for b in SHARED}

        # --- 式 (5) 局部求解（两个 worker 并行）---
        q0 = local_solve(0, q0, z, u, rho)
        q1 = local_solve(1, q1, z, u, rho)

        # --- 式 (6) 共识更新（去中心化）---
        z = consensus_update(q0, q1, u, rho)

        # --- 式 (7) 对偶更新 ---
        u = dual_update(q0, q1, z, u)

        # --- 残差与停止判据（论文式 8-10）---
        r_b, s_b, r_inf, s_inf = residuals(q0, q1, z, z_prev, rho)
        mse = mse_to_reference(q0, q1)

        if verbose and (k <= 12 or k % 10 == 0):
            print(f"  iter {k:3d} | r_inf={r_inf:.3e}  s_inf={s_inf:.3e}  "
                  f"MSE={mse:.3e}  rho={[round(rho[b],3) for b in SHARED]}")

        if r_inf < theta and s_inf < theta:
            converged_iter = k
            if verbose:
                print(f"  >>> 收敛于第 {k} 次迭代 (r_inf,s_inf < theta={theta})")
            break

        # --- 式 (12) 残差驱动 rho 自适应 ---
        # 注意：按 Boyd 2011 的实践，自适应**周期性**触发（而非每次迭代），
        # 否则乘性更新会让 rho 来回振荡。这里每 ADAPT_PERIOD 次迭代调整一次。
        if adaptive_rho and (k % ADAPT_PERIOD == 0):
            rho = adapt_rho(rho, r_b, s_b, rho0)

    final_mse = mse_to_reference(q0, q1)
    return converged_iter, final_mse


# ======================================================================
# 5. 演示
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("共识 ADMM 代理 demo —— 论文 §4.1 / §4.3")
    print("场景: 2 个 worker, 共享物体 =", SHARED)
    print("=" * 70)

    # 论文 §6.2 的消融固定 beta=1.0 后切换"是否启用残差驱动自适应"。
    # 但若初始 rho 本就接近最优，自适应空间有限。为清楚展示自适应的价值，
    # 我们从一个**偏离最优的初始 rho**（beta=0.05，偏小 -> 共识太弱）出发：
    #   [A] 保持该偏小的 rho 固定不变；
    #   [B] 同样初始化，但启用残差驱动自适应 —— 它会因原始残差持续偏大而
    #       自动放大 rho，把收敛拉回快车道。
    bad_beta = 0.05
    print(f"\n[A] 固定 rho (beta={bad_beta}, 偏小, 无残差驱动自适应):")
    it_fixed, mse_fixed = run_admm(adaptive_rho=False, beta=bad_beta,
                                   max_iter=400)

    print(f"\n[B] 同样 beta={bad_beta} 初始化, 但启用残差驱动自适应 (式 12):")
    it_adapt, mse_adapt = run_admm(adaptive_rho=True, beta=bad_beta,
                                   max_iter=400)

    print("\n" + "=" * 70)
    print("结果对比 (镜像论文 '残差驱动自适应减少迭代数 22%-80%'):")
    print(f"  固定 rho (偏小)         : 收敛迭代数 = {it_fixed:3d}, 最终 MSE = {mse_fixed:.3e}")
    print(f"  + 残差驱动自适应        : 收敛迭代数 = {it_adapt:3d}, 最终 MSE = {mse_adapt:.3e}")
    if it_fixed > 0:
        reduction = 100.0 * (it_fixed - it_adapt) / it_fixed
        print(f"  迭代数减少              : {reduction:.1f}%")
    print("=" * 70)

    print("\n[C] 质量感知初始化敏感性 (固定迭代上限, 比较不同 beta 的收敛迭代数):")
    for beta in [0.01, 0.1, 1.0, 10.0, 100.0]:
        it_b, mse_b = run_admm(adaptive_rho=False, beta=beta,
                               max_iter=500, verbose=False)
        print(f"  beta={beta:6.2f} -> 收敛迭代数 = {it_b:3d}, 最终 MSE = {mse_b:.3e}")
    print("\n说明: beta 过小 -> 共识惩罚过弱 -> 收敛慢 (对应论文 10kg/m^3 + beta=0.01 异常);")
    print("      beta 适中 (~1) -> 接近最优, 与论文 'beta=1 给出最低/近最低迭代数' 一致。")
