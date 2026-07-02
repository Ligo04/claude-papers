"""
FLASH 核心数值机制的可运行概念实现
====================================

论文: FLASH: Fast Learning via GPU-Accelerated Simulation for
      High-Fidelity Deformable Manipulation in Minutes (arXiv:2604.17513)
      项目页: https://siyuanluo.com/flash

本 demo 用纯 Python 标准库(无 numpy/torch)复刻 FLASH 的两个数值核心:

  (A) 投影动力学 local-global 求解器(§III,式 1-4):
      用一根悬挂的质点-弹簧链(1D 可变形体)演示隐式欧拉 + local 投影 +
      global 稀疏线性解的迭代,让它在重力下下垂并收敛到稳定形态。

  (B) FLASH 的关键技巧 —— 惯性主导近似带来的稀疏性(§IV-B,式 10):
      对比 Schur 补 Z = J A⁻¹ Jᵀ + E(全隐式,A⁻¹ 稠密) vs
                Z ≈ J M⁻¹ Jᵀ + E(惯性近似,M⁻¹ 稀疏对角),
      统计非零元数量,直观展示"为什么近似让系统保持稀疏"。

  (C) 多环境块对角拼装(§IV-B-2,式 11-12)的线性扩展:
      把 n 个环境块对角拼成一个大系统,展示计算量随环境数线性增长。

注意: 这是教学骨架。真实 FLASH 在 GPU 上对 FEM 三角网格 + 摩擦接触
(Signorini-Coulomb)做这套运算,并用 A⁻¹=SᵀS 稀疏分解化为稀疏矩阵乘。
这里用 1D 弹簧链 + 朴素稠密线代代替,重点在揭示 local-global 结构与
"惯性近似 → 稀疏 → 块对角 → 线性扩展"这条主线。

运行:  python3 projective_dynamics_local_global_demo.py
"""

import math
import time


# =============================================================================
# 朴素线代工具(纯 Python,够用即可)
# =============================================================================
def matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def mat_inv(A):
    """高斯-约当求逆(仅用于演示对比稀疏性,真实 FLASH 不显式稠密求逆)。"""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        if abs(d) < 1e-12:
            d = 1e-12
        M[col] = [v / d for v in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [row[n:] for row in M]


def count_nonzeros(A, tol=1e-9):
    return sum(1 for row in A for v in row if abs(v) > tol)


# =============================================================================
# (A) 投影动力学 local-global:悬挂质点-弹簧链
# =============================================================================
# 状态: 每个质点的 1D 竖直位置 q[i](顶端 q[0] 固定)。
# 弹性能: 相邻质点间的弹簧,rest length = L0,刚度 w。
# 隐式欧拉(式1): q = argmin 1/(2h²)||M^½(q-q̃)||² + Σ ψ_i(q)
# PD(式3): local 步求每根弹簧的投影目标 p_i(把当前长度投影回 rest length);
#          global 步(式4): (M/h² + Σ w G_iᵀG_i) q = M/h² q̃ + Σ w G_iᵀ p_i

def build_chain(n, L0=1.0):
    """n 个质点的链,初始竖直等间距悬挂。"""
    return [-i * L0 for i in range(n)]   # q[0]=0 顶端, 向下为负


def pd_local_global_step(q, v, h, mass, stiff, L0, g=-9.8, iters=10, fixed=(0,)):
    """一步隐式欧拉 + local-global 迭代。返回新 q, v。"""
    n = len(q)
    # 预测态 q̃ = q + h v + h² M⁻¹ f_ext(重力)
    q_pred = [q[i] + h * v[i] + h * h * g for i in range(n)]
    # 全局系统矩阵 A = M/h² + Σ w G_iᵀG_i (这里 1D,弹簧 i 连接 i,i+1)
    inv_h2 = 1.0 / (h * h)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] += mass * inv_h2
    for i in range(n - 1):       # 弹簧 i: 质点 i 与 i+1
        A[i][i] += stiff
        A[i + 1][i + 1] += stiff
        A[i][i + 1] -= stiff
        A[i + 1][i] -= stiff
    # 固定点: 行置为单位(狄利克雷边界)
    for f in fixed:
        A[f] = [1.0 if j == f else 0.0 for j in range(n)]
    A_inv = mat_inv(A)

    qk = q_pred[:]
    for _ in range(iters):
        # ---- local 步: 每根弹簧把当前向量投影回 rest length ----
        # 投影目标 p_i 给出"理想"的相邻位移 = ±L0(保持弹簧自然长度)
        rhs = [mass * inv_h2 * q_pred[i] for i in range(n)]
        for i in range(n - 1):
            d = qk[i] - qk[i + 1]
            # 投影: 期望长度为 L0(方向保持)
            dirn = 1.0 if d >= 0 else -1.0
            p = dirn * L0
            # global RHS 累加 w G_iᵀ p
            rhs[i] += stiff * p
            rhs[i + 1] -= stiff * p
        for f in fixed:
            rhs[f] = q[f]   # 固定点保持原位
        # ---- global 步: 解 A q = rhs ----
        qk = matvec(A_inv, rhs)
    v_new = [(qk[i] - q[i]) / h for i in range(n)]
    return qk, v_new


def demo_A():
    print("=" * 68)
    print("(A) 投影动力学 local-global:悬挂弹簧链在重力下下垂收敛")
    print("=" * 68)
    n = 6
    q = build_chain(n)
    v = [0.0] * n
    h, mass, stiff, L0 = 0.02, 1.0, 200.0, 1.0
    print(f"链长 n={n}, 顶端固定, 时间步 h={h}, 刚度 w={stiff}")
    print(f"{'step':>5} | 末端高度 q[n-1] | 总弹簧能(偏离 rest 的平方和)")
    for step in range(0, 81):
        q, v = pd_local_global_step(q, v, h, mass, stiff, L0, iters=8)
        if step % 20 == 0:
            energy = sum((abs(q[i] - q[i + 1]) - L0) ** 2 for i in range(n - 1))
            print(f"{step:>5} | {q[-1]:>14.4f} | {energy:>12.5f}")
    print("  → 链条下垂后弹簧能趋于稳定(local-global 迭代收敛到准静态形态)\n")


# =============================================================================
# (B) 惯性主导近似的稀疏性对比(式 10)
# =============================================================================
def demo_B():
    print("=" * 68)
    print("(B) FLASH 关键技巧:惯性近似让 Schur 补保持稀疏(式 10)")
    print("=" * 68)
    n = 8
    # 构造一个链的系统矩阵 A = M + 弹簧耦合(三对角, 稀疏)
    mass, stiff = 1.0, 50.0
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = mass
    for i in range(n - 1):
        A[i][i] += stiff; A[i+1][i+1] += stiff
        A[i][i+1] -= stiff; A[i+1][i] -= stiff
    M = [[mass if i == j else 0.0 for j in range(n)] for i in range(n)]

    # 接触 Jacobian J: 假设 m 个接触约束(每个约束牵涉相邻两点)
    m = n - 1
    J = [[0.0] * n for _ in range(m)]
    for c in range(m):
        J[c][c] = 1.0; J[c][c + 1] = -1.0
    E = [[0.0] * m for _ in range(m)]   # 接触柔度项(此处取 0)

    def schur(Minv_like):
        # Z = J (Minv_like) Jᵀ + E
        JT = [[J[r][c] for r in range(m)] for c in range(n)]
        MJt = [[sum(Minv_like[i][k] * JT[k][c] for k in range(n)) for c in range(m)] for i in range(n)]
        Z = [[sum(J[r][i] * MJt[i][c] for i in range(n)) + E[r][c] for c in range(m)] for r in range(m)]
        return Z

    A_inv = mat_inv(A)                 # 全隐式: A⁻¹ 一般稠密
    M_inv = [[1.0/mass if i == j else 0.0 for j in range(n)] for i in range(n)]  # 惯性: M⁻¹ 对角稀疏

    Z_full = schur(A_inv)              # Z = J A⁻¹ Jᵀ + E  (稠密)
    Z_approx = schur(M_inv)            # Z ≈ J M⁻¹ Jᵀ + E  (稀疏)

    print(f"系统规模 n={n} 自由度, m={m} 接触约束")
    print(f"  A⁻¹ 非零元 (全隐式接触度量) : {count_nonzeros(A_inv):>3} / {n*n}  ← 稠密")
    print(f"  M⁻¹ 非零元 (惯性主导度量)   : {count_nonzeros(M_inv):>3} / {n*n}  ← 对角稀疏")
    print(f"  Schur 补 Z=JA⁻¹Jᵀ+E  非零元 : {count_nonzeros(Z_full):>3} / {m*m}  ← 稠密(瓶颈)")
    print(f"  Schur 补 Z≈JM⁻¹Jᵀ+E  非零元 : {count_nonzeros(Z_approx):>3} / {m*m}  ← 稀疏(三对角)")
    print("  → 惯性近似把稠密 Schur 补变回稀疏带状,这是多环境块对角并行的前提。\n")


# =============================================================================
# (C) 多环境块对角拼装的线性扩展(式 11-12)
# =============================================================================
def demo_C():
    print("=" * 68)
    print("(C) 多环境块对角拼装:计算量随环境数线性增长(式 11-12)")
    print("=" * 68)
    base_n = 5

    def block_diag_solve(n_envs):
        """把 n_envs 个 base_n 链块对角拼成一个大系统并解一次(模拟一步 global)。"""
        N = base_n * n_envs
        Ā = [[0.0] * N for _ in range(N)]
        for e in range(n_envs):
            off = e * base_n
            for i in range(base_n):
                Ā[off+i][off+i] = 1.0 + (2.0 if 0 < i < base_n-1 else 1.0)
                if i < base_n - 1:
                    Ā[off+i][off+i+1] = -1.0; Ā[off+i+1][off+i] = -1.0
        b = [1.0] * N
        t0 = time.perf_counter()
        x = matvec(mat_inv(Ā), b)   # 朴素:稠密求逆(真实 FLASH 用稀疏因子)
        return time.perf_counter() - t0

    print(f"  每个环境 = {base_n} 自由度的链;块对角拼装(无额外跨环境耦合)")
    print(f"{'#envs':>6} | {'总自由度':>8} | {'求解墙钟(ms)':>14}")
    prev = None
    for ne in (1, 2, 4, 8):
        dt = block_diag_solve(ne) * 1000
        ratio = f"  (×{dt/prev:.1f})" if prev else ""
        print(f"{ne:>6} | {base_n*ne:>8} | {dt:>14.3f}{ratio}")
        prev = dt
    print("  注:朴素稠密求逆是 O(N³),增长快于线性;FLASH 用块对角稀疏结构")
    print("      使每个环境独立、无额外耦合,计算量随环境数【线性】增长。\n")


def main():
    print("\nFLASH 数值核心 demo —— arXiv:2604.17513\n")
    demo_A()
    demo_B()
    demo_C()
    print("=" * 68)
    print("主线回顾(贯穿全文的工程主线 = 保持稀疏):")
    print("  1. 投影动力学 local-global:local 并行投影 + global 稀疏线性解。")
    print("  2. 惯性近似 Z≈JM⁻¹Jᵀ+E:把稠密 Schur 补变稀疏 → GPU 友好。")
    print("  3. 稀疏 → 块对角拼多环境 → 计算量随环境数线性增长 → 大规模并行学习。")
    print("  真实 FLASH: FEM 三角网格 + 摩擦接触, A⁻¹=SᵀS 稀疏分解, 全程 GPU。")
    print("=" * 68)


if __name__ == "__main__":
    main()
