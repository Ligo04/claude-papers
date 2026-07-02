"""
DeformGen 概念演示：合法流形 S_real 与四种状态增强策略
=========================================================

这个脚本用一个【玩具版可变形物体】（一段受"保长约束"的链）来直观说明论文 Sec. 3.1
的核心论点：物理可行状态只占整个构型空间的一个薄薄的流形 S_real；不同增强策略
能不能留在 S_real 上，决定了它们生成的数据是否可用。

物理约束（玩具版）：相邻粒子间距离应近似等于静止长度 L0（不可拉伸/压缩太多）。
  - 满足约束 -> 物理合法（∈ S_real）
  - 违反约束 -> 物理非法（∉ S_real）

对比四种策略（对应论文 Table 1）：
  (i)   全局刚性变换      —— 一定合法，但只有 6-DoF 空间多样性
  (ii)  逐粒子加噪        —— 破坏连接性，跌出 S_real
  (iii) 运动学形变场      —— 看似连贯但忽略约束，可能跌出 S_real
  (iv)  动力学(本文)      —— 施加局部力 + 投影回约束(模拟仿真器演化)，始终在 S_real

只依赖 numpy。运行：
    python state_manifold_demo.py
"""

import numpy as np


L0 = 1.0   # 相邻粒子静止间距
N = 12     # 粒子数


def make_source_state():
    """源合法状态：一条沿 x 轴均匀排布的链（满足保长约束）。"""
    s = np.stack([np.arange(N) * L0, np.zeros(N), np.zeros(N)], axis=1)
    return s


def constraint_violation(s):
    """物理合法性度量：相邻粒子间距偏离 L0 的均方根。越接近 0 越合法。"""
    seg = np.linalg.norm(np.diff(s, axis=0), axis=1)   # 各段长度
    return np.sqrt(np.mean((seg - L0) ** 2))


def is_plausible(s, tol=0.05):
    return constraint_violation(s) < tol


# --- (i) 全局刚性变换：绕 z 轴旋转 + 平移 ----------------------------------
def aug_rigid(s, rng):
    theta = rng.uniform(-np.pi, np.pi)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    t = rng.uniform(-2, 2, size=3)
    return s @ Rz.T + t


# --- (ii) 逐粒子独立加噪 --------------------------------------------------
def aug_per_particle(s, rng, sigma=0.3):
    return s + rng.normal(0, sigma, size=s.shape)


# --- (iii) 运动学形变场：施加一个平滑的弯曲位移场（不管约束）-------------
def aug_kinematic(s, rng):
    amp = rng.uniform(0.3, 0.8)
    out = s.copy()
    out[:, 1] += amp * (s[:, 0] / (N * L0)) ** 2 * (N * L0)   # 抛物线弯曲
    return out


# --- (iv) 动力学(本文)：局部力 + 迭代投影回约束(模拟 PBD 式仿真器演化)-----
def project_constraints(s, iters=50):
    """位置约束投影(类似 Position-Based Dynamics)：反复把每段拉回 L0。
    这模拟论文里 Φ_sim 用自身动力学把状态保持在 S_real 上。"""
    s = s.copy()
    for _ in range(iters):
        for i in range(N - 1):
            d = s[i + 1] - s[i]
            dist = np.linalg.norm(d) + 1e-9
            corr = (dist - L0) * d / dist
            s[i] += 0.5 * corr        # 两端各修一半
            s[i + 1] -= 0.5 * corr
    return s


def aug_dynamics(s, rng):
    """从合法状态出发，对几个粒子施加随机局部力(位移)，再让"仿真器"投影回约束。"""
    out = s.copy()
    n_push = rng.integers(1, 4)
    for _ in range(n_push):
        idx = rng.integers(0, N)
        force = rng.normal(0, 0.6, size=3)
        force[2] = 0
        # 局部影响：以 idx 为中心高斯衰减地施加
        for j in range(N):
            w = np.exp(-((j - idx) ** 2) / 4.0)
            out[j] += w * force
    # 关键一步：用约束投影模拟仿真器演化 -> 始终回到 S_real
    return project_constraints(out)


def main():
    rng = np.random.default_rng(42)
    s0 = make_source_state()
    print(f"源状态合法性(约束违反 RMS) = {constraint_violation(s0):.4f}  -> 合法\n")

    strategies = {
        "(i)   全局刚性  ": aug_rigid,
        "(ii)  逐粒子加噪": aug_per_particle,
        "(iii) 运动学场  ": aug_kinematic,
        "(iv)  动力学(本文)": aug_dynamics,
    }

    M = 200  # 每种策略采样数
    print(f"每种策略生成 {M} 个增强状态，统计有多少留在 S_real，以及形状多样性。\n")
    print(f"{'策略':<16} | {'合法率':>8} | {'平均约束违反':>12} | {'形状多样性(非刚性)':>18}")
    print("-" * 70)

    for name, fn in strategies.items():
        states = [fn(s0, rng) for _ in range(M)]
        plausible = [is_plausible(st) for st in states]
        viol = np.mean([constraint_violation(st) for st in states])

        # 形状多样性：去掉刚性分量后(用 Procrustes 粗略对齐到 s0)的残差，
        # 模拟论文 Fig.8 的"非刚性残差"。越大说明产生了真正的形状变化。
        residuals = []
        for st in states:
            # 简单刚性对齐：去质心
            a = st - st.mean(0)
            b = s0 - s0.mean(0)
            H = b.T @ a
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            aligned = a @ R
            residuals.append(np.sqrt(np.mean((aligned - b) ** 2)))
        diversity = np.mean(residuals)

        print(f"{name:<16} | {np.mean(plausible)*100:>6.1f}% | "
              f"{viol:>12.4f} | {diversity:>18.4f}")

    print("\n解读(对照论文 Table 1 / Fig 8)：")
    print("  (i)  全局刚性  : 合法率 100%，但非刚性残差≈0 —— 只有空间多样性，无形状变化。")
    print("  (ii) 逐粒子加噪: 合法率低 —— 破坏链连接性，大量跌出 S_real。")
    print("  (iii)运动学场  : 有形状变化但约束被违反，合法率受损 —— 跌出 S_real。")
    print("  (iv) 动力学本文: 合法率高 且 非刚性残差大 —— 既留在 S_real 又有真拓扑多样性。")
    print("\n这正是 DeformGen 的核心论证：用动力学投影/演化保证合法，同时获得形状多样性。")


if __name__ == "__main__":
    main()
