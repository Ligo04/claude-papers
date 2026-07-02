#!/usr/bin/env python3
"""
wrench_estimation_demo.py
=========================
演示 IMPACT 的"虚拟力传感器"：仅从关节力矩残差恢复末端执行器处的
6-DOF 外部力旋量 —— 无需腕部力/力矩传感器（论文 2.2 节，式 1）。

此处复现的管线：
    1. 一个随机的 6-DOF 外部力旋量 w_ext 作用于末端执行器。
    2. 它在关节产生力矩         tau_ext = J^T @ w_ext       （物理）
    3. 我们测得带噪关节力矩     tau_meas = tau_ext + 噪声
    4. 通过正则化最小二乘逆恢复力旋量：
           w_hat = (J J^T + lambda^2 I)^-1 J tau_res        （式 1）
    5. 将 w_hat 与真值 w_ext 比较。

我们还扫描正则化权重 lambda 和机械臂的条件数，以说明为何 lambda 重要：
接近运动学奇异时 J J^T 病态，无正则化的逆会爆炸；lambda^2 I 项让估计
保持理智（Tikhonov / 阻尼最小二乘）。

运行：
    python3 wrench_estimation_demo.py
依赖：numpy（仅此）。
"""

import numpy as np

# 6-DOF 力旋量，7-DOF 机械臂（Franka FR3 有 7 个关节）-> J 是 6x7。
N_JOINTS = 7
LAMBDA_PAPER = 0.02  # 论文的力旋量正则化权重


def estimate_wrench(J, tau_res, lam):
    """式 1 闭式：w_hat = (J J^T + lam^2 I)^-1 J tau_res。
    注意：我们在力旋量空间（6x6）求逆，这是论文使用的阻尼右形式。
    lam^2 I 对接近奇异的方向做阻尼。"""
    JJt = J @ J.T                      # 6x6
    reg = (lam ** 2) * np.eye(6)
    return np.linalg.solve(JJt + reg, J @ tau_res)


def make_jacobian(condition="good", rng=None):
    """返回一个 6x7 雅可比。'singular' 使两个任务方向近乎共线，从而 J J^T
    病态（模拟运动学奇异）。"""
    rng = rng or np.random.default_rng(0)
    J = rng.normal(size=(6, N_JOINTS))
    if condition == "singular":
        # 削弱机械臂沿两个轴产生力的能力：
        # 让第 0 和第 1 行近乎相同 -> 极小的奇异值。
        J[1] = J[0] + 1e-3 * rng.normal(size=N_JOINTS)
        J[2] = J[0] + 1e-3 * rng.normal(size=N_JOINTS)
    return J


def trial(condition, lam, noise=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    J = make_jacobian(condition, rng)

    # 真值外部力旋量 [Fx Fy Fz | Tx Ty Tz]
    w_ext = np.array([rng.uniform(-20, 20) for _ in range(3)]
                     + [rng.uniform(-2, 2) for _ in range(3)])

    tau_ext = J.T @ w_ext                       # 物理：tau = J^T w
    tau_meas = tau_ext + rng.normal(0, noise, size=N_JOINTS)
    tau_model = np.zeros(N_JOINTS)              # 假设内部模型完美
    tau_res = tau_meas - tau_model             # 残差（2.2 节）

    w_hat = estimate_wrench(J, tau_res, lam)
    err = np.linalg.norm(w_hat - w_ext)
    cond = np.linalg.cond(J @ J.T)
    return w_ext, w_hat, err, cond


def main():
    rng = np.random.default_rng(42)
    print("IMPACT 虚拟力传感器 —— 从关节力矩残差求力旋量\n")
    print("从 tau_res = J^T w_ext 恢复 w_ext = [Fx Fy Fz Tx Ty Tz]\n")

    # --- 1. 良态机械臂，论文的 lambda ---
    w, wh, err, cond = trial("good", LAMBDA_PAPER, rng=np.random.default_rng(1))
    print("[良态雅可比，lambda = 0.02]")
    print("  真值  w_ext :", np.array2string(w,  precision=2, suppress_small=True))
    print("  估计  w_hat :", np.array2string(wh, precision=2, suppress_small=True))
    print(f"  ||误差|| = {err:.3f}   cond(JJ^T) = {cond:.1f}\n")

    # --- 2. 接近奇异机械臂：lambda 扫描展示正则化的收益 ---
    print("[接近奇异雅可比 —— 扫描 lambda]")
    print(f"  此处 cond(JJ^T) 巨大；观察 lambda 如何驯服估计。\n")
    print(f"  {'lambda':>8} | {'||误差||':>10} | 说明")
    print("  " + "-" * 44)
    for lam in [0.0, 1e-4, 1e-2, 0.02, 0.1, 0.5]:
        # 每次用相同几何与力旋量，只变 lambda
        r = np.random.default_rng(7)
        _, _, err, cond = trial("singular", lam, rng=r)
        note = ""
        if lam == 0.0:
            note = "无正则化 -> 估计爆炸"
        elif lam == 0.02:
            note = "<- 论文取值"
        elif lam >= 0.5:
            note = "过阻尼 -> 估计偏向 0"
        print(f"  {lam:>8.4f} | {err:>10.3f} | {note}")
    print(f"\n  （奇异情形下 cond(JJ^T) ~ {cond:.1e}）\n")

    print("要点：")
    print("  * 几何良好时，仅凭关节力矩即可准确恢复 6-DOF 力旋量 ——")
    print("    无需腕部力/力矩传感器。")
    print("  * 接近奇异时 J J^T 病态；lambda^2 I 项（Tikhonov 阻尼）")
    print("    正是让逆保持稳定的关键。")
    print("  * lambda 太大会过阻尼并把估计偏向零，")
    print("    所以 0.02 是刻意的甜点，而非随意选择。\n")


if __name__ == "__main__":
    main()
