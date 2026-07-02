"""
spd_1d_toy.py
=============
论文 Section 3.2 "A Toy Example" 的可运行复现。

目标：在一个单自由度系统上，对比传统 PD 与 SPD 控制器。
我们要让位置 q 收敛到目标 q_bar，并展示：
  - 传统 PD 在大增益 + 大步长下震荡/发散；
  - SPD 在任意大增益 + 任意步长下保持稳定，甚至两步精确收敛。

核心区别（全部代码就这一处不同）：
  PD :  tau = -kp*(q   - q_bar) - kd*qdot
  SPD:  控制力用"下一帧"状态，泰勒展开后解出加速度：
        qddot = -[kp*(q + dt*qdot - q_bar) + kd*qdot] / (m + kd*dt)
        注意分母里的 (m + kd*dt) —— 这就是稳定性的全部魔法。

只依赖标准库，直接 `python3 spd_1d_toy.py` 运行。
"""

m = 1.0          # 质量
q_bar = 0.0      # 目标位置
q0, qdot0 = 1.0, 0.0   # 初始位置/速度（初始偏离目标 1.0）


def simulate_pd(kp, kd, dt, steps):
    """传统 PD：控制力用当前状态 q_n, qdot_n（标准前向欧拉）。"""
    q, qdot = q0, qdot0
    traj = [q]
    for _ in range(steps):
        tau = -kp * (q - q_bar) - kd * qdot
        qddot = tau / m
        q_new = q + dt * qdot          # 位置用旧速度（前向欧拉）
        qdot = qdot + dt * qddot       # 速度更新
        q = q_new
        traj.append(q)
        if abs(q) > 1e6:               # 发散检测
            traj.append(float('inf'))
            break
    return traj


def simulate_spd(kp, kd, dt, steps):
    """
    SPD：控制力用下一帧状态，泰勒展开后解出加速度。
    1D 闭式解（论文式 8）：
        qddot = -[kp*(q + dt*qdot - q_bar) + kd*qdot] / (m + kd*dt)
    """
    q, qdot = q0, qdot0
    traj = [q]
    for _ in range(steps):
        # ★ 唯一的核心区别：分母 (m + kd*dt)，分子用预测的下一帧位置 (q + dt*qdot)
        qddot = -(kp * (q + dt * qdot - q_bar) + kd * qdot) / (m + kd * dt)
        q_new = q + dt * qdot          # 前向欧拉位置
        qdot = qdot + dt * qddot       # 速度更新
        q = q_new
        traj.append(q)
        if abs(q) > 1e6:
            traj.append(float('inf'))
            break
    return traj


def fmt(traj, n=8):
    out = []
    for v in traj[:n]:
        if v == float('inf'):
            out.append("   DIVERGED")
        else:
            out.append(f"{v:10.4f}")
    return " ".join(out)


def main():
    print("=" * 78)
    print("1D Toy Example: PD vs SPD  (目标 q_bar=0, 初始 q0=1.0)")
    print("=" * 78)

    dt = 1.0 / 30.0   # 大时间步长 33ms（论文卖点）

    print(f"\n时间步长 dt = {dt*1000:.1f} ms\n")

    for kp in [30, 300, 3000, 30000]:
        kd = kp * dt          # 满足稳定条件 kd = kp*dt (alpha=1, 收敛最快)
        print(f"--- kp = {kp:>6}, kd = kp*dt = {kd:.3f} ---")
        pd = simulate_pd(kp, kd, dt, 20)
        spd = simulate_spd(kp, kd, dt, 20)
        print(f"  PD : {fmt(pd)}")
        print(f"  SPD: {fmt(spd)}")
        print()

    print("=" * 78)
    print("论文极端情形：kp -> 无穷大, kd = kp*dt  →  SPD 应两步精确收敛到 q_bar")
    print("=" * 78)
    kp = 1e15
    kd = kp * dt
    spd = simulate_spd(kp, kd, dt, 5)
    print(f"  SPD (kp=1e15): {fmt(spd, 6)}")
    print("  → q2 已收敛到目标 0（论文证明：显式积分从加速度到位移最少两步）")
    print()
    pd = simulate_pd(kp, kd, dt, 5)
    print(f"  PD  (kp=1e15): {fmt(pd, 6)}")
    print("  → 传统 PD 在如此大增益下立刻发散。")


if __name__ == "__main__":
    main()
