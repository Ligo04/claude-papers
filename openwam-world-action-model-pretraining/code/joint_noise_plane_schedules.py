"""
OpenWAM 核心机制之一：联合噪声平面与去噪调度
==============================================

论文 §3.2 / §3.3 / §4.2.3，式 (3)。

一句话：因为训练时视频和动作的时间步 t_v、t_a 是**独立采样**的，训练覆盖了
整个 (t_v, t_a) 二维平面；于是任何推理调度（同步的对角线、异步的曲线）都只是
这个平面上的一条路径，始终分布内。同步推理是 (α, o) = (1, 0) 的特例，而不是
单独的代码路径。

这个 demo 做三件事：
  1. 实现式 (3) 的两族调度曲线，验证它们在 (α,o)=(1,0) 时精确还原对角线
  2. 用一个玩具联合 flow-matching 模型，对比"独立采样两个时间步"和"共用一个
     时间步"两种训练方式，在异步调度下的表现差异
  3. 画出全部调度路径（ASCII，可选 matplotlib）

运行：python3 joint_noise_plane_schedules.py
"""
import numpy as np

# ============================================================================
# 1. 式 (3)：两族调度曲线
# ============================================================================
# s = i/N 是全局进度。t = 0 是纯噪声，t = 1 是干净数据。
# 领先流被抬到对角线上方（更早到达干净数据），滞后流留在下方。

def variance_shift(s, alpha):
    """f_α(s) = αs / (1 + (α−1)s)。α > 1 时把曲线抬到对角线上方；α = 1 时是恒等。"""
    return alpha * s / (1.0 + (alpha - 1.0) * s)


def linear_offset(s, o):
    """h_o(s) = max{(s − o)/(1 − o), 0}。滞后流在全局进度超过 o 前一直停在纯噪声。"""
    if o >= 1.0:
        raise ValueError("offset o 必须 < 1")
    return np.maximum((s - o) / (1.0 - o), 0.0)


def make_schedule(N, family="sync", lead="video", param=None):
    """
    构造一条去噪调度路径 τ = {(t_v^i, t_a^i)}_{i=0..N}。

    family: "sync" | "variance_shift" | "linear_offset"
    lead:   哪条流领先（更早到达干净数据）—— "video" | "action"
    param:  variance_shift 用 α，linear_offset 用 o
    """
    s = np.linspace(0.0, 1.0, N + 1)

    if family == "sync":
        # 两流齐步走对角线
        return s.copy(), s.copy()

    if family == "variance_shift":
        # 领先流走抬高的曲线，滞后流保持在对角线上
        warped, plain = variance_shift(s, param), s.copy()
    elif family == "linear_offset":
        # 领先流保持在对角线上，滞后流被延迟
        warped, plain = s.copy(), linear_offset(s, param)
        # 注意方向：这里 warped 是领先的（=对角线），plain 是滞后的（被延迟）
        return (warped, plain) if lead == "video" else (plain, warped)
    else:
        raise ValueError(f"未知的调度族: {family}")

    return (warped, plain) if lead == "video" else (plain, warped)


# ============================================================================
# 2. 玩具联合 flow-matching 模型
# ============================================================================
# 真实 WAM 里世界流是 5B DiT、动作流是 1B ActionDiT。这里用两个耦合的线性
# 速度场代替，保留唯一重要的结构：一次前向同时预测两条流的速度，且每条流的
# 预测依赖**两个**时间步。
#
# 数据分布：动作 a 是视频隐变量 z 的一个线性函数（模拟"动作由世界状态决定"），
# 这样 action-sees-video 的信息通路才有意义。

RNG = np.random.default_rng(0)
D_V, D_A = 4, 2                       # 视频隐维度、动作维度
W_TRUE = RNG.normal(size=(D_A, D_V))  # 真实的 world → action 映射


def sample_data(n):
    """采样干净数据对 (z, a)，其中 a = W_true @ z + 小噪声。"""
    z = RNG.normal(size=(n, D_V))
    a = z @ W_TRUE.T + 0.1 * RNG.normal(size=(n, D_A))
    return z, a


class ToyJointVelocityField:
    """
    一次前向预测两条流的速度：(v_z, v_a) = model(z_t, a_t, t_v, t_a)。

    参数化刻意保留 OpenWAM 的两个关键结构：
      - 动作速度的预测依赖 z_t（即 action-sees-video 的通路）
      - 两个时间步各自作为特征进入两条流（AdaLN 注入 t_v / t_a 的简化版）
    """

    def __init__(self):
        # 特征 = [z_t (D_V), a_t (D_A), t_v, t_a] → 两条流各自的速度
        d_in = D_V + D_A + 2
        self.Wz = np.zeros((D_V, d_in))
        self.Wa = np.zeros((D_A, d_in))

    def _feat(self, z_t, a_t, t_v, t_a):
        n = z_t.shape[0]
        tv = np.full((n, 1), t_v) if np.isscalar(t_v) else t_v.reshape(n, 1)
        ta = np.full((n, 1), t_a) if np.isscalar(t_a) else t_a.reshape(n, 1)
        return np.concatenate([z_t, a_t, tv, ta], axis=1)

    def __call__(self, z_t, a_t, t_v, t_a):
        f = self._feat(z_t, a_t, t_v, t_a)
        return f @ self.Wz.T, f @ self.Wa.T

    def fit(self, independent_timesteps, steps=4000, batch=256, lr=0.02):
        """
        用式 (2) 的联合 flow-matching 目标训练（λ_v = λ_a = 1，w(·) ≡ 1）。

        independent_timesteps=True  → t_v、t_a 独立采样，覆盖整个噪声平面（论文做法）
        independent_timesteps=False → 两流共用一个 t，只覆盖对角线（对照组）
        """
        for _ in range(steps):
            z, a = sample_data(batch)
            eps_v = RNG.normal(size=z.shape)
            eps_a = RNG.normal(size=a.shape)

            if independent_timesteps:
                t_v = RNG.uniform(size=(batch, 1))
                t_a = RNG.uniform(size=(batch, 1))
            else:
                t_v = RNG.uniform(size=(batch, 1))
                t_a = t_v                      # ← 唯一的差别

            # 插值：z_t = t·z + (1−t)·ε，t=0 纯噪声、t=1 干净数据
            z_t = t_v * z + (1.0 - t_v) * eps_v
            a_t = t_a * a + (1.0 - t_a) * eps_a

            # flow matching 目标速度
            tgt_z, tgt_a = z - eps_v, a - eps_a

            f = self._feat(z_t, a_t, t_v, t_a)
            pred_z, pred_a = f @ self.Wz.T, f @ self.Wa.T

            # MSE 的解析梯度
            gz = (pred_z - tgt_z) * (2.0 / batch)
            ga = (pred_a - tgt_a) * (2.0 / batch)
            self.Wz -= lr * gz.T @ f
            self.Wa -= lr * ga.T @ f
        return self


def denoise(model, n, t_v_path, t_a_path):
    """
    沿给定调度路径做耦合 Euler 更新（论文 §3.3）：
        z_{i+1} = z_i + (t_v^{i+1} − t_v^i) · v_z
        a_{i+1} = a_i + (t_a^{i+1} − t_a^i) · v_a
    每步一次联合前向。返回最终的动作 chunk。
    """
    z = RNG.normal(size=(n, D_V))
    a = RNG.normal(size=(n, D_A))
    for i in range(len(t_v_path) - 1):
        v_z, v_a = model(z, a, t_v_path[i], t_a_path[i])
        z = z + (t_v_path[i + 1] - t_v_path[i]) * v_z
        a = a + (t_a_path[i + 1] - t_a_path[i]) * v_a
    return z, a


def action_consistency_error(z_final, a_final):
    """
    评估动作质量：生成的动作是否满足数据分布的约束 a ≈ W_true @ z。
    这是"动作是否真的读到了世界信息"的代理指标。
    """
    return float(np.mean(np.abs(a_final - z_final @ W_TRUE.T)))


# ============================================================================
# 3. ASCII 可视化
# ============================================================================

def ascii_plane(paths, size=17):
    """把若干条调度路径画在 (t_v, t_a) 平面上。左下 = 纯噪声，右上 = 干净数据。"""
    grid = [[" " for _ in range(size)] for _ in range(size)]
    # 先画对角线参考
    for k in range(size):
        grid[size - 1 - k][k] = "."
    for mark, (tv, ta) in paths.items():
        for x, y in zip(tv, ta):
            c = int(round(np.clip(x, 0, 1) * (size - 1)))
            r = size - 1 - int(round(np.clip(y, 0, 1) * (size - 1)))
            grid[r][c] = mark
    lines = ["  t_a=1 ┤" + "".join(row) for row in grid[:1]]
    lines += ["        │" + "".join(row) for row in grid[1:-1]]
    lines += ["  t_a=0 ┤" + "".join(grid[-1])]
    lines.append("        └" + "─" * size)
    lines.append("         t_v=0" + " " * (size - 12) + "t_v=1")
    return "\n".join(lines)


# ============================================================================
# 自检
# ============================================================================

def demo():
    print("=" * 74)
    print("OpenWAM 联合噪声平面与去噪调度")
    print("=" * 74)

    # --- 检查 1：(α, o) = (1, 0) 必须精确还原同步对角线 ---
    N = 10
    s = np.linspace(0, 1, N + 1)
    assert np.allclose(variance_shift(s, alpha=1.0), s), "α=1 时 f_α 必须是恒等映射"
    assert np.allclose(linear_offset(s, o=0.0), s), "o=0 时 h_o 必须是恒等映射"
    tv_sync, ta_sync = make_schedule(N, "sync")
    assert np.allclose(tv_sync, ta_sync), "同步调度两流时间步必须相同"
    print("\n[检查 1] (α,o)=(1,0) 精确还原对角线 —— 同步是异步的特例，不是单独代码路径  ✓")

    # --- 检查 2：领先流必须真的在对角线上方 ---
    tv, ta = make_schedule(N, "variance_shift", lead="video", param=8.0)
    assert np.all(tv >= ta - 1e-12), "video-lead 时视频流应不低于动作流"
    assert tv[N // 2] > ta[N // 2] + 0.05, "中途视频流应明显领先"
    tv, ta = make_schedule(N, "variance_shift", lead="action", param=8.0)
    assert np.all(ta >= tv - 1e-12), "action-lead 时动作流应不低于视频流"

    tv, ta = make_schedule(N, "linear_offset", lead="video", param=0.6)
    assert ta[0] == 0.0 and ta[int(0.5 * N)] == 0.0, "linear_offset 下滞后流应在 o 前保持纯噪声"
    assert np.all(tv >= ta - 1e-12), "video-lead 时视频流应不低于动作流"
    print("[检查 2] 两族曲线的领先/滞后方向正确，linear_offset 的滞后流在 o 前停在纯噪声  ✓")

    # --- 检查 3：所有路径都从纯噪声出发、到干净数据结束 ---
    for family, param in [("sync", None), ("variance_shift", 16.0), ("linear_offset", 0.4)]:
        for lead in ("video", "action"):
            tv, ta = make_schedule(N, family, lead, param)
            assert abs(tv[0]) < 1e-12 and abs(ta[0]) < 1e-12, f"{family} 起点必须是纯噪声"
            assert abs(tv[-1] - 1) < 1e-12 and abs(ta[-1] - 1) < 1e-12, f"{family} 终点必须是干净数据"
    print("[检查 3] 全部调度路径都从 (0,0) 走到 (1,1)  ✓")

    # --- 可视化 ---
    print("\n调度路径在联合噪声平面上的形状（. = 同步对角线参考）：\n")
    print("variance shift, α=8：")
    print(ascii_plane({
        "V": make_schedule(40, "variance_shift", "video", 8.0),
        "A": make_schedule(40, "variance_shift", "action", 8.0),
    }))
    print("\nlinear offset, o=0.5：")
    print(ascii_plane({
        "V": make_schedule(40, "linear_offset", "video", 0.5),
        "A": make_schedule(40, "linear_offset", "action", 0.5),
    }))

    # --- 检查 4：独立采样时间步为什么是枢纽 ---
    print("\n" + "=" * 74)
    print("独立采样 vs 共用时间步：异步调度下的表现")
    print("=" * 74)

    m_indep = ToyJointVelocityField().fit(independent_timesteps=True)
    m_shared = ToyJointVelocityField().fit(independent_timesteps=False)

    schedules = {
        "sync (对角线)":            ("sync", None, "video"),
        "variance_shift α=8 video": ("variance_shift", 8.0, "video"),
        "variance_shift α=8 action":("variance_shift", 8.0, "action"),
        "linear_offset o=0.6 video":("linear_offset", 0.6, "video"),
    }

    # 两个模型的误差绝对值不可直接横向比较：它们拟合的是不同的时间步分布，
    # 共用 t 的模型把全部容量投在对角线上，对角线误差自然更低。
    # 真正要看的是**离开对角线后相对自身同步基线的退化倍数**。
    errs = {}
    for name, (family, param, lead) in schedules.items():
        tv, ta = make_schedule(20, family, lead, param)
        errs[name] = (
            action_consistency_error(*denoise(m_indep, 2000, tv, ta)),
            action_consistency_error(*denoise(m_shared, 2000, tv, ta)),
        )

    e_i_sync, e_s_sync = errs["sync (对角线)"]
    print(f"\n{'调度':<28} {'独立采样':>16} {'共用 t':>16}")
    print(f"{'':<28} {'(相对自身同步)':>16} {'(相对自身同步)':>16}")
    print("-" * 74)
    for name, (e_i, e_s) in errs.items():
        r_i, r_s = e_i / e_i_sync, e_s / e_s_sync
        flag = "" if name.startswith("sync") else ("   ← 共用 t 退化更多" if r_s > r_i else "")
        print(f"{name:<28} {r_i:>15.2f}× {r_s:>15.2f}×{flag}")
    print(f"\n（同步基线绝对误差：独立采样 {e_i_sync:.4f}，共用 t {e_s_sync:.4f}；")
    print("  绝对值不可横向比较——共用 t 的模型把全部容量投在对角线上，")
    print("  对角线上更准是预期的，代价是离开对角线就没有训练信号。）")

    for name in ("variance_shift α=8 video", "linear_offset o=0.6 video"):
        e_i_async, e_s_async = errs[name]
        degrade_shared = e_s_async / max(e_s_sync, 1e-9)
        degrade_indep = e_i_async / max(e_i_sync, 1e-9)
        assert degrade_shared > degrade_indep, (
            f"共用时间步的模型在异步调度 {name} 上应比独立采样的模型退化更多"
            f"(共用 {degrade_shared:.2f}× vs 独立 {degrade_indep:.2f}×)"
        )
    print("\n[检查 4] 共用时间步训练的模型一旦离开对角线，相对退化更严重  ✓")
    print("         → 这就是论文为什么必须独立采样 t_v 和 t_a：")
    print("           只有覆盖整个噪声平面，'同步最优'才是关于机制的结论，")
    print("           而不是'异步碰上了分布外状态'的伪影。")

    # --- 论文实测结果对照 ---
    print("\n" + "=" * 74)
    print("论文 §4.2.3 实测（RoboTwin2.0-Full 成功率 %，mutual 掩码）")
    print("=" * 74)
    paper = {
        "sync":                        93.0,
        "variance_shift α=4  video":   91.8,
        "variance_shift α=8  video":   91.1,
        "variance_shift α=16 video":   91.9,
        "variance_shift α=32 video":   91.5,
        "variance_shift α=4  action":  91.0,
        "variance_shift α=8  action":  90.7,
        "variance_shift α=16 action":  89.7,
        "variance_shift α=32 action":  88.5,
        "linear_offset  o=0.2 video":  91.7,
        "linear_offset  o=0.4 video":  91.1,
        "linear_offset  o=0.6 video":  91.5,
        "linear_offset  o=0.8 video":  91.2,
        "linear_offset  o=0.2 action": 92.2,
        "linear_offset  o=0.4 action": 92.0,
        "linear_offset  o=0.6 action": 92.3,
        "linear_offset  o=0.8 action": 92.3,
    }
    best_async = max(v for k, v in paper.items() if k != "sync")
    for k, v in paper.items():
        bar = "█" * int(round((v - 88) * 4))
        tag = "  ← 最优" if k == "sync" else ""
        print(f"  {k:<28} {v:5.1f}  {bar}{tag}")
    assert paper["sync"] > best_async, "论文结论：同步优于全部 16 种异步调度"
    print(f"\n  同步 {paper['sync']:.1f} > 最好的异步 {best_async:.1f}（16 种异步全部落后）")
    print("  两族的方向偏好还相反（variance_shift 偏 video-lead，linear_offset 偏 action-lead），")
    print("  这种不一致本身说明'引领哪条流'不是一个有真实机制的旋钮。")
    print("\n  结论：支持表征学习假说（视频预测作为辅助任务塑造表征），")
    print("        反驳测试时规划假说（先想象未来再反推动作）。")
    print("        Table 1 里显式两阶段的 IDM 变体只有 87.95，是同一结论的独立证据。")

    print("\n" + "=" * 74)
    print("全部检查通过 ✓")
    print("=" * 74)


if __name__ == "__main__":
    demo()
