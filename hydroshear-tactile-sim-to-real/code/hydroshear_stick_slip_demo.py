#!/usr/bin/env python3
"""
HydroShear 核心机制演示：递归力追踪器 + stick-slip + 剪切场汇聚
=================================================================
arXiv 2603.00446 (HydroShear) 的最小教学实现（2D 俯视 + 1D 深度）。

复现论文的三个核心组件：
  1. 递归力追踪器 F（式 5-12）：α 接触截取 → 位移累积成力 → 库仑摩擦锥截断 → 脱离重置
  2. 路径依赖性演示：同一终点位姿、不同路径 → 不同剪切场（non-holonomic 的含义）
  3. 剪切场汇聚（式 16-17）：以投影点为中心、穿透深度加权、高斯衰减（触觉阴影）

场景简化：
  - 弹性膜 = 平面 z<=0（SDF: phi_E(p) = p_z，解析、无需网格）
  - 压头   = 半径 R 的球（表面点在下半球采样），只做平移运动（不失一般性地演示路径依赖）
  - marker 网格 = 7x9（与论文 GelSight Mini 配置一致）

运行:  python3 hydroshear_stick_slip_demo.py
输出:  stick_slip_curve.png / path_dependence.png （无显示环境时仅保存文件）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境也能跑
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 参数
R = 0.0175          # 球形压头半径 17.5mm（论文用 35mm 直径球）
E = K = 1.0e4       # 法向/切向刚度（论文假设 E=K 以启用式15的一器两用）
A = 1.0e-6          # 每表面点关联面积（均匀，假设 A_j = A）
MU = 0.8            # 摩擦系数（决定 stick->slip 转换阈值）
LAM_S = 8.0e4       # 剪切场高斯衰减系数 lambda_s（触觉阴影范围）
EPS = 1e-12

# ---------------------------------------------------------------- 几何
def sphere_surface_points(n_theta=12, n_phi=24):
    """在球的下半表面采样 M 个点（压头局部坐标, 球心为原点）。
    只取下半球是因为只有朝向膜的一侧可能接触。"""
    th = np.linspace(np.pi / 2 + 0.05, np.pi, n_theta)   # 极角: 赤道以下
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    pts = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1) * R
    return pts.reshape(-1, 3)                             # (M,3)

O_LOCAL = sphere_surface_points()
# 球面法向 = 径向单位向量（指向球外）；论文用压头表面法向 n_hat_j 做法/切分解
N_LOCAL = O_LOCAL / np.linalg.norm(O_LOCAL, axis=1, keepdims=True)

phi_E = lambda p: p[..., 2]   # 平面膜 SDF：z 坐标本身（膜内为负）

# ---------------------------------------------------------------- 递归力追踪器 F（式 5-12）
def force_tracker_step(f_prev, o_now, o_prev, n_hat, E=E, K=K, A=A, mu=MU):
    """一步递归 F：输入上一帧累积力与两帧表面点位置，输出本帧累积力。
    完全向量化（所有 M 个点并行）——对应论文 GPU-parallelizable 的性质。"""
    sd_now, sd_prev = phi_E(o_now), phi_E(o_prev)

    # --- 式6: ReLU 差商 alpha，截取"在膜内发生"的那部分位移（3情形统一）
    # 注意: 论文原文分母写作 phi(o_t)-phi(o_{t-1})，但按其自述语义
    # (两帧均在膜内 => alpha=1) 验算应为 phi(o_{t-1})-phi(o_t)，此处取正确符号。
    num = np.maximum(-sd_now, 0.0) - np.maximum(-sd_prev, 0.0)
    den = sd_prev - sd_now
    alpha = np.where(np.abs(den) > EPS, num / np.where(np.abs(den) > EPS, den, 1.0),
                     (sd_now < 0).astype(float))  # 分母~0: 两帧同深度, 在膜内则全算
    d = alpha[:, None] * (o_now - o_prev)                       # 式7

    # --- 法向/切向分解（法向取压头表面法向）
    d_n = np.sum(d * n_hat, axis=1)
    d_xy = d - d_n[:, None] * n_hat

    # --- 式8-9: 位移 -> 力 增量累积（stick 阶段的剪切 build-up）
    f_n = np.sum(f_prev * n_hat, axis=1) + E * A * d_n
    f_xy = (f_prev - np.sum(f_prev * n_hat, axis=1)[:, None] * n_hat) + K * A * d_xy

    # --- 式10-11: 法向非负 + 库仑摩擦锥截断（slip: 切向力钳位在 mu*f_n）
    f_n_bar = np.maximum(f_n, 0.0)
    tan_mag = np.linalg.norm(f_xy, axis=1)
    scale = np.minimum(1.0, MU * f_n_bar / np.maximum(tan_mag, EPS))
    f_xy_bar = scale[:, None] * f_xy

    # --- 式12: 脱离接触(Heaviside) -> 力清零（下次接触重新累积）
    in_contact = (sd_now < 0).astype(float)[:, None]
    return in_contact * (f_n_bar[:, None] * n_hat + f_xy_bar), scale

def rollout(traj):
    """沿位姿轨迹 traj[(T,3) 球心位置] 运行追踪器，返回每帧累积力与投影点。
    式13-15: E=K、mu 相同、A 均匀 => 投影位移 f_hat = f_tilde/(K*A)，一器两用。"""
    f = np.zeros_like(O_LOCAL)
    frames = []
    for t in range(1, len(traj)):
        o_now, o_prev = O_LOCAL + traj[t], O_LOCAL + traj[t - 1]
        f, slip_scale = force_tracker_step(f, o_now, o_prev, N_LOCAL)
        o_hat = o_now + f / (K * A)          # 式13+15: 投影点（滑移时在膜面漂移）
        frames.append(dict(f=f.copy(), o=o_now.copy(), o_hat=o_hat.copy(),
                           slip=slip_scale.copy()))
    return frames

def shear_field(frame, grid_x, grid_y, lam_s=LAM_S):
    """式16-17: 把逐点切向力汇聚成 marker 网格上的剪切场。
    权重=穿透深度, 中心=投影点 o_hat（stick-slip 的空间体现）, 高斯衰减=触觉阴影。"""
    o, o_hat, f = frame["o"], frame["o_hat"], frame["f"]
    depth = np.maximum(-phi_E(o), 0.0)                    # -phi_E, 仅接触点>0
    f_xy = f[:, :2]
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="ij")
    q = np.stack([gx, gy], -1)                            # (H,W,2)
    v = q[None] - o_hat[:, None, None, :2]                # (M,H,W,2)
    w = depth[:, None, None] * np.exp(-lam_s * np.sum(v**2, -1))
    return np.einsum("mhw,mc->hwc", w, -f_xy)             # 注意 -f_xy: marker 被往回拖

# ---------------------------------------------------------------- 实验1: stick-slip 曲线
def press_then(offsets, n_press=20):
    """构造轨迹: 从膜上方 0.5mm 处经 n_press 步压入 1mm（满足递归基例
    "t=0 未接触"），随后按 offsets 序列做切向运动。返回 (T,3) 球心位置。"""
    z0, z1 = R + 0.0005, R - 0.001
    press = [np.array([0, 0, z0 + (z1 - z0) * t / n_press]) for t in range(n_press + 1)]
    deep = np.array([0, 0, z1])
    return np.array(press + [deep + np.array([dx, dy, 0]) for dx, dy in offsets])

def demo_stick_slip():
    """压头先压入 1mm，再匀速切向拖动：切向合力先线性累积(stick)后饱和(slip)。
    这是惩罚式(瞬时)模型原理上无法产生的曲线——它们的力只依赖当前帧。"""
    N_PRESS, N_DRAG = 20, 100
    traj = press_then([(t * 8e-5, 0.0) for t in range(1, N_DRAG + 1)], N_PRESS)
    frames = rollout(traj)
    tan = [np.linalg.norm(fr["f"][:, :2].sum(0)) for fr in frames]
    # 法向合力: 各点力在 -z(膜法向朝上支撑压头)上的合成幅值
    nor = [abs(fr["f"][:, 2].sum()) for fr in frames]
    slip_frac = [1 - fr["slip"][phi_E(fr["o"]) < 0].mean()
                 if (phi_E(fr["o"]) < 0).any() else 0.0 for fr in frames]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(tan, label="tangential |sum f_xy|")
    ax[0].plot(np.array(nor) * MU, "--", label="friction limit mu*|sum f_n|")
    ax[0].axvline(N_PRESS, color="gray", lw=0.5)
    ax[0].set(xlabel="timestep", ylabel="force [N]",
              title="Stick-slip: shear builds up, then clamped by friction cone")
    ax[0].legend()
    ax[1].plot(slip_frac, color="crimson")
    ax[1].set(xlabel="timestep", ylabel="mean slip amount (1-scale)",
              title="Slip fraction: 0 = full stick")
    fig.tight_layout()
    fig.savefig("stick_slip_curve.png", dpi=130)
    print(f"[1] stick_slip_curve.png  切向力峰值={max(tan):.4f}N, "
          f"末帧摩擦极限={MU*nor[-1]:.4f}N, 末帧滑移比例={slip_frac[-1]:.2f}")

# ---------------------------------------------------------------- 实验2: 路径依赖(non-holonomic)
def demo_path_dependence():
    """两条轨迹到达同一终点位姿（原点、压入 1mm）：
       A: 压入 -> 向右拖 2mm -> 拖回原点（有切向历史）
       B: 仅竖直压入（无切向历史）
       惩罚式(瞬时)模型对两者给出相同观测；HydroShear 的路径 A 留下剪切残余(滞回)。
       注: 路径 B 的 M^s 精确为 0 是模型的正确行为——E=K 且全程 stick 时
       总力=刚度x总位移(纯竖直), xy 分量为零; 纯压入的径向 marker 运动由
       dilation 场 M^d 负责(论文式2的分解), 不出现在 shear 场里。"""
    n = 40
    go_back = [((t / n) * 0.002, 0.0) for t in range(1, n + 1)] + \
              [(0.002 - (t / n) * 0.002, 0.0) for t in range(1, n + 1)]
    trajA = press_then(go_back)                            # 有历史
    trajB = press_then([(0.0, 0.0)] * len(go_back))        # 无历史(压住不动)

    gx = np.linspace(-0.012, 0.012, 9)                     # 9x7 marker 网格(论文配置)
    gy = np.linspace(-0.009, 0.009, 7)
    SA = shear_field(rollout(trajA)[-1], gx, gy)
    SB = shear_field(rollout(trajB)[-1], gx, gy)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    vmax = max(np.hypot(SA[..., 0], SA[..., 1]).max(), 1e-30)
    for a, S, name in [(ax[0], SA, "Path A: press, drag +2mm, drag back"),
                       (ax[1], SB, "Path B: press only (never moved)")]:
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        a.quiver(X, Y, S[..., 0], S[..., 1], np.hypot(S[..., 0], S[..., 1]),
                 cmap="viridis", clim=(0, vmax),
                 scale=vmax * 12 if vmax > 1e-20 else 1)
        a.add_patch(plt.Circle((0, 0), 0.006, fill=False, ls="--", color="red"))
        a.set(title=f"{name}\n(same final pose)", xlabel="x [m]", ylabel="y [m]", aspect=1)
    fig.suptitle("Non-holonomic: same final pose, different history -> different shear",
                 y=1.02)
    fig.tight_layout()
    fig.savefig("path_dependence.png", dpi=130, bbox_inches="tight")
    magA = np.hypot(SA[..., 0], SA[..., 1]).max()
    magB = np.hypot(SB[..., 0], SB[..., 1]).max()
    print(f"[2] path_dependence.png  剪切场峰值 路径A={magA:.3e} vs 路径B={magB:.3e} "
          f"→ 历史留下了剪切残余(滞回), 瞬时模型无法区分两者")

if __name__ == "__main__":
    demo_stick_slip()
    demo_path_dependence()
    print("完成。两张图演示了 HydroShear 相对惩罚式模型的两个本质差异: "
          "剪切累积+摩擦饱和、路径依赖。")
