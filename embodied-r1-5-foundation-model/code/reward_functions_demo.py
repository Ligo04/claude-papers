#!/usr/bin/env python3
"""
reward_functions_demo.py
================================================================================
Embodied-R1.5 的**五类多任务奖励函数**（论文 §4.2.2），用 ASCII 可视化。

核心是统一的**分段线性衰减（piecewise-linear decay）**框架（论文式1）：

    φ(d; τ_p, τ_z) = clip( (τ_z - d) / (τ_z - τ_p), 0, 1 )

    d < τ_p        → 满分 1.0    （足够接近，给全额奖励）
    τ_p ≤ d < τ_z  → 线性衰减     （部分正确，给部分奖励 —— 稠密梯度信号）
    d ≥ τ_z        → 0           （太远，零奖励）

为什么这么设计？二元奖励(对/错)在 RL 里**梯度稀疏**；分段线性衰减提供**部分奖励**，
让"差一点点"和"差很多"得到不同反馈 → 稠密梯度 → RL 更好优化。

本 demo 展示：
  1. φ 衰减曲线（点距、轨迹 RMSE 用的就是它，只是阈值不同）
  2. 五类奖励家族各自的计算与示例
  3. 防 reward-hacking 的护栏（单点零分、长度/计数惩罚）
  4. 格式奖励加权：R = (1-λ)·R_acc + λ·R_fmt, λ=0.1

依赖：仅 Python 标准库。运行：python reward_functions_demo.py
================================================================================
"""

import math


def clip(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# 统一分段线性衰减（论文式1）
# ---------------------------------------------------------------------------
def phi(d, tau_p, tau_z):
    """φ(d; τ_p, τ_z) = clip((τ_z - d)/(τ_z - τ_p), 0, 1)"""
    if tau_z <= tau_p:
        raise ValueError("需要 τ_p < τ_z")
    return clip((tau_z - d) / (tau_z - tau_p), 0.0, 1.0)


def plot_phi(tau_p, tau_z, d_max, unit, title):
    """在终端用 ASCII 画出 φ 随距离 d 的衰减曲线。"""
    print(f"\n  {title}")
    print(f"  φ(d; τ_p={tau_p}, τ_z={tau_z})   [{unit}]")
    rows = 10
    cols = 56
    d_vals = [d_max * c / (cols - 1) for c in range(cols)]
    r_vals = [phi(d, tau_p, tau_z) for d in d_vals]
    for row in range(rows, -1, -1):
        thr = row / rows
        line = []
        for r in r_vals:
            line.append("█" if r >= thr - 0.5 / rows else " ")
        axis = f"{thr:0.1f}|" if row % 2 == 0 else "   |"
        print("   " + axis + "".join(line))
    print("      +" + "-" * cols)
    # x 轴刻度
    ticks = " " * 6 + f"0{'':<{cols//2-2}}d={d_max/2:.0f}{'':<{cols//2-6}}{d_max:.0f} ({unit})"
    print(ticks)
    print(f"      满分区 d<{tau_p}  |  衰减区 {tau_p}~{tau_z}  |  零分区 d≥{tau_z}")


# ---------------------------------------------------------------------------
# 五类奖励家族
# ---------------------------------------------------------------------------
def reward_exact_match(pred, gold):
    """(1) 精确匹配：选择题/数值/数学 → {0,1}。这里演示选择题灵活匹配。"""
    norm = lambda s: s.strip().upper().split(".")[0]
    return 1.0 if norm(pred) == norm(gold) else 0.0


def reward_iou(box_p, box_g):
    """(2) IoU：空间 grounding(bbox)。box=(x1,y1,x2,y2)。"""
    xa, ya = max(box_p[0], box_g[0]), max(box_p[1], box_g[1])
    xb, yb = min(box_p[2], box_g[2]), min(box_p[3], box_g[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_p = (box_p[2]-box_p[0]) * (box_p[3]-box_p[1])
    area_g = (box_g[2]-box_g[0]) * (box_g[3]-box_g[1])
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0


def reward_point_distance(pred_pts, gold_pts, tau_p=40, tau_z=150, count_penalty=0.3):
    """(3) 点距：平均最近邻距离 → φ(d_nn; 40,150)；点数不匹配减 0.3。"""
    def nn_dist(a, B):
        return min(math.dist(a, b) for b in B)
    d_nn = sum(nn_dist(p, gold_pts) for p in pred_pts) / len(pred_pts)
    r = phi(d_nn, tau_p, tau_z)
    if len(pred_pts) != len(gold_pts):
        r = max(0.0, r - count_penalty)
    return r, d_nn


def reward_trajectory(pred_traj, gold_traj, tau_p=50, tau_z=120, len_penalty=0.35):
    """(4) 轨迹 RMSE：线性插值对齐后逐点 RMSE → φ(RMSE; 50,120)。
    护栏：单点输出给 0（防 hack）；长度不匹配减 0.35。"""
    if len(pred_traj) < 2:
        return 0.0, None  # 单点输出零分，防 reward hacking
    n = max(len(pred_traj), len(gold_traj))
    P = resample(pred_traj, n)
    G = resample(gold_traj, n)
    rmse = math.sqrt(sum(math.dist(p, g) ** 2 for p, g in zip(P, G)) / n)
    r = phi(rmse, tau_p, tau_z)
    if len(pred_traj) != len(gold_traj):
        r = max(0.0, r - len_penalty)
    return r, rmse


def resample(traj, n):
    """把折线轨迹线性插值重采样到 n 个点。"""
    if len(traj) == n:
        return list(traj)
    out = []
    for i in range(n):
        t = i / (n - 1) * (len(traj) - 1)
        lo = int(math.floor(t))
        hi = min(lo + 1, len(traj) - 1)
        f = t - lo
        out.append((traj[lo][0]*(1-f) + traj[hi][0]*f,
                    traj[lo][1]*(1-f) + traj[hi][1]*f))
    return out


def reward_semantic(pred_text, gold_text):
    """(5) 语义相似度：真实用 Skywork-Reward 打分→sigmoid归一化。
    这里用词重叠 + sigmoid 模拟（真实系统用奖励模型，失败回退 BLEU）。"""
    a, b = set(pred_text.lower().split()), set(gold_text.lower().split())
    overlap = len(a & b) / max(1, len(a | b))           # Jaccard 作 raw score
    raw = (overlap - 0.5) * 6                            # 拉开后过 sigmoid
    return 1 / (1 + math.exp(-raw))


def format_combined(r_acc, fmt_ok, lam=0.1):
    """格式奖励加权（论文式2）：R = (1-λ)·R_acc + λ·R_fmt, λ=0.1。"""
    return (1 - lam) * r_acc + lam * (1.0 if fmt_ok else 0.0)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" Embodied-R1.5 · 五类多任务奖励函数 demo")
    print("=" * 70)

    # --- 统一分段线性衰减曲线 ---
    print("\n【统一框架】分段线性衰减 φ(d; τ_p, τ_z)  —— 点距与轨迹奖励的共同底座")
    plot_phi(40, 150, 200, "px", title="点距奖励所用 (τ_p=40, τ_z=150)")
    plot_phi(50, 120, 160, "px", title="2D 轨迹 RMSE 奖励所用 (τ_p=50, τ_z=120)")

    # --- 五类奖励逐一示例 ---
    print("\n" + "=" * 70)
    print(" 五类奖励家族 · 计算示例")
    print("=" * 70)

    print("\n(1) 精确匹配 (选择题/数值/数学) → {0,1}")
    for pred, gold in [("A", "A.dog"), ("B", "A.dog"), ("A.dog", "A")]:
        print(f"     pred={pred!r:<8} gold={gold!r:<8} → R={reward_exact_match(pred, gold):.0f}"
              f"  （灵活匹配：'A' 与 'A.dog' 视为一致）")

    print("\n(2) IoU (空间 grounding / bbox) → [0,1]")
    for bp in [(10, 10, 50, 50), (20, 20, 60, 60), (40, 40, 80, 80)]:
        print(f"     pred_box={bp}  gt=(10,10,50,50) → R_IoU={reward_iou(bp, (10,10,50,50)):.3f}")

    print("\n(3) 点距 (点定位) → φ(d_nn; 40,150)，点数不匹配 -0.3")
    cases = [
        ([(100, 100)], [(110, 105)], "很近"),
        ([(100, 100)], [(180, 140)], "中等距离→部分奖励"),
        ([(100, 100), (200, 200)], [(110, 105)], "预测2点/真值1点→计数惩罚"),
    ]
    for pp, gp, note in cases:
        r, d = reward_point_distance(pp, gp)
        print(f"     {note:<22} d_nn={d:6.1f}px → R={r:.3f}")

    print("\n(4) 轨迹 RMSE (2D/3D 视觉轨迹) → φ(RMSE; 50,120)，单点=0，长度不匹配 -0.35")
    gold_traj = [(0, 0), (50, 40), (100, 90), (150, 120)]
    cases = [
        ([(2, 1), (52, 42), (98, 88), (150, 121)], "几乎重合"),
        ([(0, 0), (80, 95), (150, 175), (150, 120)], "偏离→部分奖励"),
        ([(75, 60)], "单点输出→0 (防 reward hacking)"),
    ]
    for pt, note in cases:
        r, rmse = reward_trajectory(pt, gold_traj)
        rmse_s = f"{rmse:6.1f}px" if rmse is not None else "  N/A "
        print(f"     {note:<32} RMSE={rmse_s} → R={r:.3f}")

    print("\n(5) 语义相似度 (开放文本: 纠错建议/长程计划) → 奖励模型打分→[0,1]")
    gold = "regrasp the cup because the gripper missed it"
    cases = [
        ("regrasp the cup the gripper missed it", "高度相关"),
        ("move the cup to the left side slowly", "部分相关"),
        ("the weather is nice today", "无关"),
    ]
    for pred, note in cases:
        print(f"     {note:<10} R={reward_semantic(pred, gold):.3f}  | pred={pred!r}")

    # --- 格式奖励加权 ---
    print("\n" + "=" * 70)
    print(" 格式奖励加权 (论文式2)：R = (1-λ)·R_acc + λ·R_fmt,  λ=0.1")
    print("=" * 70)
    for r_acc, fmt_ok, note in [
        (0.90, True,  "答案好+格式对"),
        (0.90, False, "答案好但格式错(缺<answer>标签/字段)"),
        (0.20, True,  "答案差但格式对"),
    ]:
        print(f"     R_acc={r_acc:.2f}  fmt={'✓' if fmt_ok else '✗'}  → R={format_combined(r_acc, fmt_ok):.3f}  ({note})")

    print("\n" + "=" * 70)
    print(" 要点：连续奖励用分段线性衰减提供**稠密部分奖励**(避免二元稀疏)；")
    print("       到处加护栏(单点=0、计数/长度惩罚)防 reward hacking；")
    print("       格式奖励以小权重 λ=0.1 保证输出结构可解析。")
    print("=" * 70)


if __name__ == "__main__":
    main()
