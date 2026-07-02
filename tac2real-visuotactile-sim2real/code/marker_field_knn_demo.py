#!/usr/bin/env python3
"""
marker_field_knn_demo.py
===========================================================================
概念演示:Tac2Real 如何从 IPC 网格的形变计算出 **标记点位移场**(marker
displacement field),以及为什么它对四种接触模式(静止/下压/前移/后移)如此敏感。

复现论文第 3.2 节的两个要点:
  1. 标记点 <-> 网格节点 的映射用 **k 近邻(KNN)** 在初始状态建立;
     形变后,标记点位移由网格节点位置 **加权插值** 得到。
  2. 标记点位移场在不同接触模式间变化显著(这正是选它而非 RGB 的理由)。

这 **不是** Tac2Real。它是一个教学模型:用一张规则的弹性体表面网格 + 一组
9x7 标记点(GelSight Mini 的规格),施加四种典型形变场,打印每种模式下的位移场
统计与一张 ASCII 箭头图,直观感受"位移场对接触模式的敏感性"。

运行(只用标准库,到处能跑):
    python3 marker_field_knn_demo.py
===========================================================================
"""

from __future__ import annotations
import math

# --------------------------------------------------------------------------
# 0. 传感器规格(GelSight Mini):9x7 标记点网格
# --------------------------------------------------------------------------
NX, NY = 9, 7          # 标记点阵列(论文用 9x7)
SPACING = 1.0          # 标记点间距(任意单位)

def make_grid(nx, ny, spacing):
    """生成 nx*ny 的二维点阵,返回 [(x, y), ...]。"""
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append((i * spacing, j * spacing))
    return pts

# 标记点初始位置,以及(更密的)底层 IPC 网格节点初始位置
markers_init = make_grid(NX, NY, SPACING)
mesh_init    = make_grid(2 * NX, 2 * NY, SPACING / 2.0)   # 更密的"网格节点"


# --------------------------------------------------------------------------
# 1. KNN 映射:为每个标记点找最近的 k 个网格节点 + 反距离权重
#    (论文:初始状态用 KNN 建映射,形变后加权插值)
# --------------------------------------------------------------------------
def knn_mapping(markers, mesh, k=4):
    """返回 mapping[m] = (邻居索引列表, 归一化权重列表)。
    权重用反距离加权(IDW),距离越近权重越大。"""
    mapping = []
    for (mx, my) in markers:
        dists = []
        for idx, (gx, gy) in enumerate(mesh):
            d = math.hypot(mx - gx, my - gy)
            dists.append((d, idx))
        dists.sort()
        nbrs = [idx for _, idx in dists[:k]]
        # 反距离权重(加 1e-6 防止除零)
        raw = [1.0 / (dists[r][0] + 1e-6) for r in range(k)]
        s = sum(raw)
        w = [r / s for r in raw]
        mapping.append((nbrs, w))
    return mapping

mapping = knn_mapping(markers_init, mesh_init, k=4)


# --------------------------------------------------------------------------
# 2. 四种接触模式 -> 施加到底层网格节点的形变场
#    (静止 / 下压 / 前移 / 后移;论文图 2)
#    这里用解析的位移场近似真实弹性体在各模式下的形变趋势。
# --------------------------------------------------------------------------
def deform_mesh(mode):
    """返回形变后的网格节点位置(以及为可视化保留 z 方向"下压"分量)。
    我们在 2D 平面上返回 (x, y) 位移;press 模式额外返回"法向压入"强度。"""
    W = (2 * NX - 1) * (SPACING / 2.0)   # 网格宽度
    H = (2 * NY - 1) * (SPACING / 2.0)
    cx, cy = W / 2.0, H / 2.0
    deformed = []
    for (gx, gy) in mesh_init:
        dx = dy = 0.0
        r = math.hypot(gx - cx, gy - cy)
        bump = math.exp(-(r * r) / (2 * (W * 0.22) ** 2))   # 接触区高斯包络
        if mode == "static":
            dx, dy = 0.0, 0.0                                # 无接触:零位移
        elif mode == "press":
            # 下压:接触区整体向中心轻微汇聚(法向压入的切向投影)
            dx = -0.15 * (gx - cx) / (W / 2) * bump
            dy = -0.15 * (gy - cy) / (H / 2) * bump
        elif mode == "forward":
            # 前移:剪切位移,标记点整体被向 +x 拖拽,接触区最强
            dx = 0.45 * bump
            dy = 0.0
        elif mode == "backward":
            # 后移:相反方向的剪切
            dx = -0.45 * bump
            dy = 0.0
        deformed.append((gx + dx, gy + dy))
    return deformed


# --------------------------------------------------------------------------
# 3. 从形变后的网格节点,用 KNN 权重插值出每个标记点的位移
# --------------------------------------------------------------------------
def marker_displacements(mode):
    mesh_def = deform_mesh(mode)
    disp = []
    for m_idx, (mx, my) in enumerate(markers_init):
        nbrs, w = mapping[m_idx]
        # 形变后该标记点的位置 = 邻居网格节点形变后位置的加权和
        px = sum(w[j] * mesh_def[nbrs[j]][0] for j in range(len(nbrs)))
        py = sum(w[j] * mesh_def[nbrs[j]][1] for j in range(len(nbrs)))
        disp.append((px - mx, py - my))
    return disp


# --------------------------------------------------------------------------
# 4. 可视化 + 统计
# --------------------------------------------------------------------------
ARROWS = {(-1, 0): "←", (1, 0): "→", (0, -1): "↓", (0, 1): "↑",
          (0, 0): "·", (-1, -1): "↙", (1, 1): "↗", (-1, 1): "↖", (1, -1): "↘"}

def arrow_for(dx, dy, thresh=0.02):
    if math.hypot(dx, dy) < thresh:
        return "·"
    sx = 0 if abs(dx) < thresh else (1 if dx > 0 else -1)
    sy = 0 if abs(dy) < thresh else (1 if dy > 0 else -1)
    return ARROWS.get((sx, sy), "•")

def field_stats(disp):
    mags = [math.hypot(dx, dy) for dx, dy in disp]
    mean = sum(mags) / len(mags)
    mx = max(mags)
    return mean, mx

def show_mode(mode):
    disp = marker_displacements(mode)
    mean, mx = field_stats(disp)
    print(f"  接触模式 = {mode:<9}  平均位移 = {mean:6.3f}   最大位移 = {mx:6.3f}")
    # 打印 9x7 的箭头图(从上到下)
    for j in range(NY - 1, -1, -1):
        row = ""
        for i in range(NX):
            dx, dy = disp[j * NX + i]
            row += " " + arrow_for(dx, dy)
        print("      " + row)
    return mean


def l2_between(mode_a, mode_b):
    """两种模式的位移场之间的 L2 差异(衡量'对接触模式的敏感度')。"""
    a, b = marker_displacements(mode_a), marker_displacements(mode_b)
    return math.sqrt(sum((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2
                         for i in range(len(a))))


# ==========================================================================
if __name__ == "__main__":
    print("\nTac2Real 概念演示:标记点位移场(9x7)+ KNN 插值 + 四种接触模式\n")
    print("=" * 70)
    print(f"标记点阵列: {NX}x{NY} = {NX*NY} 个   底层网格节点: {2*NX}x{2*NY} = {4*NX*NY} 个")
    print(f"KNN: 每个标记点用最近 4 个网格节点做反距离加权插值")
    print("=" * 70)

    means = {}
    for mode in ["static", "press", "forward", "backward"]:
        print()
        means[mode] = show_mode(mode)

    print("\n" + "=" * 70)
    print("位移场对接触模式的【敏感度】—— 两两 L2 差异(越大越易区分):")
    print("=" * 70)
    modes = ["static", "press", "forward", "backward"]
    print("           " + "".join(f"{m:>10}" for m in modes))
    for a in modes:
        row = f"  {a:<9}"
        for b in modes:
            row += f"{l2_between(a, b):>10.3f}"
        print(row)

    print("\n结论:不同接触模式(尤其 forward vs backward)的标记点位移场差异显著,")
    print("这正是论文选择【标记点位移场】而非 RGB 触觉图像的原因 —— 对接触状态敏感、")
    print("维度低(9x7x2),利于大规模在线 RL 的规律性与效率。")
    print("=" * 70)
