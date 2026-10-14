#!/usr/bin/env python3
"""2-D conceptual proxy for the paper's linear-displacement direction search.

This is an educational toy example, not an official reproduction of
"Type-Free Global Intersection Analysis with Linear Displacement Fields".
It replaces VF/EE/FV mesh ray casting with point-to-rectangle rays in 2-D,
while keeping the useful idea: compare candidate separation directions using
a maximum depth and a mass/area weight.

Run from any directory, for example::

    python linear_displacement_demo.py --output untangling_demo.png

NumPy is required.  Matplotlib is optional; when it is absent the script
prints a small ASCII view instead of failing.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the environment
    print(
        "错误：这个教学 demo 需要 NumPy。请先安装：python -m pip install numpy",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


# 目标 primitive 的 2-D 代理：一个包住了交叉源线的矩形。
BOX_MIN = np.array([-1.05, -0.70], dtype=float)
BOX_MAX = np.array([1.05, 0.70], dtype=float)


def sample_intersection_contour(count: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """返回穿过矩形的 source 线及其落在矩形内的 hit 点和权重。

    这些 hit 点代替论文中由 VF/EE/FV ray casting 得到并经过 cluster
    culling 的有效 support。权重可理解为简化的 primitive area/mass。
    """

    # 这条斜线故意穿过矩形，首尾伸出矩形，形成一个简单的“穿透”。
    source = np.array([[-1.55, -0.92], [1.55, 0.92]], dtype=float)
    t = np.linspace(0.0, 1.0, count)
    points = source[0] + t[:, None] * (source[1] - source[0])
    inside = np.all((points >= BOX_MIN) & (points <= BOX_MAX), axis=1)
    hits = points[inside]

    # 中间的 hit 面积稍大，使代价公式中的 weight 不是装饰项。
    if len(hits) == 0:  # 防御性检查，当前固定场景不会触发。
        raise RuntimeError("交叉场景没有生成有效 hit 点")
    center = np.mean(hits, axis=0)
    radial = np.linalg.norm(hits - center, axis=1)
    weights = 0.8 + 0.6 * (1.0 - radial / (radial.max() + 1e-12))
    return source, hits, weights


def ray_exit_distance(point: np.ndarray, direction: np.ndarray) -> float:
    """从矩形内部 point 沿 +direction 到边界的首个 ray travel distance。"""

    px, py = point
    dx, dy = direction
    candidates: list[float] = []
    eps = 1e-12
    if dx > eps:
        candidates.append(float((BOX_MAX[0] - px) / dx))
    elif dx < -eps:
        candidates.append(float((BOX_MIN[0] - px) / dx))
    if dy > eps:
        candidates.append(float((BOX_MAX[1] - py) / dy))
    elif dy < -eps:
        candidates.append(float((BOX_MIN[1] - py) / dy))

    positive = [distance for distance in candidates if distance >= -eps]
    if not positive:  # direction 应始终是单位向量，且至少有一个非零分量。
        return 0.0
    return max(0.0, min(positive))


def evaluate_candidates(
    hits: np.ndarray, weights: np.ndarray, angles_deg: np.ndarray
) -> list[dict[str, object]]:
    """计算候选方向的 ray depth 和简化位移代价。

    论文中同一 contour 的最大 ray travel distance 记为 D_C；这里使用
    ``J = max(delta_h)^2 * sum(weights_h)``，保留其“最坏深度 × support
    质量”排序直觉。真实实现还包含 mesh primitive adjacency/filtering。
    """

    results: list[dict[str, object]] = []
    total_weight = float(np.sum(weights))
    for angle in angles_deg:
        radians = math.radians(float(angle))
        direction = np.array([math.cos(radians), math.sin(radians)], dtype=float)
        depths = np.array([ray_exit_distance(point, direction) for point in hits])
        max_depth = float(np.max(depths))
        cost = max_depth * max_depth * total_weight
        results.append(
            {
                "angle_deg": float(angle),
                "direction": direction,
                "depths": depths,
                "max_depth": max_depth,
                "weight_sum": total_weight,
                "cost": cost,
            }
        )
    return results


def ascii_view(
    source: np.ndarray,
    hits: np.ndarray,
    winner: dict[str, object],
    width: int = 60,
    height: int = 20,
) -> None:
    """无 matplotlib 时的轻量 ASCII 视图。"""

    canvas = [[" " for _ in range(width)] for _ in range(height)]
    xmin, xmax = -1.7, 1.7
    ymin, ymax = -1.05, 1.05

    def to_cell(point: np.ndarray) -> tuple[int, int]:
        x = int(round((float(point[0]) - xmin) / (xmax - xmin) * (width - 1)))
        y = int(round((ymax - float(point[1])) / (ymax - ymin) * (height - 1)))
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    # 矩形边框。
    corners = np.array(
        [
            [BOX_MIN[0], BOX_MIN[1]],
            [BOX_MAX[0], BOX_MIN[1]],
            [BOX_MAX[0], BOX_MAX[1]],
            [BOX_MIN[0], BOX_MAX[1]],
            [BOX_MIN[0], BOX_MIN[1]],
        ]
    )
    for a, b in zip(corners[:-1], corners[1:]):
        steps = max(2, int(np.linalg.norm(b - a) * 40))
        for point in np.linspace(a, b, steps):
            x, y = to_cell(point)
            canvas[y][x] = "+"

    # source 线与 hit 点。
    for a, b in zip(source[:-1], source[1:]):
        steps = max(2, int(np.linalg.norm(b - a) * 40))
        for point in np.linspace(a, b, steps):
            x, y = to_cell(point)
            canvas[y][x] = "-"
    for point in hits:
        x, y = to_cell(point)
        canvas[y][x] = "o"

    # 赢家方向箭头：从 hit support 中心延伸到矩形外。
    direction = np.asarray(winner["direction"])
    center = np.mean(hits, axis=0)
    arrow_end = center + direction * float(winner["max_depth"])
    steps = max(2, int(np.linalg.norm(arrow_end - center) * 40))
    for point in np.linspace(center, arrow_end, steps):
        x, y = to_cell(point)
        canvas[y][x] = "*"

    print("\nASCII 场景：+ 目标矩形，- source 线，o 有效 hit，* 选中的分离方向")
    print("\n".join("".join(row) for row in canvas))


def draw_scene(
    output: Path,
    source: np.ndarray,
    hits: np.ndarray,
    winner: dict[str, object],
    results: list[dict[str, object]],
) -> bool:
    """尽量用 matplotlib 保存图像；不可用时返回 False。"""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return False

    fig, (scene_ax, cost_ax) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    scene_ax.add_patch(
        Rectangle(
            tuple(BOX_MIN),
            float(BOX_MAX[0] - BOX_MIN[0]),
            float(BOX_MAX[1] - BOX_MIN[1]),
            facecolor="#88bde6",
            edgecolor="#1769aa",
            alpha=0.28,
            linewidth=2,
            label="destination primitive (proxy)",
        )
    )
    scene_ax.plot(source[:, 0], source[:, 1], color="#d1495b", linewidth=2.5, label="source path")
    scene_ax.scatter(hits[:, 0], hits[:, 1], s=30, color="#f4a261", edgecolor="black", zorder=3, label="valid hits")

    center = np.mean(hits, axis=0)
    direction = np.asarray(winner["direction"])
    max_depth = float(winner["max_depth"])
    scene_ax.arrow(
        center[0], center[1], direction[0] * max_depth, direction[1] * max_depth,
        width=0.008, head_width=0.09, head_length=0.12, color="#2a9d8f",
        length_includes_head=True, zorder=4,
        label=f"winner {winner['angle_deg']:.0f}°",
    )
    scene_ax.scatter([center[0]], [center[1]], marker="x", s=70, color="#2a9d8f", zorder=5)
    scene_ax.set_aspect("equal")
    scene_ax.set_xlim(-1.7, 1.7)
    scene_ax.set_ylim(-1.1, 1.1)
    scene_ax.set_xlabel("x")
    scene_ax.set_ylabel("y")
    scene_ax.set_title("2-D linear displacement proxy")
    scene_ax.grid(alpha=0.2)
    scene_ax.legend(fontsize=8, loc="upper left")

    angle_values = [float(row["angle_deg"]) for row in results]
    costs = [float(row["cost"]) for row in results]
    colors = ["#2a9d8f" if row is winner else "#b8c4ce" for row in results]
    cost_ax.bar(angle_values, costs, width=9.0, color=colors, edgecolor="#34495e")
    cost_ax.set_xlabel("candidate direction (degrees)")
    cost_ax.set_ylabel(r"J = max($\delta_h$)$^2$ × Σ weight")
    cost_ax.set_title("ray/depth cost (lower is better)")
    cost_ax.grid(axis="y", alpha=0.2)
    cost_ax.set_xticks(angle_values[::2])

    fig.suptitle(
        "Conceptual proxy — candidate rays, maximum depth, and minimum displacement",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2-D conceptual linear displacement direction search")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("untangling_demo.png"),
        help="matplotlib 可用时的 PNG 输出路径（默认：./untangling_demo.png）",
    )
    parser.add_argument(
        "--directions",
        type=int,
        default=24,
        help="均匀候选方向数量（默认：24，每 15 度一个）",
    )
    args = parser.parse_args(argv)
    if args.directions < 4:
        parser.error("--directions 至少需要 4 个候选方向")

    source, hits, weights = sample_intersection_contour()
    angles = np.linspace(0.0, 360.0, args.directions, endpoint=False)
    results = evaluate_candidates(hits, weights, angles)
    winner = min(results, key=lambda row: float(row["cost"]))

    print("2-D 线性位移方向搜索（conceptual proxy，非官方复现）")
    print(f"有效 hit 数量: {len(hits)}；support weight 总和: {float(np.sum(weights)):.3f}")
    print("角度(°)   max depth   J=max(depth)^2*Σweight")
    for row in results:
        print(f"{float(row['angle_deg']):7.1f}   {float(row['max_depth']):10.4f}   {float(row['cost']):14.4f}")
    print(
        f"\n选择方向: {float(winner['angle_deg']):.1f}° "
        f"r=({np.asarray(winner['direction'])[0]:.3f}, {np.asarray(winner['direction'])[1]:.3f}), "
        f"max depth={float(winner['max_depth']):.4f}, J={float(winner['cost']):.4f}"
    )

    if draw_scene(args.output, source, hits, winner, results):
        print(f"已保存可视化: {args.output.resolve()}")
    else:
        print("未检测到 matplotlib，改用 ASCII 输出。")
        ascii_view(source, hits, winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
