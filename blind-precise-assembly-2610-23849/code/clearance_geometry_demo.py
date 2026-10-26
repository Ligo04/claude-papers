#!/usr/bin/env python3
"""用标准库展示论文 §2.2 的间隙公式；不运行接触动力学或原论文策略。"""

import argparse
import csv
import math
from pathlib import Path


def extra_play(clearance_um, sides):
    """固定方向、外接正多边形的最大额外平移游隙，单位 μm。"""
    return clearance_um * (1.0 / math.cos(math.pi / sides) - 1.0)


def feasible_vertices(clearance_um, sides):
    # 每条边距原点 c，顶点是相邻法向约束的交点。
    radius = clearance_um / math.cos(math.pi / sides)
    return [
        (
            radius * math.cos((2 * i + 1) * math.pi / sides),
            radius * math.sin((2 * i + 1) * math.pi / sides),
        )
        for i in range(sides)
    ]


def render_svg(clearance_um, sides):
    points = feasible_vertices(clearance_um, sides)
    scale = 170 / clearance_um
    polygon = " ".join(f"{250+x*scale:.3f},{260-y*scale:.3f}" for x, y in points)
    error = extra_play(clearance_um, sides)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 530">
<rect width="760" height="530" fill="#f7f6f1"/>
<g font-family="sans-serif" fill="#183128">
<text x="30" y="38" font-size="23">固定方向下，销中心允许平移到哪里？</text>
<polygon points="{polygon}" fill="#b4dfcd" stroke="#187653" stroke-width="3"/>
<circle cx="250" cy="260" r="170" fill="#f7f6f1" stroke="#2b66bd" stroke-width="2" stroke-dasharray="7 5"/>
<circle cx="250" cy="260" r="4"/>
<line x1="250" y1="260" x2="420" y2="260" stroke="#2b66bd" stroke-width="2"/>
<text x="304" y="250" font-size="18">c</text>
<text x="472" y="145" font-size="19">理想圆孔：蓝色圆盘</text>
<text x="472" y="180" font-size="19">外接离散化：绿色区域</text>
<text x="472" y="234" font-size="18">c = {clearance_um:g} μm</text>
<text x="472" y="269" font-size="18">n = {sides}</text>
<text x="472" y="304" font-size="18">额外游隙 = {error:.6f} μm</text>
<text x="472" y="339" font-size="18">相对间隙 = {100*error/clearance_um:.4f}%</text>
<text x="30" y="465" font-size="17">e = c × (sec(π/n) − 1) · 由论文公式计算，非仿真测量</text>
<text x="30" y="499" font-size="16">绿色角落增加可行平移；这里画的是中心位移集合，不是实物孔的尺寸。</text>
</g></svg>"""


def verify_geometry():
    for c in (20, 100, 200):
        previous = float("inf")
        for n in (7, 16, 32, 63, 128, 281):
            vertices = feasible_vertices(c, n)
            error = extra_play(c, n)
            assert 0 < error < previous
            previous = error
            # 检查所有顶点都满足各条边的半平面约束。
            for x, y in vertices:
                for i in range(n):
                    normal_angle = 2 * math.pi * i / n
                    projection = x * math.cos(normal_angle) + y * math.sin(normal_angle)
                    assert projection <= c + 1e-8
            assert math.isclose(math.hypot(*vertices[0]) - c, error, abs_tol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clearance-um", type=float, default=100)
    parser.add_argument("--sides", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )
    args = parser.parse_args()
    if not math.isfinite(args.clearance_um) or args.clearance_um <= 0:
        parser.error("间隙必须是有限正数")
    if not 3 <= args.sides <= 10000:
        parser.error("边数必须在 3 到 10000 之间")
    verify_geometry()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clearance-feasible-region.svg").write_text(
        render_svg(args.clearance_um, args.sides), encoding="utf-8"
    )
    with (args.output_dir / "clearance-derived.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["间隙_um", "多边形边数", "公式额外游隙_um", "数值来源"])
        for c in (20, 100, 200):
            for n in (7, 16, 32, 63, 128, 281):
                writer.writerow([c, n, f"{extra_play(c,n):.9f}", "§2.2公式推导，非测量"])
    print("几何约束与边数加密检查通过。")
    print(f"c={args.clearance_um:g} μm，n={args.sides}，额外游隙={extra_play(args.clearance_um,args.sides):.6f} μm")
    print("这是几何公式的教学演示，没有运行 MuJoCo 或测量动态接触误差。")
    print(f"输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
