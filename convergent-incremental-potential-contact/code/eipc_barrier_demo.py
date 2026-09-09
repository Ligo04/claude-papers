#!/usr/bin/env python3
"""EIPC 核心公式的零依赖教学演示。

本脚本实现：
1. 论文第 2 节的接触障碍函数及一阶导数；
2. 论文式 (3.3) 的摩擦平滑函数及前两阶导数；
3. 两条共线边共享顶点时，直接求和的重复计数与 EIPC 减项修正；
4. 论文图注中明确报告的收敛阶清单。

它不是完整 EIPC 求解器：没有 FEM、CCD、Newton、摩擦外迭代或 3D stencil。
只依赖 Python 标准库，可直接运行：

    python3 eipc_barrier_demo.py
    python3 eipc_barrier_demo.py --json
    python3 eipc_barrier_demo.py --dhat 0.5 --kappa 1 --epsilon-v 0.1
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable


Point2 = tuple[float, float]


@dataclass(frozen=True)
class RateRecord:
    experiment: str
    integrator: str
    quantity: str
    rate: float
    source: str


PAPER_RATES = (
    RateRecord("1D 冲击", "Newmark", "位移", 1.0389, "Figure 8.5"),
    RateRecord("1D 冲击", "Newmark", "接触压力", 0.4709, "Figure 8.5"),
    RateRecord("1D 冲击", "BDF-2", "位移", 1.0669, "Figure 8.6"),
    RateRecord("1D 冲击", "BDF-2", "接触压力", 0.4402, "Figure 8.6"),
    RateRecord("1D 反弹", "Newmark", "能量", 0.9877, "Figure 8.7"),
    RateRecord("1D 反弹", "Newmark", "位移", 1.0092, "Figure 8.7"),
    RateRecord("1D 反弹", "BDF-2", "能量", 0.9384, "Figure 8.8"),
    RateRecord("1D 反弹", "BDF-2", "位移", 0.9661, "Figure 8.8"),
    RateRecord("2D 静态方块", "BDF-2", "能量", 1.5688, "Figure 9.1"),
    RateRecord("2D 静态方块", "BDF-2", "位移", 1.3850, "Figure 9.1"),
    RateRecord("2D 静态方块", "BDF-2", "接触压力", 1.1952, "Figure 9.1"),
    RateRecord("2D 动态方块", "BDF-2", "能量", 1.0718, "Figure 9.2"),
    RateRecord("2D 动态方块", "BDF-2", "位移", 1.1527, "Figure 9.2"),
    RateRecord("2D 动态方块", "BDF-2", "接触压力", 0.5065, "Figure 9.2"),
    RateRecord("2D 网格平面滑动", "BDF-2", "横向轨迹", 1.0761, "Figure 9.3"),
    RateRecord("3D 网格平面滑动", "未标注", "横向轨迹", 0.8879, "Figure 9.4"),
    RateRecord("3D 临界斜面", "Implicit Euler", "速度误差", 0.9959, "Figure 9.5"),
    RateRecord("3D 临界斜面", "Implicit Euler", "加速度误差", 0.9926, "Figure 9.5"),
)


def barrier_energy(distance: float, dhat: float, stiffness: float = 1.0) -> float:
    """计算论文障碍 b(d, d̂)。"""
    if dhat <= 0.0:
        raise ValueError("dhat 必须为正数")
    if stiffness <= 0.0:
        raise ValueError("stiffness 必须为正数")
    if distance <= 0.0:
        return math.inf
    if distance >= dhat:
        return 0.0
    ratio = distance / dhat
    return -stiffness * (ratio - 1.0) ** 2 * math.log(ratio)


def barrier_derivative(distance: float, dhat: float, stiffness: float = 1.0) -> float:
    """计算 ∂b/∂d；激活区内应为负。"""
    if dhat <= 0.0:
        raise ValueError("dhat 必须为正数")
    if stiffness <= 0.0:
        raise ValueError("stiffness 必须为正数")
    if distance <= 0.0:
        return -math.inf
    if distance >= dhat:
        return 0.0
    ratio = distance / dhat
    bracket = 2.0 * (ratio - 1.0) * math.log(ratio) + (ratio - 1.0) ** 2 / ratio
    return -stiffness * bracket / dhat


def friction_mollifier(speed: float, epsilon_v: float) -> float:
    """计算论文式 (3.3) 的 m_εv(s)。"""
    if speed < 0.0:
        raise ValueError("speed 必须非负")
    if epsilon_v <= 0.0:
        raise ValueError("epsilon_v 必须为正数")
    if speed >= epsilon_v:
        return speed
    return (
        -(speed**3) / (3.0 * epsilon_v**2)
        + speed**2 / epsilon_v
        + epsilon_v / 3.0
    )


def friction_mollifier_derivative(speed: float, epsilon_v: float) -> float:
    """计算 m'_εv(s)，决定归一化摩擦力幅值。"""
    if speed < 0.0:
        raise ValueError("speed 必须非负")
    if epsilon_v <= 0.0:
        raise ValueError("epsilon_v 必须为正数")
    if speed >= epsilon_v:
        return 1.0
    return -(speed**2) / epsilon_v**2 + 2.0 * speed / epsilon_v


def friction_mollifier_second_derivative(speed: float, epsilon_v: float) -> float:
    """计算 m''_εv(s)。"""
    if speed < 0.0:
        raise ValueError("speed 必须非负")
    if epsilon_v <= 0.0:
        raise ValueError("epsilon_v 必须为正数")
    if speed >= epsilon_v:
        return 0.0
    return -2.0 * speed / epsilon_v**2 + 2.0 / epsilon_v


def point_distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_distance(point: Point2, start: Point2, end: Point2) -> float:
    """二维点到闭线段距离。"""
    vx, vy = end[0] - start[0], end[1] - start[1]
    wx, wy = point[0] - start[0], point[1] - start[1]
    denom = vx * vx + vy * vy
    if denom == 0.0:
        return point_distance(point, start)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    projection = (start[0] + t * vx, start[1] + t * vy)
    return point_distance(point, projection)


def overlap_rows(dhat: float, stiffness: float) -> list[dict[str, float]]:
    """展示共享顶点附近的直接求和与减项修正。

    边 e1=[-1,0]→[0,0]、e2=[0,0]→[1,0] 共线且共享原点。
    对上方查询点，较远线段的最近点就是共享顶点，因此
    b(e1)+b(e2)-b(vertex) 精确等于 max(b(e1), b(e2))。
    """
    left = (-1.0, 0.0), (0.0, 0.0)
    right = (0.0, 0.0), (1.0, 0.0)
    shared = (0.0, 0.0)
    rows: list[dict[str, float]] = []
    for x in (-0.4, -0.2, 0.0, 0.2, 0.4):
        point = (x, 0.2)
        d_left = point_segment_distance(point, *left)
        d_right = point_segment_distance(point, *right)
        d_vertex = point_distance(point, shared)
        e_left = barrier_energy(d_left, dhat, stiffness)
        e_right = barrier_energy(d_right, dhat, stiffness)
        e_vertex = barrier_energy(d_vertex, dhat, stiffness)
        exact = max(e_left, e_right)
        direct = e_left + e_right
        corrected = direct - e_vertex
        rows.append(
            {
                "x": x,
                "y": point[1],
                "exact_max": exact,
                "direct_sum": direct,
                "corrected": corrected,
                "direct_overcount": direct - exact,
                "corrected_error": corrected - exact,
            }
        )
    return rows


def barrier_rows(dhat: float, stiffness: float) -> list[dict[str, float]]:
    fractions = (0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0, 1.2)
    return [
        {
            "d_over_dhat": fraction,
            "distance": fraction * dhat,
            "energy": barrier_energy(fraction * dhat, dhat, stiffness),
            "derivative": barrier_derivative(fraction * dhat, dhat, stiffness),
        }
        for fraction in fractions
    ]


def friction_rows(epsilon_v: float) -> list[dict[str, float]]:
    fractions = (0.0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0, 1.5)
    return [
        {
            "speed_over_epsilon_v": fraction,
            "speed": fraction * epsilon_v,
            "m": friction_mollifier(fraction * epsilon_v, epsilon_v),
            "m_prime": friction_mollifier_derivative(fraction * epsilon_v, epsilon_v),
            "m_second": friction_mollifier_second_derivative(
                fraction * epsilon_v, epsilon_v
            ),
        }
        for fraction in fractions
    ]


def verify(dhat: float, stiffness: float, epsilon_v: float) -> None:
    """检查论文公式应满足的局部不变量。"""
    assert barrier_energy(dhat, dhat, stiffness) == 0.0
    assert barrier_energy(0.01 * dhat, dhat, stiffness) > barrier_energy(
        0.1 * dhat, dhat, stiffness
    )
    assert barrier_derivative(0.5 * dhat, dhat, stiffness) < 0.0
    assert math.isclose(
        friction_mollifier(epsilon_v, epsilon_v), epsilon_v, rel_tol=1e-12
    )
    assert math.isclose(
        friction_mollifier_derivative(epsilon_v, epsilon_v), 1.0, rel_tol=1e-12
    )
    assert math.isclose(
        friction_mollifier_second_derivative(epsilon_v, epsilon_v),
        0.0,
        abs_tol=1e-12,
    )
    for row in overlap_rows(dhat, stiffness):
        assert row["direct_overcount"] >= -1e-12
        assert math.isclose(
            row["corrected"], row["exact_max"], rel_tol=1e-10, abs_tol=1e-12
        )


def print_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    headers = tuple(headers)
    rendered = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dhat", type=float, default=0.5, help="障碍激活距离")
    parser.add_argument("--kappa", type=float, default=1.0, help="障碍缩放参数")
    parser.add_argument(
        "--epsilon-v", type=float, default=0.1, help="摩擦平滑速度阈值"
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    verify(args.dhat, args.kappa, args.epsilon_v)
    barriers = barrier_rows(args.dhat, args.kappa)
    overlaps = overlap_rows(args.dhat, args.kappa)
    frictions = friction_rows(args.epsilon_v)

    if args.json:
        print(
            json.dumps(
                {
                    "parameters": {
                        "dhat": args.dhat,
                        "kappa": args.kappa,
                        "epsilon_v": args.epsilon_v,
                    },
                    "barrier": barriers,
                    "overlap_correction": overlaps,
                    "friction_mollifier": frictions,
                    "paper_reported_rates": [asdict(record) for record in PAPER_RATES],
                    "verification": "passed",
                    "scope": "教学公式演示，不是完整 EIPC 求解器",
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return

    print("\n障碍函数 b(d, d̂)")
    print_table(
        ("d/d̂", "d", "energy", "db/dd"),
        (
            (
                f"{row['d_over_dhat']:.2f}",
                f"{row['distance']:.6g}",
                f"{row['energy']:.8g}",
                f"{row['derivative']:.8g}",
            )
            for row in barriers
        ),
    )

    print("\n共享顶点重复计数：exact=max，direct=两边求和，corrected=direct-顶点项")
    print_table(
        ("x", "exact", "direct", "corrected", "overcount"),
        (
            (
                f"{row['x']:+.1f}",
                f"{row['exact_max']:.8g}",
                f"{row['direct_sum']:.8g}",
                f"{row['corrected']:.8g}",
                f"{row['direct_overcount']:.8g}",
            )
            for row in overlaps
        ),
    )

    print("\n摩擦平滑 m_εv(s)")
    print_table(
        ("s/εv", "s", "m", "m'", "m''"),
        (
            (
                f"{row['speed_over_epsilon_v']:.2f}",
                f"{row['speed']:.6g}",
                f"{row['m']:.8g}",
                f"{row['m_prime']:.8g}",
                f"{row['m_second']:.8g}",
            )
            for row in frictions
        ),
    )

    print("\n论文报告的部分收敛阶")
    print_table(
        ("实验", "积分器", "量", "阶", "来源"),
        (
            (
                record.experiment,
                record.integrator,
                record.quantity,
                f"{record.rate:.4f}",
                record.source,
            )
            for record in PAPER_RATES
        ),
    )
    print("\n验证通过：障碍、C² 摩擦连接和共线双边减项不变量均成立。")
    print("范围声明：这是教学公式演示，不是完整 EIPC/FEM/CCD 复现。")


if __name__ == "__main__":
    main()

