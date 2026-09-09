#!/usr/bin/env python3
"""UniVTAC 的标准库概念 demo。

这个程序不依赖 Isaac Sim、TacEx、PyTorch 或真实传感器，而是生成一个小型
视触觉样本，并把论文中的三条监督路径显式写出来：

* Shape：带 marker / 无 marker 的 RGB 观察；
* Contact：gelpad 深度与 marker 位移；
* Pose：物体相对传感器的 3D 平移 + 四元数。

它还演示论文中的接触安全速度规则。输出使用 PPM/PGM，因此在最小 Python
环境中即可运行。所有“latent”都是可解释的手工特征，旨在帮助理解数据流，
不是 UniVTAC Encoder 的神经网络复现。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

RGB = tuple[float, float, float]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def shape_value(kind: str, x: float, y: float) -> float:
    """返回一个连续的形状场；正值表示压头内部。"""
    if kind == "circle":
        return 0.30 - math.hypot(x, y)
    if kind == "cross":
        return 0.12 - min(max(abs(x), abs(y)), max(abs(x), abs(y))) + 0.10 * (abs(x) < 0.12 or abs(y) < 0.12)
    if kind == "cubehole":
        outer = 0.27 - max(abs(x), abs(y))
        hole = max(abs(x), abs(y)) - 0.075
        return min(outer, hole)
    if kind == "star":
        radius = math.hypot(x, y)
        angle = math.atan2(y, x)
        boundary = 0.22 + 0.065 * math.cos(5.0 * angle)
        return boundary - radius
    if kind == "torus":
        return 0.075 - abs(math.hypot(x, y) - 0.18)
    raise ValueError(f"unknown shape: {kind}")


@dataclass
class Sample:
    width: int
    height: int
    shape: str
    indentation: float
    shear: float
    yaw_deg: float
    marked: list[list[RGB]]
    pure: list[list[RGB]]
    depth: list[list[float]]
    markers: list[tuple[float, float]]
    pose: list[float]


def marker_grid() -> list[tuple[float, float]]:
    """九列七行的 marker 网格，便于观察 2D 位移。"""
    return [
        (0.18 + 0.08 * col, 0.28 + 0.075 * row)
        for row in range(7)
        for col in range(9)
    ]


def render_sample(shape: str, indentation: float, shear: float, yaw_deg: float, width: int = 320, height: int = 240) -> Sample:
    pure: list[list[RGB]] = []
    marked: list[list[RGB]] = []
    depth: list[list[float]] = []
    angle = math.radians(yaw_deg)
    for py in range(height):
        pure_row: list[RGB] = []
        marked_row: list[RGB] = []
        depth_row: list[float] = []
        for px in range(width):
            # 把像素坐标映射到以 gelpad 中心为原点的局部平面。
            x = (px / (width - 1) - 0.5) * 0.72
            y = (py / (height - 1) - 0.5) * 0.72
            xr, yr = rotate(x, y, -angle)
            field = shape_value(shape, xr, yr)
            inside = clamp(field / 0.08 + 0.5)
            # 深度在形状中心较大，并带少量剪切方向变化。
            local_depth = clamp(indentation * inside * (0.55 + 0.45 * (1.0 - min(1.0, math.hypot(xr, yr) / 0.3))))
            local_depth *= clamp(1.0 + 0.25 * shear * xr / 0.3)
            grain = 0.025 * math.sin(px * 0.21 + py * 0.13)
            base = (
                clamp(0.16 + 0.52 * inside + grain),
                clamp(0.20 + 0.26 * inside + 0.015 * shear),
                clamp(0.24 + 0.18 * inside),
            )
            pure_row.append(base)
            # marker 纹理先由后面的网格叠加；这里保留原始 RGB 观察。
            marked_row.append(base)
            depth_row.append(local_depth)
        pure.append(pure_row)
        marked.append(marked_row)
        depth.append(depth_row)

    # 计算 marker 的剪切位移，并将 marker 画入带标记图像。
    markers: list[tuple[float, float]] = []
    for u, v in marker_grid():
        # 归一化位置相对中心，剪切使 x/y 发生可解释的偏移。
        du = shear * 0.055 * (v - 0.54)
        dv = shear * 0.032 * (u - 0.50) + indentation * 0.008
        markers.append((clamp(u + du, 0.0, 1.0), clamp(v + dv, 0.0, 1.0)))
        cx, cy = int(round((u + du) * (width - 1))), int(round((v + dv) * (height - 1)))
        for yy in range(max(0, cy - 2), min(height, cy + 3)):
            for xx in range(max(0, cx - 2), min(width, cx + 3)):
                if (xx - cx) ** 2 + (yy - cy) ** 2 <= 5:
                    marked[yy][xx] = (0.96, 0.91, 0.28)

    # 物体相对 gelpad 中心的简化 pose（平移 + yaw 四元数）。
    tx = shear * 0.018
    ty = indentation * 0.006
    tz = indentation * 0.025
    half = angle / 2.0
    pose = [tx, ty, tz, 0.0, 0.0, math.sin(half), math.cos(half)]
    return Sample(width, height, shape, indentation, shear, yaw_deg, marked, pure, depth, markers, pose)


def write_ppm(path: Path, image: list[list[RGB]]) -> None:
    height, width = len(image), len(image[0])
    payload = bytearray()
    for row in image:
        for pixel in row:
            payload.extend(int(round(clamp(channel) * 255.0)) for channel in pixel)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)


def write_pgm(path: Path, image: list[list[float]]) -> None:
    height, width = len(image), len(image[0])
    payload = bytearray()
    for row in image:
        for value in row:
            payload.append(int(round(clamp(value) * 255.0)))
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + payload)


def mse_rgb(a: list[list[RGB]], b: list[list[RGB]]) -> float:
    total = 0.0
    count = 0
    for row_a, row_b in zip(a, b):
        for px_a, px_b in zip(row_a, row_b):
            total += sum((x - y) ** 2 for x, y in zip(px_a, px_b)) / 3.0
            count += 1
    return total / max(1, count)


def flatten_depth(image: list[list[float]]) -> list[float]:
    return [value for row in image for value in row]


def toy_pathway_report(sample: Sample) -> dict:
    """用可解释统计量展示三条路径携带的互补信息。

    这不是训练出来的神经网络。数值的作用是告诉读者：不同标签看的是
    不同物理因素，不能只用一张 RGB 图的单一像素误差代替全部监督。
    """
    depth_values = flatten_depth(sample.depth)
    depth_mean = sum(depth_values) / len(depth_values)
    depth_max = max(depth_values)
    base_markers = marker_grid()
    displacements = [
        math.hypot((x - bx), (y - by))
        for (x, y), (bx, by) in zip(sample.markers, base_markers)
    ]
    marker_mean = sum(displacements) / len(displacements)
    # 只用纯图像重建带标记图像时，marker 纹理不可预测；反之亦然。
    marker_artifact_mse = mse_rgb(sample.marked, sample.pure)
    shape_signal = {
        "area_fraction": sum(1 for row in sample.depth for d in row if d > 0.02) / (sample.width * sample.height),
        "marker_artifact_mse": marker_artifact_mse,
    }
    contact_signal = {
        "depth_mean": depth_mean,
        "depth_max": depth_max,
        "marker_displacement_mean": marker_mean,
        "shear_control": sample.shear,
    }
    pose_signal = {
        "translation_xyz": sample.pose[:3],
        "quaternion_xyzw": sample.pose[3:],
        "yaw_deg": sample.yaw_deg,
    }
    return {
        "shape_pathway": shape_signal,
        "contact_pathway": contact_signal,
        "pose_pathway": pose_signal,
        "full_latent_fields": ["shape", "contact", "pose"],
        "interpretation": {
            "shape": "双视图 RGB 让表示区分物体形状与 marker/光照伪影",
            "contact": "深度与 marker 位移把局部像素变化连接到压入和剪切",
            "pose": "平移与四元数把局部触觉放回外部度量空间",
        },
    }


def adaptive_grasp_velocity(d_min: float, d_max: float, threshold: float, v_fast: float, v_slow: float) -> float:
    """论文式 Grasp 反馈规则的标量版本。"""
    if abs(d_min - d_max) < 1e-12:
        return v_fast
    return min(abs(d_min - threshold), v_slow)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniVTAC 视触觉多路径监督概念 demo")
    parser.add_argument("--shape", choices=("circle", "cross", "cubehole", "star", "torus"), default="star")
    parser.add_argument("--indentation", type=float, default=0.65, help="压入程度 [0,1]")
    parser.add_argument("--shear", type=float, default=0.35, help="剪切方向/幅度 [-1,1]")
    parser.add_argument("--yaw", type=float, default=18.0, help="物体相对传感器的 yaw（度）")
    parser.add_argument("--out-dir", type=Path, default=Path("univtac_demo_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.indentation <= 1.0:
        raise SystemExit("indentation 必须在 [0,1] 内")
    if not -1.0 <= args.shear <= 1.0:
        raise SystemExit("shear 必须在 [-1,1] 内")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample = render_sample(args.shape, args.indentation, args.shear, args.yaw)
    write_ppm(args.out_dir / "tactile_marked.ppm", sample.marked)
    write_ppm(args.out_dir / "tactile_pure.ppm", sample.pure)
    write_pgm(args.out_dir / "gelpad_depth.pgm", sample.depth)

    pathway = toy_pathway_report(sample)
    grasp = {
        "no_contact_qdot": adaptive_grasp_velocity(1.0, 1.0, 0.72, 0.08, 0.012),
        "contact_qdot": adaptive_grasp_velocity(0.84, 1.0, 0.72, 0.08, 0.012),
        "inputs": {"d_min_no_contact": 1.0, "d_min_contact": 0.84, "d_max": 1.0, "delta_th": 0.72},
        "note": "无接触快速接近，接触后以 v_slow 和 δ_th 限制闭合；这是数据生成安全机制。",
    }
    report = {
        "demo": "conceptual-surrogate-not-univtac",
        "controls": {"shape": args.shape, "indentation": args.indentation, "shear": args.shear, "yaw_deg": args.yaw},
        "observation_shapes": {"marked_rgb": [3, sample.height, sample.width], "pure_rgb": [3, sample.height, sample.width], "depth": [sample.height, sample.width], "markers": [2, len(sample.markers)], "pose": [7]},
        "pathways": pathway,
        "adaptive_grasp": grasp,
        "files": ["tactile_marked.ppm", "tactile_pure.ppm", "gelpad_depth.pgm"],
        "note": "Hand-crafted features illustrate supervision semantics; no Isaac Sim, TacEx, PyTorch, weights, or real sensor data are used.",
    }
    report_path = args.out_dir / "run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()[:16]
    print("UniVTAC 概念 demo 已完成（不是论文实现）。")
    print(f"输出目录: {args.out_dir.resolve()}")
    print(f"样本: shape={args.shape}, indentation={args.indentation:.2f}, shear={args.shear:.2f}, yaw={args.yaw:.1f}°")
    print(f"marker 数量: {len(sample.markers)}; pose 维度: {len(sample.pose)}; 报告摘要哈希: {digest}")
    print(f"Grasp 规则: no-contact qdot={grasp['no_contact_qdot']:.4f}, contact qdot={grasp['contact_qdot']:.4f}")
    print("输出: tactile_marked.ppm, tactile_pure.ppm, gelpad_depth.pgm, run_report.json")


if __name__ == "__main__":
    main()
