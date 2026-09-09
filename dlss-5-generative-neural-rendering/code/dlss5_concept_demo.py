#!/usr/bin/env python3
"""DLSS 5 的概念性 CPU demo（不含 NVIDIA 模型、权重或 SDK）。

这个小程序把报告公开的四个接口直觉化：
* 当前 RGB 渲染帧是内容锚点；
* structure / tone / chroma 控制决定外观变化幅度；
* manual / skin mask 决定变化作用在哪里；
* 一个携带的标量状态模拟因果逐帧处理，而不是等待未来帧。

实现故意使用 Python 标准库和 PPM 图像格式，便于在没有 GPU、Pillow 或
深度学习框架的环境中运行。它是可视化教学替代物，不实现真实的一步
像素空间扩散，也不能用来推断 DLSS 5 的画质或性能。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RGB = tuple[float, float, float]
Image = list[list[RGB]]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def draw_disc(image: Image, cx: float, cy: float, radius: float, color: RGB) -> None:
    """在合成的渲染帧里画一个有清晰边界的圆形物体。"""
    height = len(image)
    width = len(image[0])
    x0 = max(0, int(cx - radius - 1))
    x1 = min(width, int(cx + radius + 2))
    y0 = max(0, int(cy - radius - 1))
    y1 = min(height, int(cy + radius + 2))
    radius_sq = radius * radius
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius_sq:
                image[y][x] = color


def draw_rect(image: Image, left: int, top: int, right: int, bottom: int, color: RGB) -> None:
    height = len(image)
    width = len(image[0])
    for y in range(max(0, top), min(height, bottom)):
        for x in range(max(0, left), min(width, right)):
            image[y][x] = color


def render_frame(frame_index: int, width: int, height: int) -> Image:
    """生成一个确定性的、带角色/材质/边缘的简化 CG 输入帧。"""
    image: Image = []
    phase = frame_index * 0.18
    for y in range(height):
        row: list[RGB] = []
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            # 天空到地面的渐变，故意保留较低对比度供 tone 控制观察。
            if v < 0.62:
                k = v / 0.62
                base = (0.12 + 0.10 * (1 - k), 0.19 + 0.12 * (1 - k), 0.27 + 0.16 * (1 - k))
            else:
                k = (v - 0.62) / 0.38
                grain = 0.018 * math.sin(x * 0.31 + y * 0.17 + phase)
                base = (0.20 + 0.16 * k + grain, 0.18 + 0.11 * k + grain, 0.12 + 0.07 * k)
            row.append(tuple(clamp(c) for c in base))
        image.append(row)

    # 窗格与地面边线：用来观察结构是否被保留。
    draw_rect(image, int(width * 0.06), int(height * 0.12), int(width * 0.09), int(height * 0.62), (0.28, 0.24, 0.20))
    draw_rect(image, int(width * 0.91), int(height * 0.08), int(width * 0.95), int(height * 0.64), (0.25, 0.22, 0.19))
    for frac in (0.25, 0.42, 0.59):
        y = int(height * frac)
        draw_rect(image, int(width * 0.05), y, int(width * 0.95), y + max(1, height // 120), (0.34, 0.29, 0.23))

    # 左侧叶片/植被：高频材质区域。
    for i in range(18):
        cx = width * 0.13 + (i % 5) * width * 0.026
        cy = height * 0.30 + (i // 5) * height * 0.08
        draw_disc(image, cx, cy, max(2.0, width * 0.018), (0.08, 0.24 + 0.01 * (i % 3), 0.13))

    # 中央角色（脸/头发/衣服），位置随帧小幅移动，但几何边界由输入决定。
    shift = math.sin(phase) * width * 0.008
    body_left = int(width * 0.38 + shift)
    body_right = int(width * 0.66 + shift)
    draw_rect(image, body_left, int(height * 0.58), body_right, int(height * 0.96), (0.25, 0.19, 0.16))
    draw_disc(image, width * 0.52 + shift, height * 0.40, width * 0.105, (0.55, 0.29, 0.20))
    draw_disc(image, width * 0.52 + shift, height * 0.34, width * 0.118, (0.11, 0.07, 0.055))
    # 眼睛、徽章和衣服纹理，作为细小结构标记。
    eye_y = int(height * 0.39)
    for eye_x in (int(width * 0.49 + shift), int(width * 0.55 + shift)):
        draw_disc(image, eye_x, eye_y, max(1.0, width * 0.008), (0.82, 0.86, 0.90))
        draw_disc(image, eye_x, eye_y, max(0.5, width * 0.003), (0.03, 0.03, 0.03))
    draw_disc(image, width * 0.52 + shift, height * 0.70, max(2.0, width * 0.018), (0.72, 0.45, 0.12))
    for i in range(6):
        y = int(height * 0.64 + i * height * 0.042)
        draw_rect(image, body_left + int(width * 0.025), y, body_right - int(width * 0.025), y + 1, (0.38, 0.28, 0.22))
    return image


def blur_luma(image: Image, x: int, y: int) -> float:
    """小窗口均值，作为低频/高频分解的透明替代。"""
    height = len(image)
    width = len(image[0])
    total = 0.0
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            xx = min(width - 1, max(0, x + dx))
            yy = min(height - 1, max(0, y + dy))
            r, g, b = image[yy][xx]
            total += 0.2126 * r + 0.7152 * g + 0.0722 * b
            count += 1
    return total / count


def mask_weight(mask: str, x: int, y: int, width: int, height: int) -> float:
    """返回概念性的空间/语义增强权重。"""
    u = x / max(1, width - 1)
    v = y / max(1, height - 1)
    if mask == "global":
        return 1.0
    if mask == "manual":
        # 手工掩码：角色和右下角 hero 区域强，边缘平滑衰减。
        character = max(0.0, 1.0 - math.hypot((u - 0.52) / 0.20, (v - 0.53) / 0.45))
        hero = max(0.0, 1.0 - math.hypot((u - 0.80) / 0.22, (v - 0.72) / 0.28))
        return clamp(max(character, 0.65 * hero))
    # “skin” 只是合成场景中已知脸部区域的标签，并非真实语义模型。
    face = math.hypot((u - 0.52) / 0.115, (v - 0.40) / 0.13)
    return clamp(1.0 - face)


def appearance_step(
    image: Image,
    state_mean: float | None,
    structure: float,
    tone: float,
    chroma: float,
    mask: str,
    model: str,
) -> tuple[Image, float]:
    """执行一次确定性的外观变换；state 只来自当前/过去帧。"""
    height = len(image)
    width = len(image[0])
    # 三个模型对应不同的色温/绿色倾向，仅用于展示“离散外观模型子集”。
    style = {
        "A": (0.06, 0.00, -0.02),  # 暖色、偏皮肤
        "B": (0.00, 0.04, 0.02),  # 中性、偏植被
        "C": (-0.03, 0.02, 0.07),  # 冷色、高对比
    }[model]
    output: Image = []
    means: list[float] = []
    for y in range(height):
        row: list[RGB] = []
        for x in range(width):
            r, g, b = image[y][x]
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            local = blur_luma(image, x, y)
            high = luma - local
            w = mask_weight(mask, x, y, width, height)

            # tone：低频亮度/色温变化；state 只使用已到达的历史均值。
            historical = luma if state_mean is None else 0.92 * luma + 0.08 * state_mean
            tone_delta = tone * w * (0.10 * (historical - 0.42) + 0.07)
            rr = r + tone_delta + tone * w * style[0]
            gg = g + tone_delta + tone * w * style[1]
            bb = b + tone_delta + tone * w * style[2]

            # structure：高频细节增强；不改变像素位置或几何边界。
            detail = structure * w * 0.72 * high
            rr += detail
            gg += detail
            bb += detail

            # chroma：围绕亮度调整饱和度。
            saturation = 1.0 + chroma * w
            rr = luma + (rr - luma) * saturation
            gg = luma + (gg - luma) * saturation
            bb = luma + (bb - luma) * saturation
            row.append((clamp(rr), clamp(gg), clamp(bb)))
            means.append((0.2126 * clamp(rr) + 0.7152 * clamp(gg) + 0.0722 * clamp(bb)))
        output.append(row)
    return output, sum(means) / len(means)


def write_ppm(path: Path, image: Image) -> None:
    """写入无依赖的二进制 PPM（P6）。"""
    height = len(image)
    width = len(image[0])
    payload = bytearray()
    for row in image:
        for r, g, b in row:
            payload.extend(int(round(clamp(c) * 255.0)) for c in (r, g, b))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)


def image_difference(a: Image, b: Image) -> float:
    total = 0.0
    count = 0
    for row_a, row_b in zip(a, b):
        for px_a, px_b in zip(row_a, row_b):
            total += sum(abs(x - y) for x, y in zip(px_a, px_b)) / 3.0
            count += 1
    return total / max(1, count)


def run_sequence(args: argparse.Namespace, out_dir: Path, write_files: bool) -> tuple[list[Image], list[Image], list[float]]:
    inputs: list[Image] = []
    outputs: list[Image] = []
    states: list[float] = []
    state: float | None = None
    for frame in range(args.frames):
        source = render_frame(frame, args.width, args.height)
        enhanced, state = appearance_step(
            source,
            state,
            args.structure,
            args.tone,
            args.chroma,
            args.mask,
            args.model,
        )
        inputs.append(source)
        outputs.append(enhanced)
        states.append(state)
        if write_files:
            write_ppm(out_dir / f"frame_{frame:03d}_input.ppm", source)
            write_ppm(out_dir / f"frame_{frame:03d}_output.ppm", enhanced)
    return inputs, outputs, states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DLSS 5 受约束生成的概念性 CPU 可视化")
    parser.add_argument("--width", type=int, default=320, help="输出宽度（默认 320）")
    parser.add_argument("--height", type=int, default=180, help="输出高度（默认 180）")
    parser.add_argument("--frames", type=int, default=3, help="连续帧数（默认 3）")
    parser.add_argument("--structure", type=float, default=0.65, help="结构强度 [0,1]")
    parser.add_argument("--tone", type=float, default=0.55, help="色调强度 [0,1]")
    parser.add_argument("--chroma", type=float, default=0.20, help="色度增强 [0,1]")
    parser.add_argument("--mask", choices=("global", "manual", "skin"), default="manual")
    parser.add_argument("--model", choices=("A", "B", "C"), default="A", help="概念外观模型子集")
    parser.add_argument("--out-dir", type=Path, default=Path("dlss5_concept_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width < 8 or args.height < 8 or args.frames < 1:
        raise SystemExit("width/height 至少为 8，frames 至少为 1")
    for name in ("structure", "tone", "chroma"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} 必须在 [0,1] 内")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs, outputs, states = run_sequence(args, args.out_dir, write_files=True)
    # 第二次运行用于验证确定性：同一序列/设置应得到完全相同的像素值。
    _, repeat_outputs, _ = run_sequence(args, args.out_dir, write_files=False)
    deterministic = all(image_difference(a, b) == 0.0 for a, b in zip(outputs, repeat_outputs))
    changes = [image_difference(a, b) for a, b in zip(inputs, outputs)]
    report = {
        "demo": "conceptual-surrogate-not-DLSS5",
        "resolution": [args.width, args.height],
        "frames": args.frames,
        "controls": {
            "structure": args.structure,
            "tone": args.tone,
            "chroma": args.chroma,
            "mask": args.mask,
            "model": args.model,
        },
        "causal_state_mean": states,
        "mean_absolute_rgb_change": changes,
        "deterministic_repeat": deterministic,
        "outputs": [f"frame_{i:03d}_output.ppm" for i in range(args.frames)],
        "note": "PPM images illustrate control semantics only; no NVIDIA weights or SDK are used.",
    }
    (args.out_dir / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DLSS 5 概念 demo 已完成（这是教学替代物，不是 NVIDIA 实现）。")
    print(f"输出目录: {args.out_dir.resolve()}")
    print(f"设置: S={args.structure:.2f}, T={args.tone:.2f}, chroma={args.chroma:.2f}, mask={args.mask}, model={args.model}")
    print(f"确定性复跑: {'PASS' if deterministic else 'FAIL'}")
    print("可用 PPM: frame_000_input.ppm 与 frame_000_output.ppm（以及后续帧）。")


if __name__ == "__main__":
    main()
