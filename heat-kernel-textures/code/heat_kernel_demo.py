#!/usr/bin/env python3
"""无需外部依赖的 HKTex 教学演示。

使用平面高斯作为热核代理，演示各向异性、式 (6) 软阈值和式 (7)
残差颜色混合。没有实现网格谱分解、双调和检索或 PBR。
运行：python3 heat_kernel_demo.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Kernel:
    x: float
    y: float
    angle: float
    anisotropy: float
    threshold: float
    sharpness: float
    color: tuple[float, float, float]


def anisotropic_heat(p: tuple[float, float], k: Kernel) -> float:
    """评价归一化各向异性高斯：流形热核的平面教学代理。"""
    dx, dy = p[0] - k.x, p[1] - k.y
    ca, sa = math.cos(k.angle), math.sin(k.angle)
    # 转到核的主轴坐标。
    u, v = ca * dx + sa * dy, -sa * dx + ca * dy
    # 各向异性增大使短轴更窄；这里用形状比例代理论文的 eta。
    raw = math.exp(-0.5 * (u * u + k.anisotropy * v * v) / 0.08)
    # 严格按式 (6) 的端点归一化，确保 alpha(0)=0，alpha(1)=1。
    sig = lambda z: 1.0 / (1.0 + math.exp(-z))
    hi, lo = sig(k.sharpness * (raw - k.threshold)), sig(-k.sharpness * k.threshold)
    upper = sig(k.sharpness * (1.0 - k.threshold))
    return max(0.0, min(1.0, (hi - lo) / (upper - lo)))


def color_at(p: tuple[float, float], kernels: list[Kernel], top_k: int = 3) -> tuple[float, float, float]:
    weighted = sorted(((anisotropic_heat(p, k), k) for k in kernels), reverse=True, key=lambda z: z[0])[:top_k]
    denom = sum(a for a, _ in weighted)
    if denom < 1e-9:
        return (0.05, 0.05, 0.05)  # 教学实现显式处理零权重回退。
    return tuple(0.05 + sum(a * k.color[i] for a, k in weighted) / denom for i in range(3))


def ppm(path: str, kernels: list[Kernel], size: int = 96) -> None:
    """输出混合颜色图；仅在保存显示像素时截断到 [0,1]。"""
    with open(path, "w", encoding="ascii") as f:
        f.write(f"P3\n{size} {size}\n255\n")
        for j in range(size):
            y = j / (size - 1)
            for i in range(size):
                x = i / (size - 1)
                rgb = color_at((x, y), kernels)
                f.write("%d %d %d " % tuple(round(255 * max(0.0, min(1.0, c))) for c in rgb))
            f.write("\n")


def main() -> None:
    kernels = [
        Kernel(0.30, 0.38, 0.2, 3.0, 0.55, 18.0, (0.95, 0.15, 0.10)),
        Kernel(0.65, 0.55, -0.5, 8.0, 0.60, 22.0, (0.10, 0.65, 0.95)),
        Kernel(0.45, 0.72, 1.1, 2.0, 0.50, 16.0, (0.20, 0.90, 0.25)),
    ]
    ppm("heat_kernel_demo.ppm", kernels)
    for p in [(0.30, 0.38), (0.50, 0.50), (0.90, 0.10)]:
        print(f"C{p} = {tuple(round(v, 3) for v in color_at(p, kernels))}")

    # 平面上指数映射退化为加法；这不演示跨三角形的测地线追踪。
    p = [0.30, 0.38]
    gradient = [0.04, -0.03]
    step = 0.5
    p_new = [p[i] - step * gradient[i] for i in range(2)]
    print(f"平面上的位置更新: {tuple(p)} -> {tuple(round(v, 3) for v in p_new)}")
    assert abs(anisotropic_heat((kernels[0].x, kernels[0].y), kernels[0]) - 1.0) < 1e-12
    assert color_at((1e6, 1e6), kernels) == (0.05, 0.05, 0.05)
    print("已生成 heat_kernel_demo.ppm；中心归一化与零权重回退检查通过。")


if __name__ == "__main__":
    main()
