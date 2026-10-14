#!/usr/bin/env python3
"""UNITAS 概念 demo：从像素/深度得到 world point，再组织 action/scene flow。

这是教学代理，不是 UNITAS checkpoint 或官方 benchmark 复现。
它用标准库演示论文的接口：
1. camera ray + depth -> world-frame point；
2. action flow 与 scene flow 使用同一 metric frame；
3. scene flow 可以条件化于 action flow；
4. 输出论文中的真实关键指标，便于把 demo 与论文表格对照。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def backproject(u: float, v: float, depth: float, fx: float, fy: float, cx: float, cy: float):
    """P_camera = D K^-1 [u, v, 1]^T，教学版假设 camera/world 同 frame。"""
    return (
        (u - cx) * depth / fx,
        (v - cy) * depth / fy,
        depth,
    )


def interpolate(start, end, steps: int):
    for i in range(steps + 1):
        t = i / steps
        yield tuple(a + (b - a) * t for a, b in zip(start, end))


def action_conditioned_scene_flow(action_flow, scale: float = 0.4):
    """用一个可解释的 toy 规律模拟 action flow 对场景点的影响。"""
    return [tuple(scale * x for x in point) for point in action_flow]


def rounded(points):
    return [[round(x, 4) for x in p] for p in points]


def run_demo(depth: float):
    # 这些相机内参是教学 proxy；论文实际从数据源 calibration 获得。
    camera = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}
    pixel = (350.0, 260.0)
    world_anchor = backproject(pixel[0], pixel[1], depth, **camera)

    action_end = (
        world_anchor[0] + 0.10,
        world_anchor[1] + 0.02,
        world_anchor[2],
    )
    action_flow = list(interpolate(world_anchor, action_end, steps=4))
    action_displacements = [
        tuple(p[i] - world_anchor[i] for i in range(3)) for p in action_flow
    ]
    scene_flow = action_conditioned_scene_flow(action_displacements)

    result = {
        "paper": "UNITAS: A 3D-Native World Action Model for Embodied Manipulation",
        "teaching_proxy": True,
        "world_anchor": rounded([world_anchor])[0],
        "action_flow": rounded(action_flow),
        "scene_flow_conditioned_on_action": rounded(scene_flow),
        "paper_metrics": {
            "parameters_billion": 1.7,
            "libero_average_success_percent": 99.8,
            "libero_plus_success_percent": 89.1,
            "robotwin_clean_success_percent": 92.94,
            "robotwin_randomized_success_percent": 93.04,
            "vlabench_success_percent": 56.2,
            "action_only_latency_ms_rtx5090d": 55.77,
            "actions_and_both_flows_1024_queries_latency_ms": 305.49,
            "moving_ade_cm": 0.427,
            "moving_fde_cm": 0.825,
            "moving_acc_at_1cm_percent": 92.31,
        },
        "boundary": "toy flow is not an official reproduction; camera intrinsics and scene coupling are illustrative.",
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=float, default=2.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    result = run_demo(args.depth)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

