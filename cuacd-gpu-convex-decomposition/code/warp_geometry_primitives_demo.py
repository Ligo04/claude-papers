#!/usr/bin/env python3
"""CuACD 原语教学：CPU 模拟 ballot 压缩、分段排序与二维凸包安全预筛选。

仅用标准库。数据是固定种子的合成教学输入，没有 GPU 计时或论文复现结果。
短段排序使用 Python sorted；三维整数凸包、切割、搜索和 allocator 均未实现。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

WARP = 32


def compact(values, keep):
    """模拟 ballot + prefix popcount；不足 32 个元素时只设置有效 lane 位。"""
    output, trace = [], []
    for start in range(0, len(values), WARP):
        chunk = values[start:start + WARP]
        mask = sum(1 << lane for lane, value in enumerate(chunk) if keep(value))
        base = len(output)
        packed = [None] * mask.bit_count()
        slots = []
        for lane, value in enumerate(chunk):
            if mask & (1 << lane):
                rank = (mask & ((1 << lane) - 1)).bit_count()
                packed[rank] = value
                slots.append({"lane": lane, "destination": base + rank})
        output.extend(packed)
        trace.append({"input_start": start, "active_lanes": len(chunk),
                      "mask_hex": f"0x{mask:08x}", "kept": len(packed),
                      "destinations": slots})
    return output, trace


def warp_sort_model(values):
    """语义示例：32 样本 pivot、三路划分、迭代段栈；不模拟原版原地写回。"""
    result = list(values)
    stack = [(0, len(result))]
    partitions = 0
    while stack:
        begin, end = stack.pop()
        segment = result[begin:end]
        if len(segment) <= WARP:
            # 论文这里使用 warp bitonic network；教学只保留其排序语义。
            result[begin:end] = sorted(segment)
            continue
        sample = [segment[i * (len(segment) - 1) // (WARP - 1)] for i in range(WARP)]
        pivot = sorted(sample)[WARP // 2]
        smaller, _ = compact(segment, lambda x: x < pivot)
        equal, _ = compact(segment, lambda x: x == pivot)
        larger, _ = compact(segment, lambda x: x > pivot)
        result[begin:end] = smaller + equal + larger
        partitions += 1
        # 等值区无需再次排序，保证全相等输入也终止。
        stack.append((begin, begin + len(smaller)))
        stack.append((begin + len(smaller) + len(equal), end))
    return result, partitions


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull2d(points):
    """Andrew monotone chain，二维教学用；返回逆时针极点。"""
    pts = sorted(set(points))
    if len(pts) < 2:
        return pts
    lower, upper = [], []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def prefilter2d(points):
    """八方向二维类比；论文三维用 40 方向 / 最多 80 个极值见证点。"""
    if not points:
        return [], []
    directions = [(math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)) for i in range(8)]
    witnesses = [max(points, key=lambda p: p[0] * d[0] + p[1] * d[1]) for d in directions]
    inner = hull2d(witnesses)
    if len(inner) < 3:
        return list(points), inner
    def strict_inside(point):
        # 保守处理边界：不删除边上或非常接近边的点。
        return all(cross(inner[i], inner[(i + 1) % len(inner)], point) > 1e-12
                   for i in range(len(inner)))
    survivors = [point for point in points if not strict_inside(point)]
    return survivors, inner


def self_test():
    """检查守恒、排序语义、最后不足一 warp 的批次、重复值和 hull 不变性。"""
    rng = random.Random(31)
    cases = [[], [1], [5] * 129, list(range(97)), list(range(97))[::-1]]
    cases += [[rng.randrange(-8, 9) for _ in range(n)] for n in (31, 32, 33, 63, 64, 65, 257)]
    for values in cases:
        got, _ = warp_sort_model(values)
        assert got == sorted(values), "排序与参考结果不一致"
        assert Counter(got) == Counter(values), "元素丢失或重复"
        kept, trace = compact(values, lambda x: x % 2 == 0)
        assert kept == [x for x in values if x % 2 == 0]
        positions = [x["destination"] for batch in trace for x in batch["destinations"]]
        assert positions == list(range(len(kept))), "输出位置冲突或存在空洞"
    point_sets = [[], [(0, 0)], [(0, 0), (1, 0), (2, 0)],
                  [(0, 0), (1, 0), (1, 1), (0, 1), (.5, .5), (.5, 0)]]
    point_sets += [[(rng.random(), rng.random()) for _ in range(n)] for n in (33, 128, 2048)]
    for points in point_sets:
        survivors, _ = prefilter2d(points)
        assert hull2d(survivors) == hull2d(points), "预筛选改变了凸包"
    return {"sorting_and_compaction_cases": len(cases), "hull_cases": len(point_sets), "passed": True}


def svg_geometry(points, survivors, inner, hull):
    def xy(p):
        return 25 + 450 * p[0], 475 - 450 * p[1]
    def polygon(vertices, color):
        coords = " ".join(f"{xy(p)[0]:.2f},{xy(p)[1]:.2f}" for p in vertices)
        return f'<polygon points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="500" height="540" viewBox="0 0 500 540">',
             '<rect width="500" height="540" fill="#faf8f2"/>']
    kept = set(survivors)
    for point in points:
        x, y = xy(point)
        color = "#bc654f" if point in kept else "#c5cdc6"
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.5" fill="{color}"/>')
    parts += [polygon(inner, "#bc654f"), polygon(hull, "#216454"),
              '<text x="25" y="510" font-size="13">二维教学输入：绿色为最终 hull，珊瑚色为幸存点 / 内部 hull</text>', '</svg>']
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not 3 <= args.points <= 100000:
        parser.error("--points 必须在 3 到 100000 之间")
    rng = random.Random(args.seed)
    values = [rng.randrange(16) for _ in range(65)]
    ordered, partitions = warp_sort_model(values)
    compacted, trace = compact(values, lambda x: x < 8)
    points = [(rng.random(), rng.random()) for _ in range(args.points)]
    survivors, inner = prefilter2d(points)
    before, after = hull2d(points), hull2d(survivors)
    assert ordered == sorted(values) and before == after
    result = {
        "scope": "标准库 CPU 教学模拟；合成输入；不是官方 CUDA 实现或性能复现",
        "seed": args.seed,
        "sort": {"input": values, "output": ordered, "three_way_partitions": partitions},
        "compaction": {"predicate": "value < 8", "output": compacted, "batches": trace},
        "prefilter2d": {"points": len(points), "survivors": len(survivors),
                        "inner_hull_vertices": len(inner), "final_hull_vertices": len(after),
                        "same_hull_as_unfiltered": before == after},
        "tests": self_test() if args.self_test else "未请求",
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        args.json.with_suffix(".svg").write_text(svg_geometry(points, survivors, inner, after))
    print(json.dumps({k: result[k] for k in ("scope", "seed", "prefilter2d", "tests")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
