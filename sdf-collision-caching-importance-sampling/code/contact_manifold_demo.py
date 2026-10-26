#!/usr/bin/env python3
"""二维 SDF 接触流形教学演示，只使用 Python 标准库。

对应论文 Eq. 2、3、6、7 和 Algorithm 1。场景是带两个支脚的组合形状
缓慢平移经过地面，两个支脚产生分离的接触区域。几何与运动是教学设置，
不是论文实验；这里没有刚体动力学、摩擦、CCD 或 GPU 后端。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

Point = tuple[float, float]
Box = tuple[Point, Point]


def add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def scale(a: Point, s: float) -> Point:
    return a[0] * s, a[1] * s


def norm(a: Point) -> float:
    return math.hypot(*a)


def box_sdf(p: Point, center: Point, half: Point) -> float:
    q = (abs(p[0] - center[0]) - half[0], abs(p[1] - center[1]) - half[1])
    return math.hypot(max(q[0], 0), max(q[1], 0)) + min(max(q), 0)


COMPONENTS: tuple[Box, ...] = (
    ((-0.3, 0.14), (0.09, 0.20)),
    ((0.3, 0.14), (0.09, 0.20)),
    ((0.0, 0.31), (0.30, 0.04)),
)
GROUND: Box = ((0.0, -0.15), (0.60, 0.15))


def body_sdf(p: Point, offset: float = 0) -> float:
    # min 保持并集的内外符号；重叠组件内部不一定保持精确欧氏距离。
    local = (p[0] - offset, p[1])
    return min(box_sdf(local, c, h) for c, h in COMPONENTS)


def laplacian(p: Point, delta: float = 0.001) -> float:
    # 二维版本的 Laplacian，用中心差分累加两个坐标方向。
    center = body_sdf(p)
    return sum(
        (body_sdf(add(p, e)) - 2 * center + body_sdf(sub(p, e))) / delta**2
        for e in ((delta, 0.0), (0.0, delta))
    )


def curvature_seeds() -> list[Point]:
    candidates: set[Point] = set()
    for c, h in COMPONENTS:
        for i in range(17):
            t = 2 * i / 16 - 1
            for p in ((c[0] + t * h[0], c[1] - h[1]),
                      (c[0] + t * h[0], c[1] + h[1]),
                      (c[0] - h[0], c[1] + t * h[1]),
                      (c[0] + h[0], c[1] + t * h[1])):
                if abs(body_sdf(p)) < 1e-8 and abs(laplacian(p)) > 4:
                    candidates.add(p)
    return sorted(candidates)


def smooth_max(a: float, b: float, epsilon: float) -> float:
    return 0.5 * (a + b + math.sqrt((a - b) ** 2 + epsilon))


def project(p: Point, domain: tuple[Point, Point]) -> Point:
    lo, hi = domain
    return tuple(max(lo[i], min(hi[i], p[i])) for i in range(2))


def nelder_mead(objective, seed: Point, domain, tolerance: float, max_iterations: int):
    """有界二维 simplex 搜索，返回极小值点和迭代数。"""
    seed = project(seed, domain)
    lo, hi = domain
    # 教学实现的额外处理：边界种子略向盒中心移动，避免投影把 simplex 压成一条线。
    if any(abs(seed[i] - lo[i]) < 1e-9 or abs(seed[i] - hi[i]) < 1e-9 for i in range(2)):
        seed = add(scale(seed, 0.9), scale(add(lo, hi), 0.05))
    vertices = [seed]
    for axis in range(2):
        step = max((hi[axis] - lo[axis]) * 0.1, 0.5 * tolerance)
        direction = step if seed[axis] + step <= hi[axis] else -step
        p = list(seed)
        p[axis] += direction
        vertices.append(project(tuple(p), domain))
    values = [objective(p) for p in vertices]
    for iteration in range(max_iterations):
        ordered = sorted(zip(values, vertices), key=lambda z: z[0])
        values, vertices = [z[0] for z in ordered], [z[1] for z in ordered]
        if max(norm(sub(p, vertices[0])) for p in vertices[1:]) < tolerance:
            return vertices[0], iteration
        centroid = scale(add(vertices[0], vertices[1]), 0.5)
        reflected = project(sub(scale(centroid, 2), vertices[2]), domain)
        fr = objective(reflected)
        if fr < values[0]:
            expanded = project(add(centroid, scale(sub(reflected, centroid), 2)), domain)
            fe = objective(expanded)
            vertices[2], values[2] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[1]:
            vertices[2], values[2] = reflected, fr
        else:
            target = reflected if fr < values[2] else vertices[2]
            contracted = scale(add(centroid, target), 0.5)
            fc = objective(contracted)
            if fc < min(fr, values[2]):
                vertices[2], values[2] = contracted, fc
            else:
                for i in (1, 2):
                    vertices[i] = scale(add(vertices[0], vertices[i]), 0.5)
                    values[i] = objective(vertices[i])
    best = min(range(3), key=lambda i: values[i])
    return vertices[best], max_iterations


@dataclass
class Contact:
    x: Point
    phi_a: float
    phi_b: float
    cached: bool

    @property
    def active(self) -> bool:
        # 接触判断查两个原始场。softmax 或含排斥项的 g 的符号不能代替它。
        return self.phi_a < 0 and self.phi_b < 0


def simulate(args) -> tuple[list[dict], list[Contact]]:
    rng = random.Random(args.seed)
    feature_points = curvature_seeds()
    cache: list[Contact] = []
    frames = []
    dt, speed = 0.016, 0.08
    for frame in range(args.frames):
        offset = speed * dt * frame
        domain = ((max(offset - 0.39, -0.60), -0.06), (min(offset + 0.39, 0.60), 0.0))
        lo, hi = domain
        if any(lo[i] >= hi[i] for i in range(2)):
            cache, manifold = [], []
            frames.append({'frame': frame, 'offset': offset, 'objective_evaluations': 0,
                           'individual_sdf_queries_in_objective': 0, 'iterations': 0,
                           'candidate_count': 0, 'active_count': 0,
                           'curvature_candidates_in_overlap': 0,
                           'contact_regions_found': [], 'points': []})
            continue
        features = [add(p, (offset, 0.0)) for p in feature_points
                    if all(lo[i] - 1e-9 <= add(p, (offset, 0))[i] <= hi[i] + 1e-9 for i in range(2))]
        manifold: list[Contact] = []
        queries, iterations = 0, 0

        def objective(p: Point) -> float:
            nonlocal queries
            queries += 1
            base = smooth_max(body_sdf(p, offset), box_sdf(p, *GROUND), args.epsilon)
            return base - args.alpha * sum(norm(sub(p, c.x)) for c in manifold)

        seeds = [(add(c.x, (speed * dt, 0)), True) for c in cache]
        for _ in range(args.random + max(0, args.cache - len(cache))):
            if args.sampling == 'curvature' and features:
                seed = rng.choice(features)
            else:
                seed = (rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]))
            seeds.append((seed, False))
        for seed, cached in seeds:
            p, count = nelder_mead(objective, seed, domain, args.tolerance, 50)
            iterations += count
            manifold.append(Contact(p, body_sdf(p, offset), box_sdf(p, *GROUND), cached))
        unique: list[Contact] = []
        for c in manifold:
            if all(norm(sub(c.x, q.x)) >= args.tolerance for q in unique):
                unique.append(c)
        manifold = unique

        def rank(c: Contact):
            return smooth_max(c.phi_a, c.phi_b, args.epsilon) - args.alpha * sum(
                norm(sub(c.x, q.x)) for q in manifold)

        cache = sorted(manifold, key=rank)[:args.cache]
        active = [c for c in manifold if c.active]
        frames.append({
            'frame': frame, 'offset': offset, 'objective_evaluations': queries,
            'individual_sdf_queries_in_objective': 2 * queries,
            'iterations': iterations, 'candidate_count': len(manifold),
            'active_count': len(active), 'curvature_candidates_in_overlap': len(features),
            'contact_regions_found': sorted({('left' if c.x[0] < offset else 'right') for c in active}),
            'points': [{'x': c.x, 'phi_a': c.phi_a, 'phi_b': c.phi_b,
                        'smooth_max': smooth_max(c.phi_a, c.phi_b, args.epsilon),
                        'active': c.active, 'cached': c.cached} for c in manifold],
        })
    return frames, manifold


def svg_result(frames: list[dict], contacts: list[Contact]) -> str:
    offset = frames[-1]['offset']
    def sx(x): return 430 + 490 * x
    def sy(y): return 280 - 490 * y
    shapes = []
    for c, h in (*COMPONENTS, GROUND):
        moving = (c, h) != GROUND
        cx = c[0] + (offset if moving else 0)
        shapes.append(f'<rect x="{sx(cx-h[0]):.2f}" y="{sy(c[1]+h[1]):.2f}" width="{980*h[0]:.2f}" height="{980*h[1]:.2f}" fill="{"#216d69" if moving else "#d6d0c4"}" fill-opacity="0.68" stroke="#285954"/>')
    for c in contacts:
        color = '#b55220' if c.active else '#777'
        shapes.append(f'<circle cx="{sx(c.x[0]):.2f}" cy="{sy(c.x[1]):.2f}" r="5" fill="{color}" stroke="white"/>')
    return '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="490" viewBox="0 0 860 490">' + \
        '<rect width="860" height="490" fill="#faf7ef"/><g font-family="sans-serif" fill="#243c3a"><text x="35" y="38" font-size="22">二维教学场景：两个接触区域</text><text x="35" y="65" font-size="14">橙色为同时处于两个原始场内部的接触点；无动力学仿真</text></g>' + \
        ''.join(shapes) + '<text x="35" y="465" font-family="sans-serif" fill="#405c59" font-size="14">几何、运动和查询计数均来自本演示，不是论文 benchmark。</text></svg>'


def self_check() -> None:
    assert box_sdf((0, 0), (0, 0), (1, 1)) == -1
    assert math.isclose(box_sdf((2, 0), (0, 0), (1, 1)), 1)
    for a in [-1, -0.01, 0, 0.5]:
        for b in [-2, -0.05, 0, 1]:
            s = smooth_max(a, b, 0.1)
            assert -1e-12 <= s - max(a, b) <= math.sqrt(0.1) / 2 + 1e-12
    assert smooth_max(-0.03, -0.03, 0.1) > 0
    point, _ = nelder_mead(lambda p: (p[0] - 0.2)**2 + (p[1] + 0.1)**2,
                           (-0.7, 0.8), ((-1, -1), (1, 1)), 1e-6, 200)
    assert norm(sub(point, (0.2, -0.1))) < 2e-6, point
    assert curvature_seeds(), '场景必须包含高曲率种子'
    print('公式上界、浅重叠反例、有界 Nelder–Mead 和曲率种子检查通过。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--cache', type=int, default=4)
    parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--sampling', choices=['curvature', 'uniform'], default='curvature')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--tolerance', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.frames < 1 or args.cache < 0 or args.random < 0 or args.cache + args.random < 1:
        parser.error('frames 必须为正，cache/random 非负且总和至少为 1')
    if not all(math.isfinite(v) for v in (args.epsilon, args.alpha, args.tolerance)) or args.epsilon < 0 or args.alpha < 0 or args.tolerance <= 0:
        parser.error('epsilon/alpha 必须有限且非负，tolerance 必须有限且为正')
    if args.self_check:
        self_check()
    frames, manifold = simulate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {'scope': '二维几何优化教学演示，不是论文实验复现',
               'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'frames': frames}
    (args.output_dir / 'contacts.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'contact_manifold.svg').write_text(svg_result(frames, manifold))
    print(f'生成 {len(frames)} 帧，末帧接触区域：{frames[-1]["contact_regions_found"]}')
    print(f'结果目录：{args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
