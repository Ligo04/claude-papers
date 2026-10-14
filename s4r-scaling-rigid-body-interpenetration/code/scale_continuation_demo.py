#!/usr/bin/env python3
"""A tiny 2-D teaching model of S4R's scale-continuation idea.

This is deliberately not the paper's triangle-mesh/FCL solver.  Bodies are
circles and the contact QP is approximated by cyclic projections of linear
pair constraints.  It makes the important distinction visible: the
interpenetrating full-size state is first made easy by shrinking bodies, then
scale is restored while small minimum-norm corrections are applied.
"""
from __future__ import annotations
import argparse, math
from dataclasses import dataclass

@dataclass
class Disk:
    x: float
    y: float
    r: float
    x0: float
    y0: float

def pair_correction(bodies: list[Disk], scale: float, margin: float, passes: int = 80) -> None:
    """Approximate min-norm QP by repeated orthogonal projections.

    For a violated pair, split the required displacement equally along the
    separating normal.  A real S4R step solves all active rows together with
    OSQP; this small projection loop is only for intuition.
    """
    for _ in range(passes):
        changed = False
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                a, b = bodies[i], bodies[j]
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                target = scale * (a.r + b.r) + margin
                if d >= target:
                    continue
                if d < 1e-12:
                    nx, ny = 1.0, 0.0
                else:
                    nx, ny = dx / d, dy / d
                push = 0.5 * (target - d)
                a.x -= push * nx; a.y -= push * ny
                b.x += push * nx; b.y += push * ny
                changed = True
        if not changed:
            break

def penetration_count(bodies: list[Disk], scale: float, margin: float = 0.0) -> int:
    count = 0
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            if math.hypot(b.x-a.x, b.y-a.y) < scale*(a.r+b.r)+margin:
                count += 1
    return count

def solve(scale_min: float = 0.05, steps: int = 20, margin: float = 0.02) -> list[Disk]:
    # Same centers are intentionally overlapped at full size.
    bodies = [Disk(-0.20, 0.0, 0.45, -0.20, 0.0),
              Disk( 0.16, 0.03, 0.42,  0.16, 0.03),
              Disk( 0.02, 0.35, 0.30,  0.02, 0.35)]
    pair_correction(bodies, scale_min, margin)
    print(f"start scale={scale_min:.3f}: pen={penetration_count(bodies, scale_min, margin)}")
    for k in range(steps):
        s0 = scale_min + (1.0-scale_min)*k/steps
        s1 = scale_min + (1.0-scale_min)*(k+1)/steps
        pair_correction(bodies, s1, margin)
        if k in (0, steps//2, steps-1):
            print(f"step {k+1:02d}/{steps}: scale={s1:.3f}, pen={penetration_count(bodies,s1,margin)}")
    final_pen = penetration_count(bodies, 1.0, 0.0)
    rmsd = math.sqrt(sum((b.x-b.x0)**2+(b.y-b.y0)**2 for b in bodies)/len(bodies))
    print(f"full-size evaluator: pen={final_pen}, center_RMSD={rmsd:.4f}")
    return bodies

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scale-min', type=float, default=0.05)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--margin', type=float, default=0.02)
    args = p.parse_args()
    solve(args.scale_min, args.steps, args.margin)
