"""Pure-geometry error of a collision representation: where is its hole wall?

No dynamics or solver: rays are cast outward from the hole axis at many
angles and heights, and the first point each representation treats as solid
is compared with the true wall (the CoACD-style one-sided surface distance,
measured where insertion happens). Convex hulls are intersected exactly (slab
clipping); an SDF plate is evaluated from MuJoCo's own compiled octree, with
the same leaf lookup and trilinear interpolation as oct_distance().

Sign matches the probes: negative = the representation's wall stands in free
space (phantom material), positive = the wall is recessed (extra room).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from . import geometry as geo
from .geometry import Task
from .representations import PlateRep

N_THETA = 720
# Heights as fractions of the plate thickness; dense near both faces, where
# octree interpolation rounds the rim edges.
Z_FRACTIONS = (0.002, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99, 0.998)
BAND = 2e-3  # search from 2 mm inside the true wall to 2 mm into the plate


def _rays(task: Task):
    a = np.linspace(0, 2 * math.pi, N_THETA, endpoint=False)
    u = np.column_stack([np.cos(a), np.sin(a), np.zeros_like(a)])
    zs = np.array(Z_FRACTIONS) * task.thickness
    origins = np.repeat(np.column_stack([np.zeros((len(zs), 2)), zs]), len(u), axis=0)
    dirs = np.tile(u, (len(zs), 1))
    # True wall crossing along each ray (circle: R; square: first face hit).
    ux, uy = np.abs(dirs[:, 0]), np.abs(dirs[:, 1])
    t_true = (np.full(len(dirs), task.size) if task.shape == 'circle'
              else task.size / np.maximum(ux, uy))
    return origins, dirs, t_true


def _first_hit_hulls(hulls, T: float, origins, dirs, t_lo, t_hi) -> np.ndarray:
    """Smallest t in [t_lo, t_hi] at which the ray enters any hull (inf if none)."""
    lift = np.array([0.0, 0.0, T / 2])  # plate body frame -> world
    best = np.full(len(dirs), np.inf)
    for h in hulls:
        eq = ConvexHull(h + lift).equations  # n.x + d <= 0 inside
        n, d = eq[:, :3], eq[:, 3]
        nu = dirs @ n.T                       # (rays, planes)
        rhs = -(origins @ n.T + d)            # need t * nu <= rhs
        # A ray running along a face (e.g. the seam shared by two adjacent
        # wedges) is parallel to it: it counts as inside unless it lies
        # measurably outside the face.
        par = np.abs(nu) < 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            t = rhs / nu
        enter = np.where(~par & (nu < 0), t, -np.inf).max(axis=1)
        leave = np.where(~par & (nu > 0), t, np.inf).min(axis=1)
        blocked = (par & (rhs < -1e-12)).any(axis=1)
        enter = np.maximum(enter, t_lo)
        hit = (~blocked) & (enter <= np.minimum(leave, t_hi))
        best = np.where(hit, np.minimum(best, enter), best)
    return best


def octree_sdf(model: mujoco.MjModel, meshid: int, p: np.ndarray) -> np.ndarray:
    """Vectorised MuJoCo oct_distance() for points p (N, 3) in the mesh frame."""
    adr, num = model.mesh_octadr[meshid], model.mesh_octnum[meshid]
    aabb = model.oct_aabb[adr:adr + num].reshape(-1, 6)
    child = model.oct_child[adr:adr + num].reshape(-1, 8)
    coeff = model.oct_coeff[adr:adr + num].reshape(-1, 8)
    node = np.zeros(len(p), dtype=int)
    for _ in range(64):
        leaf = (child[node] == -1).all(axis=1)
        if leaf.all():
            break
        c = aabb[node]
        coord = (p - (c[:, :3] - c[:, 3:])) / (2 * c[:, 3:])
        octant = ((coord[:, 0] >= .5) + 2 * (coord[:, 1] >= .5)
                  + 4 * (coord[:, 2] >= .5)).astype(int)
        node = np.where(leaf, node, child[node, octant])
    c = aabb[node]
    coord = (p - (c[:, :3] - c[:, 3:])) / (2 * c[:, 3:])
    w = np.ones((len(p), 8))
    for j in range(8):
        for axis, bit in ((0, 1), (1, 2), (2, 4)):
            w[:, j] *= np.where(j & bit, coord[:, axis], 1 - coord[:, axis])
    sdf = (w * coeff[node]).sum(axis=1)
    root = aabb[0]
    box = np.linalg.norm(np.maximum(np.abs(p - root[:3]) - root[3:], 0), axis=1)
    return np.where(box > 0, sdf + box, sdf)


def _first_hit_sdf(task: Task, rep: PlateRep, origins, dirs, t_lo, t_hi,
                   step: float = 5e-6) -> np.ndarray:
    from .scene import build_scene
    sc = build_scene(task, rep, 0.0, joints='xy', gravity=False,
                     peg_pos=(0, 0, 10 * task.peg_length))  # peg far away
    m, d = sc.model, mujoco.MjData(sc.model)
    mujoco.mj_kinematics(m, d)
    g = int(np.flatnonzero(sc.plate_geoms)[0])
    pos, rot = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
    ts = np.arange(0.0, 1.0 + 1e-12, step / (t_hi[0] - t_lo[0]))
    best = np.full(len(dirs), np.inf)
    prev_t = prev_s = None
    for f in ts:
        t = t_lo + f * (t_hi - t_lo)
        pts = origins + t[:, None] * dirs
        s = octree_sdf(m, m.geom_dataid[g], (pts - pos) @ rot)
        if prev_s is None:
            best = np.where(s <= 0, t, best)  # already solid at the band start
        else:
            cross = np.isinf(best) & (prev_s > 0) & (s <= 0)
            # Linear interpolation of the zero crossing (the SDF is trilinear).
            frac = prev_s / np.where(prev_s - s == 0, 1, prev_s - s)
            best = np.where(cross, prev_t + frac * (t - prev_t), best)
        prev_t, prev_s = t, s
    return best


def wall_offset(task: Task, rep: PlateRep) -> dict:
    """Signed wall position error over N_THETA x len(Z_FRACTIONS) rays (µm)."""
    origins, dirs, t_true = _rays(task)
    t_lo, t_hi = np.maximum(t_true - BAND, 0.0), t_true + BAND
    if rep.sdf_mesh is not None:
        t_rep = _first_hit_sdf(task, rep, origins, dirs, t_lo, t_hi)
    else:
        t_rep = _first_hit_hulls(rep.hulls, task.thickness, origins, dirs, t_lo, t_hi)
    t_rep = np.where(np.isinf(t_rep), t_hi, t_rep)  # no wall within the band
    hit = origins + t_rep[:, None] * dirs
    # Distance of the representation's wall point from the true wall, signed
    # so that a wall standing in free space is negative.
    offset = geo.hole_sdf2(task, hit[:, :2]) * 1e6
    # Mid-wall (10-90 % of the thickness) separates the wall itself from
    # rounding of the rim edges within a few tens of µm of the plate faces.
    frac = origins[:, 2] / task.thickness
    mid = offset[(frac >= 0.1) & (frac <= 0.9)]
    return {'wall_min_um': float(offset.min()), 'wall_max_um': float(offset.max()),
            'wall_rms_um': float(np.sqrt(np.mean(offset ** 2))),
            'wall_mid_min_um': float(mid.min()), 'wall_mid_max_um': float(mid.max()),
            'wall_frac_phantom': float(np.mean(offset < -1.0)),
            'wall_band_um': BAND * 1e6}
