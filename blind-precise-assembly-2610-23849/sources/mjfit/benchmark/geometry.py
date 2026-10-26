"""Exact task geometry and simulator-independent ground truth.

A task is a static square plate occupying 0 <= z <= T with a vertical
through-hole, and a straight peg whose cross-section is the hole outline shrunk
by a radial clearance c (c < 0 makes the peg wider than the hole). The peg is
always an exact MuJoCo primitive (cylinder or box), so only the plate's
collision representation varies between methods.

Ground truth comes from the analytic outlines (a true circle, not a
tessellation), so every representation, mjfit's included, is charged for its
own approximation error. All lengths are meters.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

SHAPES = ('circle', 'square')


@dataclass(frozen=True)
class Task:
    shape: str                 # 'circle' | 'square'
    size: float = 0.008        # hole radius (circle) or half-width (square)
    thickness: float = 0.015   # plate thickness T
    plate_half: float = 0.024  # half-width of the square plate
    peg_length: float = 0.040  # long enough to span the plate while tilted

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"Unknown shape {self.shape!r}; known: {SHAPES}")

    def key(self) -> dict:
        return asdict(self)

    def peg_half(self, clearance: float) -> float:
        """Peg radius (circle) or half-width (square)."""
        return self.size - clearance


# ----- signed distance functions ------------------------------------------

def _box2(xy: np.ndarray, half: float) -> np.ndarray:
    q = np.abs(xy) - half
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def hole_sdf2(task: Task, xy: np.ndarray) -> np.ndarray:
    """2D signed distance to the hole outline (negative inside the hole)."""
    if task.shape == 'circle':
        return np.linalg.norm(xy, axis=-1) - task.size
    return _box2(xy, task.size)


def plate_sdf(task: Task, p: np.ndarray) -> np.ndarray:
    """Signed distance to the plate solid (negative inside material)."""
    region = np.maximum(-hole_sdf2(task, p[..., :2]), _box2(p[..., :2], task.plate_half))
    slab = np.abs(p[..., 2] - task.thickness / 2) - task.thickness / 2
    w = np.stack([region, slab], axis=-1)
    return np.minimum(w.max(axis=-1), 0) + np.linalg.norm(np.maximum(w, 0), axis=-1)


def peg_sdf(task: Task, clearance: float, p_local: np.ndarray) -> np.ndarray:
    """Signed distance to the peg, with p_local in the peg body frame."""
    h, half_len = task.peg_half(clearance), task.peg_length / 2
    if task.shape == 'circle':
        radial = np.linalg.norm(p_local[..., :2], axis=-1) - h
    else:
        radial = _box2(p_local[..., :2], h)
    w = np.stack([radial, np.abs(p_local[..., 2]) - half_len], axis=-1)
    return np.minimum(w.max(axis=-1), 0) + np.linalg.norm(np.maximum(w, 0), axis=-1)


# ----- surface samples for penetration checks ----------------------------------

def _outline(task: Task, half: float, spacing: float) -> np.ndarray:
    """Points on a closed outline of the given radius / half-width."""
    if task.shape == 'circle':
        n = max(64, math.ceil(2 * math.pi * half / spacing))
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return half * np.column_stack([np.cos(a), np.sin(a)])
    n = max(8, math.ceil(2 * half / spacing))
    s = np.linspace(-half, half, n, endpoint=False)
    ones = np.full(n, half)
    return np.vstack([np.column_stack([s, -ones]), np.column_stack([ones, s]),
                      np.column_stack([-s, ones]), np.column_stack([-ones, -s])])


def peg_samples(task: Task, clearance: float, spacing: float = 1e-4,
                z_spacing: float = 5e-4) -> np.ndarray:
    """Peg side-wall rings from end to end, peg frame. The end rings are the
    end-face rims, where a prism first enters a plate; rim_samples covers the
    plate edges entering the peg, so rings can be sparser along z."""
    ring = _outline(task, task.peg_half(clearance), spacing)
    zs = np.linspace(-task.peg_length / 2, task.peg_length / 2,
                     max(2, math.ceil(task.peg_length / z_spacing) + 1))
    return np.concatenate([np.column_stack([ring, np.full(len(ring), z)]) for z in zs])


def rim_samples(task: Task, spacing: float = 5e-5) -> np.ndarray:
    """Hole edges at the plate's top and bottom faces, world frame."""
    ring = _outline(task, task.size, spacing)
    return np.vstack([np.column_stack([ring, np.full(len(ring), z)])
                      for z in (0.0, task.thickness)])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def penetration(task: Task, clearance: float, pos: np.ndarray, rot: np.ndarray,
                peg_pts: np.ndarray, rim_pts: np.ndarray) -> float:
    """True penetration depth of the peg at pose (pos, rot) into the plate.

    The larger of two one-sided depths: peg surface points inside the plate,
    and plate hole-rim points inside the peg. Converges to the exact depth as
    sampling densifies because prism/prism overlap is deepest at edges.
    """
    into_plate = -plate_sdf(task, peg_pts @ rot.T + pos).min()
    into_peg = -peg_sdf(task, clearance, (rim_pts - pos) @ rot).min()
    return max(0.0, into_plate, into_peg)


def true_signed_distance(task: Task, clearance: float, pos: np.ndarray, rot: np.ndarray,
                         peg_pts: np.ndarray, rim_pts: np.ndarray) -> float:
    """True signed distance between peg and plate at a pose: the separation
    if apart, minus the penetration depth if overlapping. Evaluated whenever
    the simulator reports contact, |sd| is the `min-|sd|` measure of Son & Kim
    (LOCC, RSS 2023): zero for a perfect simulator, > 0 a false contact
    through a gap, < 0 real overlap."""
    pen = penetration(task, clearance, pos, rot, peg_pts, rim_pts)
    if pen > 0:
        return -pen
    gap_peg = plate_sdf(task, peg_pts @ rot.T + pos).min()
    gap_rim = peg_sdf(task, clearance, (rim_pts - pos) @ rot).min()
    return float(max(0.0, min(gap_peg, gap_rim)))


def contact_stats(sd: np.ndarray) -> dict:
    """Summary of true signed distances at contact-reporting states (µm)."""
    if len(sd) == 0:
        return {'sd_n': 0, 'sd_abs_mean_um': np.nan, 'sd_false_max_um': np.nan,
                'sd_pen_max_um': np.nan}
    sd = np.asarray(sd) * 1e6
    return {'sd_n': len(sd), 'sd_abs_mean_um': float(np.abs(sd).mean()),
            'sd_false_max_um': float(max(sd.max(), 0.0)),
            'sd_pen_max_um': float(max(-sd.min(), 0.0))}


def contact_gap(task: Task, clearance: float, pos: np.ndarray, rot: np.ndarray,
                points: np.ndarray) -> np.ndarray:
    """True local gap at simulated contact points: sd_plate + sd_peg.

    Zero for a contact on touching true surfaces, positive for a phantom
    contact (force through empty space), negative for true overlap.
    """
    return plate_sdf(task, points) + peg_sdf(task, clearance, (points - pos) @ rot)


# ----- oracles: exact answers for the probes ---------------------------------

def translation_gap(task: Task, clearance: float, xy: np.ndarray) -> np.ndarray:
    """Signed gap for an aligned peg translated by xy (rotation locked).

    The feasible translation set is a disk (circle) or square (square) of
    radius / half-width c. Inside it the value is the true peg-to-wall gap;
    outside it is minus the translation needed to fit again.
    """
    if task.shape == 'circle':
        return clearance - np.linalg.norm(xy, axis=-1)
    return -_box2(xy, clearance)


def translation_support(task: Task, clearance: float, u: np.ndarray) -> np.ndarray:
    """Max displacement along unit u of a rotation-locked peg (S2 answer)."""
    if task.shape == 'circle':
        return np.full(u.shape[:-1], clearance)
    return clearance * np.abs(u).sum(axis=-1)


def max_tilt(task: Task, clearance: float, phi: float = 0.0) -> float:
    """Max tilt about horizontal axis (cos phi, sin phi, 0), lateral free.

    The peg spans both plate faces, so it fits iff its slices at z = 0 and
    z = T both fit the hole under one common translation.
    """
    r, R, half_t = task.peg_half(clearance), task.size, task.thickness / 2
    if task.shape == 'circle':
        # Slices are ellipses offset along their major axis:
        # (T/2) tan(theta) + r / cos(theta) <= R.
        return math.acos(r / math.hypot(R, half_t)) - math.atan2(half_t, R)

    def fits(theta: float) -> bool:
        pts = _tilted_slices(r, half_t, phi, theta)
        return bool(np.all(pts.max(axis=0) - pts.min(axis=0) <= 2 * R))
    return _bisect(fits, 0.0, 0.5)


def _tilted_slices(half: float, half_t: float, phi: float, theta: float) -> np.ndarray:
    """Slice vertices of a tilted square prism at z = +-T/2 (rotation center)."""
    k = np.array([math.cos(phi), math.sin(phi), 0.0])
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
    axis = rot[:, 2]
    corners = np.array([[s1 * half, s2 * half, 0.0] for s1 in (-1, 1) for s2 in (-1, 1)]) @ rot.T
    out = [c + ((z - c[2]) / axis[2]) * axis for z in (-half_t, half_t) for c in corners]
    return np.array(out)[:, :2]


def max_yaw(task: Task, clearance: float) -> float:
    """Max rotation about z for a square peg in a square hole, lateral free."""
    if task.shape != 'square':
        raise ValueError("Yaw is unconstrained for a circular peg.")
    a, s = task.size, task.peg_half(clearance)
    # Rotated square's half-width s (cos psi + sin psi) must stay <= a.
    return math.asin(min(1.0, a / (s * math.sqrt(2)))) - math.pi / 4


def _bisect(fits, lo: float, hi: float, iters: int = 80) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


# ----- meshes for the baselines ------------------------------------------------

def hole_polygon(task: Task, n_circle: int = 1024) -> np.ndarray:
    """Hole outline as a CCW polygon; circles use vertices on the true circle
    (sagitta R (1 - cos(pi/n)) is 0.04 um at R = 8 mm, n = 1024)."""
    if task.shape == 'circle':
        a = np.linspace(0, 2 * math.pi, n_circle, endpoint=False)
        return task.size * np.column_stack([np.cos(a), np.sin(a)])
    a = task.size
    return np.array([[-a, -a], [a, -a], [a, a], [-a, a]])


def plate_mesh(task: Task, n_circle: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Watertight plate-with-hole triangle mesh, plate body frame (z centered)."""
    import shapely.geometry
    import trimesh
    p = task.plate_half
    outer = [(-p, -p), (p, -p), (p, p), (-p, p)]
    poly = shapely.geometry.Polygon(outer, [hole_polygon(task, n_circle)[::-1]])
    mesh = trimesh.creation.extrude_polygon(poly, task.thickness)
    mesh.apply_translation([0, 0, -task.thickness / 2])
    if not mesh.is_watertight:
        raise RuntimeError("plate mesh is not watertight")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
