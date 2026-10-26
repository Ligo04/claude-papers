"""Exact geometry of a bolt and nut, and the geometric ground truth.

The parts are the ones mjfit.thread intends to build: an ISO 68-1 thread
whose teeth are shrunk by c/2 (radially at the crest, along the normal on each
flank), so that facing flanks are c apart when the nut is centred. A thread is
a meridian profile swept along a helix, so a point (r, phi, z) is inside it
exactly when (r, u) lies in the profile, with u = z - P * phi / 2pi taken
modulo the pitch. Membership is therefore exact. Distances are measured in the
meridian plane, which differs from the 3D distance by under 0.1 % at M10
(helix angle 3 deg), far below anything the probes resolve.

Frames: the bolt is a threaded rod along +z with threads on 0 <= z <= L; the
nut's threads span 0 <= z <= Hn of its own frame. Frames whose origins differ
by a whole number of pitches (and no rotation) put the nut centred in its play.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from mjfit import _polygon, _thread

@dataclass(frozen=True)
class ThreadTask:
    major_diameter: float = 0.010   # M10
    pitch: float = 0.0015           # coarse, P = 1.5 mm
    bolt_length: float = 0.016      # threaded rod, threads over [0, L]
    nut_height: float = 0.008
    nut_outer_radius: float = 0.008
    density: float = 7800.0
    shape: str = 'thread'           # row label shared with the peg tables
    # mjfit thread form and its form_params, as sorted (key, value) pairs
    # (values hashable: tuples, not lists).
    form: str = 'iso'
    form_params: tuple = ()
    starts: int = 1                 # threads per pitch window: lead = starts * pitch
    handedness: str = 'right'
    chamfer: float = 0.0            # mjfit.thread's chamfer (both rod ends, both nut faces)

    def key(self) -> dict:
        d = asdict(self)
        if self.form == 'iso' and not self.form_params:
            del d['form'], d['form_params']      # ISO keeps its cache keys
        if self.starts == 1:
            del d['starts']
        if self.handedness == 'right':
            del d['handedness']
        if self.chamfer == 0:
            del d['chamfer']
        return d

    @property
    def lead(self) -> float:
        return self.starts * self.pitch

    @property
    def hand(self) -> int:
        """+1 right-handed, -1 left: the helix rises hand * lead per turn."""
        return 1 if self.handedness == 'right' else -1

    @property
    def params(self) -> dict:
        return dict(self.form_params)

    @property
    def profile(self) -> '_thread.ThreadProfile':
        return _profile(self)

    @property
    def r_major(self) -> float:
        return self.major_diameter / 2

    @property
    def r_minor(self) -> float:
        """Bolt core radius."""
        return self.profile.r_core

    @property
    def r_root(self) -> float:
        """Nut root radius at zero clearance (the major radius for ISO)."""
        return self.profile.r_root

    def r_bore(self, clearance: float) -> float:
        """Nut root (shell inner) radius."""
        return self.r_root + clearance / 2

    def nut_home(self) -> float:
        """Nut frame height that centres the nut on the bolt, on the pitch grid."""
        return round((self.bolt_length - self.nut_height) / 2 / self.pitch) * self.pitch


# ----- meridian profiles -------------------------------------------------------------

@lru_cache(maxsize=64)
def _profile(task: ThreadTask):
    return _thread.thread_profile(task.form, task.major_diameter, task.pitch, task.params)


@lru_cache(maxsize=256)
def _teeth(task: ThreadTask, c: float):
    """mjfit's (bolt, nut) teeth as (dr, u): dr from the core for the bolt,
    from the root for the nut; u in the bolt's phase for both. Each is
    base-left, base-right, then on around (see mjfit._thread._canonical)."""
    prof = _profile(task)
    bolt, nut = prof.teeth(c)
    return bolt - [prof.r_core, 0], nut - [prof.r_root, 0]


def bolt_tooth(task: ThreadTask, c: float) -> np.ndarray:
    """(dr, u) convex polygon of one bolt tooth, dr from the core. The base is
    extended into the core so the union with it has no seam."""
    t = _teeth(task, c)[0]
    b = -2e-4
    return np.vstack([[[b, t[0, 1]], [b, t[1, 1]]], t[1:], t[:1]])


def nut_tooth(task: ThreadTask, c: float) -> np.ndarray:
    """(dr, u) convex polygon of one nut tooth, dr from the root (negative
    toward the axis), u in the bolt's phase. The base is extended out into
    the shell."""
    t = _teeth(task, c)[1]
    b = c / 2 + 2e-4
    return np.vstack([[[b, t[0, 1]]], t[:1], t[:0:-1], [[b, t[1, 1]]]])


def _polygon_sd(q: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Exact signed distance from points q (N, 2) to a polygon (convex, or
    any simple one: see _polygon_sd_general)."""
    if not _polygon.is_convex(poly):
        return _polygon_sd_general(q, poly)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1]
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    length = np.linalg.norm(e, axis=1)
    normal = np.column_stack([e[:, 1], -e[:, 0]]) / length[:, None]  # outward
    rel = q[:, None, :] - a[None]
    inside = (rel * normal[None]).sum(-1).max(axis=1)
    t = np.clip((rel * e[None]).sum(-1) / length**2, 0, 1)
    seg = np.linalg.norm(rel - t[..., None] * e[None], axis=-1).min(axis=1)
    return np.where(inside <= 0, inside, seg)


def _polygon_sd_general(q: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Signed distance to a simple polygon: the distance to its nearest edge,
    negative inside (crossing number)."""
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    rel = q[:, None, :] - a[None]
    t = np.clip((rel * e[None]).sum(-1) / (e ** 2).sum(-1), 0, 1)
    dist = np.linalg.norm(rel - t[..., None] * e[None], axis=-1).min(axis=1)
    y = q[:, 1:2]
    crosses = (a[None, :, 1] > y) != (b[None, :, 1] > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_at = a[None, :, 0] + (y - a[None, :, 1]) * e[None, :, 0] / e[None, :, 1]
    inside = (crosses & (q[:, :1] < x_at)).sum(axis=1) % 2 == 1
    return np.where(inside, -dist, dist)


def _periodic_sd(dr: np.ndarray, u: np.ndarray, poly: np.ndarray, P: float) -> np.ndarray:
    """Distance to the tooth repeated every pitch along u."""
    u = np.mod(u, P)
    return np.minimum.reduce([_polygon_sd(np.column_stack([dr, u + k * P]), poly)
                              for k in (-1, 0, 1)])


def _cyl(p: np.ndarray):
    r = np.hypot(p[:, 0], p[:, 1])
    phi = np.arctan2(p[:, 1], p[:, 0])
    return r, phi


def bolt_sdf(task: ThreadTask, c: float, p: np.ndarray) -> np.ndarray:
    """Signed distance to the bolt, points in the bolt frame (negative inside)."""
    P = task.pitch
    r, phi = _cyl(p)
    u = p[:, 2] - task.hand * task.lead * phi / (2 * math.pi)
    sd = np.minimum(r - task.r_minor,
                    _periodic_sd(r - task.r_minor, u, bolt_tooth(task, c), P))
    sd = np.maximum(sd, np.maximum(-p[:, 2], p[:, 2] - task.bolt_length))
    if task.chamfer:
        # The chamfer cones at both ends of the rod (r - cone radius: its
        # sign, and within the cone's slope of its distance).
        k, r0 = _chamfer_slopes(task, c)[0], task.r_minor
        for z0, towards in ((0.0, 1), (task.bolt_length, -1)):
            sd = np.maximum(sd, r - r0 - k * towards * (p[:, 2] - z0))
    return sd


def nut_sdf(task: ThreadTask, c: float, p: np.ndarray) -> np.ndarray:
    """Signed distance to the nut, points in the nut frame (negative inside)."""
    P = task.pitch
    r, phi = _cyl(p)
    u = p[:, 2] - task.hand * task.lead * phi / (2 * math.pi)
    inner = np.minimum(task.r_bore(c) - r,
                       _periodic_sd(r - task.r_root, u, nut_tooth(task, c), P))
    sd = np.maximum.reduce([inner, r - task.nut_outer_radius,
                            -p[:, 2], p[:, 2] - task.nut_height])
    if task.chamfer:
        # The countersinks at both faces.
        k, r0 = _chamfer_slopes(task, c)[1], task.r_bore(c)
        for z0, towards in ((0.0, 1), (task.nut_height, -1)):
            sd = np.maximum(sd, r0 - k * towards * (p[:, 2] - z0) - r)
    return sd


def _chamfer_slopes(task: ThreadTask, c: float):
    """Radial slopes of the bolt's chamfer (core to crest over the chamfer
    length) and of the nut's countersink (bore to tooth tips), as mjfit
    cuts them."""
    bolt, nut = _teeth(task, c)
    return (bolt[:, 0].max() / task.chamfer,
            (task.r_bore(c) - task.r_root - nut[:, 0].min()) / task.chamfer)


# ----- exact surfaces, sampled ------------------------------------------------------------

def _boundary(poly_pts: list, spacing: float) -> np.ndarray:
    """Points every `spacing` along a (dr, u) polyline."""
    out = []
    for a, b in zip(poly_pts[:-1], poly_pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, math.ceil(np.linalg.norm(b - a) / spacing))
        out.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
    out.append(np.asarray(poly_pts[-1], float)[None])
    return np.concatenate(out)


def _sweep(profile_pts: np.ndarray, base_r: float, phase: float, P: float,
           z0: float, z1: float, spacing: float, rise: float | None = None) -> np.ndarray:
    """Sweep one pitch of (dr, u) boundary along the helix (rising `rise` per
    turn, default P), keeping z in [z0, z1]. The tooth repeats every P along u,
    which covers every start of a multi-start thread."""
    rise = P if rise is None else rise
    r = base_r + profile_pts[:, 0]
    n_phi = max(8, math.ceil(2 * math.pi * r.max() / spacing))
    phi = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    ks = np.arange(math.floor((z0 - P - abs(rise)) / P) - 1, math.ceil(z1 / P) + 2)
    rr, pp, kk = np.meshgrid(np.arange(len(r)), phi, ks, indexing='ij')
    z = profile_pts[rr, 1] + rise * pp / (2 * math.pi) + phase * P + kk * P
    keep = (z >= z0) & (z <= z1)
    rad = r[rr][keep]
    return np.column_stack([rad * np.cos(pp[keep]), rad * np.sin(pp[keep]), z[keep]])


def bolt_surface(task: ThreadTask, c: float, z0: float, z1: float,
                 spacing: float = 5e-5) -> np.ndarray:
    """Points on the bolt's threaded surface between heights z0 and z1 (not
    the chamfer cones: thread cut away by a chamfer is dropped)."""
    t = _teeth(task, c)[0]
    # root from the last tooth's base-right, then over the tooth
    pts = [(0.0, t[1, 1] - task.pitch)] + [tuple(v) for v in t[:1]] + \
        [tuple(v) for v in t[:0:-1]]
    prof = _boundary(pts, spacing)
    out = _sweep(prof, task.r_minor, 0.0, task.pitch, max(z0, 0), min(z1, task.bolt_length),
                 spacing, task.hand * task.lead)
    return out[bolt_sdf(task, c, out) < TOL] if task.chamfer else out


def nut_surface(task: ThreadTask, c: float, spacing: float = 5e-5,
                z1: float | None = None) -> np.ndarray:
    """Points on the nut's threaded bore (the only surface the bolt can reach;
    not the countersinks: thread they cut away is dropped), up to height z1
    (default: all of it)."""
    t = _teeth(task, c)[1]
    root = c / 2
    pts = [(root, t[1, 1] - task.pitch), (root, t[0, 1])] + [tuple(v) for v in t[:1]] + \
        [tuple(v) for v in t[:0:-1]] + [(root, t[1, 1])]
    prof = _boundary(pts, spacing)
    top = task.nut_height if z1 is None else min(z1, task.nut_height)
    out = _sweep(prof, task.r_root, 0.0, task.pitch, 0.0, top, spacing,
                 task.hand * task.lead)
    return out[nut_sdf(task, c, out) < TOL] if task.chamfer else out


# ----- relative pose -----------------------------------------------------------------------

def rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


class Pair:
    """Sampled exact surfaces of both parts, for pose queries. The nut pose is
    (pos, R): nut frame -> bolt frame.

    By default the nut is about its home on the bolt's middle. With `tip` it
    is on the bolt's tip instead: the bolt within `tip` of its top face and
    the nut within `tip` of its base, end faces and chamfer cones included."""

    def __init__(self, task: ThreadTask, c: float, spacing: float = 5e-5, margin: float = 2e-3,
                 tip: float | None = None):
        self.task, self.c = task, c
        if tip is None:
            home = task.nut_home()
            self.nut_pts = nut_surface(task, c, spacing)
            self.bolt_pts = bolt_surface(task, c, home - margin, home + task.nut_height + margin,
                                         spacing)
        else:
            L = task.bolt_length
            bolt_face, nut_face = tip_faces(task, c, spacing)
            self.nut_pts = np.concatenate([nut_surface(task, c, spacing, tip), nut_face])
            self.bolt_pts = np.concatenate([bolt_surface(task, c, L - tip, L, spacing),
                                            bolt_face])

    def signed_distance(self, pos, R) -> float:
        """Signed distance between the parts: the gap when apart, minus the
        penetration depth when overlapping (Son & Kim's min-|sd| convention)."""
        t, c = self.task, self.c
        pos = np.asarray(pos, float)
        a = bolt_sdf(t, c, self.nut_pts @ R.T + pos).min()
        b = nut_sdf(t, c, (self.bolt_pts - pos) @ R).min()
        return float(min(a, b))

    def penetration(self, pos, R) -> float:
        return max(0.0, -self.signed_distance(pos, R))


def _bisect(ok, lo: float, hi: float, iters: int = 50) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


# Contact is declared at this depth: well below every probe's resolution, and
# above the sampling noise of the surfaces.
TOL = 1e-8


def axial_play(pair: Pair, sign: int) -> float:
    """How far the centred, unrotated nut can slide along +z (sign=1) or -z."""
    home = np.array([0, 0, pair.task.nut_home()])
    return _bisect(lambda s: pair.penetration(home + [0, 0, sign * s], np.eye(3)) <= TOL,
                   0.0, 4 * pair.c + 1e-4)


def radial_play(pair: Pair, theta: float) -> float:
    """How far the nut can slide sideways along theta, height and rotation fixed."""
    home = np.array([0, 0, pair.task.nut_home()])
    u = np.array([math.cos(theta), math.sin(theta), 0.0])
    return _bisect(lambda s: pair.penetration(home + s * u, np.eye(3)) <= TOL,
                   0.0, 4 * pair.c + 1e-4)


def max_tilt(pair: Pair, phi: float = 0.0) -> float:
    """Largest rotation of the nut about the horizontal axis (cos phi, sin phi, 0)
    through its centre, translation fixed."""
    t = pair.task
    home = np.array([0, 0, t.nut_home()])
    centre = np.array([0, 0, t.nut_height / 2])
    axis = np.array([math.cos(phi), math.sin(phi), 0.0])

    def ok(th):
        R = rot(axis, th)
        return pair.penetration(home + centre - R @ centre, R) <= TOL
    return _bisect(ok, 0.0, 0.05)


def helix_z(task: ThreadTask, angle: float) -> float:
    """Axial advance of a nut turned by `angle` (rad) along the helix."""
    return task.hand * task.lead * angle / (2 * math.pi)


def spin_down_accel(task: ThreadTask, mass: float, inertia_axis: float) -> float:
    """Frictionless nut descending a vertical bolt under gravity: energy
    conservation along the helix z = P phi / 2pi gives a constant acceleration
    a = g / (1 + I (2pi/P)^2 / m)."""
    k = 2 * math.pi / task.lead
    return 9.81 / (1 + inertia_axis * k * k / mass)


# ----- the teeth in the meridian plane (exact: an axial shift is a shift in u) ---------

def _pieces(poly: np.ndarray) -> list:
    """Convex pieces of a tooth (itself when convex)."""
    if _polygon.is_convex(poly):
        return [poly]
    ccw = poly if _polygon.signed_area(poly) > 0 else poly[::-1]
    return [ccw[ix] for ix in _thread._convex_partition(ccw)]


def _apart(a: np.ndarray, b: np.ndarray) -> bool:
    """Are two teeth strictly apart (every pair of their convex pieces)?"""
    return all(_separated(x, y) for x in _pieces(a) for y in _pieces(b))


def _separated(a: np.ndarray, b: np.ndarray) -> bool:
    """Are two convex polygons strictly apart (a separating edge normal)?"""
    for poly in (a, b):
        e = np.roll(poly, -1, axis=0) - poly
        n = np.column_stack([e[:, 1], -e[:, 0]])
        pa, pb = a @ n.T, b @ n.T
        if ((pa.max(0) < pb.min(0)) | (pb.max(0) < pa.min(0))).any():
            return True
    return False


def _nut_copies(task: ThreadTask, c: float, shift: float) -> list:
    """The nut tooth, in absolute (r, u), shifted by `shift` along u, one copy
    per neighbouring pitch."""
    prof, P = _profile(task), task.pitch
    nut = _teeth(task, c)[1] + [prof.r_root, shift]
    return [nut + [0, k * P] for k in (-2, -1, 0, 1, 2)]


def axial_gap(task: ThreadTask, c: float, sign: int) -> float:
    """How far the centred nut slides along +z (sign 1) or -z (sign -1)
    before its teeth touch the bolt's: the exact axial play (2c/√3 each way
    for ISO)."""
    bolt = _teeth(task, c)[0] + [_profile(task).r_core, 0]

    def free(s):
        return all(_apart(bolt, n) for n in _nut_copies(task, c, sign * s))
    return _bisect(free, 0.0, task.pitch, iters=60)


def _outward_normals(poly: np.ndarray) -> np.ndarray:
    e = np.roll(poly, -1, axis=0) - poly
    area = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return np.sign(area) * np.column_stack([e[:, 1], -e[:, 0]]) \
        / np.linalg.norm(e, axis=1)[:, None]


def seat_contact(task: ThreadTask, c: float, sign: int = -1):
    """The nut pushed along sign·z onto the bolt: (contact radius, cos of the
    contact normal's angle to the axis). The contact is where the two teeth
    touch in the meridian plane (the middle of it, for flank on flank); the
    normal is that of the edge a contact point lies inside of."""
    prof = _profile(task)
    bolt = _teeth(task, c)[0] + [prof.r_core, 0]
    tol = 1e-7 * task.pitch
    pts, normals = [], []
    for nut in _nut_copies(task, c, sign * axial_gap(task, c, sign)):
        # vertices of either tooth lying on the other's boundary
        cp = np.vstack([nut[np.abs(_polygon_sd(nut, bolt)) < tol],
                        bolt[np.abs(_polygon_sd(bolt, nut)) < tol]])
        if not len(cp):
            continue
        pts.append(cp)
        for poly, flip in ((bolt, 1.0), (nut, -1.0)):   # normals out of the bolt
            n = _outward_normals(poly)
            for i in range(len(poly)):
                a, b = poly[i], poly[(i + 1) % len(poly)]
                inside = (_segment_distance(cp, a, b) < tol) & \
                    (np.linalg.norm(cp - a, axis=1) > tol) & (np.linalg.norm(cp - b, axis=1) > tol)
                normals += [flip * n[i]] * int(inside.sum())
    pts = np.vstack(pts)
    normal = np.mean(normals, axis=0)
    return float(pts[:, 0].mean()), float(abs(normal[1]) / np.linalg.norm(normal))


def _segment_distance(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    e = b - a
    t = np.clip(((q - a) @ e) / (e @ e), 0, 1)
    return np.linalg.norm(q - (a + t[:, None] * e), axis=1)


def pitch_radius(task: ThreadTask) -> float | None:
    """Radius where the basic bolt tooth is half a pitch wide (the pitch
    line), or None when it never is (a loose wire)."""
    prof = _profile(task)
    t = prof.bolt_tooth
    rs = np.linspace(prof.r_core, t[:, 0].max(), 4001)[1:-1]
    width = np.empty_like(rs)
    for i, r in enumerate(rs):
        u = []
        for a, b in zip(t, np.roll(t, -1, axis=0)):
            if (a[0] - r) * (b[0] - r) <= 0 and a[0] != b[0]:
                u.append(a[1] + (r - a[0]) / (b[0] - a[0]) * (b[1] - a[1]))
        width[i] = max(u) - min(u) if u else 0.0
    half = task.pitch / 2
    on = np.abs(width - half) < 1e-9 * task.pitch
    if on.any():                           # constant width (square): mid-way
        return float(rs[on].mean())
    cross = np.flatnonzero(np.diff(np.sign(width - half)))
    if not len(cross):
        return None
    i = cross[0]
    return float(rs[i] + (half - width[i]) / (width[i + 1] - width[i]) * (rs[i + 1] - rs[i]))


def screw_torque(task: ThreadTask, load: float, mu: float, raising: bool,
                 c: float = 0.0) -> float:
    """Torque that turns a nut pressed down a vertical bolt by an axial load
    (N), with Coulomb friction mu on the flanks: the power-screw formula,
    raising the load (turning against it) T = F r (mu' + tan lam) / (1 - mu'
    tan lam), lowering it (turning with it) T = F r (mu' - tan lam) / (1 + mu'
    tan lam). The contact normal's tilt is folded into mu' = mu / cos(beta)
    (beta = 30° for ISO), and the load is taken at the pitch radius (ISO:
    r2 = (d - 0.6495 P) / 2), or at the contact for a wire that never reaches
    half a pitch wide (lead angle lam). Lowering is negative when the thread
    does not self-lock (mu' < tan lam). Textbook, not exact: the real load
    spreads over the tooth."""
    r_contact, cos_beta = seat_contact(task, c)
    r = pitch_radius(task) or r_contact
    tan_lam = task.lead / (2 * math.pi * r)
    mu_f = mu / cos_beta
    if raising:
        return load * r * (mu_f + tan_lam) / (1 - mu_f * tan_lam)
    return load * r * (mu_f - tan_lam) / (1 + mu_f * tan_lam)


def rundown_torque(task: ThreadTask, mass: float, mu: float, c: float = 0.0) -> float:
    """Torque that turns a nut down a vertical bolt at constant speed, carrying
    its own weight: screw_torque lowering m g."""
    return screw_torque(task, 9.81 * mass, mu, raising=False, c=c)


# ----- the nut on the bolt's tip, coaxial: exact contact in the meridian plane -------------

@lru_cache(maxsize=64)
def tip_window(task: ThreadTask, c: float):
    """(lo, hi): the helical phases (see tip_overlap) at which the coaxial
    nut's teeth clear the bolt's, about the centred phase 0: its axial play."""
    return -axial_gap(task, c, -1), axial_gap(task, c, 1)


def _convex_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Intersection of two convex polygons (either winding), or None."""
    out = _polygon.ensure_ccw(a)
    b = _polygon.ensure_ccw(b)
    for p, q in zip(b, np.roll(b, -1, axis=0)):
        n = np.array([q[1] - p[1], p[0] - q[0]])     # outward, for CCW
        out = _thread._clip_halfplane(out, n, n @ p)
        if out is None:
            return None
    return out


@lru_cache(maxsize=64)
def _meridian_parts(task: ThreadTask, c: float):
    """Convex pieces, in absolute (r, u), of the bolt (core and teeth) and of
    the nut at phase 0 (shell and teeth), over a few pitches about u = 0."""
    P, big = task.pitch, 4 * task.pitch
    rb, ro = task.r_bore(c), task.nut_outer_radius
    core = np.array([[0.0, -big], [task.r_minor, -big], [task.r_minor, big], [0.0, big]])
    shell = np.array([[rb, -big], [ro, -big], [ro, big], [rb, big]])
    bolt = [core] + [q + [task.r_minor, k * P] for k in range(-2, 3)
                     for q in _pieces(bolt_tooth(task, c))]
    nut = [shell] + [q + [task.r_root, k * P] for k in range(-2, 3)
                     for q in _pieces(nut_tooth(task, c))]
    return bolt, nut


def tip_overlap(task: ThreadTask, c: float, sigma: float) -> float:
    """The coaxial nut on the bolt's tip at helical phase sigma (its frame's
    height less helix_z of its angle; 0 mod P centres it in its play): how far
    its base face can pass below the bolt's top face before the parts meet,
    or inf when their teeth clear at this phase (within the axial play).

    Exact, for overlaps shorter than the nut. At one angle phi each part is its
    meridian section, and turning phi slides both tooth patterns along z
    together while the end faces stay put. So the parts meet at some phi
    exactly when the teeth, at relative phase sigma, overlap at a point (r, u)
    that both chamfers leave: the bolt's cone cuts its thread (r - r_core)/k_b
    below its top face, the nut's countersink (r_bore - r)/k_n above its base.
    The faces must overlap by more than the sum; the answer is its least value
    over the teeth's overlap. Without chamfers it is 0."""
    lo, hi = tip_window(task, c)
    s = (sigma - lo) % task.pitch + lo
    if s <= hi:
        return math.inf
    if not task.chamfer:
        return 0.0
    kb, kn = _chamfer_slopes(task, c)
    rc, rb = task.r_minor, task.r_bore(c)
    bolt, nut = _meridian_parts(task, c)
    best = math.inf
    for a in bolt:
        for b in nut:
            b = b + [0.0, s]
            if (a[:, 0].max() <= b[:, 0].min() or b[:, 0].max() <= a[:, 0].min()
                    or a[:, 1].max() <= b[:, 1].min() or b[:, 1].max() <= a[:, 1].min()):
                continue
            ov = _convex_overlap(a, b)
            if ov is None:
                continue
            # The sum is convex and piecewise linear in r, with kinks at r_core
            # and r_bore: least at a vertex or where the polygon spans a kink.
            r = np.concatenate([ov[:, 0], [x for x in (rc, rb)
                                           if ov[:, 0].min() < x < ov[:, 0].max()]])
            best = min(best, float((np.maximum(0, (r - rc) / kb)
                                    + np.maximum(0, (rb - r) / kn)).min()))
    return best


class Tip:
    """The coaxial nut on the bolt's tip (top face z = L), exactly: which
    (angle, height) poses of the nut frame overlap the bolt, where the nut
    rests, and the path it follows when turned under an axial load
    (quasi-static, frictionless)."""

    def __init__(self, task: ThreadTask, c: float, n: int = 3000):
        self.task, self.c = task, c
        self.L, self.P = task.bolt_length, task.pitch
        self.lo, self.hi = tip_window(task, c)
        self.deepest = 0.0
        if task.chamfer:
            # tip_overlap on the phases where the teeth overlap (continuous
            # there; the window's edges are exact, a sliver at them is not).
            s = self.hi + np.linspace(0, 1, n + 1)[1:-1] * (self.P - (self.hi - self.lo))
            d = np.array([tip_overlap(task, c, x) for x in s])
            ok = np.isfinite(d)
            self._s, self._d = s[ok], d[ok]
            self.deepest = float(d[ok].max())

    def overlap(self, sigma) -> np.ndarray:
        """tip_overlap, vectorized (interpolated between exact values when
        chamfered)."""
        s = (np.asarray(sigma, float) - self.lo) % self.P + self.lo
        out = np.full(s.shape, math.inf)
        teeth = s > self.hi
        out[teeth] = np.interp(s[teeth], self._s, self._d) if self.task.chamfer else 0.0
        return out

    def hits(self, theta: float, z) -> np.ndarray:
        """Does the nut frame at this angle and these heights overlap the bolt?"""
        z = np.asarray(z, float)
        e = self.L - z
        return (e > 0) & (e > self.overlap(z - helix_z(self.task, theta)))

    def _hit(self, theta: float, z: float) -> bool:
        return bool(self.hits(theta, np.array([z]))[0])

    def _edge(self, theta: float, z_hit: float, z_free: float, iters: int = 40) -> float:
        """Bisect to the free side of the boundary between two heights."""
        for _ in range(iters):
            mid = (z_hit + z_free) / 2
            z_hit, z_free = (mid, z_free) if self._hit(theta, mid) else (z_hit, mid)
        return z_free

    def _scan(self, theta: float, z: float, sign: int, want: bool, step: float):
        """(first, before): the first height from z in direction sign (+1 up,
        -1 down) where hits() == want, and the height tried before it; (inf,
        z) if none within reach. Tried: every `step`, and just either side of
        the top face and of each window edge, where layers thinner than a
        step begin."""
        reach = self.P + self.deepest + 4 * step
        h, eps = helix_z(self.task, theta), 1e-12
        k = np.arange(math.floor((z - reach - h - self.hi) / self.P) - 1,
                      math.ceil((z + reach - h - self.lo) / self.P) + 2)
        edges = np.concatenate([[self.L], h + self.lo + k * self.P, h + self.hi + k * self.P])
        edges = np.concatenate([edges - eps, edges + eps, [z + sign * eps]])
        grid = np.arange(1, int(reach / step) + 2) * step
        edges = sign * (edges - z)                      # as distances from z
        edges = edges[(edges > 0) & (edges <= reach)]
        prev = z
        for near in (64 * step, math.inf):
            d = np.sort(np.concatenate([grid[grid <= near], edges[edges <= near]]))
            zs = z + sign * d
            where = np.flatnonzero(self.hits(theta, zs) == want)
            if len(where):
                i = where[0]
                return float(zs[i]), float(zs[i - 1]) if i else prev
        return math.inf, z

    def rest(self, theta: float, step: float = 5e-7) -> float:
        """Height of the nut frame resting on the tip at this angle: the first
        contact lowering it from above."""
        z, free = self._scan(theta, self.L + 1e-9, -1, True, step)
        return self._edge(theta, z, free)

    def path(self, thetas: np.ndarray, z0: float | None = None, step: float = 5e-7):
        """Quasi-static height of the nut frame turned through `thetas`
        (monotone and fine) under a downward load: at each angle it is pushed
        up out of anything it now overlaps, then drops to the bottom of the
        free gap it is in. Returns the heights and the steepest push, as a
        multiple of the helix's rise per radian (about 1 on a flank; a wall,
        which a frictionless nut could not climb, shows far steeper)."""
        z = self.rest(thetas[0], step) if z0 is None else z0
        out = np.empty(len(thetas))
        steep, rise = 0.0, abs(self.task.lead) / (2 * math.pi)
        for i, th in enumerate(thetas):
            if self._hit(th, z):
                up, below = self._scan(th, z, 1, False, step)
                znew = self._edge(th, below, up)
                if i:
                    steep = max(steep, (znew - z) / abs(th - thetas[i - 1]) / rise)
                z = znew
            down, above = self._scan(th, z, -1, True, step)
            if not math.isfinite(down):
                raise RuntimeError('the nut fell off the tip')
            z = self._edge(th, down, above)
            out[i] = z
        return out, steep

    def click(self) -> tuple[float, float]:
        """(angle, drop) of the back-spin click: the nut resting on the tip,
        turned the unscrewing way, first falls into the thread where its phase
        at the resting height enters the play window from above. Angle in
        [0, 2 pi / starts)."""
        z = self.L - self.overlap(np.array([self.hi + 1e-9]))[0]
        th = (z - self.hi) * 2 * math.pi / (self.task.hand * self.task.lead)
        th %= 2 * math.pi / self.task.starts
        return th, z - self.rest(th + self.task.hand * 1e-9)

    def entry(self) -> float:
        """Angle at which the nut resting on the tip, turned the screwing way,
        starts down the thread: its phase at the resting height reaches the
        play window from below. In [0, 2 pi / starts)."""
        z = self.L - self.overlap(np.array([self.lo - 1e-9]))[0]
        th = (z - self.lo) * 2 * math.pi / (self.task.hand * self.task.lead)
        return th % (2 * math.pi / self.task.starts)


# ----- the tip's surfaces, and the nut threaded partway on -----------------------------------

def _crest_tip(task: ThreadTask, c: float):
    """(bolt crest radius, nut tooth tip radius)."""
    bolt, nut = _teeth(task, c)
    return task.r_minor + bolt[:, 0].max(), task.r_root + nut[:, 0].min()


def _annulus(r0: float, r1: float, z: float, spacing: float) -> np.ndarray:
    out = []
    for r in np.arange(max(r0, spacing / 2), r1 + spacing / 2, spacing):
        n = max(8, math.ceil(2 * math.pi * r / spacing))
        phi = np.linspace(0, 2 * math.pi, n, endpoint=False)
        out.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), np.full(n, z)]))
    return np.concatenate(out)


def tip_faces(task: ThreadTask, c: float, spacing: float = 5e-5, reach: float = 3e-4):
    """Points on the end faces and chamfer cones that meet at the tip: (bolt
    top face z = L, and its cone), (nut base z = 0, and its countersink).
    Only the annuli within `reach` of the other part's radii."""
    import dataclasses
    L = task.bolt_length
    crest, tip = _crest_tip(task, c)
    bolt = _annulus(tip - reach, crest, L, spacing)
    bolt = bolt[bolt_sdf(task, c, bolt - [0, 0, 1e-9]) < 0]
    nut = _annulus(tip, crest + reach, 0.0, spacing)
    nut = nut[nut_sdf(task, c, nut + [0, 0, 1e-9]) < 0]
    if task.chamfer:
        flat = dataclasses.replace(task, chamfer=0.0)
        kb, kn = _chamfer_slopes(task, c)
        cones_b, cones_n = [], []
        for d in np.arange(0, 1.05 * task.chamfer, spacing / 2):
            cones_b.append(_annulus(task.r_minor + kb * d, task.r_minor + kb * d, L - d, spacing))
            cones_n.append(_annulus(task.r_bore(c) - kn * d, task.r_bore(c) - kn * d, d, spacing))
        cb, cn = np.concatenate(cones_b), np.concatenate(cones_n)
        # on a cone where it cuts solid thread
        bolt = np.concatenate([bolt, cb[bolt_sdf(flat, c, cb) < 0]])
        nut = np.concatenate([nut, cn[nut_sdf(flat, c, cn) < 0]])
    return bolt, nut


def partial_pose(task: ThreadTask, turns: float):
    """(height, angle) of the nut frame threaded `turns` leads onto the bolt's
    tip: its base face turns * lead below the top face, centred in its play."""
    z = task.bolt_length - turns * task.lead
    return z, 2 * math.pi * (z - task.nut_home()) / (task.hand * task.lead)


def _first_hit(hit, step: float, hi: float) -> float:
    """Smallest x in (0, hi] with hit(x): scanned in `step`s, then bisected.
    inf when there is none."""
    x0, x = 0.0, step
    while x <= hi + 1e-15:
        if hit(x):
            return _bisect(lambda y: not hit(y), x0, x, iters=30)
        x0, x = x, x + step
    return math.inf


def partial_ray(pair: 'Pair', turns: float, psi: float) -> float:
    """How far the partly threaded nut slides sideways along the ray psi (bolt
    frame) before it touches, height and rotation fixed."""
    z, a = partial_pose(pair.task, turns)
    u = np.array([math.cos(psi), math.sin(psi), 0.0])
    base, R = np.array([0.0, 0.0, z]), rot([0, 0, 1], a)
    return _first_hit(lambda s: pair.penetration(base + s * u, R) > TOL,
                      pair.c / 8, 2e-3)


def partial_radial_play(pair: 'Pair', turns: float, theta: float, fine: 'Pair' = None,
                        spread: float = math.radians(90), step: float = math.radians(5)) -> float:
    """Reach along theta of the partly threaded nut free to slide in the
    plane (height and rotation fixed): the support, along theta, of its free
    sideways translations. A frictionless nut pushed along theta slides
    around a contact whose normal is off theta, so this is more than the
    first contact straight along theta wherever that normal is off it (a
    quarter turn on: the tip arc's end on the bolt core's rim). The free set,
    taken as star-shaped about the centred pose, is traced by first-contact
    rays every `step` within `spread` of theta (on `pair`), and the best is
    refined by golden section (on `fine`, default `pair`)."""
    fine = pair if fine is None else fine

    def reach(p, psi):
        return partial_ray(p, turns, psi) * math.cos(psi - theta)
    psis = theta + np.arange(-spread, spread + 1e-9, step)
    best = psis[int(np.argmax([reach(pair, p) for p in psis]))]
    lo, hi = best - step, best + step
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = reach(fine, x1), reach(fine, x2)
    for _ in range(14):                       # to ~0.01 deg
        if f1 > f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = reach(fine, x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = reach(fine, x2)
    return max(f1, f2, reach(fine, theta))


def partial_max_tilt(pair: 'Pair', turns: float, phi: float, pivot: float) -> float:
    """Largest rotation of the partly threaded nut about the horizontal axis
    (cos phi, sin phi, 0) of the bolt frame through the point `pivot` above
    its base on the thread axis."""
    z, a = partial_pose(pair.task, turns)
    axis = np.array([math.cos(phi), math.sin(phi), 0.0])
    p = np.array([0.0, 0.0, z + pivot])
    Rz = rot([0, 0, 1], a)

    def hit(th):
        Rt = rot(axis, th)
        return pair.penetration(p + Rt @ [0.0, 0.0, -pivot], Rt @ Rz) > TOL
    return _first_hit(hit, 1e-3, 0.3)


# ----- watertight meshes (the input every baseline decomposes) ------------------------------

def _tube(profile: np.ndarray, base_r: float, phase: float, P: float, z0: float, z1: float,
          per_turn: int = 360, rise: float | None = None):
    """Profile (dr, u) swept along the helix (rising `rise` per turn, default
    P) as a closed triangle mesh, covering heights z0..z1 (plus a pitch and a
    turn each side, trimmed later)."""
    rise = P if rise is None else rise
    phi0 = 2 * math.pi * (z0 - P - phase * P - abs(rise)) / rise
    phi1 = 2 * math.pi * (z1 + P - phase * P + abs(rise)) / rise
    n = math.ceil(abs(phi1 - phi0) / (2 * math.pi) * per_turn)
    phi = np.linspace(phi0, phi1, n + 1)
    m = len(profile)
    r = base_r + profile[:, 0]
    z = profile[None, :, 1] + rise * phi[:, None] / (2 * math.pi) + phase * P
    verts = np.stack([r[None] * np.cos(phi)[:, None], r[None] * np.sin(phi)[:, None], z], -1)
    verts = verts.reshape(-1, 3)
    faces = []
    for i in range(n):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            faces += [(a, b, b + m), (a, b + m, a + m)]
    last = n * m
    tris = _cap_triangles(profile)
    cap0 = [(a, c_, b) for a, b, c_ in tris]
    cap1 = [(last + a, last + b, last + c_) for a, b, c_ in tris]
    faces = np.array(faces + cap0 + cap1)
    return verts, faces


def _cap_triangles(profile: np.ndarray) -> list:
    """Triangles (vertex indices, in the profile's own winding) covering the
    profile: a fan for a convex one, fans of its convex pieces otherwise."""
    m = len(profile)
    if _polygon.is_convex(profile):
        return [(0, j, j + 1) for j in range(1, m - 1)]
    flip = _polygon.signed_area(profile) < 0
    ccw = profile[::-1] if flip else profile
    out = []
    for ix in _thread._convex_partition(ccw):
        ix = [m - 1 - i for i in ix][::-1] if flip else ix     # back to profile indices
        out += [(ix[0], ix[j], ix[j + 1]) for j in range(1, len(ix) - 1)]
    return out


def _manifold(v, f):
    import manifold3d as m3d
    mesh = m3d.Mesh(vert_properties=np.asarray(v, np.float32), tri_verts=np.asarray(f, np.uint32))
    out = m3d.Manifold(mesh)
    if out.is_empty():
        # Wrong winding: flip it.
        mesh = m3d.Mesh(vert_properties=np.asarray(v, np.float32),
                        tri_verts=np.asarray(f, np.uint32)[:, ::-1].copy())
        out = m3d.Manifold(mesh)
    if out.is_empty():
        raise RuntimeError(f'not a manifold solid: {out.status()}')
    return out


def _to_arrays(man):
    mesh = man.to_mesh()
    return (np.asarray(mesh.vert_properties[:, :3], np.float64),
            np.asarray(mesh.tri_verts, np.int64))


def _threads(tooth, base_r, task: ThreadTask, z0: float, z1: float):
    """One helical tube per start, each a pitch further along."""
    out = None
    for k in range(task.starts):
        tube = _manifold(*_tube(tooth, base_r, float(k), task.pitch, z0, z1,
                                rise=task.hand * task.lead))
        out = tube if out is None else out + tube
    return out


def bolt_mesh(task: ThreadTask, c: float, segments: int = 720):
    """Watertight bolt: core cylinder united with the thread, trimmed to [0, L]."""
    import manifold3d as m3d
    P, L = task.pitch, task.bolt_length
    tube = _threads(bolt_tooth(task, c), task.r_minor, task, 0.0, L)
    core = m3d.Manifold.cylinder(L + 4 * P, task.r_minor, task.r_minor, segments).translate(
        (0, 0, -2 * P))
    # Circumscribe the core polygon so its chords never cut into the true core.
    core = core.scale((1 / math.cos(math.pi / segments),) * 2 + (1,))
    slab = m3d.Manifold.cube((4 * task.r_major, 4 * task.r_major, L), True).translate((0, 0, L / 2))
    bolt = (tube + core) ^ slab
    if task.chamfer:
        # Keep a double-ended frustum: core radius at each end face, opening
        # past the crest `chamfer` in.
        k, ch, big = _chamfer_slopes(task, c)[0], task.chamfer, 2 * task.r_major
        grow = big - task.r_minor
        keep = m3d.Manifold.cylinder(grow / k, task.r_minor, big, segments)
        keep = keep + m3d.Manifold.cylinder(L - 2 * ch, big, big, segments).translate((0, 0, ch))
        keep = keep + m3d.Manifold.cylinder(grow / k, big, task.r_minor, segments).translate(
            (0, 0, L - grow / k))
        bolt = bolt ^ keep
    return _to_arrays(bolt)


def nut_mesh(task: ThreadTask, c: float, segments: int = 720):
    """Watertight nut: shell (bore radius r_root + c/2) united with the
    inward teeth, trimmed to [0, Hn]."""
    import manifold3d as m3d
    P, Hn = task.pitch, task.nut_height
    tube = _threads(nut_tooth(task, c), task.r_root, task, 0.0, Hn)
    # Circumscribe the tube: its 1-degree chords would otherwise bulge 0.2 um
    # toward the bolt.
    tube = tube.scale((1 / math.cos(math.pi / 360),) * 2 + (1,))
    outer = m3d.Manifold.cylinder(Hn, task.nut_outer_radius, task.nut_outer_radius, segments)
    # The bore polygon is circumscribed so it never protrudes into the gap.
    bore = m3d.Manifold.cylinder(Hn + 2 * P, task.r_bore(c), task.r_bore(c), segments)
    bore = bore.scale((1 / math.cos(math.pi / segments),) * 2 + (1,)).translate((0, 0, -P))
    slab = m3d.Manifold.cube((4 * task.nut_outer_radius,) * 2 + (Hn,), True).translate(
        (0, 0, Hn / 2))
    nut = ((outer - bore) + tube) ^ slab
    if task.chamfer:
        # Countersinks: cones from the bore at each face down to the axis.
        k, r0 = _chamfer_slopes(task, c)[1], task.r_bore(c)
        for z0, towards in ((0.0, 1), (Hn, -1)):
            cone = m3d.Manifold.cylinder(r0 / k, r0, 0.0, segments)
            cone = cone.scale((1 / math.cos(math.pi / segments),) * 2 + (1,))
            if towards < 0:
                cone = cone.mirror((0, 0, 1))
            nut = nut - cone.translate((0, 0, z0))
    return _to_arrays(nut)


def nut_inertia(task: ThreadTask, c: float, mesh=None):
    """(mass, centre of mass, inertia tensor about the COM) of the exact nut,
    and its moment about the thread axis. Every representation's nut gets
    these, so dynamics differ only through contact."""
    import trimesh
    v, f = mesh if mesh is not None else nut_mesh(task, c)
    m = trimesh.Trimesh(v, f, process=False)
    m.density = task.density
    com = np.asarray(m.center_mass)
    I = np.asarray(m.moment_inertia)
    i_axis = I[2, 2] + m.mass * (com[0] ** 2 + com[1] ** 2)
    return float(m.mass), com, I, float(i_axis)
