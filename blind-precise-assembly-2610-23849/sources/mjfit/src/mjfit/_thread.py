"""Threaded bolt and nut pairs.

A thread is a tooth profile swept along a helix. Each short angular slice of
a turn is a twisted prism, which is convex, so the bolt is a core cylinder
plus one convex wedge per slice, and the nut is a convex-decomposed shell
plus inward-facing wedges. Discretization only ever removes material: the
nut's wedges are pushed out by their chord sag and helicoid twist, so slices
add play but never interference.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import _mjcf as mj
from . import _polygon as pg

BOLT_RGBA = '0.55 0.58 0.62 1'
HEAD_RGBA = '0.35 0.38 0.42 1'
NUT_RGBA = '0.75 0.72 0.68 1'
FLOOR_GAP = 0.001          # the lower part starts this far above the floor


# ----- thread forms -----------------------------------------------------------
#
# A thread form is a pair of convex teeth in the meridian half-plane, in
# coordinates (r, u): radius, and axial position u = z - (the helix's advance
# at that angle) taken mod P. The bolt's tooth stands on its core (radius
# r_core) and the nut's hangs from its root (radius r_root). Sweeping each
# tooth along the helix gives the part. Both are given at zero clearance in
# one shared phase, so with the two part frames a whole number of pitches
# apart the teeth fill each other's gaps. Clearance c then moves every exposed
# edge of both teeth back c/2 along its normal (the base on the core or root
# stays put), and the nut's root out by c/2. For ISO that is exactly the
# standard's rule; for any tooth it keeps facing surfaces c apart.
#
# Teeth need not be convex (hooks, T-slots, barbs): a non-convex tooth is
# offset as a whole (each edge back c/2, mitred corners, which at reflex
# corners only keeps more clear) and then cut into convex pieces, each swept
# as its own wedges. The cuts lie inside the tooth, so they are never moved.

THREAD_FORMS = ('iso', 'trapezoidal', 'acme', 'square', 'buttress', 'wire', 'custom')


TRAPEZOID_FORMS = {
    'iso':         {'flank_angles': (30.0, 30.0), 'depth': 5 * np.sqrt(3) / 16,
                    'crest_flat': 1 / 8},                                   # ISO 68-1
    'trapezoidal': {'flank_angles': (15.0, 15.0), 'depth': 0.5, 'crest_flat': 0.366},  # ISO 2904
    'acme':        {'flank_angles': (14.5, 14.5), 'depth': 0.5, 'crest_flat': 0.3707},
    'square':      {'flank_angles': (0.0, 0.0), 'depth': 0.5, 'crest_flat': 0.5},
    # ANSI B1.9 7°/45°: the steep flank carries the load of a nut pressed
    # toward the bolt tip (a bolt in tension).
    'buttress':    {'flank_angles': (45.0, 7.0), 'depth': 0.6, 'crest_flat': 0.1631},
}


THREAD_LENGTH_PARAMS = ('wire_diameter', 'nut_wire_offset', 'bore_diameter',
                        'core_diameter', 'root_diameter', 'bolt_tooth', 'nut_tooth')


class ThreadProfile:
    """Zero-clearance teeth of a thread form (see the section comment)."""

    def __init__(self, bolt_tooth, nut_tooth, r_core: float, r_root: float,
                 pitch: float):
        self.pitch = pitch
        self.r_core, self.r_root = r_core, r_root
        # Each tooth ends at its base line, so the base is one edge on it.
        self.bolt_tooth = _clip_radius(_ensure_tooth(bolt_tooth, 'bolt'), r_core, above=True)
        self.nut_tooth = _clip_radius(_ensure_tooth(nut_tooth, 'nut'), r_root, above=False)
        if self.bolt_tooth is None or self.nut_tooth is None:
            raise ValueError("a tooth does not reach past its base line "
                             "(bolt tooth above r_core, nut tooth below r_root)")

    @property
    def r_crest(self) -> float:
        """Bolt major radius (tip of the bolt tooth)."""
        return float(self.bolt_tooth[:, 0].max())

    def symmetric(self, tol: float = 1e-12) -> bool:
        """Is the bolt tooth mirror-symmetric in u about its centre?"""
        t = self.bolt_tooth
        m = t.copy()
        m[:, 1] = t[:, 1].min() + t[:, 1].max() - t[:, 1]
        a = t[np.lexsort((t[:, 1], t[:, 0]))]
        b = m[np.lexsort((m[:, 1], m[:, 0]))]
        return len(a) == len(b) and np.allclose(a, b, atol=tol * self.pitch)

    def teeth(self, clearance: float):
        """(bolt, nut) teeth at this clearance, as (r, u) arrays in the order
        base-left, base-right, then around (see _canonical). Raises when the
        clearance leaves no tooth."""
        out = []
        for name, tooth, line in (('bolt', self.bolt_tooth, self.r_core),
                                  ('nut', self.nut_tooth, self.r_root)):
            base = _base_edges(tooth, line)
            deltas = np.where(base, 0.0, clearance / 2)
            if pg.is_convex(tooth):
                t = _offset_polygon(tooth, deltas)
            else:
                t = _offset_mitre(tooth, deltas)
            if t is None:
                raise ValueError(f"clearance ({clearance}) leaves no {name} tooth "
                                 f"for pitch {self.pitch}"
                                 + ("" if pg.is_convex(tooth) else
                                    " (or is too large for its smallest feature)"))
            out.append(_canonical(t, line))
        return tuple(out)


def _ensure_tooth(tooth, name: str) -> np.ndarray:
    t = pg.ensure_ccw(np.asarray(tooth, dtype=float).reshape(-1, 2))
    if len(t) < 3 or not pg.is_simple(t):
        raise ValueError(f"the {name} tooth must be a simple polygon of (r, u) points")
    return t


def _offset_mitre(poly: np.ndarray, deltas) -> np.ndarray | None:
    """Move each edge i -> i+1 of a simple polygon (either winding) inward by
    deltas[i], each new vertex where its two edges' moved lines meet. Keeps
    the vertex order. Returns None when an edge would vanish or flip, or the
    result would cross itself (the offset is too large for the polygon)."""
    n = len(poly)
    deltas = np.broadcast_to(np.asarray(deltas, dtype=float), (n,))
    e = np.roll(poly, -1, axis=0) - poly
    normal = np.column_stack([e[:, 1], -e[:, 0]]) / np.linalg.norm(e, axis=1)[:, None]
    if pg.signed_area(poly) < 0:
        normal = -normal                                  # outward for either winding
    out = np.empty_like(poly)
    for i in range(n):
        j = (i - 1) % n                                   # edges j = i-1 and i meet at i
        # points x with n_j·x = n_j·p_j - d_j and n_i·x = n_i·p_i - d_i
        A = np.array([normal[j], normal[i]])
        b = np.array([normal[j] @ poly[j] - deltas[j], normal[i] @ poly[i] - deltas[i]])
        if abs(np.linalg.det(A)) < 1e-12:                 # collinear edges
            out[i] = poly[i] - deltas[i] * normal[i]
        else:
            out[i] = np.linalg.solve(A, b)
    e2 = np.roll(out, -1, axis=0) - out
    if (np.sum(e * e2, axis=1) <= 0).any() or not pg.is_simple(out) or \
            np.sign(pg.signed_area(out)) != np.sign(pg.signed_area(poly)):
        return None
    return out


def _convex_partition(poly: np.ndarray) -> list[list[int]]:
    """Convex pieces of a simple CCW polygon, as vertex index lists (CCW):
    ear-clip into triangles, then merge neighbours while the union stays
    convex (Hertel-Mehlhorn, at most four times the fewest pieces)."""
    n = len(poly)
    idx = list(range(n))
    pieces = []
    while len(idx) > 3:
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if pg.cross2(b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]) <= 1e-18:
                continue                                  # reflex or flat: not an ear
            if any(pg.point_in_triangle(poly[j], a, b, c) for j in idx
                   if j not in (i0, i1, i2) and not any(pg.points_equal(poly[j], q, 1e-15)
                                                          for q in (a, b, c))):
                continue
            pieces.append([i0, i1, i2])
            idx.pop(k)
            break
        else:
            raise ValueError("could not cut the tooth into convex pieces "
                             "(is it a simple polygon?)")
    pieces.append(idx)

    def convex(ix):
        q = poly[ix]
        e = np.roll(q, -1, axis=0) - q
        return (e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1) >= -1e-18).all()

    merged = True
    while merged:
        merged = False
        for a in range(len(pieces)):
            for b in range(a + 1, len(pieces)):
                A, B = pieces[a], pieces[b]
                for k in range(len(A)):
                    u, v = A[k], A[(k + 1) % len(A)]          # edge u -> v in A, v -> u in B
                    if v in B and B[(B.index(v) + 1) % len(B)] == u:
                        ra = A[k + 1:] + A[:k + 1]            # v ... u
                        rb = B[B.index(u):] + B[:B.index(u)]  # u ... v
                        union = ra + rb[1:-1]
                        if convex(union):
                            pieces[a] = union
                            pieces.pop(b)
                            merged = True
                        break
                if merged:
                    break
            if merged:
                break
    return pieces


def _clip_halfplane(poly: np.ndarray, n: np.ndarray, d: float) -> np.ndarray | None:
    """Convex polygon (CCW) intersected with n·x <= d."""
    out = []
    k = len(poly)
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        fa, fb = n @ a - d, n @ b - d
        if fa <= 0:
            out.append(a)
        if fa * fb < 0:
            out.append(a + (b - a) * (fa / (fa - fb)))
    if len(out) < 3:
        return None
    out = np.array(out)
    keep = np.linalg.norm(out - np.roll(out, 1, axis=0), axis=1) > 1e-15
    out = out[keep]
    return out if len(out) >= 3 and pg.signed_area(out) > 1e-20 else None


def _offset_polygon(poly: np.ndarray, deltas) -> np.ndarray | None:
    """Move each edge i -> i+1 of a convex polygon (either winding) inward
    by deltas[i] along its normal. Returns a CCW polygon, or None when
    nothing is left."""
    n = len(poly)
    deltas = np.broadcast_to(np.asarray(deltas, dtype=float), (n,))
    if pg.signed_area(poly) < 0:
        # Reversed, edge k runs between the original vertices n-1-k and n-2-k.
        poly = poly[::-1]
        deltas = deltas[(n - 2 - np.arange(n)) % n]
    out = poly
    for i in range(len(poly)):
        if deltas[i] == 0:
            continue
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        n = np.array([e[1], -e[0]]) / np.linalg.norm(e)      # outward for CCW
        out = _clip_halfplane(out, n, n @ a - deltas[i])
        if out is None:
            return None
    return out


def _edge_normals(poly: np.ndarray) -> np.ndarray:
    """Outward unit normals (n_r, n_u) of a convex polygon's edges i -> i+1,
    for either winding."""
    e = np.roll(poly, -1, axis=0) - poly
    n = np.column_stack([e[:, 1], -e[:, 0]]) / np.linalg.norm(e, axis=1)[:, None]
    return n if pg.signed_area(poly) > 0 else -n


def _base_edges(poly: np.ndarray, line: float) -> np.ndarray:
    """Edges i -> i+1 lying on the radius `line`."""
    tol = 1e-9 * max(abs(line), 1e-3)
    on = np.abs(poly[:, 0] - line) < tol
    return on & np.roll(on, -1)


def _clip_radius(poly: np.ndarray, line: float, above: bool) -> np.ndarray | None:
    """Keep r >= line (above) or r <= line."""
    n = np.array([-1.0, 0.0]) if above else np.array([1.0, 0.0])
    return _clip_halfplane(pg.ensure_ccw(poly), n, -line if above else line)


def _canonical(poly: np.ndarray, line: float) -> np.ndarray:
    """Order a tooth as base-left, base-right, then on around, the order the
    wedge writer's face table assumes (the ISO profiles' order)."""
    base = _base_edges(poly, line)
    if not base.any():
        return poly
    i = int(np.flatnonzero(base)[0])
    j = (i + 1) % len(poly)
    if poly[i, 1] > poly[j, 1]:          # walk the other way round
        poly = poly[::-1]
        i, j = len(poly) - 1 - j, len(poly) - 1 - i
    return np.roll(poly, -i, axis=0)


def _trapezoid_profile(major_radius: float, pitch: float, flank_angles=(30.0, 30.0),
                       depth: float = 5 * np.sqrt(3) / 16,
                       crest_flat: float = 1 / 8) -> ThreadProfile:
    """Trapezoid family (angles in degrees, depth and crest_flat in pitches).
    The bolt tooth starts after its root gap, so ISO lands on the ISO
    profiles' phase (bolt base from P/4 to P). Negative angles undercut the
    flank: a dovetail, whose teeth widen away from their base (both teeth
    stay convex)."""
    P, R = pitch, major_radius
    h, fc = depth * P, crest_flat * P
    t1, t2 = (np.tan(np.radians(a)) for a in flank_angles)
    fr = P - fc - h * (t1 + t2)                      # root gap of the bolt
    if h <= 0 or fc < 0 or fr < -1e-15 * P or P - fr <= 0 or P - fc <= 0 \
            or max(abs(a) for a in flank_angles) >= 90:
        raise ValueError(f"trapezoid thread does not fit one pitch: depth {depth}, "
                         f"crest_flat {crest_flat}, flank_angles {flank_angles} "
                         f"(root gap {fr / P:.4g} P)")
    fr = max(fr, 0.0)
    bolt = [(R - h, fr), (R - h, P), (R, fr + h * t1 + fc), (R, fr + h * t1)]
    # The nut tooth is the gap between bolt teeth: its tip is the bolt's root
    # gap, its base spans the bolt crests' gap.
    nut = [(R - h, 0.0), (R - h, fr), (R, fr + h * t1), (R, fr + h * t1 + fc - P)]
    if fr == 0:          # sharp bolt root: the nut tooth is a triangle
        nut = [nut[0], nut[2], nut[3]]
    if fc == 0:
        bolt = bolt[:3]
    return ThreadProfile(bolt, nut, R - h, R, P)


def _circle_tooth(r_c: float, u_c: float, radius: float, n: int) -> np.ndarray:
    """Inscribed n-gon of a circle in (r, u), with vertices at the circle's
    radial and axial extremes (n a multiple of 4)."""
    a = 2 * np.pi * np.arange(n) / n
    return np.column_stack([r_c + radius * np.cos(a), u_c + radius * np.sin(a)])


def _wire_profile(major_radius: float, pitch: float, wire_diameter: float | None = None,
                  nut_wire_offset: float | None = None, bore_diameter: float | None = None,
                  wire_segments: int = 16) -> ThreadProfile:
    """Round-wire thread: a wire of circular section wound on each part (the
    FurnitureBench legs). The bolt wire is centred on the core (radius
    major - wire/2); the nut wire half a pitch along, `nut_wire_offset`
    further out, on a bore of `bore_diameter` (default: through the nut wire's
    centre). The defaults are the tight form: wire_diameter P/√3 and the
    offset at which the two wires touch at zero clearance. The circles are
    inscribed polygons, so the teeth never exceed the round wire."""
    P = pitch
    d_w = P / np.sqrt(3) if wire_diameter is None else float(wire_diameter)
    rho = d_w / 2
    if nut_wire_offset is None:
        if d_w < P / 2:
            raise ValueError("wires thinner than P/2 never touch; give nut_wire_offset")
        nut_wire_offset = np.sqrt(d_w ** 2 - P ** 2 / 4)
    if wire_segments < 8 or wire_segments % 4:
        raise ValueError("wire_segments must be a multiple of 4, at least 8")
    r_b = major_radius - rho
    r_n = r_b + float(nut_wire_offset)
    r_root = r_n if bore_diameter is None else float(bore_diameter) / 2
    bolt = _circle_tooth(r_b, rho, rho, wire_segments)
    nut = _circle_tooth(r_n, rho + P / 2, rho, wire_segments)
    return ThreadProfile(bolt, nut, r_b, r_root, P)


def thread_profile(form: str, major_diameter: float, pitch: float,
                   params: dict | None = None) -> ThreadProfile:
    """The zero-clearance teeth of a named thread form.

    form: one of THREAD_FORMS. params override the form's defaults:
      trapezoid family (iso, trapezoidal, acme, square, buttress):
        flank_angles (deg, lower and upper), depth, crest_flat (pitches);
      wire: wire_diameter, nut_wire_offset, bore_diameter (m), wire_segments;
      custom: bolt_tooth and nut_tooth, lists of (r, u) points (m, one
        shared phase), core_diameter and root_diameter (m).
    major_diameter is the bolt's outer diameter (ignored by custom).
    """
    params = dict(params or {})
    if form in TRAPEZOID_FORMS:
        return _trapezoid_profile(major_diameter / 2, pitch,
                                  **{**TRAPEZOID_FORMS[form], **params})
    if form == 'wire':
        return _wire_profile(major_diameter / 2, pitch, **params)
    if form == 'custom':
        need = {'bolt_tooth', 'nut_tooth', 'core_diameter', 'root_diameter'}
        if need - params.keys():
            raise ValueError(f"custom thread form needs {sorted(need)}")
        return ThreadProfile(params['bolt_tooth'], params['nut_tooth'],
                             params['core_diameter'] / 2, params['root_diameter'] / 2, pitch)
    raise ValueError(f"thread_form must be one of {THREAD_FORMS}, got {form!r}")


def _thread_wedge_vertices(profile: np.ndarray, theta: float, dtheta: float,
                           z_advance: float, base_radius: float,
                           z_offset: float) -> np.ndarray:
    """Sweep a 2D meridian profile through (θ, θ+Δθ) with axial advance.

    profile: (N, 2) array of (dr, z_local) points. Radius at vertex = base_radius + dr.
    theta, dtheta: angular range (radians).
    z_advance: axial shift between the θ face and the θ+Δθ face (signed; sign
        = handedness × lead × dtheta / (2π)).
    base_radius: reference radius (minor for bolt, major for nut).
    z_offset: global z offset applied to z_local at θ.

    Returns (2N, 3) vertices, first N at θ, next N at θ+Δθ, same profile order.
    """
    verts = np.zeros((2 * len(profile), 3))
    for j, (ang, dz) in enumerate([(theta, 0.0), (theta + dtheta, z_advance)]):
        c, s = np.cos(ang), np.sin(ang)
        for i, (dr, z_local) in enumerate(profile):
            r = base_radius + dr
            verts[j * len(profile) + i] = [r * c, r * s, z_offset + z_local + dz]
    return verts


def _profile_ring(profile: np.ndarray, angle: float, base_radius: float,
                  z: float) -> np.ndarray:
    """The (dr, z_local) profile placed in the meridian plane at `angle`,
    with its z_local measured from `z`."""
    c, s = np.cos(angle), np.sin(angle)
    r = base_radius + profile[:, 0]
    return np.column_stack([r * c, r * s, z + profile[:, 1]])


def _wedge_has_volume(verts: np.ndarray, n_ring: int = 4, eps: float = 1e-15) -> bool:
    """Conservative check that a wedge (rings of n_ring profile vertices)
    spans a nonzero-volume hull: the tetrahedron of a first-ring triangle
    and the second ring's first vertex has volume. The triangle is the
    first-ring vertices 0, k, j that span the most area (a piece of a
    non-convex tooth can start with collinear vertices)."""
    ring = verts[:n_ring]
    v0 = ring[0]
    d = np.linalg.norm(ring - v0, axis=1)
    k = int(np.argmax(d))
    area = np.linalg.norm(np.cross(ring[k] - v0, ring - v0), axis=1)
    j = int(np.argmax(area))
    m = np.column_stack([ring[k] - v0, ring[j] - v0, verts[n_ring] - v0])
    return abs(np.linalg.det(m)) > eps


def _clip_z(verts: np.ndarray, z0: float, z1: float,
            min_volume: float = 1e-13) -> np.ndarray | None:
    """Vertices of the convex hull of `verts` clipped to z0 <= z <= z1.

    The clipped hull is the hull of the vertices inside the slab plus every
    crossing of the planes by a segment between two vertices (all such
    segments lie in the hull, and its clipped edges are among them). Returns
    `verts` unchanged when already inside, and None when what remains is a
    nub too small for MuJoCo to hull reliably (its volume, estimated from the
    principal extents, under `min_volume`); dropping one only removes a sliver
    of tooth at the part's end face.
    """
    z = verts[:, 2]
    if z.min() >= z0 and z.max() <= z1:
        return verts
    if z.max() <= z0 or z.min() >= z1:
        return None
    pts = [verts[(z >= z0) & (z <= z1)]]
    for a in range(len(verts)):
        for b in range(a + 1, len(verts)):
            for zc in (z0, z1):
                if (z[a] - zc) * (z[b] - zc) < 0:
                    t = (zc - z[a]) / (z[b] - z[a])
                    pts.append((verts[a] + t * (verts[b] - verts[a]))[None])
    pts = np.unique(np.concatenate(pts), axis=0)
    if len(pts) < 4:
        return None
    centered = pts - pts.mean(axis=0)
    extents = np.ptp(centered @ np.linalg.svd(centered, full_matrices=False)[2].T, axis=0)
    return pts if np.prod(extents) / 6 > min_volume else None


def _clip_half(verts: np.ndarray, n: np.ndarray, d: float,
               min_volume: float = 1e-13) -> np.ndarray | None:
    """Vertices of the convex hull of `verts` clipped to n·x <= d: the hull
    of the vertices inside plus every crossing of the plane by a segment
    between two vertices. None when nothing, or a nub too small to hull, is
    left (see _clip_z)."""
    side = verts @ n - d
    if (side <= 0).all():
        return verts
    if (side >= 0).all():
        return None
    i, j = np.nonzero((side[:, None] < 0) & (side[None, :] > 0))
    t = side[i] / (side[i] - side[j])
    pts = np.unique(np.concatenate([verts[side <= 0],
                                    verts[i] + t[:, None] * (verts[j] - verts[i])]), axis=0)
    if len(pts) < 4:
        return None
    centered = pts - pts.mean(axis=0)
    extents = np.ptp(centered @ np.linalg.svd(centered, full_matrices=False)[2].T, axis=0)
    return pts if np.prod(extents) / 6 > min_volume else None


def _cone_chord(r0: float, slope: float, z0: float, towards: int,
                theta0: float, theta1: float):
    """(n, d): a half-space n·x <= d whose part in the sector [theta0,
    theta1] lies inside the cone r <= r0 + slope·towards·(z - z0) (a bolt
    chamfer at the end z0, the part on the side `towards`). Its plane holds
    the cone's generators at theta0 and theta1, which lie on the cone: the
    chord of a convex surface lies inside it."""
    a = slope * towards
    n = np.array([a * (np.sin(theta0) - np.sin(theta1)), a * (np.cos(theta1) - np.cos(theta0)),
                  a * a * np.sin(theta1 - theta0)])
    apex = np.array([0.0, 0.0, z0 - r0 / a])
    if n @ (np.array([0.0, 0.0, z0]) - apex) > 0:     # keep the axis side
        n = -n
    return n, float(n @ apex)


def _cone_tangent(r0: float, slope: float, z0: float, towards: int, theta: float):
    """(n, d): the half-space n·x <= d outside the tangent plane at angle
    theta of the cone r = r0 - slope·towards·(z - z0) (a countersink at the
    face z0, narrowing towards the part). A cone is convex, so everything
    beyond a tangent plane lies outside it, whatever the angle."""
    n = -np.array([np.cos(theta), np.sin(theta), slope * towards])
    return n, float(-r0 - slope * towards * z0)


def _hex_polygon(across_flats: float) -> np.ndarray:
    """Regular hexagon CCW, with two flats parallel to the x-axis.
    `across_flats` is the flat-to-flat distance (standard hex-nut sizing)."""
    # circumradius = across_flats / cos(30°) = across_flats * 2/√3
    R = across_flats / np.sqrt(3)
    a = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    return np.column_stack([R * np.cos(a), R * np.sin(a)])


def _circle_polygon(radius: float, n: int = 32) -> np.ndarray:
    """CCW circle polygon."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def _decompose_nut_shell_cylinder(
    r_inner: float,
    r_outer: float,
    n_sectors: int,
) -> list[np.ndarray]:
    """Return a list of 2D (x, y) trapezoidal polygons tiling the annular
    cylinder (r_inner, r_outer). Inner arc is chord-approximated so the
    *midpoint* of each chord sits at r_inner — i.e. we inflate the polygon
    inner radius so the flat chord never crosses the nominal bore.

    n_sectors trapezoidal prisms; each is a 4-vertex convex polygon.
    """
    if n_sectors < 3:
        raise ValueError("n_sectors must be >= 3 for a cylinder shell.")
    # Inflate so chord midpoint sits at r_inner.
    # chord midpoint distance = r_inflated * cos(π/n_sectors). Set = r_inner.
    r_inner_inflated = r_inner / np.cos(np.pi / n_sectors)
    # Outer: chord endpoints touch nominal outer; the midpoint then sits
    # slightly inside. Acceptable (doesn't affect contact with the bolt).
    sectors = []
    for k in range(n_sectors):
        a0 = 2 * np.pi * k / n_sectors
        a1 = 2 * np.pi * (k + 1) / n_sectors
        # CCW: outer(a0) -> outer(a1) -> inner(a1) -> inner(a0)
        poly = np.array([
            [r_outer * np.cos(a0), r_outer * np.sin(a0)],
            [r_outer * np.cos(a1), r_outer * np.sin(a1)],
            [r_inner_inflated * np.cos(a1), r_inner_inflated * np.sin(a1)],
            [r_inner_inflated * np.cos(a0), r_inner_inflated * np.sin(a0)],
        ])
        sectors.append(poly)
    return sectors


def _decompose_nut_shell_hex(
    r_inner: float,
    across_flats: float,
    n_bore: int = 32,
) -> list[np.ndarray]:
    """Hex-outline nut: (hexagon) minus (cylindrical bore). Decompose via
    bridge + ear-clip (handles any simple polygon-with-hole). Returns triangle
    soup — many small convex pieces — which MuJoCo handles cheaply."""
    hex_outer = _hex_polygon(across_flats)
    # Circumscribe the bore polygon so its chords never cut inside r_inner.
    bore = _circle_polygon(r_inner / np.cos(np.pi / n_bore), n_bore)
    # Make sure bore fits well inside hexagon inscribed radius (= across_flats/2).
    if r_inner >= across_flats / 2 * 0.98:
        raise ValueError(
            f"bore radius {r_inner} too large for hex across_flats {across_flats} "
            f"(min flat-to-center distance = {across_flats/2}).")
    return pg.triangulate(hex_outer, [bore])


_WEDGE_FACES_QUAD = [
    # end cap at θ (vertices 0,1,2,3 — trapezoid in meridian plane)
    (0, 1, 2, 3),
    # end cap at θ+Δθ (4,5,6,7 — reversed to face the other way)
    (7, 6, 5, 4),
    # base barrel (0,1,5,4): at core cylinder, outward normal = into the core
    # (interior of wedge is above, i.e. +dr). This face is internal when core
    # cylinder is also present, so orientation doesn't matter for contact.
    (4, 5, 1, 0),
    # crest barrel (3,2,6,7): outward = +dr (bolt apex)
    (3, 2, 6, 7),
    # rising flank (1,2,6,5): connects base-right to crest-right
    (1, 2, 6, 5),
    # falling flank (0,4,7,3): connects base-left to crest-left
    (0, 4, 7, 3),
]


def _write_wedge_obj(path: Path, wedges: list[np.ndarray], flip: bool = False,
                     faces: bool = True) -> None:
    """Write a batch of 8-vertex thread wedges as a single OBJ. Face topology
    is the trapezoid-swept prism connectivity above. `flip` reverses triangle
    winding (used for nut threads, which have -dr profile)."""
    lines = ["# Generated by mkscrew.py"]
    face_lines: list[str] = []
    n_verts = 0
    if not faces or any(len(v) != 8 for v in wedges):
        # A wedge clipped at a thread end: vertices only, MuJoCo hulls them.
        if len(wedges) != 1:
            raise ValueError("clipped wedges must be written one per OBJ")
        lines += [f"v {mj.num(x)} {mj.num(y)} {mj.num(z)}" for x, y, z in wedges[0]]
        path.write_text("\n".join(lines) + "\n")
        return

    for verts in wedges:
        base_idx = n_verts + 1
        for (x, y, z) in verts:
            lines.append(f"v {mj.num(x)} {mj.num(y)} {mj.num(z)}")
            n_verts += 1
        for quad in _WEDGE_FACES_QUAD:
            a, b, c, d = (base_idx + i for i in quad)
            if flip:
                face_lines.append(f"f {a} {c} {b}")
                face_lines.append(f"f {a} {d} {c}")
            else:
                face_lines.append(f"f {a} {b} {c}")
                face_lines.append(f"f {a} {c} {d}")

    lines.extend(face_lines)
    path.write_text("\n".join(lines) + "\n")


# ----- wedges -----------------------------------------------------------------

def _wedges(*, major_diameter, pitch, thread_length, n_starts, segments_per_turn,
            thread_form, thread_params, handedness, clearance, seam_relief, thread_chamfer,
            include_bolt, include_nut, nut_driven, nut_height, bolt_shank_length,
            bolt_head_height) -> dict:
    """Thread wedges of the bolt and the nut, in each part's own frame.

    The bolt's thread spans z in [0, thread_length] from its tip; the nut's
    spans [0, nut_height] from its bottom face. Every wedge is the convex hull
    of its vertices.
    """
    D = major_diameter
    P = pitch
    lead = n_starts * P
    form = thread_profile(thread_form, D, P, thread_params)
    r_minor = form.r_core            # bolt core
    r_root = form.r_root             # nut root at zero clearance

    if clearance < 0:
        raise ValueError("clearance must be >= 0")
    if seam_relief < 0:
        raise ValueError("seam_relief must be >= 0")
    if thread_chamfer < 0:
        raise ValueError("thread_chamfer must be >= 0")
    # form.teeth rejects a clearance that leaves no tooth (for ISO,
    # clearance >= √3·P/8, about 0.22·P).

    sign = 1 if handedness == 'right' else -1
    dtheta = 2 * np.pi / segments_per_turn

    # Bolt threads span [0, thread_length] in the bolt local frame.
    # Nut threads span only [0, nut_height] (nut's bore is shorter than the
    # bolt shaft in general). Interleave the two with a half-pitch axial
    # offset so teeth of one fit into the root flats of the other.
    # Teeth as (r, u) in the bolt's phase, both parts' frames on the pitch
    # grid (the nut tooth already sits in the bolt's gaps).
    bolt_profile, nut_profile = form.teeth(clearance)
    # Nut-driven scenes flip the bolt (below), which mirrors its tooth in u.
    # A symmetric tooth only moves the nut's phase; an asymmetric one needs
    # the nut's tooth mirrored too, as a nut turned over with its bolt.
    mirror_nut = nut_driven and not form.symmetric()
    if mirror_nut:
        nut_profile = _canonical(nut_profile * [1, -1], r_root)
    # Seam relief: the same teeth with flanks and crest moved in by
    # `seam_relief` (i.e. clearance + 2 * seam_relief), used at wedge ends.
    bolt_relief = nut_relief = None
    relief_angle = 0.0
    if seam_relief > 0:
        bolt_relief, nut_relief = form.teeth(clearance + 2 * seam_relief)
        if mirror_nut:
            nut_relief = _canonical(nut_relief * [1, -1], r_root)
        # Ramp about 20x as long as it is deep, measured at mid-flank; at most
        # a third of the wedge at each end so every wedge keeps full teeth.
        relief_angle = min(20 * seam_relief / ((r_minor + form.r_crest) / 2),
                           2 * np.pi / segments_per_turn / 3)

    def _gen_wedges(profile, base_radius, axial_extent, internal=False, relief=None,
                    pieces=None):
        """Generate wedge vertex arrays for all starts × turns × segments.

        The helix covers [0, axial_extent] axially: the turns run from the
        lowest one that still rises above z = 0 (the partial first turn at
        the z = 0 end face) to past the other end. The profile's u is its phase:
        the nut's is already in the bolt's, so the teeth interlock.

        Multi-start threads are interleaved AXIALLY. Each start k has its
        first tooth at z = k·P; all N helices share the same angular range
        [0, 2π·n_turns] — that produces the "tooth every P" axial density
        regardless of n_starts.

        Each wedge is the hull of the profile at θ and θ+Δθ, so its faces
        are chords of the helical surfaces. On an external thread they fall
        inside the true tooth (or inside the core). On an internal thread
        they bulge toward the bolt two ways, and `internal` removes both so
        that discretization only ever adds play:
          - the tip and flank arcs sag by r(1 - cos(Δθ/2)) (36 µm at M10 with
            24 segments): the wedge is pushed out radially by 1/cos(Δθ/2);
          - each flank is a twisted quad of the helicoid, and its hull
            diagonal sits off the helicoid axially by
            lead·Δθ·(r1 - r2) / (4π(r1 + r2)) (2.6 µm at M10 with 24
            segments): both flanks are moved back into the tooth by that.
        Wedges are clipped to [0, axial_extent] so no tooth sticks out past
        the part's end faces.

        With `relief` (the tooth shrunk by the seam relief) each wedge is the
        hull of four copies instead of two: `relief` at θ and θ+Δθ, the full
        tooth `relief_angle` in from each end (see seam_relief).

        `pieces`: for a non-convex tooth (then CCW), the vertex index lists of
        its convex pieces. The corrections move the tooth's own edges (mitred,
        so reflex corners come out right), and each piece is swept as its own
        wedges; the cuts between pieces lie inside the tooth.
        """
        n_turns = int(np.ceil(axial_extent / lead)) + 2
        # A turn's tooth rises lead per turn from u in [u_min, u_max], so the
        # turn before turn 0 still pokes above z = 0 over most of the first
        # turn. It is clipped at the face like the rest.
        first_turn = -int(np.ceil(max(profile[:, 1].max(), 0.0) / lead)) - 1
        wedges = []
        r = base_radius + profile[:, 0]
        normals = _edge_normals(profile)
        base = _base_edges(profile, r_root if internal else r_minor)   # never moved
        # Moving a flank back along u by t moves edge i in by |n_u| t.
        n_u = np.where(base, 0.0, np.abs(normals[:, 1]))
        # The internal-thread corrections must cover every chord of the hull.
        # Without relief that is the one chord across the wedge. With relief
        # it is the larger of the full-height middle chord (span
        # dtheta - 2 * relief_angle) and the chord between the two relieved
        # ends (across the whole wedge, but starting `seam_relief` further
        # back, which is an axial setback of seam_relief / |n_u| on a flank).
        span = dtheta if relief is None else dtheta - 2 * relief_angle
        radial_scale = 1 / np.cos(span / 2)
        offsets = np.zeros(len(profile))
        if internal:
            twist = lead * span * (r.max() - r.min()) / (4 * np.pi * (r.max() + r.min()))
            if relief is not None:
                r_tip = r.min()
                radial_scale = max(radial_scale,
                                   r_tip / ((r_tip + seam_relief) * np.cos(dtheta / 2)))
                full = lead * dtheta * (r.max() - r.min()) / (4 * np.pi * (r.max() + r.min()))
                setback = seam_relief / max(n_u.max(), 0.1)
                twist = max(twist, full - setback)
            offsets = n_u * twist
        else:
            # External thread. A flank quad's hull diagonal sits off the
            # helicoid axially by lead·Δθ·(r1 - r2) / (4π(r1 + r2)), but the
            # chord also sags inward by r(1 - cos(Δθ/2)), which on a flank
            # that widens toward the core (ISO, trapezoidal) more than makes
            # up for it. On steep flanks (square, the sides of a round wire)
            # nothing makes up for it, so those edges are moved back.
            r1, r2 = r, np.roll(r, -1)
            bulge = lead * dtheta * np.abs(r1 - r2) / (4 * np.pi * (r1 + r2))
            sag = np.minimum(r1, r2) * (1 - np.cos(dtheta / 2))
            offsets = np.where(base, 0.0, np.maximum(0.0, n_u * bulge - normals[:, 0] * sag))
            if relief is not None:
                # With relief, the short ramp quads between a relieved and a
                # full copy can bulge out by their own twist. Those quads join
                # two different profiles, so the first-order twist bound is
                # doubled (measured bulge stays below it).
                twist = (2 * lead * relief_angle * (r.max() - r.min())
                         / (4 * np.pi * (r.max() + r.min())))
                offsets = np.maximum(offsets, n_u * twist)
        line = r_root if internal else r_minor
        full, full_relief = profile, relief

        def shrink(offsets):
            out = []
            for prof in (full, full_relief):
                if prof is not None and offsets.any():
                    # relief shares the profile's edge directions, so the
                    # same per-edge offsets apply edge by edge.
                    if pieces is not None:       # keep the vertex order the pieces index
                        prof = _offset_mitre(prof, offsets)
                    else:
                        prof = _offset_polygon(prof, offsets if len(prof) == len(full)
                                               else offsets.max())
                    if prof is None:
                        raise ValueError("segments_per_turn too low for this thread: "
                                         "the flank corrections leave no tooth")
                    if pieces is None:
                        prof = _canonical(prof, line)
                out.append(prof)
            return out

        n_full = len(full)
        parts = pieces if pieces is not None else [list(range(n_full))]

        def excess(prof, rel):
            """How far each edge of the full tooth is overshot by the hull of
            one wedge per piece (all wedges are congruent), measured in each
            hull point's own meridian plane. Samples every facet the hull can
            have between two rings: both triangle pairs of each side quad
            (either diagonal can split it), interiors included, since the
            curved flank can bow out past a facet's middle (2 nm at 12
            segments) as well as along its edges."""
            req = np.zeros(n_full)
            if len(prof) != n_full or (rel is not None and len(rel) != n_full):
                return req
            ii, jj = np.meshgrid(np.arange(33), np.arange(33), indexing='ij')
            on = ii + jj <= 32
            w = np.column_stack([ii[on], jj[on], 32 - ii[on] - jj[on]]) / 32

            def overshoot(tri, k):
                """Largest overshoot of the full tooth's edge k over a
                triangle of hull points: a barycentric grid, then zooming in
                on its peak (the overshoot is curved; a grid alone misses the
                peak by ~10 nm)."""
                tri, peak, at, ww = np.array(tri), -np.inf, None, w
                for level in range(4):
                    x = ww @ tri
                    u = x[:, 2] - sign * lead * np.arctan2(x[:, 1], x[:, 0]) / (2 * np.pi)
                    rr = np.hypot(x[:, 0], x[:, 1]) - base_radius
                    dev = (np.column_stack([rr, u]) - full[k]) @ normals[k]
                    i = int(np.argmax(dev))
                    if dev[i] > peak:
                        peak, at = dev[i], ww[i]
                    ww = np.clip(at + (w - 1 / 3) / 8 ** (level + 1), 0.0, None)
                    ww /= ww.sum(axis=1, keepdims=True)
                return peak
            for ix in parts:
                rings = [(prof[ix], 0.0), (prof[ix], dtheta)] if rel is None else \
                    [(rel[ix], 0.0), (prof[ix], relief_angle),
                     (prof[ix], dtheta - relief_angle), (rel[ix], dtheta)]
                pts = []
                for pr, a in rings:
                    v = _profile_ring(pr, a, base_radius, sign * a * lead / (2 * np.pi))
                    if internal:
                        v[:, :2] *= radial_scale
                    pts.append(v)
                m = len(ix)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        a, b = pts[i], pts[j]
                        for e in range(m):
                            k = ix[e]
                            if (ix[(e + 1) % m] - k) % n_full != 1 or base[k]:
                                continue            # a cut between pieces, or the base
                            e1 = (e + 1) % m
                            for tri in ((a[e], a[e1], b[e1]), (a[e], b[e1], b[e]),
                                        (a[e], a[e1], b[e]), (a[e1], b[e1], b[e])):
                                req[k] = max(req[k], overshoot(tri, k))
            return req

        # The formulas above are first order. Top them up from the geometry:
        # measure one wedge against the full tooth and move back any edge
        # that still sticks out (a few nm at most), until none does.
        for _ in range(5):
            profile, relief = shrink(offsets)
            over = excess(profile, relief)
            if over.max() <= 1e-13:
                break
            offsets = offsets + 1.1 * over
        z_adv = sign * dtheta * lead / (2 * np.pi)
        for ix in parts:
            prof_p = profile[ix]
            rel_p = None if relief is None else relief[ix]
            if pieces is not None and not (pg.is_convex(prof_p) and
                                           (rel_p is None or pg.is_convex(rel_p))):
                raise ValueError("clearance or seam_relief too large for this tooth's "
                                 "convex pieces")
            for start_idx in range(n_starts):
                for turn in range(first_turn, n_turns):
                    z_turn_base = start_idx * P + turn * lead
                    for seg in range(segments_per_turn):
                        theta_i = seg * dtheta
                        z_at_theta = z_turn_base + sign * theta_i * lead / (2 * np.pi)
                        if rel_p is None:
                            v = _thread_wedge_vertices(
                                prof_p, theta_i, dtheta, z_adv,
                                base_radius, z_at_theta)
                        else:
                            rings = [(rel_p, 0.0), (prof_p, relief_angle),
                                     (prof_p, dtheta - relief_angle), (rel_p, dtheta)]
                            v = np.concatenate([
                                _profile_ring(prof, theta_i + a, base_radius,
                                              z_at_theta + sign * a * lead / (2 * np.pi))
                                for prof, a in rings])
                        if internal:
                            v[:, :2] *= radial_scale
                        if _wedge_has_volume(v, len(prof_p)):
                            v = _clip_z(v, 0.0, axial_extent)
                            if v is not None and thread_chamfer > 0:
                                v = _chamfer(v, internal, theta_i, theta_i + dtheta)
                            if v is not None:
                                wedges.append(v)
        return wedges

    # The teeth carry absolute radii (base_radius 0). The nut tooth's u is in
    # the bolt's phase; normalize it to end within the first pitch so its
    # first turn starts at or below the nut's base (the wedges are clipped to
    # the part), as the bolt's does.
    def _first_pitch(prof, rel, shift):
        return prof + [0, shift], (None if rel is None else rel + [0, shift])

    bolt_profile, bolt_relief = _first_pitch(
        bolt_profile, bolt_relief, -P * np.floor(bolt_profile[:, 1].min() / P))
    nut_profile, nut_relief = _first_pitch(
        nut_profile, nut_relief, -P * np.ceil(nut_profile[:, 1].max() / P - 1))

    # Chamfers: (radius at the end face, slope, face z, direction into the
    # part) for each chamfered end of each part.
    bolt_ends = nut_faces = ()
    if thread_chamfer > 0:
        r_crest = bolt_profile[:, 0].max()
        r_tip, r_bore = nut_profile[:, 0].min(), r_root + clearance / 2
        k_bolt = (r_crest - r_minor) / thread_chamfer
        k_nut = (r_bore - r_tip) / thread_chamfer
        bare = bolt_shank_length == 0 and bolt_head_height == 0
        bolt_ends = ((0.0, 1),) + (((thread_length, -1),) if bare else ())
        nut_faces = ((0.0, 1), (nut_height, -1))

    def _chamfer(v, internal, theta0, theta1):
        """Cut a wedge by the chamfer cones near its part's ends."""
        z = v[:, 2]
        for z0, towards in (nut_faces if internal else bolt_ends):
            if towards * (z.min() - z0) > 1.05 * thread_chamfer and \
                    towards * (z.max() - z0) > 1.05 * thread_chamfer:
                continue                        # wholly past this chamfer
            if internal:
                n, d = _cone_tangent(r_bore, k_nut, z0, towards, (theta0 + theta1) / 2)
            else:
                n, d = _cone_chord(r_minor, k_bolt, z0, towards, theta0, theta1)
            v = _clip_half(v, n, d)
            if v is None:
                return None
            z = v[:, 2]
        return v

    def _split(tooth, rel):
        """A non-convex tooth made CCW (with its relief), and its convex
        pieces; a convex tooth as it is, with no pieces."""
        if pg.is_convex(tooth):
            return tooth, rel, None
        if pg.signed_area(tooth) < 0:
            tooth, rel = tooth[::-1], (None if rel is None else rel[::-1])
        return tooth, rel, _convex_partition(tooth)

    bolt_wedges = []
    if include_bolt:
        tooth, rel, pieces = _split(bolt_profile, bolt_relief)
        bolt_wedges = _gen_wedges(tooth, 0.0, thread_length, relief=rel, pieces=pieces)
    nut_wedges = []
    if include_nut:
        tooth, rel, pieces = _split(nut_profile, nut_relief)
        nut_wedges = _gen_wedges(tooth, 0.0, nut_height, internal=True, relief=rel,
                                 pieces=pieces)
    # The OBJ writer's face table needs a plain four-sided tooth in canonical
    # order; pieces of a non-convex tooth are written as bare vertices.
    faces = pg.is_convex(bolt_profile) and pg.is_convex(nut_profile)
    return dict(bolt=bolt_wedges, nut=nut_wedges, faces=faces, form=form,
                r_minor=r_minor, r_root=r_root, lead=lead, mirror_nut=mirror_nut,
                bolt_profile=bolt_profile)


# ----- scene --------------------------------------------------------------------

def thread(
    *,
    major_diameter: float = 0.010,
    pitch: float = 0.0015,
    thread_length: float = 0.020,
    form: str = 'iso',
    form_params: dict | None = None,
    n_starts: int = 1,
    handedness: str = 'right',
    clearance: float = 1e-4,
    segments_per_turn: int = 24,
    seam_relief: float = 0.0,
    chamfer: float = 0.0,
    bolt: bool = True,
    bolt_shank_length: float = 0.0,
    bolt_head: str = 'hex',
    bolt_head_size: float = 0.016,
    bolt_head_height: float = 0.0064,
    bolt_density: float = 7800.0,
    nut: bool = True,
    nut_shape: str = 'hex',
    nut_size: float = 0.016,
    nut_height: float = 0.008,
    nut_sectors: int = 16,
    nut_density: float = 7800.0,
    driven: str = 'bolt',
    fixed: bool = True,
    friction=0.3,
    condim: int = 3,
    timestep: float = 0.001,
    standalone: bool = True,
    floor: bool = True,
    mesh_dir: str | Path | None = None,
    scale: float = 1.0,
) -> str:
    """A bolt engaged in a nut, as an MJCF string. Defaults: a steel M10x1.5.

    Args:
        major_diameter: Outer diameter of the bolt's thread, in meters.
        pitch: Axial distance between neighboring teeth.
        thread_length: Length of the bolt's threaded section.
        form: Tooth shape: ``iso`` (ISO 68-1 metric, and UTS), ``trapezoidal``
            (ISO 2904), ``acme``, ``square``, ``buttress``, ``wire`` (a round
            wire wound on each part), or ``custom``.
        form_params: Overrides of the form's defaults; see
            :func:`thread_profile`.
        n_starts: Number of interleaved helices; the lead is ``n_starts *
            pitch``.
        handedness: ``right`` or ``left``.
        clearance: Gap between the facing surfaces of the two teeth. Every
            exposed tooth edge moves back half of it.
        segments_per_turn: Wedges per helix turn. Fewer segments only add
            play; 24 suits clearances of 100 um, use 48-96 for tens of um.
        seam_relief: Dips the teeth by this depth at the seams between wedges,
            ramping back over about 20 times the depth, so a tooth sliding
            across a seam is not caught by the next wedge's end face. About
            5 um suits MuJoCo's default contact softness; 0 is exact.
        chamfer: Chamfers the thread ends over this axial length, as on
            machined parts (45 degrees when it equals the thread depth).
        bolt: Include the bolt.
        bolt_shank_length: Unthreaded length between the thread and the head.
        bolt_head: ``hex`` or ``cylinder``.
        bolt_head_size: Across flats (hex) or diameter (cylinder).
        bolt_head_height: Height of the head; 0 for a bare threaded rod.
        bolt_density: kg/m^3.
        nut: Include the nut.
        nut_shape: ``hex`` or ``cylinder``.
        nut_size: Across flats (hex) or outer diameter (cylinder).
        nut_height: Height of the nut, all of it threaded.
        nut_sectors: Sectors of a cylindrical nut's shell.
        nut_density: kg/m^3.
        driven: Which part sits on top, ready to be turned into the other:
            ``bolt`` (tip down into a nut) or ``nut`` (onto an upright bolt).
        fixed: Weld the lower part to the world. False leaves both parts free.
        friction: Sliding friction, or the (sliding, torsional, rolling) triple.
            Friction above the tangent of the lead angle makes a thread
            self-locking; 0.1-0.3 lets a spun part run down freely.
        condim: MuJoCo contact dimensionality (1, 3, 4 or 6).
        timestep: Simulation time step in seconds (standalone files). The
            stiff thread contacts need 1 ms or less.
        standalone: Add a floor, a light and solver options so the file runs on
            its own. False emits only the parts.
        floor: Add a ground plane (standalone files).
        mesh_dir: Write meshes as OBJ files into this folder, to be saved next
            to the XML file, instead of inline vertex lists.
        scale: Multiplies every length, including lengths in ``form_params``.

    Returns:
        The MJCF model as a string, with bodies ``bolt`` and ``nut``. The bolt's
        frame is at the center of its tip with its axis along +z; the nut's at
        the center of its bottom face.
    """
    for name, value, choices in (('handedness', handedness, ('right', 'left')),
                                 ('driven', driven, ('bolt', 'nut')),
                                 ('bolt_head', bolt_head, ('hex', 'cylinder')),
                                 ('nut_shape', nut_shape, ('hex', 'cylinder'))):
        _require(value in choices, f"{name} must be one of {', '.join(choices)}; got {value!r}.")
    _require(int(n_starts) == n_starts and n_starts >= 1, "n_starts must be a positive integer.")
    _require(int(segments_per_turn) == segments_per_turn and segments_per_turn >= 6,
             "segments_per_turn must be an integer of at least 6.")
    _require(bolt or nut, "Include the bolt, the nut, or both.")
    _require(scale > 0, "scale must be positive.")
    for name, value in dict(major_diameter=major_diameter, pitch=pitch, thread_length=thread_length,
                            nut_height=nut_height, nut_size=nut_size, bolt_head_size=bolt_head_size,
                            bolt_density=bolt_density, nut_density=nut_density).items():
        _require(value > 0, f"{name} must be positive.")
    for name, value in dict(clearance=clearance, seam_relief=seam_relief, chamfer=chamfer,
                            bolt_shank_length=bolt_shank_length,
                            bolt_head_height=bolt_head_height).items():
        _require(value >= 0, f"{name} must be nonnegative.")

    s = scale
    D, P, length = major_diameter * s, pitch * s, thread_length * s
    c, shank, head_h = clearance * s, bolt_shank_length * s, bolt_head_height * s
    nut_h = nut_height * s
    geo = _wedges(
        major_diameter=D, pitch=P, thread_length=length, n_starts=int(n_starts),
        segments_per_turn=int(segments_per_turn), thread_form=form,
        thread_params=_scale_form_params(form_params, s), handedness=handedness,
        clearance=c, seam_relief=seam_relief * s, thread_chamfer=chamfer * s,
        include_bolt=bolt, include_nut=nut, nut_driven=driven == 'nut', nut_height=nut_h,
        bolt_shank_length=shank, bolt_head_height=head_h)
    r_minor, r_root, lead = geo['r_minor'], geo['r_root'], geo['lead']

    # Nut shell: the root moved out c/2 like the teeth.
    shell: list[np.ndarray] = []
    if nut:
        r_bore = r_root + c / 2
        if nut_shape == 'cylinder':
            _require(nut_size * s / 2 > r_bore / np.cos(np.pi / nut_sectors),
                     "nut_size is too small for the thread.")
            shell = _decompose_nut_shell_cylinder(r_bore, nut_size * s / 2, int(nut_sectors))
        else:
            shell = _decompose_nut_shell_hex(r_bore, nut_size * s)

    # Layout. The bolt-driven scene puts the bolt's tip about a pitch below
    # the nut's top, on the pitch grid relative to the nut, where the teeth
    # interlock with the clearance centered. The nut-driven scene turns the
    # bolt tip up (a half turn about y, which keeps the handedness).
    bolt_full = length + shank + head_h
    if driven == 'nut':
        z_bolt = FLOOR_GAP + bolt_full
        # The flip maps the bolt's u to z_bolt - lead/2 - u (mod P): a
        # mirrored nut tooth interlocks at z_bolt - lead/2, a symmetric one
        # (its own mirror about its center u_c) at z_bolt - lead/2 - 2 u_c.
        prof = geo['bolt_profile']
        u_c = (prof[:, 1].min() + prof[:, 1].max()) / 2
        phase = (-lead / 2 - (0.0 if geo['mirror_nut'] else 2 * u_c)) % P
        z_nut = z_bolt - nut_h + P
        z_nut = z_bolt + phase + P * round((z_nut - z_bolt - phase) / P)
    else:
        z_nut = FLOOR_GAP
        z_bolt = z_nut + P * max(0, round((nut_h - P) / P))

    model = mj.Model('thread')
    model.contact_defaults(friction=friction, condim=condim)
    meshes = mj.Meshes(model, mesh_dir)
    if standalone:
        model.standalone(timestep=timestep, floor=floor, floor_size=max(0.2, 4 * bolt_full),
                         light_pos=(0.3, 0.3, 0.3))
    wb = model.section('worldbody')

    def wedge_mesh(name, verts, flip):
        if meshes.exporting:
            _write_wedge_obj(meshes.dir / f'{name}.obj', [verts], flip=flip, faces=geo['faces'])
            ET.SubElement(meshes.asset, 'mesh', name=name, file=f'{name}.obj')
            return name
        return meshes.points(name, verts)

    def geom(body, name, rgba, density, **attrs):
        ET.SubElement(body, 'geom', {'name': name, **attrs, 'rgba': rgba,
                                     'density': mj.num(density)})

    if nut:
        free = driven == 'nut' or not fixed
        body = ET.SubElement(wb, 'body', name='nut', childclass=mj.CLASS, pos=mj.vec((0, 0, z_nut)))
        if free:
            ET.SubElement(body, 'freejoint', name='nut_joint')
        for k, poly in enumerate(shell):
            name = meshes.prism(f'nut_shell_{k}', poly, 0.0, nut_h)
            geom(body, name, NUT_RGBA, nut_density, type='mesh', mesh=name)
        for k, verts in enumerate(geo['nut']):
            name = wedge_mesh(f'nut_thread_{k}', verts, flip=True)
            geom(body, name, NUT_RGBA, nut_density, type='mesh', mesh=name)

    if bolt:
        free = driven == 'bolt' or not fixed
        attrs = {'name': 'bolt', 'childclass': mj.CLASS, 'pos': mj.vec((0, 0, z_bolt))}
        if driven == 'nut':
            attrs['quat'] = '0 0 1 0'
        body = ET.SubElement(wb, 'body', attrs)
        if free:
            ET.SubElement(body, 'freejoint', name='bolt_joint')
        geom(body, 'bolt_core', BOLT_RGBA, bolt_density, type='cylinder',
             size=mj.vec((r_minor, length / 2)), pos=mj.vec((0, 0, length / 2)))
        if shank > 0:
            geom(body, 'bolt_shank', BOLT_RGBA, bolt_density, type='cylinder',
                 size=mj.vec((D / 2, shank / 2)), pos=mj.vec((0, 0, length + shank / 2)))
        if head_h > 0:
            z0 = length + shank
            if bolt_head == 'cylinder':
                geom(body, 'bolt_head', HEAD_RGBA, bolt_density, type='cylinder',
                     size=mj.vec((bolt_head_size * s / 2, head_h / 2)),
                     pos=mj.vec((0, 0, z0 + head_h / 2)))
            else:
                name = meshes.prism('bolt_head', _hex_polygon(bolt_head_size * s), z0, z0 + head_h)
                geom(body, 'bolt_head', HEAD_RGBA, bolt_density, type='mesh', mesh=name)
        for k, verts in enumerate(geo['bolt']):
            name = wedge_mesh(f'bolt_thread_{k}', verts, flip=False)
            geom(body, name, BOLT_RGBA, bolt_density, type='mesh', mesh=name)
    return model.to_xml()


def _scale_form_params(params: dict | None, scale: float) -> dict:
    """form_params with its lengths scaled."""
    out = {}
    for key, val in dict(params or {}).items():
        if key in THREAD_LENGTH_PARAMS:
            val = (np.asarray(val, dtype=float) * scale).tolist() \
                if isinstance(val, (list, tuple)) else float(val) * scale
        out[key] = val
    return out


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)
