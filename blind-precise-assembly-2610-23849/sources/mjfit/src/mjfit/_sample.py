"""Random peg-in-hole assemblies, for training on many different parts.

Each sample is a config (a plain dict, ready for :func:`mjfit.build` or a
YAML file) drawn from its own random stream: sample ``index`` of ``seed`` is
the same however many samples you draw.

Two families:

``primitive``
    A circle, rectangle, stadium, regular polygon or star, with up to three
    islands (circles, rectangles, stadiums, polygons) standing in the hole,
    and an optional handle.
``convex``
    Random convex outlines with an exact number of vertices and Bezier-
    rounded corners, an optional convex handle and a convex plate. With the
    defaults this is the peg family of the paper's training distribution.
"""
from __future__ import annotations

import random

import numpy as np

from . import _polygon as pg
from . import shapes

FAMILIES = ('primitive', 'convex')
HOLE_SHAPES = ('circle', 'rect', 'stadium', 'ngon', 'star')
ISLAND_SHAPES = ('circle', 'rect', 'stadium', 'ngon')
ROUNDING_STYLES = ('polygon', 'mixed', 'smooth')


def sample_pegs(n: int, seed: int = 0, **kwargs) -> list[dict]:
    """``n`` random peg configs; keyword arguments go to :func:`sample_peg`."""
    return [sample_peg(seed, i, **kwargs) for i in range(n)]


def sample_peg(seed: int = 0, index: int = 0, *, family: str = 'primitive', **kwargs) -> dict:
    """One random peg config.

    Args:
        seed: Seed of the whole set.
        index: Which sample of the set.
        family: ``primitive`` or ``convex`` (see the module docs).
        **kwargs: Options of the family, each a fixed value or a ``(low,
            high)`` range to draw from uniformly (see ``docs/sampling.md``).
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {', '.join(FAMILIES)}, got {family!r}.")
    rng = random.Random(f'mjfit:{family}:{seed}:{index}')
    return (primitive_peg if family == 'primitive' else convex_peg)(rng, **kwargs)


def primitive_peg(
    rng: random.Random,
    *,
    size=0.025,
    clearance=5e-4,
    islands=(0, 3),
    hole_shapes=HOLE_SHAPES,
    island_shapes=ISLAND_SHAPES,
    plate_thickness=(0.01, 0.03),
    plate_scale=(5.0, 10.0),
    peg_overhang=(0.02, 0.06),
    handle_probability: float = 0.5,
    handle_height=(0.01, 0.04),
    handle_margin=(0.002, 0.006),
    min_wall: float = 0.001,
    attempts: int = 200,
) -> dict:
    """A primitive hole with islands.

    Args:
        rng: Random stream.
        size: Circumradius of the hole's shape.
        clearance: Peg-to-hole clearance.
        islands: Number of islands; fewer are placed if they do not fit.
        hole_shapes: Shape types to draw the hole from.
        island_shapes: Shape types to draw islands from (convex ones).
        plate_thickness: Thickness of the plate.
        plate_scale: Side of the square plate, as a multiple of ``size``.
        peg_overhang: How much longer the peg is than the plate is thick.
        handle_probability: Chance of a handle.
        handle_height: Height of the handle.
        handle_margin: How far the handle extends past the peg (at least
            twice the clearance, so it cannot pass the hole).
        min_wall: Thinnest wall of the hollow peg, between its pockets and
            around them.
        attempts: Placement tries per island.
    """
    size, c = _draw(rng, size), _draw(rng, clearance)
    thick = _draw(rng, plate_thickness)
    cfg = {'kind': 'peg', 'clearance': c, 'plate_size': size * _draw(rng, plate_scale),
           'plate_thickness': thick, 'peg_length': thick + _draw(rng, peg_overhang)}
    for _ in range(attempts):
        hole = _primitive(rng, hole_shapes, size)
        outline = shapes.from_spec(hole)
        peg_outline = pg.offset_inward(outline, c)
        if pg.offset_is_valid(outline, peg_outline):
            break
    else:
        raise ValueError(f"No hole from {hole_shapes} leaves a peg at clearance {c:g} m.")
    cfg['hole'] = hole
    placed: list[np.ndarray] = []
    specs = []
    for _ in range(int(round(_draw(rng, islands)))):
        for _ in range(attempts):
            spec = _primitive(rng, island_shapes, size * rng.uniform(0.10, 0.35),
                              center=(rng.uniform(-0.6, 0.6) * size, rng.uniform(-0.6, 0.6) * size))
            island = shapes.from_spec(spec)
            if not pg.contains(peg_outline, pg.offset_outward(island, c + min_wall)):
                continue
            grown = pg.offset_outward(island, c + min_wall / 2)
            if all(pg.disjoint(grown, pg.offset_outward(p, c + min_wall / 2)) for p in placed):
                placed.append(island)
                specs.append(spec)
                break
        else:
            break                      # give up on further islands rather than fail
    if specs:
        cfg['islands'] = specs
    if rng.random() < handle_probability:
        cfg['handle_height'] = _draw(rng, handle_height)
        margin = _draw(rng, handle_margin)
        cfg['handle_margin'] = max(margin, 2 * c)
    return cfg


def convex_peg(
    rng: random.Random,
    *,
    vertices=(3, 10),
    width=(0.032, 0.048),
    aspect: float = 20 / 7,
    rounding=(1 / 3, 1 / 3, 1 / 3),
    corner_rounding=(0.2, 0.5),
    clearance=5e-4,
    total_height=(0.05, 0.10),
    min_insertion: float = 0.022,
    handle_probability: float = 0.5,
    handle_height=(0.02, 0.04),
    handle_min_margin: float = 0.001,
    handle_max_margin: float = 0.01,
    plate_thickness=(0.01, 0.02),
    plate_min_rim: float = 0.005,
    plate_max_rim: float = 0.03,
    max_error: float = 1e-4,
    attempts: int = 1000,
) -> dict:
    """A random convex outline with rounded corners, a handle and a plate.

    Args:
        rng: Random stream.
        vertices: Number of control vertices of each outline.
        width: Span of the hole along x.
        aspect: Width over height of the hole's span.
        rounding: Probabilities of the corner styles: all sharp, a random
            subset rounded, all fully rounded.
        corner_rounding: Rounding fraction of the corners in the mixed style.
        clearance: Peg-to-hole clearance.
        total_height: Height of the peg including its handle.
        min_insertion: Shortest insertion section under a handle.
        handle_probability: Chance of a handle.
        handle_height: Height of the handle.
        handle_min_margin: Least distance from the peg outline to the handle's.
        handle_max_margin: Farthest any handle vertex may be from the peg.
        plate_thickness: Thickness of the plate.
        plate_min_rim: Least distance from the plate's outline to the hole and
            handle.
        plate_max_rim: Farthest any plate vertex may be from them.
        max_error: Chord error of the rounded corners.
        attempts: Tries per outline before falling back to a rounded hull.
    """
    c = _draw(rng, clearance)
    w = _draw(rng, width)
    h = w / _draw(rng, aspect)
    total = _draw(rng, total_height)
    for _ in range(attempts):
        hole = _rounded_convex(rng, vertices, w, h, rounding, corner_rounding, max_error)
        peg_outline = pg.offset_inward(hole, c)
        if pg.offset_is_valid(hole, peg_outline):
            break
    else:
        raise ValueError(f"Could not draw a convex hole that leaves a peg at clearance {c:g} m.")
    shift = _centroid(peg_outline)
    hole, peg_outline = hole - shift, peg_outline - shift
    cfg = {'kind': 'peg', 'hole': _plain(hole), 'clearance': c,
           'plate_thickness': _draw(rng, plate_thickness), 'center': False}
    covered = hole
    handle_h = 0.0
    if rng.random() < handle_probability:
        handle_h = min(_draw(rng, handle_height), total - min_insertion)
    if handle_h > 0:
        handle = _containing(rng, peg_outline, handle_min_margin, handle_max_margin, vertices,
                             rounding, corner_rounding, max_error, attempts)
        cfg['handle'], cfg['handle_height'] = _plain(handle), handle_h
        covered = np.vstack([hole, handle])
    cfg['peg_length'] = total - handle_h
    cfg['plate'] = _plain(_containing(rng, covered, plate_min_rim, plate_max_rim, vertices,
                                      rounding, corner_rounding, max_error, attempts))
    return cfg


# ----- helpers -----------------------------------------------------------------

def _draw(rng: random.Random, value):
    """A fixed value, or a uniform draw from a (low, high) range."""
    if isinstance(value, (tuple, list)):
        lo, hi = value
        if isinstance(lo, int) and isinstance(hi, int):
            return rng.randint(lo, hi)
        return rng.uniform(lo, hi)
    return value


def _primitive(rng: random.Random, pool, size: float, center=(0.0, 0.0)) -> dict:
    """A shape spec whose extent is bounded by `size`."""
    kind = rng.choice(list(pool))
    spec: dict = {'type': kind}
    if kind == 'circle':
        spec['radius'] = size * rng.uniform(0.5, 1.0)
    elif kind == 'rect':
        spec['width'] = size * rng.uniform(0.6, 1.0)
        spec['height'] = size * rng.uniform(0.4, 1.0)
    elif kind == 'stadium':
        spec['width'] = size * rng.uniform(0.7, 1.0)
        spec['height'] = size * rng.uniform(0.4, 0.9)
    elif kind == 'ngon':
        spec['n'] = rng.randint(3, 8)
        spec['radius'] = size * rng.uniform(0.6, 1.0)
    elif kind == 'star':
        spec['points'] = rng.randint(4, 8)
        spec['outer_radius'] = size * rng.uniform(0.7, 1.0)
        spec['inner_radius'] = spec['outer_radius'] * rng.uniform(0.4, 0.7)
    else:
        raise ValueError(f"Unknown shape type {kind!r} in the pool.")
    if kind != 'circle':
        spec['rotate'] = rng.uniform(0.0, 360.0)
    if any(center):
        spec['center'] = [float(center[0]), float(center[1])]
    return spec


def _random_convex(rng: random.Random, n: int, width: float, height: float) -> np.ndarray:
    """A random convex polygon with exactly n vertices spanning width x height.

    Valtr's algorithm: split sorted random coordinates into two monotone
    chains per axis, pair the x and y steps at random, and chain the steps
    in angle order.
    """
    def steps() -> list[float]:
        xs = sorted(rng.random() for _ in range(n))
        lo = hi = xs[0]
        out = []
        for x in xs[1:-1]:
            if rng.random() < 0.5:
                out.append(x - hi)
                hi = x
            else:
                out.append(lo - x)
                lo = x
        return out + [xs[-1] - hi, lo - xs[-1]]

    dx, dy = steps(), steps()
    rng.shuffle(dy)
    edges = np.column_stack([dx, dy])
    edges = edges[np.argsort(np.arctan2(edges[:, 1], edges[:, 0]))]
    verts = np.vstack([np.zeros((1, 2)), np.cumsum(edges[:-1], axis=0)])
    verts = verts * [width, height] / np.ptp(verts, axis=0)
    return verts - _centroid(verts)


def _rounded_convex(rng, vertices, width, height, styles, corner_rounding, max_error):
    n = int(_draw(rng, vertices))
    control = _random_convex(rng, n, width, height)
    style = rng.choices(ROUNDING_STYLES, weights=styles)[0]
    alphas = np.zeros(n)
    if style == 'smooth':
        alphas[:] = 0.5
    elif style == 'mixed':
        for i in rng.sample(range(n), rng.randint(1, n - 1)):
            alphas[i] = _draw(rng, corner_rounding)
    return shapes.bezier(control, alphas, max_error=max_error)


def _containing(rng, inner, min_rim, max_rim, vertices, rounding, corner_rounding, max_error,
                attempts):
    """A random rounded convex outline around `inner`: every point of `inner`
    at least `min_rim` inside it, and none of its vertices farther than
    `max_rim` from `inner`'s hull. Falls back to the hull grown by the rim."""
    hull = pg.convex_hull(inner)
    span = np.ptp(hull, axis=0) + (min_rim + max_rim)
    for _ in range(attempts):
        template = _rounded_convex(rng, vertices, *span, rounding, corner_rounding, max_error)
        outline = _grow_to_contain(template, hull, min_rim)
        if _farthest(outline, hull) <= max_rim:
            return outline
    return pg.offset_outward(hull, min_rim * (1 + 1e-9))


def _grow_to_contain(template: np.ndarray, inner: np.ndarray, margin: float) -> np.ndarray:
    """Scale a convex outline about the origin (inside it) so that every
    point of `inner` is at least `margin` inside it, touching at the nearest."""
    edges = np.roll(template, -1, axis=0) - template
    inward = np.column_stack([-edges[:, 1], edges[:, 0]]) / np.linalg.norm(edges, axis=1)[:, None]
    reach = -np.einsum('ij,ij->i', inward, template)          # origin-to-edge distances
    if np.any(reach <= 0):
        raise ValueError("The outline to grow must contain the origin.")
    need = (margin - inward @ inner.T) / reach[:, None]        # scale each edge needs per point
    return template * float(need.max()) * (1 + 1e-12)


def _farthest(points: np.ndarray, hull: np.ndarray) -> float:
    """Largest distance from `points` (outside or on it) to a convex polygon."""
    a, e = hull, np.roll(hull, -1, axis=0) - hull
    t = np.clip(np.einsum('pij,ij->pi', points[:, None] - a[None], e) / (e * e).sum(1), 0, 1)
    nearest = a[None] + t[..., None] * e[None]
    return float(np.linalg.norm(points[:, None] - nearest, axis=2).min(axis=1).max())


def _centroid(poly: np.ndarray) -> np.ndarray:
    following = np.roll(poly, -1, axis=0)
    cross = poly[:, 0] * following[:, 1] - following[:, 0] * poly[:, 1]
    return ((poly + following) * cross[:, None]).sum(axis=0) / (3.0 * cross.sum())


def _plain(poly: np.ndarray) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in poly]
