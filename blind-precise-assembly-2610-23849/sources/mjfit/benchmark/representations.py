"""Plate collision representations under test.

Each builder turns a Task into a PlateRep: convex vertex sets (MuJoCo hulls
each mesh geom) or one SDF mesh, in the plate body frame (origin at the plate
center, so the plate spans -T/2 <= z <= T/2). The plate depends only on the
hole, never on the clearance, so one decomposition serves a whole clearance
sweep. Decompositions are cached on disk because CoACD can take minutes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Task, plate_mesh

import mjfit
from mjfit import shapes

CACHE_DIR = Path(__file__).resolve().parent / 'cache'
CACHE_VERSION = 1
METHODS = ('ours', 'coacd', 'vhacd', 'sdf')


@dataclass
class PlateRep:
    method: str
    params: dict
    hulls: list[np.ndarray] = field(default_factory=list)
    sdf_mesh: tuple[np.ndarray, np.ndarray] | None = None
    setup_s: float = 0.0   # decomposition time (SDF octree time is in compile)

    @property
    def label(self) -> str:
        if not self.params:
            return self.method
        return self.method + '(' + ','.join(f'{k}={v}' for k, v in self.params.items()) + ')'


def cache_path(task: Task, method: str, params: dict | None = None) -> Path:
    key = json.dumps([CACHE_VERSION, task.key(), method, dict(params or {})], sort_keys=True)
    return CACHE_DIR / f"{method}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"


def build_plate(task: Task, method: str, params: dict | None = None,
                use_cache: bool = True) -> PlateRep:
    """The plate for one setting. use_cache: load the cached decomposition,
    or build it and store it; False builds afresh and leaves the cache alone."""
    params = dict(params or {})
    if method == 'sdf':
        return PlateRep(method, params, sdf_mesh=plate_mesh(task))
    builders = {'ours': _ours, 'coacd': _coacd, 'vhacd': _vhacd}
    if method not in builders:
        raise ValueError(f"Unknown method {method!r}; known: {METHODS}")
    path = cache_path(task, method, params)
    if use_cache and path.exists():
        z = np.load(path)
        hulls = [z[f'hull_{i}'] for i in range(int(z['n']))]
        return PlateRep(method, params, hulls=hulls, setup_s=float(z['setup_s']))
    t0 = time.perf_counter()
    hulls = builders[method](task, **params)
    rep = PlateRep(method, params, hulls=hulls, setup_s=time.perf_counter() - t0)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Atomic: parallel workers may load this path while it is being written.
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
        np.savez(tmp, n=len(hulls), setup_s=rep.setup_s,
                 **{f'hull_{i}': h for i, h in enumerate(hulls)})
        tmp.replace(path)
    return rep


def circle_sides(radius: float, chord_error: float) -> int:
    """Sides of the circumscribed n-gon whose corners stick out <= chord_error."""
    return math.ceil(math.pi / math.acos(radius / (radius + chord_error)))


def _ours(task: Task, chord_error_um: float | None = None) -> list[np.ndarray]:
    """mjfit's own pipeline: mjfit.peg, then read back the plate prisms."""
    if task.shape == 'circle':
        hole = shapes.circle(task.size, n=circle_sides(task.size, chord_error_um * 1e-6))
    else:
        a = task.size
        hole = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
    xml = mjfit.peg(
        hole=hole, clearance=1e-5, plate_size=2 * task.plate_half,
        plate_thickness=task.thickness, peg_length=task.peg_length,
        include_peg=False, floor=False, center=False)
    meshes = ET.fromstring(xml).findall('./asset/mesh')
    return [np.fromstring(m.attrib['vertex'], sep=' ').reshape(-1, 3)
            for m in meshes if m.attrib['name'].startswith('plate_')]


def _solid_hulls(parts) -> list[np.ndarray]:
    """Drop flat slivers (MuJoCo cannot hull coplanar points); they enclose no
    material, so removing them does not change the represented solid."""
    from scipy.spatial import ConvexHull, QhullError
    out = []
    for pv, _ in parts:
        pv = np.asarray(pv, dtype=np.float64)
        try:
            if ConvexHull(pv).volume > 1e-15:
                out.append(pv)
        except QhullError:
            pass
    return out


def _coacd(task: Task, threshold: float = 0.05, **kw) -> list[np.ndarray]:
    import coacd
    coacd.set_log_level('error')
    v, f = plate_mesh(task)
    return _solid_hulls(coacd.run_coacd(coacd.Mesh(v, f), threshold=threshold, seed=0, **kw))


def _vhacd(task: Task, max_hulls: int = 64, resolution: int = 400_000,
           max_verts: int = 64, min_error: float = 1.0, max_depth: int = 10) -> list[np.ndarray]:
    """V-HACD knobs: voxel resolution, hull budget, vertices per hull, the
    volume error (%) below which a part stops splitting, and recursion depth
    (at most 2^max_depth parts)."""
    import vhacdx
    v, f = plate_mesh(task)
    # vhacdx takes VTK-style faces: [3, a, b, c, 3, ...], flattened.
    vtk_faces = np.column_stack([np.full(len(f), 3), f]).ravel().astype(np.uint32)
    return _solid_hulls(vhacdx.compute_vhacd(
        v, vtk_faces, maxConvexHulls=max_hulls, resolution=resolution,
        maxNumVerticesPerCH=max_verts, minimumVolumePercentErrorAllowed=min_error,
        maxRecursionDepth=max_depth))
