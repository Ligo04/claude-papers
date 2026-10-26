"""Bolt and nut collision representations under test.

Unlike the peg's plate, both parts are concave here (the bolt's thread is a
ridge wound around a cylinder), so every method represents both. Each part is
a set of convex vertex sets and primitive cylinders, or one SDF mesh, in the
part's own frame (bolt threads on 0 <= z <= L, nut threads on 0 <= z <= Hn).
The parts depend on the clearance (both teeth shrink by c/2), so one
decomposition serves one clearance. Decompositions are cached on disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from ..representations import CACHE_DIR, _solid_hulls
from . import geometry as geo
import mjfit

from .geometry import ThreadTask

CACHE_VERSION = 1
# Part of the 'ours' cache key: bumped whenever the wedge generator's output
# changes, so that only 'ours' is rebuilt.
OURS_VERSION = 3
# sdf_analytic: the exact ISO parts as an analytic SDF plugin (plugin/), so its
# only error is MuJoCo's SDF contact search; its knob is that search's effort.
METHODS = ('ours', 'coacd', 'vhacd', 'sdf', 'sdf_analytic')


@dataclass
class PartRep:
    hulls: list[np.ndarray] = field(default_factory=list)
    cylinders: list[tuple[float, float, float]] = field(default_factory=list)  # (r, z0, z1)
    sdf_mesh: tuple[np.ndarray, np.ndarray] | None = None
    analytic: dict | None = None   # iso.sdf.thread plugin config

    @property
    def n_geoms(self) -> int:
        return (len(self.hulls) + len(self.cylinders) + (self.sdf_mesh is not None)
                + (self.analytic is not None))


@dataclass
class ThreadRep:
    method: str
    params: dict
    clearance: float
    bolt: PartRep
    nut: PartRep
    setup_s: float = 0.0

    @property
    def label(self) -> str:
        if not self.params:
            return self.method
        return self.method + '(' + ','.join(f'{k}={v}' for k, v in self.params.items()) + ')'


def exact_meshes(task: ThreadTask, clearance: float):
    """(bolt, nut) watertight meshes, built once and cached: manifold3d's
    booleans are not bit-reproducible (and have crashed when many processes
    ran them at once), so every method must start from the same file."""
    key = json.dumps([CACHE_VERSION, task.key(), clearance], sort_keys=True)
    path = CACHE_DIR / f"thread_mesh_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"
    if not path.exists():
        (bv, bf), (nv, nf) = geo.bolt_mesh(task, clearance), geo.nut_mesh(task, clearance)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + '.tmp.npz')
        np.savez(tmp, bv=bv, bf=bf, nv=nv, nf=nf)
        tmp.replace(path)
    z = np.load(path)
    return (z['bv'], z['bf']), (z['nv'], z['nf'])


def cache_path(task: ThreadTask, method: str, params: dict | None, clearance: float):
    key = [CACHE_VERSION, task.key(), method, dict(params or {}), clearance]
    if method == 'ours' and OURS_VERSION > 1:
        key.append(OURS_VERSION)
    key = json.dumps(key, sort_keys=True)
    return CACHE_DIR / f"thread_{method}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"


def build_thread(task: ThreadTask, method: str, params: dict | None, clearance: float,
                 use_cache: bool = True) -> ThreadRep:
    """Both parts for one setting. use_cache: load the cached decomposition,
    or build it and store it; False builds afresh and leaves the cache alone
    (the exact meshes the baselines start from are still read from, or added
    to, the cache: see exact_meshes)."""
    params = dict(params or {})
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; known: {METHODS}")
    if method == 'sdf_analytic':
        common = {'diameter': task.major_diameter, 'pitch': task.pitch, 'clearance': clearance,
                  'outer': task.nut_outer_radius}
        bolt, nut = {'part': 0, 'length': task.bolt_length}, {'part': 1, 'length': task.nut_height}
        if task.form != 'iso' or task.form_params or task.starts != 1 or task.hand != 1 \
                or task.chamfer:
            # Other threads pass their tooth and helix (the plugin's
            # single-start right-handed ISO tooth is built in).
            for cfg, tooth, base in ((bolt, geo.bolt_tooth(task, clearance), task.r_minor),
                                     (nut, geo.nut_tooth(task, clearance), task.r_root)):
                cfg.update(nvert=len(tooth), base=base, shell=task.r_bore(clearance),
                           lead=task.lead, hand=task.hand)
                if task.chamfer:
                    cfg.update(chamfer_r=base if cfg is bolt else task.r_bore(clearance),
                               chamfer_k=geo._chamfer_slopes(task, clearance)[0 if cfg is bolt else 1])
                for i, (dr, u) in enumerate(tooth):
                    cfg[f'v{i}r'], cfg[f'v{i}u'] = dr, u
        return ThreadRep(method, params, clearance, PartRep(analytic={**bolt, **common}),
                         PartRep(analytic={**nut, **common}))
    path = cache_path(task, method, params, clearance)
    if use_cache and path.exists():
        return _load(path, method, params, clearance)
    t0 = time.perf_counter()
    if method == 'ours':
        bolt, nut = _ours(task, clearance, **params)
    else:
        bolt_mesh, nut_mesh = exact_meshes(task, clearance)
        if method == 'sdf':
            bolt, nut = PartRep(sdf_mesh=bolt_mesh), PartRep(sdf_mesh=nut_mesh)
        else:
            decompose = _coacd if method == 'coacd' else _vhacd
            bolt = PartRep(hulls=decompose(bolt_mesh, **params))
            nut = PartRep(hulls=decompose(nut_mesh, **params))
    rep = ThreadRep(method, params, clearance, bolt, nut, time.perf_counter() - t0)
    if use_cache:
        _save(path, rep)
    return rep


def _save(path, rep: ThreadRep):
    arrays = {'setup_s': rep.setup_s}
    for name, part in (('bolt', rep.bolt), ('nut', rep.nut)):
        arrays[f'{name}_n'] = len(part.hulls)
        arrays.update({f'{name}_hull_{i}': h for i, h in enumerate(part.hulls)})
        arrays[f'{name}_cyl'] = np.array(part.cylinders, dtype=float).reshape(-1, 3)
        if part.sdf_mesh is not None:
            arrays[f'{name}_v'], arrays[f'{name}_f'] = part.sdf_mesh
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic: parallel workers may load this path while it is being written.
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _load(path, method, params, clearance) -> ThreadRep:
    z = np.load(path)
    parts = []
    for name in ('bolt', 'nut'):
        parts.append(PartRep(
            hulls=[z[f'{name}_hull_{i}'] for i in range(int(z[f'{name}_n']))],
            cylinders=[tuple(c) for c in z[f'{name}_cyl']],
            sdf_mesh=(z[f'{name}_v'], z[f'{name}_f']) if f'{name}_v' in z else None))
    return ThreadRep(method, params, clearance, *parts, setup_s=float(z['setup_s']))


def _ours(task: ThreadTask, clearance: float, segments: int = 24, relief_um: float = 0.0):
    """mjfit's own pipeline: mjfit.thread, then read back the parts.
    relief_um is the seam relief (µm)."""
    xml = mjfit.thread(
        major_diameter=task.major_diameter, pitch=task.pitch, thread_length=task.bolt_length,
        form=task.form, form_params=dict(task.params), n_starts=task.starts,
        handedness=task.handedness, chamfer=task.chamfer,
        segments_per_turn=segments, clearance=clearance, seam_relief=relief_um * 1e-6,
        bolt_head_height=0.0,
        nut_height=task.nut_height, nut_size=2 * task.nut_outer_radius,
        nut_shape='cylinder', floor=False)
    root = ET.fromstring(xml)
    meshes = {m.attrib['name']: np.fromstring(m.attrib['vertex'], sep=' ').reshape(-1, 3)
              for m in root.findall('./asset/mesh')}
    bolt = PartRep(hulls=[v for k, v in meshes.items() if k.startswith('bolt_thread')])
    core = root.find(".//geom[@name='bolt_core']")
    r, half = (float(x) for x in core.attrib['size'].split())
    zc = float(core.attrib['pos'].split()[2])
    bolt.cylinders.append((r, zc - half, zc + half))
    nut = PartRep(hulls=[v for k, v in meshes.items() if k.startswith('nut_')])
    return bolt, nut


def _coacd(mesh, threshold: float = 0.05, **kw) -> list[np.ndarray]:
    """CoACD on the exact mesh as given. Its preprocess_mode='auto' aborts
    (std::logic_error) on the threaded bolt, and 'on' re-voxelizes the mesh at
    a resolution far coarser than the thread; the input is already a
    watertight manifold, so preprocessing is off."""
    import coacd
    coacd.set_log_level('error')
    v, f = mesh
    return _solid_hulls(coacd.run_coacd(coacd.Mesh(v, f), threshold=threshold, seed=0,
                                        preprocess_mode='off', **kw))


def _vhacd(mesh, max_hulls: int = 64, resolution: int = 400_000, max_verts: int = 64,
           min_error: float = 1.0, max_depth: int = 10) -> list[np.ndarray]:
    import vhacdx
    v, f = mesh
    vtk_faces = np.column_stack([np.full(len(f), 3), f]).ravel().astype(np.uint32)
    return _solid_hulls(vhacdx.compute_vhacd(
        v, vtk_faces, maxConvexHulls=max_hulls, resolution=resolution,
        maxNumVerticesPerCH=max_verts, minimumVolumePercentErrorAllowed=min_error,
        maxRecursionDepth=max_depth))
