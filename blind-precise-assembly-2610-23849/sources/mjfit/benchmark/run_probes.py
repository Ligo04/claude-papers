"""Run the passive-probe sweep on CPU MuJoCo; rows go to results/peg/probes_cpu/.

    python -m benchmark.run_probes                       # full sweep, configs/probes.yaml
    python -m benchmark.run_probes --shapes circle --methods ours sdf --probes lateral
    python -m benchmark.run_probes --config benchmark/configs/circle_cpu.yaml \
        --probes geometry memory lateral tilt            # extended circle sweep

Decompositions are built once (cached in benchmark/cache), then every
(representation, MULTICCD) job runs all selected probes in its own process.
Rows are merged into the stored tables (see tables.py): a rerun replaces only
the measurements it repeats.
"""
from __future__ import annotations

import os

# One BLAS thread per worker; the pool provides the parallelism.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from . import probes, tables
from .geometry import Task
from .representations import build_plate
from .scene import build_scene
from .wall import wall_offset

HERE = Path(__file__).resolve().parent
PROBES = ('geometry', 'memory', 'static', 'drop', 'lateral', 'tilt', 'yaw')


def rep_configs(cfg: dict, shape: str, methods) -> list[tuple[str, dict]]:
    out = []
    for method, grid in cfg['methods'].items():
        if methods and method not in methods:
            continue
        if isinstance(grid, dict):
            grid = grid[shape]
        out += [(method, dict(p)) for p in grid]
    return out


def run_job(task: Task, method: str, params: dict, multiccd: bool,
            pcfg: dict, which: list[str]) -> dict[str, list[dict]]:
    rep = build_plate(task, method, params)
    out = {p: [] for p in which}
    um = 1e-6
    if 'geometry' in which and multiccd:
        # Solver-free wall position error: independent of MULTICCD, so it runs
        # once, in the MULTICCD-on job every representation has.
        out['geometry'].append({'shape': task.shape, 'method': method, 'label': rep.label,
                                'multiccd': True, **wall_offset(task, rep)})
    if 'memory' in which and multiccd:
        # Compiled model size: the memory the collision representation costs
        # (mesh hulls and their graphs, or the SDF octree).
        sc = build_scene(task, rep, 0.0, joints='xy', gravity=False,
                         peg_pos=(0, 0, 10 * task.peg_length))
        m = sc.model
        out['memory'].append({'shape': task.shape, 'method': method, 'label': rep.label,
                              'multiccd': True, 'model_MB': m.nbuffer / 2**20,
                              'noct': m.noct, 'nmeshvert': m.nmeshvert,
                              'n_plate_geoms': sc.n_plate_geoms, 'compile_s': sc.compile_s})
    if 'static' in which:
        out['static'] += probes.probe_static(task, rep, pcfg['static']['clearance_um'] * um,
                                             n_dir=pcfg['static']['n_dir'], multiccd=multiccd)
    if 'drop' in which:
        for c in pcfg['drop']['clearance_um']:
            for friction in pcfg['drop']['friction']:
                out['drop'].append(probes.probe_drop(task, rep, c * um, friction=friction,
                                                     multiccd=multiccd))
    for c in pcfg['lateral']['clearance_um'] if 'lateral' in which else []:
        out['lateral'] += probes.probe_lateral(task, rep, c * um, force=pcfg['lateral']['force_N'],
                                               n_dir=pcfg['lateral']['n_dir'], nthread=1,
                                               multiccd=multiccd)
    for c in pcfg['tilt']['clearance_um'] if 'tilt' in which else []:
        for phi in pcfg['tilt']['phi_deg'][task.shape]:
            out['tilt'].append(probes.probe_tilt(task, rep, c * um, phi=math.radians(phi),
                                                 multiccd=multiccd))
    if 'yaw' in which and task.shape == 'square':
        for c in pcfg['yaw']['clearance_um']:
            out['yaw'].append(probes.probe_yaw(task, rep, c * um, multiccd=multiccd))
    for rows in out.values():
        for r in rows:
            r.update(setup_s=rep.setup_s, n_hulls=len(rep.hulls) or 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, default=HERE / 'configs' / 'probes.yaml')
    ap.add_argument('--results', type=Path, default=tables.ROOT,
                    help='raw results root (default: benchmark/results)')
    ap.add_argument('--shapes', nargs='*')
    ap.add_argument('--methods', nargs='*')
    ap.add_argument('--probes', nargs='*', choices=PROBES, default=list(PROBES))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    shapes = args.shapes or cfg['shapes']
    reps, jobs = [], []
    for shape in shapes:
        task = Task(shape, **cfg['task'])
        for method, params in rep_configs(cfg, shape, args.methods):
            reps.append((task, method, params))
            for multiccd in cfg['multiccd'] if method != 'sdf' else [True]:
                jobs.append((task, method, params, multiccd))
    print(f'{len(reps)} plates, {len(jobs)} jobs on {args.workers} workers', flush=True)

    done: dict[int, dict[str, list[dict]]] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        # Build and cache every decomposition first, in parallel (CoACD at fine
        # thresholds takes minutes each), so probe jobs only load the cache.
        list(pool.map(build_plate, *zip(*reps)))
        print(f'plates ready after {time.perf_counter() - t0:.0f}s', flush=True)
        futures = {pool.submit(run_job, *job, cfg['probes'], args.probes): j
                   for j, job in enumerate(jobs)}
        for i, fut in enumerate(as_completed(futures), 1):
            task, method, params, multiccd = jobs[futures[fut]]
            done[futures[fut]] = fut.result()
            print(f'[{i}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] {task.shape} {method} '
                  f'{params} multiccd={multiccd}', flush=True)

    out: dict[str, list[dict]] = {p: [] for p in args.probes}
    for j in sorted(done):  # in job order, whatever order the jobs finished in
        for probe, rows in done[j].items():
            out[probe] += rows
    for probe, rows in out.items():
        if rows:
            print(f'wrote {tables.write(f"peg/probes_cpu/{probe}", rows, args.results)}')


if __name__ == '__main__':
    main()
