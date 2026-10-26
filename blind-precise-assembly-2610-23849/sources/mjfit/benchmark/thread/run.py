"""Run the bolt/nut sweep: probes (accuracy) or speed (CPU time per step).

    python -m benchmark.thread.run --parts probes           # all probes, all settings
    python -m benchmark.thread.run --parts speed             # timing: idle machine
    python -m benchmark.thread.run --methods ours --clearance 100 --probes axial screw

Every (representation, clearance) is built in its own process (cached on
disk) and its probe jobs, one per MULTICCD setting, start as soon as it is
ready, so slow decompositions do not hold up the rest. Rows go to
results/thread/probes_cpu/<probe> and results/thread/speed_cpu, merged into
the stored tables after every job.
"""
from __future__ import annotations

import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import math
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .. import tables
from . import probes as T
from .geometry import ThreadTask
from .representations import build_thread, cache_path, exact_meshes
from .scene import build_scene

HERE = Path(__file__).resolve().parents[1]
PROBES = ('memory', 'axial', 'radial', 'tilt', 'screw', 'spin', 'start', 'partial', 'torque')
TILT_PHI = (0.0, math.pi / 2)
# T6 torque ratios: T4's, and a lower one that the flanks' facets can stop
START_RATIOS = (T.START_TORQUE_RATIO, 2.0)


def label(method: str, params: dict) -> str:
    """ThreadRep.label, without building the representation."""
    if not params:
        return method
    return method + '(' + ','.join(f'{k}={v}' for k, v in params.items()) + ')'


def rep_configs(cfg: dict, methods) -> list[tuple[str, dict]]:
    return [(m, dict(p)) for m, grid in cfg['methods'].items()
            if not methods or m in methods for p in grid]


def run_probes(task, method, params, c, multiccd, which) -> dict[str, list[dict]]:
    rep = build_thread(task, method, params, c)
    out = {p: [] for p in which}
    if 'memory' in which and multiccd:
        out['memory'].append(T.memory(task, rep))
    if 'axial' in which:
        out['axial'] += T.probe_axial(task, rep, multiccd=multiccd)
    if 'radial' in which:
        out['radial'] += T.probe_radial(task, rep, multiccd=multiccd)
    if 'tilt' in which:
        out['tilt'] += [T.probe_tilt(task, rep, phi, multiccd=multiccd) for phi in TILT_PHI]
    if 'screw' in which:
        out['screw'].append(T.probe_screw(task, rep, multiccd=multiccd))
    if 'spin' in which:
        out['spin'] += [T.probe_spin(task, rep, mu, multiccd=multiccd)
                        for mu in (0.0, T.SPIN_FRICTION)]
    if 'start' in which:
        out['start'] += [T.probe_start(task, rep, direction, multiccd=multiccd, torque_ratio=k)
                         for direction in ('back', 'on') for k in START_RATIOS]
    if 'partial' in which:
        out['partial'] += T.probe_partial(task, rep, multiccd=multiccd)
    if 'torque' in which:
        out['torque'] += [T.probe_torque(task, rep, mu, direction, load, condim, cone, dt,
                                         multiccd=multiccd)
                          for mu in T.TORQUE_MU for direction in ('up', 'down')
                          for load, condim, cone, dt in T.TORQUE_CASES]
    for rows in out.values():
        for r in rows:
            r.update(setup_s=rep.setup_s, n_bolt=rep.bolt.n_geoms, n_nut=rep.nut.n_geoms)
    return out


def _partial_one(task, c, job):
    return T._partial_key(task, c, *job), T.compute_partial(task, c, *job)


def exact_partial_all(task, clearances, workers):
    """Fill the on-disk cache of exact T7 answers, in parallel."""
    import json
    from ..representations import CACHE_DIR
    path = CACHE_DIR / 'thread_partial.json'
    have = json.loads(path.read_text()) if path.exists() else {}
    todo = [(c, job) for c in clearances for job in T.partial_jobs(task)
            if T._partial_key(task, c, *job) not in have]
    if not todo:
        return
    with ProcessPoolExecutor(workers) as pool:
        done = dict(pool.map(_partial_one, *zip(*[(task, c, job) for c, job in todo])))
    T.exact_partial(task, clearances[0], *T.partial_jobs(task)[0], store=done)


def run_speed(task, method, params, c, multiccd, warmup=300, nstep=500, reps=3) -> dict:
    """CPU time per step (1 thread) while the nut is screwed up the thread
    under load (T4's steady state: every flank of the nut in contact)."""
    rep = build_thread(task, method, params, c)
    sc = build_scene(task, rep, joints='screw', multiccd=multiccd)
    m, d = sc.model, mujoco.MjData(sc.model)
    d.xfrc_applied[sc.nut_body] = [0, 0, -T.SCREW_LOAD, 0, 0, T.SCREW_TORQUE]
    for _ in range(warmup):
        mujoco.mj_step(m, d)
    best, ncon = math.inf, []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(nstep):
            mujoco.mj_step(m, d)
            ncon.append(d.ncon)
        best = min(best, (time.perf_counter() - t0) / nstep)
    d2 = mujoco.MjData(m)
    mujoco.mj_forward(m, d2)
    t0 = time.perf_counter()
    for _ in range(50):
        mujoco.mj_collision(m, d2)
    return {'shape': task.shape, 'method': method, 'label': rep.label,
            'clearance_um': c * 1e6, 'multiccd': multiccd, 'scenario': 'screw',
            'us_per_step': best * 1e6, 'ncon': float(np.mean(ncon)), 'nefc_last': int(d.nefc),
            'collision_us_at_rest': (time.perf_counter() - t0) / 50 * 1e6,
            'n_geoms': sc.n_geoms}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, default=HERE / 'configs' / 'thread.yaml')
    ap.add_argument('--results', type=Path, default=tables.ROOT,
                    help='raw results root (default: benchmark/results)')
    ap.add_argument('--parts', nargs='*', choices=['probes', 'speed'], default=['probes'])
    ap.add_argument('--methods', nargs='*')
    ap.add_argument('--clearance', nargs='*', type=float, help='µm; default from config')
    ap.add_argument('--probes', nargs='*', choices=PROBES, default=list(PROBES))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--resume', action='store_true',
                    help='run only the jobs missing from the stored tables')
    ap.add_argument('--only-cached', action='store_true',
                    help='skip CoACD / V-HACD settings whose decomposition is not cached yet')
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    task = ThreadTask(**(cfg.get('task') or {}))
    clearances = [c * 1e-6 for c in (args.clearance or cfg['clearance_um'])]
    reps = [(task, m, p, c) for c in clearances for m, p in rep_configs(cfg, args.methods)]
    if args.only_cached:
        reps = [r for r in reps
                if r[1] in ('ours', 'sdf', 'sdf_analytic') or cache_path(*r).exists()]
    jobs = [(task, m, p, c, mccd) for (task, m, p, c) in reps
            for mccd in (cfg['multiccd'] if not m.startswith('sdf') else [True])]
    if args.resume:
        name = (f'thread/probes_cpu/{next(p for p in args.probes if p != "memory")}'
                if 'probes' in args.parts else 'thread/speed_cpu')
        have = tables.read(name, args.results)
        done = set() if have is None else {
            (r.label, round(r.clearance_um, 6), bool(r.multiccd)) for r in have.itertuples()}
        jobs = [j for j in jobs if (label(j[1], j[2]), round(j[3] * 1e6, 6), j[4]) not in done]
        reps = [r for r in reps if any(j[1:4] == r[1:] for j in jobs)]
    print(f'{len(reps)} representations, {len(jobs)} jobs on {args.workers} workers', flush=True)

    t0 = time.perf_counter()
    if 'probes' in args.parts and 'tilt' in args.probes:
        for c in clearances:
            for phi in TILT_PHI:
                T.exact_tilt(task, c, phi)
        print(f'exact tilt limits ready after {time.perf_counter() - t0:.0f}s', flush=True)
    if 'probes' in args.parts and 'partial' in args.probes:
        exact_partial_all(task, clearances, args.workers)
        print(f'exact partial-engagement limits ready after {time.perf_counter() - t0:.0f}s',
              flush=True)
    for c in clearances:  # once, in this process (see exact_meshes)
        exact_meshes(task, c)
    if any(r[1] == 'sdf_analytic' for r in reps):
        from . import plugin
        plugin.library()  # compile before the workers need it
    def record(part, job, fut, i, n):
        task_, m, p, c, mccd = job
        try:
            res = fut.result()
        except Exception as e:  # keep the sweep going; record the failure
            print(f'FAILED {part} {m} {p} c={c * 1e6:g} mccd={mccd}: {e!r}', flush=True)
            return
        # Stored after every job, so a crash late in the sweep loses nothing.
        if part == 'probes':
            for probe, rows in res.items():
                if rows:
                    tables.write(f'thread/probes_cpu/{probe}', rows, args.results)
        else:
            tables.write('thread/speed_cpu', [res], args.results)
        print(f'[{part} {i}/{n} {time.perf_counter() - t0:6.0f}s] {m} {p} '
              f'c={c * 1e6:g} multiccd={mccd}', flush=True)

    if 'probes' in args.parts:
        with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
            builds = {pool.submit(build_thread, *r): r for r in reps}
            probes, pending, done_n = {}, set(builds), 0
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    if fut in builds:
                        task_, m, p, c = builds[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            print(f'FAILED build {m} {p} c={c * 1e6:g}: {e!r}', flush=True)
                            continue
                        for job in [j for j in jobs if j[:4] == (task_, m, p, c)]:
                            pf = pool.submit(run_probes, *job, args.probes)
                            probes[pf] = job
                            pending.add(pf)
                    else:
                        done_n += 1
                        record('probes', probes[fut], fut, done_n, len(jobs))
    if 'speed' in args.parts:  # one process at a time: timings need an idle machine
        with ProcessPoolExecutor(1, max_tasks_per_child=1) as pool:
            futures = {pool.submit(run_speed, *job): job for job in jobs
                       if job[1] in ('ours', 'sdf', 'sdf_analytic')
                       or cache_path(*job[:4]).exists()}
            for i, fut in enumerate(as_completed(futures), 1):
                record('speed', futures[fut], fut, i, len(futures))

    print(f'wrote {tables.path("thread/*", args.results).parent}')


if __name__ == '__main__':
    main()
