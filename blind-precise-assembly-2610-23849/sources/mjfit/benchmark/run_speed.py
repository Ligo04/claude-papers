"""Speed benchmark: CPU MuJoCo (1 env, 1 thread) and MuJoCo Warp (batched worlds).

    python -m benchmark.run_speed --parts cpu_single     # CPU, every rep and scenario
    python -m benchmark.run_speed --parts warp --scenarios wall --nworlds 256 512

Rows go to results/peg/speed_cpu and results/peg/speed_gpu (one row per
batch size), merged into the stored tables as each configuration finishes.

Scenarios are passive, contact-rich states with gravity and friction on and a
free peg (clearance +100 um unless noted). Every representation simulates the
same scene; only the plate's collision geometry differs.

  wall : peg inside the hole, pushed sideways by 0.1 N in a random direction
  rim  : peg 200 um wider than the hole, resting on the rim
  drop : aligned peg dropped from 1 mm (free fall, entry, floor landing)

Run on an otherwise idle machine; the numbers are wall-clock timings.
"""
from __future__ import annotations

import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import math
import platform
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml
from mujoco import rollout

from . import tables
from .geometry import Task
from .probes import FULL, XFRC
from .representations import build_plate
from .run_probes import rep_configs
from .scene import build_scene

HERE = Path(__file__).resolve().parent
SCENARIOS = ('wall', 'rim', 'drop')
def scenario(task: Task, rep, name: str, multiccd: bool):
    """Scene, initial qpos, and a per-world applied-wrench generator."""
    top, half_len = task.thickness, task.peg_length / 2
    clearance, z, push = {'wall': (1e-4, top / 2, 0.1),
                          'rim': (-2e-4, top + half_len + 5e-5, 0.0),
                          'drop': (1e-4, top + half_len + 1e-3, 0.0)}[name]
    sc = build_scene(task, rep, clearance, joints='free', gravity=True, friction=True,
                     peg_pos=(0, 0, z), multiccd=multiccd)

    def wrenches(n: int, rng: np.random.Generator) -> np.ndarray:
        a = rng.uniform(0, 2 * math.pi, n)
        w = np.zeros((n, 6))
        w[:, 0], w[:, 1] = push * np.cos(a), push * np.sin(a)
        return w
    return sc, sc.model.qpos0.copy(), wrenches


def _initial_state(m, qpos0):
    d = mujoco.MjData(m)
    d.qpos[:] = qpos0
    mujoco.mj_forward(m, d)
    s = np.empty(mujoco.mj_stateSize(m, FULL))
    mujoco.mj_getState(m, d, s, FULL)
    return s


def _control(sc, wrench: np.ndarray, nstep: int) -> np.ndarray:
    c = np.zeros((len(wrench), nstep, 6 * sc.model.nbody))
    c[:, :, 6 * sc.peg_body:6 * sc.peg_body + 6] = wrench[:, None]
    return c


def cpu_single(sc, qpos0, wrench: np.ndarray, nstep: int = 1000, repeats: int = 5) -> dict:
    """Best-of-N wall time of an nstep C rollout, plus contact statistics and
    the share of the step spent in mj_collision (replayed on sampled states)."""
    m = sc.model
    s0, ctrl = _initial_state(m, qpos0)[None], _control(sc, wrench[None], nstep)
    d = mujoco.MjData(m)
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        states, _ = rollout.rollout(m, d, s0, ctrl, control_spec=XFRC, nstep=nstep)
        best = min(best, time.perf_counter() - t0)
    ncon, nefc, coll = [], [], []
    for i, s in enumerate(states[0]):
        mujoco.mj_setState(m, d, s, FULL)
        mujoco.mj_forward(m, d)
        ncon.append(d.ncon)
        nefc.append(d.nefc)
        if i % 20 == 0:
            t0 = time.perf_counter()
            for _ in range(20):
                mujoco.mj_collision(m, d)
            coll.append((time.perf_counter() - t0) / 20)
    return {'us_per_step': best / nstep * 1e6, 'collision_us': float(np.mean(coll)) * 1e6,
            'ncon_mean': float(np.mean(ncon)), 'ncon_max': int(np.max(ncon)),
            'nefc_mean': float(np.mean(nefc)), 'nefc_max': int(np.max(nefc))}


def warp_scaling(sc, qpos0, wrench_fn, nworlds, nstep: int = 100, launches: int = 5) -> list[dict]:
    import warp as wp
    from .warp_backend import Batch, _flags
    rng = np.random.default_rng(0)
    # Calibrate per-world buffers on a small batch with generous capacity.
    t0 = time.perf_counter()
    ncal = 64
    cap = max(512, 2 * sc.n_plate_geoms)
    cal = Batch(sc, np.tile(qpos0, (ncal, 1)), wrench_fn(ncal, rng), nconmax=cap, njmax=6 * cap)
    peak_con, peak_efc, peak_pairs = 0, 0, 0
    for _ in range(4):
        cal.advance(100)
        peak_con = max(peak_con, int(cal.contacts_per_world().max()))
        peak_efc = max(peak_efc, int(cal.dw.nefc.numpy().max()))
        peak_pairs = max(peak_pairs, math.ceil(int(cal.dw.ncollision.numpy()[0]) / ncal))
    warmup_s = time.perf_counter() - t0
    # Warp's broadphase stores candidate pairs in the contact buffer too.
    nconmax = max(16, math.ceil(1.5 * peak_con), math.ceil(1.25 * peak_pairs) + 4)
    njmax = max(64, math.ceil(1.5 * peak_efc))
    cal.release()
    rows = []
    for n in nworlds:
        # Large batches reach rarer contact spikes than the calibration batch;
        # on overflow, double the buffers and time again rather than report
        # physics with dropped contacts.
        for retry in range(4):
            try:
                b = Batch(sc, np.tile(qpos0, (n, 1)), wrench_fn(n, rng), nconmax=nconmax,
                          njmax=njmax)
                b.advance(nstep)  # capture + first launch
                wp.synchronize()
                mem = wp.get_mempool_used_mem_current('cuda:0')
                t0 = time.perf_counter()
                for _ in range(launches):
                    b.advance(nstep)
                wp.synchronize()
                dt = time.perf_counter() - t0
            except Exception as e:  # out of device memory: larger batches fail too
                rows.append({'nworld': n, 'error': f'{type(e).__name__}: {str(e)[:120]}'})
                return rows
            flags = _flags(b.overflow())
            if not flags['warp_invalid'] or retry == 3:  # last try is kept, flagged
                break
            b.release()
            nconmax, njmax = 2 * nconmax, 2 * njmax
        rows.append({'nworld': n, 'env_steps_per_s': n * nstep * launches / dt,
                     'realtime_factor': n * nstep * launches * sc.model.opt.timestep / dt,
                     'ms_per_step': dt / (nstep * launches) * 1e3, 'gpu_mem_MB': mem / 2**20,
                     'nconmax': nconmax, 'njmax': njmax, 'calib_peak_contacts': peak_con,
                     'calib_peak_pairs': peak_pairs,
                     'buffer_retries': retry, 'warmup_s': warmup_s, **flags})
        b.release()
    return rows


def _warp_job(task: Task, method: str, params: dict, multiccd: bool, name: str,
              nworlds: list[int]) -> list[dict]:
    """Warp timing for one configuration, in its own process: captured CUDA
    graphs and Warp's memory pool keep device memory after a batch is freed."""
    import warp as wp
    rep = build_plate(task, method, params)
    sc, qpos0, wrench_fn = scenario(task, rep, name, multiccd)
    gpu = wp.get_device('cuda:0').name
    return [{'gpu': gpu, **r} for r in warp_scaling(sc, qpos0, wrench_fn, nworlds)]


def _fresh_process():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn'),
                               max_tasks_per_child=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, default=HERE / 'configs' / 'probes.yaml')
    ap.add_argument('--results', type=Path, default=tables.ROOT,
                    help='raw results root (default: benchmark/results)')
    ap.add_argument('--parts', nargs='*', default=['cpu_single', 'warp'],
                    choices=['cpu_single', 'warp'])
    ap.add_argument('--shapes', nargs='*')
    ap.add_argument('--methods', nargs='*')
    ap.add_argument('--scenarios', nargs='*', default=list(SCENARIOS), choices=SCENARIOS)
    ap.add_argument('--nworlds', type=int, nargs='*', default=[256])
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    machine = {'cpu': _cpu_name()}
    names = {'cpu_single': 'peg/speed_cpu', 'warp': 'peg/speed_gpu'}
    for shape in args.shapes or cfg['shapes']:
        task = Task(shape, **cfg['task'])
        for method, params in rep_configs(cfg, shape, args.methods):
            rep = build_plate(task, method, params)
            for multiccd in [True, False] if method != 'sdf' else [True]:
                for name in args.scenarios:
                    sc, qpos0, wrench_fn = scenario(task, rep, name, multiccd)
                    base = {'shape': shape, 'method': method, 'label': rep.label,
                            'multiccd': multiccd, 'scenario': name,
                            'n_plate_geoms': sc.n_plate_geoms, **machine}
                    rows = {}
                    if 'cpu_single' in args.parts:
                        w = wrench_fn(1, np.random.default_rng(0))[0]
                        rows['cpu_single'] = [{**base, **cpu_single(sc, qpos0, w)}]
                    if 'warp' in args.parts:
                        with _fresh_process() as pool:
                            res = pool.submit(_warp_job, task, method, params, multiccd,
                                              name, args.nworlds).result()
                        rows['warp'] = [{**base, **r} for r in res]
                    print(f'{shape} {rep.label} multiccd={multiccd} {name}', flush=True)
                    for part, r in rows.items():  # checkpoint as we go
                        tables.write(names[part], r, args.results)
    print(f'wrote {", ".join(str(tables.path(names[p], args.results)) for p in args.parts)}')


def _cpu_name() -> str:
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


if __name__ == '__main__':
    main()
