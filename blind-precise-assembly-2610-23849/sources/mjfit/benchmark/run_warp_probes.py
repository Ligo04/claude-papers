"""Accuracy probes on MuJoCo Warp; rows go to results/peg/probes_gpu/.

    python -m benchmark.run_warp_probes          # needs a CUDA GPU
    python -m benchmark.run_warp_probes --shapes circle --probes lateral tilt --clearances 100 20
    python -m benchmark.run_warp_probes --config benchmark/configs/circle_cpu.yaml \
        --probes lateral tilt --clearances 100 20     # extended circle sweep

Runs S2 (lateral play; one world per push direction), S3 (tilt), and S1
(drop outcome) on the same scenes and cached decompositions as run_probes.
Rows are merged into the stored tables as each configuration finishes.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import yaml

from . import tables
from .geometry import Task
from .representations import build_plate
from .run_probes import rep_configs
from .run_speed import _fresh_process
from .warp_backend import warp_drop, warp_lateral, warp_tilt

HERE = Path(__file__).resolve().parent
PROBES = ('lateral', 'tilt', 'drop')


def _job(task: Task, method: str, params: dict, multiccd: bool, pcfg: dict,
         which) -> tuple[str, dict[str, list[dict]]]:
    rep = build_plate(task, method, params)
    out = {p: [] for p in which}
    for c in pcfg['lateral']['clearance_um'] if 'lateral' in which else []:
        out['lateral'] += warp_lateral(task, rep, c * 1e-6, force=pcfg['lateral']['force_N'],
                                       n_dir=pcfg['lateral']['n_dir'], multiccd=multiccd)
    for c in pcfg['tilt']['clearance_um'] if 'tilt' in which else []:
        for phi in pcfg['tilt']['phi_deg'][task.shape]:
            out['tilt'].append(warp_tilt(task, rep, c * 1e-6, phi=math.radians(phi),
                                         multiccd=multiccd))
    for friction in pcfg['drop']['friction'] if 'drop' in which else []:
        out['drop'] += warp_drop(task, rep, pcfg['drop']['clearance_um'], friction=friction,
                                 multiccd=multiccd)
    return rep.label, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, default=HERE / 'configs' / 'probes.yaml')
    ap.add_argument('--results', type=Path, default=tables.ROOT,
                    help='raw results root (default: benchmark/results)')
    ap.add_argument('--shapes', nargs='*')
    ap.add_argument('--methods', nargs='*')
    ap.add_argument('--probes', nargs='*', choices=PROBES, default=list(PROBES))
    ap.add_argument('--clearances', type=float, nargs='*',
                    help='S2/S3 clearances in um (default: from the config)')
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    pcfg = cfg['probes']
    for probe in ('lateral', 'tilt') if args.clearances else ():
        pcfg[probe]['clearance_um'] = args.clearances
    t0 = time.perf_counter()
    for shape in args.shapes or cfg['shapes']:
        task = Task(shape, **cfg['task'])
        for method, params in rep_configs(cfg, shape, args.methods):
            for multiccd in cfg['multiccd'] if method != 'sdf' else [True]:
                # Fresh process per job: device memory held by captured graphs
                # is only reliably returned when the process exits.
                with _fresh_process() as pool:
                    label, out = pool.submit(_job, task, method, params, multiccd,
                                             pcfg, args.probes).result()
                print(f'[{time.perf_counter() - t0:6.0f}s] {shape} {label} '
                      f'multiccd={multiccd}', flush=True)
                for probe, rows in out.items():  # checkpoint as we go
                    if rows:
                        tables.write(f'peg/probes_gpu/{probe}', rows, args.results)
    print(f'wrote {tables.path("peg/probes_gpu/*", args.results)}')


if __name__ == '__main__':
    main()
