"""Rerun the benchmark behind the paper, one stage at a time.

    python -m benchmark.reproduce --stage report                  # tables + figures, minutes
    python -m benchmark.reproduce --stage probes speed gpu thread report
    python -m benchmark.reproduce --stage probes --dry-run        # print the commands only
    python -m benchmark.reproduce --stage probes --results /tmp/rerun --shapes square --methods ours

Each stage runs the commands below, in order, as subprocesses of this Python.
Run stages write into --results (default benchmark/results, the tables the
paper was built from): a rerun measurement replaces the stored row with the
same key and leaves every other row alone, so pass a fresh --results
directory to rerun from scratch, then build the report from it with
`--stage report --results DIR --out DIR/report`. --shapes, --methods and
--clearance narrow every command to a subset, for a quick check; commands
left with nothing to run are skipped.

Run times below are estimates for a 32-thread desktop (the paper's timings
used an Intel i9-13900KS and an NVIDIA RTX 5090), built from the shipped
step costs, build times and a few timed probe jobs, with the decomposition
cache in place (`python -m benchmark.fetch cache`). Without the cache, every
decomposition is rebuilt first (times given).

  probes   CPU accuracy probes, peg: configs/probes.yaml (66 jobs, about
           10 min), then the extended circle sweep configs/circle_cpu.yaml
           (18 jobs, about 15 min). Without the cache add about 17 min (CoACD
           0.01) and 8 min (CoACD 0.005). SDF octree depth 10 needs about
           13 GB of RAM per process and there are two such jobs, so the
           extended sweep peaks near 35 GB; lower --workers to bound it.
  speed    CPU timing, peg: one process, every representation and scenario,
           best of 5 runs of 1000 steps. About 15 min (35 min more without
           the cache: the plates are built one after another). Needs an
           otherwise idle machine.
  gpu      MuJoCo Warp, peg circle: S2/S3 accuracy (main and extended
           sweep), then timings at 256 worlds (main, extended) and 512 worlds
           (main, both shapes). Needs a CUDA GPU. Every one of the ~190
           configurations runs in a fresh process, which dominates: about
           1 h (the first run also compiles Warp's kernels). Device memory
           stays under 2 GB per batch; the SDF depth-10 jobs need the same
           13 GB of host RAM to compile.
  thread   Bolt/nut (configs/thread.yaml): T1-T5 and memory for every
           representation (about 20 min), T6/T7 for ours and both SDFs (about
           20 min), T8 for ours and the analytic SDF at c = 100 um (about
           25 min), then CPU timings in one process (about 20 min). Without
           the cache add about 30 min of decompositions (CoACD 0.02: 20-30 min
           per clearance) and 10-20 min of exact T7 answers. The analytic SDF
           plugin is compiled on first use (Linux, g++; see thread/plugin).
  report   benchmark/report/: the peg summaries and figures (GPU timings at
           --gpu-nworld worlds, 256 by default as in the paper's Pareto
           figure; speed_table.csv lists 256 and 512), and the bolt/nut ones,
           re-simulating the T6 traces for thread_start.png. A few minutes.

Optional, not in the shipped results:

  videos        the behavior videos (MUJOCO_GL=egl; on the order of an hour),
                or fetch the rendered ones: `python -m benchmark.fetch videos`.
  thread-coacd  CoACD at threshold 0.01 on the bolt/nut
                (configs/thread_coacd_fine.yaml): about 3,800 hulls and about
                2 h to decompose per clearance (the release cache holds both),
                then an estimated 20-40 ms per simulated step, so under an hour
                of probes and timing with the cache. In the run-down video it
                jams, like every coarser threshold.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import tables

HERE = Path(__file__).resolve().parent
CFG = HERE / 'configs'
PROBES = CFG / 'probes.yaml'
CIRCLE_EXT = CFG / 'circle_cpu.yaml'
THREAD = CFG / 'thread.yaml'
THREAD_COACD = CFG / 'thread_coacd_fine.yaml'

# stage -> [(module, args)]; every run command also gets --results.
STAGES = {
    'probes': [
        ('benchmark.run_probes', []),
        ('benchmark.run_probes', ['--config', CIRCLE_EXT,
                                  '--probes', 'geometry', 'memory', 'lateral', 'tilt']),
    ],
    'speed': [
        ('benchmark.run_speed', ['--parts', 'cpu_single']),
        ('benchmark.run_speed', ['--config', CIRCLE_EXT, '--parts', 'cpu_single',
                                 '--scenarios', 'wall']),
    ],
    'gpu': [
        ('benchmark.run_warp_probes', ['--shapes', 'circle', '--probes', 'lateral', 'tilt',
                                       '--clearances', '100', '20']),
        ('benchmark.run_warp_probes', ['--config', CIRCLE_EXT, '--probes', 'lateral', 'tilt',
                                       '--clearances', '100', '20']),
        ('benchmark.run_speed', ['--parts', 'warp', '--shapes', 'circle', '--scenarios', 'wall',
                                 '--nworlds', '256']),
        ('benchmark.run_speed', ['--config', CIRCLE_EXT, '--parts', 'warp',
                                 '--scenarios', 'wall', '--nworlds', '256']),
        ('benchmark.run_speed', ['--parts', 'warp', '--scenarios', 'wall', '--nworlds', '512']),
    ],
    'thread': [
        ('benchmark.thread.run', ['--parts', 'probes', '--probes', 'memory', 'axial', 'radial',
                                  'tilt', 'screw', 'spin']),
        ('benchmark.thread.run', ['--parts', 'probes', '--probes', 'start', 'partial',
                                  '--methods', 'ours', 'sdf', 'sdf_analytic']),
        ('benchmark.thread.run', ['--parts', 'probes', '--probes', 'torque',
                                  '--methods', 'ours', 'sdf_analytic', '--clearance', '100']),
        ('benchmark.thread.run', ['--parts', 'speed']),
    ],
    'thread-coacd': [
        ('benchmark.thread.run', ['--config', THREAD_COACD, '--parts', 'probes', '--probes',
                                  'memory', 'axial', 'radial', 'tilt', 'screw', 'spin']),
        ('benchmark.thread.run', ['--config', THREAD_COACD, '--parts', 'speed']),
    ],
    'videos': [
        ('benchmark.render', []),
        ('benchmark.thread.render', ['--behaviors', 'screw', 'spin', 'rundown']),
        ('benchmark.thread.render', ['--behaviors', 'rundown', '--clearance', '20',
                                     '--workers', '1']),
        ('benchmark.thread.render', ['--behaviors', 'compare', '--workers', '1']),
        ('benchmark.thread.render', ['--behaviors', 'compare', '--clearance', '20',
                                     '--workers', '1']),
        ('benchmark.thread.render', ['--behaviors', 'compare', '--forms', 'all',
                                     '--workers', '1']),
        ('benchmark.thread.render', ['--behaviors', 'forms', '--workers', '1']),
        ('benchmark.thread.render', ['--behaviors', 'wacky', '--workers', '1']),
        ('benchmark.thread.render', ['--behaviors', 'undercut', '--workers', '1']),
    ],
}
ORDER = ['probes', 'speed', 'gpu', 'thread', 'report', 'thread-coacd', 'videos']
PEG_RUNS = {'benchmark.run_probes', 'benchmark.run_speed', 'benchmark.run_warp_probes'}


def _value(args: list, flag: str) -> list[str] | None:
    """The values given after flag in an argument list, or None."""
    if flag not in args:
        return None
    i, out = args.index(flag) + 1, []
    while i < len(args) and not str(args[i]).startswith('--'):
        out.append(str(args[i]))
        i += 1
    return out


def _set(args: list, flag: str, values: list[str]) -> list:
    """args with flag's values replaced (or added)."""
    args = list(args)
    if flag in args:
        i = args.index(flag)
        j = i + 1
        while j < len(args) and not str(args[j]).startswith('--'):
            j += 1
        del args[i:j]
    return args + [flag, *values]


def _narrow(module: str, args: list, opt) -> list | None:
    """The command restricted to --shapes / --methods / --clearance, or None
    if nothing is left to run."""
    if module not in PEG_RUNS and module != 'benchmark.thread.run':
        return args
    cfg_path = Path(_value(args, '--config')[0]) if '--config' in args else (
        PROBES if module in PEG_RUNS else THREAD)
    cfg = yaml.safe_load(cfg_path.read_text())
    methods = _value(args, '--methods') or list(cfg['methods'])
    if opt.methods:
        methods = [m for m in methods if m in opt.methods]
        if not methods:
            return None
        args = _set(args, '--methods', methods)
    if module in PEG_RUNS:
        shapes = _value(args, '--shapes') or list(cfg['shapes'])
        if opt.shapes:
            shapes = [s for s in shapes if s in opt.shapes]
            if not shapes:
                return None
            args = _set(args, '--shapes', shapes)
        # a shape whose grid has none of the methods (e.g. ours, extended sweep)
        grids = cfg['methods']
        if not any(m in grids and (not isinstance(grids[m], dict) or s in grids[m])
                   for m in methods for s in shapes):
            return None
    elif opt.clearance:
        cs = [float(c) for c in _value(args, '--clearance') or cfg['clearance_um']]
        cs = [c for c in cs if c in opt.clearance]
        if not cs:
            return None
        args = _set(args, '--clearance', [f'{c:g}' for c in cs])
    if opt.workers and module in ('benchmark.run_probes', 'benchmark.thread.run'):
        args = _set(args, '--workers', [str(opt.workers)])
    return args


def commands(stage: str, opt) -> list[list[str]]:
    py = [sys.executable, '-m']
    if stage == 'report':
        out = [str(opt.out)]
        return [py + ['benchmark.report', '--results', str(opt.results), '--out', *out,
                      '--gpu-nworld', str(opt.gpu_nworld)],
                py + ['benchmark.thread.report', '--results', str(opt.results), '--out', *out,
                      '--traces']]
    cmds = []
    for module, args in STAGES[stage]:
        args = _narrow(module, args, opt)
        if args is None:
            continue
        if stage != 'videos':
            args = args + ['--results', opt.results]
        cmds.append(py + [module] + [str(a) for a in args])
    return cmds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', nargs='+', required=True, choices=ORDER)
    ap.add_argument('--results', type=Path, default=tables.ROOT,
                    help='raw results root to write (run stages) or read (report)')
    ap.add_argument('--out', type=Path, default=HERE / 'report', help='report directory')
    ap.add_argument('--gpu-nworld', type=int, default=256,
                    help='Warp batch size of the GPU timings in the report (256 or 512)')
    ap.add_argument('--shapes', nargs='*', help='peg: circle and/or square')
    ap.add_argument('--methods', nargs='*', help='ours coacd vhacd sdf (sdf_analytic: thread)')
    ap.add_argument('--clearance', nargs='*', type=float, help='thread clearances, µm')
    ap.add_argument('--workers', type=int, help='process pool size (CPU probe sweeps)')
    ap.add_argument('--dry-run', action='store_true', help='print the commands, run nothing')
    opt = ap.parse_args(argv)
    opt.results, opt.out = opt.results.resolve(), opt.out.resolve()

    stages = sorted(set(opt.stage), key=ORDER.index)
    env = {**os.environ, 'MUJOCO_GL': os.environ.get('MUJOCO_GL', 'egl')}
    root = Path(__file__).resolve().parents[1]
    # The checkout itself, and its src/ when mjfit is not pip-installed.
    paths = [str(root)] + ([str(root / 'src')] if (root / 'src' / 'mjfit').is_dir() else [])
    env['PYTHONPATH'] = os.pathsep.join(paths + ([env['PYTHONPATH']] if env.get('PYTHONPATH')
                                                  else []))
    if not opt.dry_run and opt.results.resolve() == tables.ROOT.resolve() and any(
            s != 'report' for s in stages):
        print(f'note: writing into {tables.ROOT}, the shipped tables; rerun rows replace '
              'theirs (git checkout benchmark/results restores them)', flush=True)
    for stage in stages:
        for cmd in commands(stage, opt):
            print('$ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
            if opt.dry_run:
                continue
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True, env=env, cwd=root)
            print(f'  done in {time.perf_counter() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
