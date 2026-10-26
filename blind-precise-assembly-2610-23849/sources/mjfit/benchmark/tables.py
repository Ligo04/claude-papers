"""Raw results: one gzipped CSV per probe or timing, in a fixed layout.

    results/peg/probes_cpu/<probe>.csv.gz     wall offset, memory, S0-S4 on CPU MuJoCo
    results/peg/probes_gpu/<probe>.csv.gz     S2, S3 (and S1) on MuJoCo Warp
    results/peg/speed_cpu.csv.gz              CPU time per step, 1 env, 1 thread
    results/peg/speed_gpu.csv.gz              MuJoCo Warp time per env-step (column nworld)
    results/thread/probes_cpu/<probe>.csv.gz  T1-T8 and memory on CPU MuJoCo
    results/thread/speed_cpu.csv.gz           CPU time per step, 1 thread

Tables are named by their path without the suffix ('peg/speed_gpu'). A run
merges its rows into a table: a new row replaces the stored row with the same
key (the measurement's identity: shape, representation, MULTICCD, clearance,
direction, ...) in place, new keys are appended, and every other stored row
is kept. So a partial rerun updates only what it measured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / 'results'
SUFFIX = '.csv.gz'
# Columns that identify one measurement; a table uses those it has.
PEG_KEYS = ['shape', 'label', 'multiccd', 'scenario', 'nworld', 'clearance_um', 'friction',
            'theta', 'phi_deg', 'gap_true_um']
THREAD_KEYS = ['label', 'multiccd', 'clearance_um', 'direction', 'theta', 'phi_deg', 'friction',
               'scenario', 'torque_ratio', 'turns', 'kind', 'angle_deg', 'mu', 'load_N',
               'condim', 'cone', 'timestep_ms']


def path(name: str, root: Path = ROOT) -> Path:
    return Path(root) / f'{name}{SUFFIX}'


def keys(name: str, columns) -> list[str]:
    ks = THREAD_KEYS if name.startswith('thread/') else PEG_KEYS
    return [k for k in ks if k in columns]


def read(name: str, root: Path = ROOT, **kw) -> pd.DataFrame | None:
    """The table, or None if it has not been written."""
    p = path(name, root)
    return pd.read_csv(p, **kw) if p.exists() else None


def write(name: str, rows, root: Path = ROOT) -> Path:
    """Merge rows (a DataFrame or a list of dicts) into the table and save it.

    Stored rows are parsed with round-trip float precision so that rewriting
    them leaves their text unchanged. The gzip header carries no timestamp, so
    equal tables give equal files."""
    new = pd.DataFrame(rows)
    old = read(name, root, float_precision='round_trip')
    df = new if old is None else pd.concat([old, new], ignore_index=True)
    k = keys(name, df.columns)
    if k:
        # The last row of each key wins, at the position of the key's first row.
        slot = df.groupby(k, dropna=False, sort=False).ngroup().to_numpy()
        last = ~df.duplicated(k, keep='last').to_numpy()
        df = df[last].iloc[np.argsort(slot[last], kind='stable')]
    p = path(name, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    df.to_csv(tmp, index=False, compression={'method': 'gzip', 'mtime': 0})
    tmp.replace(p)
    return p
