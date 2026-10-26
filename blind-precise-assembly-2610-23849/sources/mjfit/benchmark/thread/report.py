"""Bolt/nut tables and figures -> report/ (thread_*.png, thread_summary.md).

    python -m benchmark.thread.report               # from benchmark/results/thread/
    python -m benchmark.thread.report --traces      # + thread_start.png (re-simulates)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import report as R  # noqa: E402  (shared style, colors, Pareto helper)
from .. import tables  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
# The thread benchmark adds a fifth method: the exact parts as an analytic SDF.
METHODS = R.METHODS + ['sdf_analytic']
NAMES = {**R.NAMES, 'ours': 'ours (mjfit)', 'sdf_analytic': 'SDF, analytic (exact shape)'}
COLOR = {**R.COLOR, 'sdf_analytic': '#8a5cc8'}
MARKER = {**R.MARKER, 'sdf_analytic': 'v'}
SPIN_T = 0.8  # T5 duration (s)
YMAX_OVER_C = 5  # Pareto y axis stops at 5 x clearance; blow-ups sit on the edge
ACCURACY = ('axial', 'radial', 'tilt', 'screw')
PROBES = ('memory',) + ACCURACY + ('spin', 'start', 'partial', 'torque')


def load(root: Path = tables.ROOT) -> tuple[dict, pd.DataFrame | None]:
    """(probes by name, CPU timings) from the raw tables under root/thread/."""
    p = {k: tables.read(f'thread/probes_cpu/{k}', root) for k in PROBES}
    return p, tables.read('thread/speed_cpu', root)


def _order(df: pd.DataFrame) -> list[str]:
    """Labels by method, then by the method's knob (coarse to fine)."""
    def knob(label):
        kv = dict(re.findall(r'(\w+)=([\d.e+-]+)', label))
        if 'segments' in kv:
            return float(kv['segments'])
        if 'threshold' in kv:
            return -float(kv['threshold'])
        if 'max_hulls' in kv:
            return float(kv['max_hulls']) * 1e9 + float(kv.get('resolution', 0))
        if 'initpoints' in kv:
            return float(kv['initpoints']) * float(kv['iterations'])
        return float(kv.get('depth', 0))
    out = []
    for m in METHODS:
        labels = set(df.loc[df.method == m, 'label'])
        out += sorted(labels, key=knob)
    return out


def tag(label: str, mccd: bool = True) -> str:
    kv = dict(re.findall(r'(\w+)=([\d.e+-]+)', label))
    if 'segments' in kv:
        t = f'n={kv["segments"]}'
        if 'relief_um' in kv:
            t += f', {kv["relief_um"]} µm seam relief'
    elif 'threshold' in kv:
        t = f'thr {kv["threshold"]}'
    elif 'max_hulls' in kv:
        t = f'{kv["max_hulls"]} hulls, {float(kv["resolution"]) / 1e6:g}M vox'
    elif 'depth' in kv:
        t = f'depth {kv["depth"]}'
    elif 'initpoints' in kv:
        t = f'{kv["initpoints"]} pts × {kv["iterations"]} it'
    else:
        t = label
    return t + ('' if mccd else ', off')


def accuracy(p: dict, label: str, c: float, mccd: bool):
    """(worst |settled error| over T1-T4 in µm, clearance filled?)."""
    parts = [df[(df.label == label) & (df.multiccd == mccd) & (df.clearance_um == c)]
             for df in (p.get(k) for k in ACCURACY) if df is not None]
    parts = [x for x in parts if len(x)]
    if not parts:
        return None
    if any(x.overlap_at_start.any() for x in parts):
        return c, True
    return max(max(x.error_um.abs().max() for x in parts), 0.05), False


def cost(p: dict, speed, label: str, c: float, mccd: bool, xkind: str):
    if xkind == 'memory':
        mem = p.get('memory')
        pick = mem[(mem.label == label) & (mem.clearance_um == c)] if mem is not None else []
        return float(pick.model_MB.iloc[0]) if len(pick) else None
    if speed is None:
        return None
    pick = speed[(speed.label == label) & (speed.multiccd == mccd) & (speed.clearance_um == c)]
    # A filled setting can eject the nut in the timing scenario; with nothing
    # left to collide its step time (µs, against >= 460 µs for any thread in
    # contact) is not a cost of the representation. A nut merely bouncing off
    # its flanks at the last step keeps a normal step time.
    pick = pick[(pick.nefc_last > 0) | (pick.us_per_step > 100)]
    return float(pick.us_per_step.iloc[0]) if len(pick) else None


XLABEL = {'cpu': 'CPU µs per step, 1 thread (nut screwed up the thread under load)',
          'memory': 'compiled model memory (MB)'}
LABEL_OFFSET = {'ours': (5, -8, 'left'), 'coacd': (5, -9, 'left'),
                'vhacd': (5, 8, 'left'), 'sdf': (-5, 8, 'right'),
                'sdf_analytic': (6, 9, 'left')}


def pareto_panel(ax, p, speed, c, xkind):
    base = p['axial']
    for m in METHODS:
        pts = []
        for label in _order(base[base.method == m]):
            for mccd in ((True, False) if not m.startswith('sdf') else (True,)):
                x = cost(p, speed, label, c, mccd, 'memory' if xkind == 'memory' else 'cpu')
                acc = accuracy(p, label, c, mccd)
                if x is not None and acc is not None:
                    pts.append((x, *acc, label, mccd))
        if not pts:
            continue
        front = R._pareto_front(pts)
        top = YMAX_OVER_C * c
        ax.plot([q[0] for q in front], [min(q[1], top) for q in front], '-', color=COLOR[m],
                lw=1.5, zorder=2)
        for q in pts:
            x, y, filled, label, mccd = q
            on = q in front
            if y > top:   # blew up: drawn on the top edge
                ax.plot(x, top, '^', ms=5, zorder=3, color=COLOR[m], alpha=0.5, clip_on=False)
                continue
            ax.plot(x, y, MARKER[m], ms=6 if on else 4, zorder=3, color=COLOR[m],
                    alpha=1.0 if on else 0.3, mfc='#fcfcfb' if filled else COLOR[m],
                    mec=COLOR[m] if filled else '#fcfcfb', mew=1.5 if filled else 1.0)
        last = None
        for x, y, filled, label, mccd in front:
            if filled or (last and abs(np.log10(x / last)) < 0.15):
                continue
            dx, dy, ha = LABEL_OFFSET[m]
            ax.annotate(tag(label, mccd), (x, y), xytext=(dx, dy), textcoords='offset points',
                        ha=ha, va='center', fontsize=6.5, color=COLOR[m])
            last = x
    ax.set_xscale('log')
    ax.set_yscale('log')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        axis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 5)))
        axis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.tick_params(which='minor', labelsize=7)
    ax.set_ylim(top=YMAX_OVER_C * c)
    ax.axhline(c, color=R.MUTED, lw=1, ls=':')
    ax.set_xlabel(XLABEL[xkind])
    ax.set_ylabel('worst |error| over T1–T4 (µm)')
    ax.set_title(f'M10×1.5 bolt/nut, c = {c:g} µm (axial play ±{2 * c / 3 ** .5:.3g} µm)',
                 loc='left')


def fig_pareto(p, speed, out: Path, xkind: str, name: str):
    cs = sorted(set(p['axial'].clearance_um), reverse=True)
    fig, axes = plt.subplots(1, len(cs), figsize=(6.2 * len(cs), 4.8), squeeze=False)
    for ax, c in zip(axes[0], cs):
        pareto_panel(ax, p, speed, c, xkind)
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], lw=1.5, label=NAMES[m])
               for m in METHODS]
    fig.legend(handles=handles, fontsize=8, loc='upper center', ncol=len(handles),
               bbox_to_anchor=(0.5, 0.925))
    what = 'Model memory' if xkind == 'memory' else 'Compute time'
    fig.suptitle(f'{what} vs error; lower-left is better. Each line is the Pareto front of one '
                 'method over its settings (granularity × MULTICCD); faint points are dominated.\n'
                 'Dotted: error = clearance. Hollow: the representation fills the clearance. '
                 f'▲ on the top edge: off scale (> {YMAX_OVER_C:g}× clearance).',
                 x=0.01, ha='left', fontsize=9, color=R.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out / name, dpi=160)
    plt.close(fig)


def summarize(p, speed, c: float) -> pd.DataFrame:
    rows = []
    base = p['axial']
    for label in _order(base):
        m = base.loc[base.label == label, 'method'].iloc[0]
        for mccd in ((True, False) if not m.startswith('sdf') else (True,)):
            def pick(name):
                df = p.get(name)
                if df is None:
                    return df
                return df[(df.label == label) & (df.multiccd == mccd) & (df.clearance_um == c)]
            ax_, ra, ti, sc, sp = (pick(k) for k in ('axial', 'radial', 'tilt', 'screw', 'spin'))
            if ax_ is None or ax_.empty:
                continue
            filled = any(x is not None and len(x) and x.overlap_at_start.any()
                         for x in (ax_, ra, ti, sc))
            worst = lambda df: df.error_um.abs().max() if df is not None and len(df) else np.nan
            row = {'setting': NAMES[m] + ' ' + tag(label, mccd),
                   'filled at start': 'yes' if filled else '',
                   'T1 axial (µm)': ax_.error_um.abs().max(),
                   'T2 radial (µm)': worst(ra), 'T3 tilt @faces (µm)': worst(ti),
                   'T4 screw (µm)': worst(sc)}
            if sc is not None and len(sc):
                row['T4 outcome'] = sc.outcome.iloc[0]
                row['T4 lead ratio'] = sc.lead_ratio.iloc[0]
            if sp is not None and len(sp):
                f0 = sp[sp.friction == 0]
                f3 = sp[sp.friction > 0]
                if len(f0):
                    # Exact descent: settle onto the flanks (2c/sqrt3), then
                    # accelerate down the helix at the exact rate.
                    a = f0.accel_exact_mm_s2.iloc[0] / 1e3
                    exact_mm = (2 * c * 1e-6 / 3 ** .5 + a * SPIN_T ** 2 / 2) * 1e3
                    row['T5 frictionless descent (% of exact)'] = f0.drop_mm.iloc[0] / exact_mm * 100
                row['T5 friction 0.3 hold error (µm)'] = (f3.slip_um.iloc[0] if len(f3)
                                                          else np.nan)
            sds = [x for x in (ax_, ra, ti, sc) if x is not None and 'sd_false_max_um' in x]
            if sds:
                row['false contact gap (µm)'] = max(x.sd_false_max_um.max() for x in sds)
                row['true pen. at contact (µm)'] = max(x.sd_pen_max_um.max() for x in sds)
            row['CPU µs/step'] = cost(p, speed, label, c, mccd, 'cpu')
            row['model MB'] = cost(p, speed, label, c, mccd, 'memory')
            rows.append(row)
    return pd.DataFrame(rows)


# ----- T6, T7: the nut on the bolt's tip ---------------------------------------------------

def _settings(df: pd.DataFrame, c: float):
    """(label, method, multiccd) with rows at clearance c, in report order."""
    out = []
    for label in _order(df):
        m = df.loc[df.label == label, 'method'].iloc[0]
        for mccd in ((True, False) if not m.startswith('sdf') else (True,)):
            if len(df[(df.label == label) & (df.multiccd == mccd) & (df.clearance_um == c)]):
                out.append((label, m, mccd))
    return out


def _worst(df: pd.DataFrame, settled: bool = True) -> float:
    """Worst |error| (µm) over rows: of the settled (median) value, as in
    T1-T4, or of the whole run (for T6, which follows a path)."""
    if not len(df) or 'error_um' not in df:
        return np.nan
    cols = ['error_um'] if settled else ['error_max_um', 'error_min_um']
    vals = np.abs(df[cols].to_numpy(dtype=float))
    return float(np.nanmax(vals)) if not np.isnan(vals).all() else np.nan


def summarize_start(p, c: float) -> pd.DataFrame:
    """T6: outcomes at each torque ratio; at T4's ratio, the event (click or
    entry) error as axial phase along the helix, the click's drop error, and
    the worst height error off the drops."""
    from .probes import START_TORQUE_RATIO
    df, rows = p['start'], []
    for label, m, mccd in _settings(df, c):
        s = df[(df.label == label) & (df.multiccd == mccd) & (df.clearance_um == c)]
        row = {'setting': NAMES[m] + ' ' + tag(label, mccd),
               'filled at start': 'yes' if s.overlap_at_start.any() else ''}
        for direction, name in (('back', 'back-spin'), ('on', 'screw-on')):
            for k in sorted(set(s.torque_ratio), reverse=True):
                r = s[(s.direction == direction) & (s.torque_ratio == k)]
                if len(r):
                    row[f'{name} {k:g}×'] = r.outcome.iloc[0]
            r = s[(s.direction == direction) & (s.torque_ratio == START_TORQUE_RATIO)]
            if len(r):
                row[f'{name} event (µm)'] = r.event_err_um.iloc[0]
                if direction == 'back':
                    row['click drop error (µm)'] = r.drop_um.iloc[0] - r.drop_exact_um.iloc[0]
                row[f'{name} height (µm)'] = _worst(r, settled=False)
        row['true pen. at contact (µm)'] = s.sd_pen_max_um.max()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_partial(p, c: float) -> pd.DataFrame:
    """T7: worst |error| per engagement, radial (µm) and tilt (µm at the
    crest)."""
    df, rows = p['partial'], []
    for label, m, mccd in _settings(df, c):
        s = df[(df.label == label) & (df.multiccd == mccd) & (df.clearance_um == c)]
        row = {'setting': NAMES[m] + ' ' + tag(label, mccd),
               'filled at start': 'yes' if s.overlap_at_start.any() else ''}
        for kind in ('radial', 'tilt'):
            for t in sorted(set(s.turns)):
                row[f'{kind} {t:g} turn (µm)'] = _worst(s[(s.kind == kind) & (s.turns == t)])
        rows.append(row)
    return pd.DataFrame(rows)


def partial_exact(p, c: float) -> pd.DataFrame:
    """The exact T7 answers, per engagement: radial play and tilt limit, least
    and most over the directions, with full engagement (T2, T3) below."""
    df = p['partial']
    s = df[df.clearance_um == c]
    rows = []
    for t in sorted(set(s.turns)):
        ra, ti = s[(s.kind == 'radial') & (s.turns == t)], s[(s.kind == 'tilt') & (s.turns == t)]
        rows.append({'turns on': f'{t:g}',
                     'radial play (µm)': f'{ra.truth_um.min():.1f} … {ra.truth_um.max():.1f}',
                     'tilt limit (mrad)': f'{ti.truth_mrad.min():.1f} … {ti.truth_mrad.max():.1f}'})
    full = {'turns on': 'full (T2, T3: hinge at the nut centre)'}
    if p.get('radial') is not None:
        r = p['radial'][p['radial'].clearance_um == c].truth_um
        full['radial play (µm)'] = f'{r.min():.1f}' if len(r) else ''
    if p.get('tilt') is not None:
        r = p['tilt'][p['tilt'].clearance_um == c].truth_mrad
        full['tilt limit (mrad)'] = f'{r.min():.1f} … {r.max():.1f}' if len(r) else ''
    return pd.DataFrame(rows + [full])


def _best(df: pd.DataFrame, m: str, c: float):
    """A method's setting with the smallest worst error at c, preferring
    ones that keep the clearance open."""
    best = None
    for label, m_, mccd in _settings(df[df.method == m], c):
        s = df[(df.label == label) & (df.multiccd == mccd) & (df.clearance_um == c)]
        key = (bool(s.overlap_at_start.any()), _worst(s))
        if best is None or key < best[0]:
            best = (key, label, mccd)
    return best


def fig_partial(p, out: Path):
    """T7: exact play against engagement, and each method's best setting's
    worst error."""
    df = p['partial']
    cs = sorted(set(df.clearance_um), reverse=True)
    fig, axes = plt.subplots(len(cs), 3, figsize=(13, 3.6 * len(cs)), squeeze=False)
    for row, c in zip(axes, cs):
        s = df[df.clearance_um == c]
        for ax, kind, col, unit in ((row[0], 'radial', 'truth_um', 'µm'),
                                    (row[1], 'tilt', 'truth_mrad', 'mrad')):
            k = s[s.kind == kind].drop_duplicates(['turns', 'angle_deg'])
            g = k.groupby('turns')[col]
            ax.fill_between(g.min().index, g.min(), g.max(), color=R.GRID, lw=0)
            ax.plot(k.turns, k[col], '.', color=R.INK2, ms=4)
            ax.plot(g.min().index, g.min(), '-', color=R.INK, lw=1)
            ax.plot(g.max().index, g.max(), '-', color=R.INK, lw=1)
            full = p.get('radial' if kind == 'radial' else 'tilt')
            if full is not None and len(full[full.clearance_um == c]):
                v = full[full.clearance_um == c][col].min()
                ax.axhline(v, color=R.MUTED, lw=1, ls=':')
                ax.annotate('full engagement' + ('' if kind == 'radial' else
                                                 ' (T3: hinge at the nut centre)'),
                            (ax.get_xlim()[1], v), xytext=(-4, 4),
                            textcoords='offset points', ha='right', fontsize=7, color=R.MUTED)
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.set_xlabel('turns threaded on')
            ax.set_ylabel(f'exact {"radial play" if kind == "radial" else "tilt limit"} ({unit})')
            ax.set_title(f'c = {c:g} µm: exact {kind} {"play" if kind == "radial" else "limit"}'
                         ', every direction', loc='left')
        ax = row[2]
        for m in METHODS:
            b = _best(df[df.method == m], m, c)
            if b is None:
                continue
            (filled, _), label, mccd = b
            s = df[(df.label == label) & (df.multiccd == mccd) & (df.clearance_um == c)]
            t = sorted(set(s.turns))
            y = [max(_worst(s[s.turns == x]), 0.05) for x in t]
            ax.plot(t, y, '-' + MARKER[m], color=COLOR[m], ms=5, mfc='#fcfcfb' if filled else COLOR[m],
                    label=f'{NAMES[m]}, {tag(label, mccd)}' + (' (fills)' if filled else ''))
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.axhline(c, color=R.MUTED, lw=1, ls=':')
        ax.set_xlabel('turns threaded on')
        ax.set_ylabel('worst |error|, radial and tilt (µm)')
        ax.set_title('each method\'s best setting', loc='left')
        ax.legend(fontsize=6.5)
    fig.suptitle('T7: the nut threaded partway onto the tip. Grey band: least to most over '
                 'directions (radial) or hinge axes (tilt, hinge half-way along the engaged '
                 'length). Tilt errors are at the thread crest.', x=0.01, ha='left',
                 fontsize=9, color=R.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'thread_partial.png', dpi=160)
    plt.close(fig)


# ----- T8: turning torque against friction -------------------------------------------------

# (load N, condim, cone, timestep ms) -> name; the first is MuJoCo's default cone
# with the example configs' condim.
TORQUE_CASES = {(0.1, 3, 'pyramidal', 1.0): '0.1 N, condim 3',
                (0.1, 3, 'elliptic', 1.0): '0.1 N, condim 3, elliptic',
                (0.1, 4, 'pyramidal', 1.0): '0.1 N, condim 4',
                (1.0, 3, 'pyramidal', 1.0): '1 N, condim 3',
                (1.0, 3, 'elliptic', 1.0): '1 N, condim 3, elliptic',
                (1.0, 3, 'elliptic', 0.25): '1 N, condim 3, elliptic, 0.25 ms step'}


def _torque_case(df: pd.DataFrame, case) -> pd.DataFrame:
    load, condim, cone, dt = case
    return df[(df.load_N == load) & (df.condim == condim) & (df.cone == cone)
              & np.isclose(df.timestep_ms, dt)]


def _torque_cell(r: pd.DataFrame) -> str:
    """'up / down' breakaway over the formula; > 3 never turned, 0 back-drove."""
    out = []
    for d in ('up', 'down'):
        x = r[r.direction == d]
        if not len(x):
            out.append('')
        elif x.outcome.iloc[0] == 'holds':
            out.append('> 3')
        elif x.outcome.iloc[0] == 'back-drives':
            out.append('0')
        else:
            out.append(f'{x.ratio.iloc[0]:.2f}')
    return ' / '.join(out)


def summarize_torque(p, c: float) -> pd.DataFrame:
    """T8, each setting at 0.1 N with condim 3 and MuJoCo's default cone:
    breakaway torque over the formula, against / with the load, per mu."""
    df, rows = p['torque'], []
    df = _torque_case(df[df.clearance_um == c], next(iter(TORQUE_CASES)))
    for label, m, mccd in _settings(df, c):
        s = df[(df.label == label) & (df.multiccd == mccd)]
        row = {'setting': NAMES[m] + ' ' + tag(label, mccd)}
        for mu in sorted(set(s.mu)):
            row[f'μ = {mu:g}'] = _torque_cell(s[s.mu == mu])
        rows.append(row)
    return pd.DataFrame(rows)


def torque_cases(p, c: float, label: str, mccd: bool) -> pd.DataFrame:
    """T8 for one setting, by setup."""
    df = p['torque']
    s = df[(df.clearance_um == c) & (df.label == label) & (df.multiccd == mccd)]
    rows = []
    for case, name in TORQUE_CASES.items():
        r = _torque_case(s, case)
        if len(r):
            rows.append({'setup': name, **{f'μ = {mu:g}': _torque_cell(r[r.mu == mu])
                                           for mu in sorted(set(r.mu))}})
    return pd.DataFrame(rows)


def fig_torque(p, out: Path, c: float = 100, label: str = 'ours(segments=48)', mccd=False):
    """T8: breakaway torque over the formula against friction, per setup."""
    df = p['torque']
    s = df[(df.clearance_um == c) & (df.label == label) & (df.multiccd == mccd)]
    if not len(s):
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    colors = ['#2a78d6', '#1baf7a', '#eb6834', '#8a5cc8', '#eda100', '#0b0b0b']
    for ax, d in zip(axes, ('up', 'down')):
        for (case, name), col in zip(TORQUE_CASES.items(), colors):
            r = _torque_case(s, case)
            r = r[r.direction == d].sort_values('mu')
            if not len(r):
                continue
            y = np.where(r.outcome == 'holds', 3.0, np.where(r.outcome == 'back-drives', 0.02,
                                                              np.minimum(r.ratio, 3.0)))
            ax.plot(r.mu, y, '-o', color=col, ms=4, lw=1.2, label=name)
        ax.axhline(1, color=R.INK, lw=1, ls=(0, (4, 2)))
        ax.axhspan(2.9, 3.2, color=R.GRID, lw=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.015, 3.3)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.set_xlabel('friction μ (both parts)')
        ax.set_title(('turning against the load' if d == 'up' else 'turning with the load')
                     + f', {NAMES[label.split("(")[0]]} {tag(label, mccd)}', loc='left')
    axes[0].set_ylabel('breakaway torque / power-screw formula')
    axes[1].legend(fontsize=6.5, loc='lower right')
    fig.suptitle('T8: the engaged nut, pressed down by an axial load, under a slow torque ramp. '
                 'Dashed: the formula.\nTop band: 3x or more (or never turned within 3x); '
                 'bottom: back-drove (the thread did not self-lock).', x=0.01, ha='left',
                 fontsize=9, color=R.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / 'thread_torque.png', dpi=160)
    plt.close(fig)


# (method, params, multiccd) re-simulated for the T6 trace figure
TRACE_SETTINGS = [('ours', {'segments': 24}, False), ('ours', {'segments': 48}, False),
                  ('ours', {'segments': 96}, False),
                  ('sdf_analytic', {'initpoints': 100, 'iterations': 30}, True),
                  ('sdf', {'depth': 8}, True)]


def _start_trace(c_um: float, direction: str, k: float, method: str, params: dict,
                 mccd: bool):
    """One T6 run with its trace: (label, result row, trace)."""
    from .geometry import ThreadTask
    from .probes import probe_start
    from .representations import build_thread
    task = ThreadTask()
    rep = build_thread(task, method, params, c_um * 1e-6)
    r, tr = probe_start(task, rep, direction, multiccd=mccd, torque_ratio=k, trace=True)
    return rep.label, r, tr


def fig_start(out: Path, clearances=(100, 20), workers: int | None = None):
    """T6 traces, re-simulated (in parallel): back-spin at T4's torque and at
    2x, screwing on at T4's torque."""
    from concurrent.futures import ProcessPoolExecutor
    from .probes import START_TORQUE_RATIO
    from .representations import build_thread
    from .geometry import ThreadTask
    cols = (('back', START_TORQUE_RATIO), ('back', 2.0), ('on', START_TORQUE_RATIO))
    jobs = [(c_um, direction, k, method, params, mccd) for c_um in clearances
            for direction, k in cols for method, params, mccd in TRACE_SETTINGS]
    for c_um in clearances:  # build (or load) every representation once, up front
        for method, params, _ in TRACE_SETTINGS:
            build_thread(ThreadTask(), method, params, c_um * 1e-6)
    if any(m == 'sdf_analytic' for m, _, _ in TRACE_SETTINGS):
        from . import plugin
        plugin.library()
    with ProcessPoolExecutor(workers) as pool:
        runs = iter(list(pool.map(_start_trace, *zip(*jobs))))   # in job order
    fig, axes = plt.subplots(len(clearances), 3, figsize=(13, 3.6 * len(clearances)),
                             squeeze=False)
    for row, c_um in zip(axes, clearances):
        for ax, (direction, k) in zip(row, cols):
            exact = None
            # Scale: the click (drop 4c/sqrt3) for back-spin, or a few times the
            # axial play for the height error screwing on.
            span = 4 * c_um / 3 ** .5
            lo, hi = (-1.8 * span, 0.6 * span) if direction == 'back' else (-1.5 * span, 1.5 * span)
            for method, params, mccd in TRACE_SETTINGS:
                rep_label, r, tr = next(runs)
                exact = tr
                if direction == 'back':
                    y = tr['z_um']
                else:   # screwing on: the height error, as the path runs down a turn
                    y = np.interp(tr['deg'], tr['deg_exact'], tr['z_exact_um']) - tr['z_um']
                label = f'{NAMES[method]}, {tag(rep_label, mccd)}: {r["outcome"]}'
                if np.nanpercentile(np.abs(y), 50) > max(abs(lo), hi):
                    label += f' (off scale: {np.nanmedian(np.abs(y)) / 1e3:.1f} mm)'
                    y = np.full_like(y, np.nan)
                ax.plot(tr['deg'], y, color=COLOR[method], lw=1,
                        ls='-' if method != 'ours' or params['segments'] == 48 else
                        (':' if params['segments'] == 24 else '--'), label=label)
            if exact is not None and direction == 'back':
                ax.plot(exact['deg_exact'], exact['z_exact_um'], color=R.INK, lw=1.5,
                        ls=(0, (4, 2)), label='exact', zorder=5)
                # the click and the ride back up, with some flat either side
                back = exact['deg_exact'][(exact['deg_exact'] > 0)
                                          & (exact['z_exact_um'] > -1e-3)]
                ax.set_xlim(-20, (back[0] if len(back) else 90) + 30)
            ax.set_ylim(lo, hi)
            ax.set_xlabel('turned past the exact ' + ('click' if direction == 'back' else 'entry')
                          + ' (deg)')
            ax.set_ylabel('height above the resting flat (µm)' if direction == 'back'
                          else 'height error (µm; + into the bolt)')
            ax.set_title(f'c = {c_um:g} µm, ' + ('back-spin' if direction == 'back'
                                                 else 'screwing on') + f', torque {k:g}× exact',
                         loc='left')
            ax.legend(fontsize=6)
    fig.suptitle('T6: the nut on the bolt\'s tip, frictionless, turned by a constant torque '
                 '(a multiple of the exact back-off torque) against T4\'s 0.1 N load.',
                 x=0.01, ha='left', fontsize=9, color=R.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'thread_start.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=tables.ROOT,
                    help='raw results root (default: benchmark/results)')
    ap.add_argument('--out', type=Path, default=HERE / 'report')
    ap.add_argument('--traces', action='store_true',
                    help='re-simulate a few T6 runs for thread_start.png (a minute or two)')
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    p, speed = load(args.results)
    if p['axial'] is None:
        raise SystemExit('no thread results')
    md = ['# Bolt/nut benchmark: summary\n']
    for c in sorted(set(p['axial'].clearance_um), reverse=True):
        table = summarize(p, speed, c)
        table.to_csv(args.out / f'thread_summary_c{c:g}.csv', index=False)
        md += [f'## c = {c:g} µm\n', table.to_markdown(index=False, floatfmt='.3g'), '\n']
    if p['start'] is not None:
        md.append('# The nut on the bolt\'s tip (T6 start, T7 partial engagement)\n')
        for c in sorted(set(p['start'].clearance_um), reverse=True):
            table = summarize_start(p, c)
            table.to_csv(args.out / f'thread_start_c{c:g}.csv', index=False)
            md += [f'## T6, c = {c:g} µm\n', table.to_markdown(index=False, floatfmt='.3g'), '\n']
    if p['partial'] is not None:
        for c in sorted(set(p['partial'].clearance_um), reverse=True):
            md += [f'## T7 exact answers, c = {c:g} µm\n',
                   partial_exact(p, c).to_markdown(index=False), '\n']
            table = summarize_partial(p, c)
            table.to_csv(args.out / f'thread_partial_c{c:g}.csv', index=False)
            md += [f'## T7, c = {c:g} µm\n', table.to_markdown(index=False, floatfmt='.3g'), '\n']
        fig_partial(p, args.out)
    if p['torque'] is not None:
        md.append('# Turning torque against friction (T8)\n')
        for c in sorted(set(p['torque'].clearance_um), reverse=True):
            table = summarize_torque(p, c)
            table.to_csv(args.out / f'thread_torque_c{c:g}.csv', index=False)
            md += [f'## T8, c = {c:g} µm: breakaway / formula, up / down (0.1 N, condim 3)\n',
                   table.to_markdown(index=False), '\n',
                   f'## T8, c = {c:g} µm, ours n=48, MULTICCD off, by setup\n',
                   torque_cases(p, c, 'ours(segments=48)', False).to_markdown(index=False), '\n']
        fig_torque(p, args.out)
    (args.out / 'thread_summary.md').write_text('\n'.join(md))
    fig_pareto(p, speed, args.out, 'memory', 'pareto_thread_memory.png')
    if speed is not None:
        fig_pareto(p, speed, args.out, 'cpu', 'pareto_thread_cpu.png')
    if args.traces:
        fig_start(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
