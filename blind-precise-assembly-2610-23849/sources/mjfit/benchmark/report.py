"""Peg-in-hole summaries and figures from the raw tables in benchmark/results/.

    python -m benchmark.report                    # writes benchmark/report/ (tracked)
    python -m benchmark.report --gpu-nworld 512 --out /tmp/report512

GPU timings are reported at one MuJoCo Warp batch size, --gpu-nworld (256 by
default, as in the Pareto figures; the speed table lists every batch size
measured). Every table/figure is skipped if its input table is missing, so
this can run on partial results.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import tables  # noqa: E402

HERE = Path(__file__).resolve().parent
METHODS = ['ours', 'coacd', 'vhacd', 'sdf']
NAMES = {'ours': 'ours (mjfit)', 'coacd': 'CoACD', 'vhacd': 'V-HACD', 'sdf': 'SDF'}
COLOR = {'ours': '#2a78d6', 'coacd': '#eb6834', 'vhacd': '#1baf7a', 'sdf': '#eda100'}
MARKER = {'ours': 'o', 'coacd': 's', 'vhacd': '^', 'sdf': 'D'}
GPU_NWORLD = 256  # default MuJoCo Warp batch size the GPU timings are reported at
INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
GOOD, BAD = '#0ca30c', '#d03b3b'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': BASE,
    'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'legend.frameon': False, 'axes.titlesize': 10, 'axes.titlecolor': INK,
})


PROBES_CPU = ('geometry', 'memory', 'static', 'drop', 'lateral', 'tilt', 'yaw')
PROBES_GPU = ('lateral', 'tilt', 'drop')


def load(root: Path = tables.ROOT) -> tuple[dict, dict]:
    """(probes, speed) from the raw tables under root. probes: the CPU probes
    by name, plus 'warp_<name>' for the MuJoCo Warp ones; speed: 'cpu_single'
    and 'warp' (runs whose buffers overflowed are dropped: not physics)."""
    p = {k: tables.read(f'peg/probes_cpu/{k}', root) for k in PROBES_CPU}
    p.update({f'warp_{k}': tables.read(f'peg/probes_gpu/{k}', root) for k in PROBES_GPU})
    speed = {'cpu_single': tables.read('peg/speed_cpu', root),
             'warp': tables.read('peg/speed_gpu', root)}
    if speed['warp'] is not None:
        w = speed['warp']
        speed['warp'] = w[~w.get('warp_invalid', False).fillna(False).astype(bool)]
    return p, speed


def _knob(label: str) -> tuple:
    """Coarse to fine within a method: chord error and CoACD threshold
    falling, V-HACD hulls and voxels rising, SDF octree depth then contact
    search effort rising."""
    kv = {k: float(v) for k, v in re.findall(r'(\w+)=([\d.e+-]+)', label)}
    if 'chord_error_um' in kv:
        return (-kv['chord_error_um'],)
    if 'threshold' in kv:
        return (-kv['threshold'],)
    if 'max_hulls' in kv:
        return kv['max_hulls'], kv.get('resolution', 4e5)
    return kv.get('depth', 0.0), kv.get('initpoints', 0.0)


def _rep_order(df: pd.DataFrame) -> list[str]:
    """Labels grouped by method, coarse to fine within a method."""
    seen = []
    for m in METHODS:
        seen += sorted(dict.fromkeys(df.loc[df.method == m, 'label']), key=_knob)
    return seen


def _short(label: str) -> str:
    return (label.replace('chord_error_um=', 'ε=').replace('threshold=', 'thr=')
            .replace('max_hulls=', 'hulls=').replace(',resolution=4000000', ',res=4M')
            .replace(',initpoints=100,iterations=30', ',100pts/30it'))


# ----- per-representation summary ------------------------------------------------

HOLE_DIAMETER_UM = 16000.0


def ratio(c_um: float) -> str:
    """Whitney's clearance ratio (D - d) / D for a radial clearance c."""
    return f'{2 * c_um / HOLE_DIAMETER_UM:.4g}'


def contact_sd(p, shape, label, c_ref, mccd=True):
    """(worst false-contact gap, worst penetration) in µm at contact-reporting
    states over S2-S4 at clearance c_ref (min-|sd| of Son & Kim)."""
    parts = [df[(df['shape'] == shape) & (df.label == label) & (df.multiccd == mccd)
                & (df.clearance_um == c_ref)]
             for df in (p.get('lateral'), p.get('tilt'), p.get('yaw'))
             if df is not None and 'sd_false_max_um' in df]
    parts = [x for x in parts if len(x)]
    if not parts or any(x.overlap_at_start.any() for x in parts):
        return None
    return (max(x.sd_false_max_um.max() for x in parts),
            max(x.sd_pen_max_um.max() for x in parts))


def onset_error(static: pd.DataFrame) -> np.ndarray:
    """Per direction: minus the true gap at which contact is first reported.
    Same sign convention as S2-S4: negative = contact through empty space
    (phantom material), positive = contact only after real overlap."""
    return -static[static.contact].groupby('theta').gap_true_um.max().values


def summarize(p: dict[str, pd.DataFrame], speed: dict[str, pd.DataFrame], shape: str,
              c_ref: float = 100.0, nworld: int = GPU_NWORLD) -> pd.DataFrame:
    rows = []
    base = next(p[k] for k in ('lateral', 'static', 'drop', 'tilt') if p.get(k) is not None)
    for label in _rep_order(base[base['shape'] == shape]):
        for mccd in (True, False):
            sel = lambda df: df[(df['shape'] == shape) & (df.label == label) & (df.multiccd == mccd)]  # noqa: E731
            if sel(base).empty:
                continue
            r = {'representation': label, 'multiccd': 'on' if mccd else 'off'}
            r['hulls'] = int(sel(base).n_hulls.iloc[0])
            r['setup_s'] = round(float(sel(base).setup_s.iloc[0]), 2)
            geo_df = p.get('geometry')
            if geo_df is not None:
                g = geo_df[(geo_df['shape'] == shape) & (geo_df.label == label)]
                if len(g):
                    g = g.iloc[0]
                    r['wall offset um [min,max]'] = f'[{g.wall_min_um:+.1f}, {g.wall_max_um:+.1f}]'
                    r['mid-wall offset um [min,max]'] = (
                        f'[{g.wall_mid_min_um:+.1f}, {g.wall_mid_max_um:+.1f}]')
            for c in (c_ref, 20.0):
                sd = contact_sd(p, shape, label, c, mccd)
                r[f'contact sd c={c:g}: false gap / pen um'] = (
                    'filled' if sd is None else f'{sd[0]:.1f} / {sd[1]:.1f}')
            # S0 and S1 run on the main sweep only: 'not run' for the rest.
            if p.get('static') is not None:
                s0 = sel(p['static'])
                err = onset_error(s0)
                r['S0 onset err um [min,max]'] = (
                    f'[{err.min():+.1f}, {err.max():+.1f}]' if len(err)
                    else 'none' if len(s0) else 'not run')
            if p.get('drop') is not None:
                d = sel(p['drop'])
                for fr, tag in ((True, 'S1 wrong (fric)'), (False, 'S1 wrong (no fric)')):
                    dd = d[d.friction == fr].sort_values('clearance_um')
                    wrong = dd[dd.outcome != dd.expected]
                    r[tag] = (', '.join(f'{c:+g}{o[0]}' for c, o in
                                        zip(wrong.clearance_um, wrong.outcome)) or '-'
                              ) if len(dd) else 'not run'
            if p.get('lateral') is not None:
                lat = sel(p['lateral'])
                # Centered peg already intersecting the plate: the clearance is gone.
                r['filled at c (um)'] = ', '.join(
                    f'{c:g}' for c in sorted(lat[lat.overlap_at_start].clearance_um.unique())) or '-'
                for c in (c_ref, 20.0):
                    lc = lat[lat.clearance_um == c]
                    if len(lc):
                        r[f'S2 c={c:g} median um [min,max]'] = (
                            f'[{lc.error_um.min():+.1f}, {lc.error_um.max():+.1f}]')
                        r[f'S2 c={c:g} spike um'] = round(float(lc.error_max_um.max()), 1)
            for probe in ('tilt', 'yaw'):
                if p.get(probe) is not None:
                    t = sel(p[probe])
                    t = t[t.clearance_um == c_ref]
                    if len(t):
                        worst = t.error_um.iloc[np.argmax(np.abs(t.error_um.values))]
                        r[f'S{3 if probe == "tilt" else 4} c={c_ref:g} um'] = round(float(worst), 1)
            wl = p.get('warp_lateral')
            if wl is not None:
                w = sel(wl)
                w = w[w.clearance_um == c_ref]
                if len(w):
                    r[f'Warp S2 c={c_ref:g} median um [min,max]'] = (
                        f'[{w.error_um.min():+.1f}, {w.error_um.max():+.1f}]'
                        + (' INVALID' if w.warp_invalid.any() else ''))
            wd = p.get('warp_drop')
            if wd is not None and len(sel(wd)):
                dd = sel(wd)
                dd = dd[dd.friction].sort_values('clearance_um')
                wrong = dd[dd.outcome != dd.expected]
                r['Warp S1 wrong (fric)'] = ', '.join(
                    f'{c:+g}{o[0]}' for c, o in zip(wrong.clearance_um, wrong.outcome)) or '-'
            cs = speed.get('cpu_single')
            if cs is not None:
                cc = cs[(cs['shape'] == shape) & (cs.label == label) & (cs.multiccd == mccd)
                        & (cs.scenario == 'wall')]
                if len(cc):
                    r['CPU us/step (wall)'] = round(float(cc.us_per_step.iloc[0]), 1)
            gpu = _cost(speed, shape, label, 'gpu', mccd=mccd, nworld=nworld)
            if gpu is not None:
                r[f'GPU us/env-step @{nworld} (wall)'] = round(gpu, 2)
            rows.append(r)
    return pd.DataFrame(rows)


def _sig(x: float | None, digits: int = 3) -> float:
    return float(f'{x:.{digits}g}') if x is not None and np.isfinite(x) else np.nan


def speed_table(p: dict, speed: dict, scenario: str = 'wall') -> pd.DataFrame:
    """Real-time factor (simulated time / wall time at the 1 ms step) per
    setting and MULTICCD: CPU (1 env, 1 thread) and GPU summed over each
    measured Warp batch size, with the worst settled error over S2-S4 on CPU
    at c = 100 and 20 µm ('filled': the representation fills the clearance)."""
    cs, warp = speed.get('cpu_single'), speed.get('warp')
    if p.get('lateral') is None or cs is None:
        return pd.DataFrame()
    nworlds = sorted(set(warp.nworld.dropna().astype(int))) if warp is not None else []
    rows = []
    for shape in ('circle', 'square'):
        lat = p['lateral'][p['lateral']['shape'] == shape]
        for label in _rep_order(lat):
            m = lat.loc[lat.label == label, 'method'].iloc[0]
            for mccd in (True, False) if m != 'sdf' else (True,):
                cpu = _cost(speed, shape, label, 'cpu', scenario, mccd)
                if cpu is None:
                    continue
                r = {'shape': shape, 'representation': label,
                     'multiccd': 'on' if mccd else 'off',
                     'hulls': int(lat.loc[lat.label == label, 'n_hulls'].iloc[0]),
                     'CPU real-time factor': _sig(1e3 / cpu)}
                for n in nworlds:
                    g = warp[(warp['shape'] == shape) & (warp.label == label)
                             & (warp.multiccd == mccd) & (warp.scenario == scenario)
                             & (warp.nworld == n)].dropna(subset=['realtime_factor'])
                    r[f'GPU real-time factor @{n}'] = (
                        _sig(float(g.realtime_factor.iloc[0])) if len(g) else np.nan)
                for c in (100.0, 20.0):
                    acc = _accuracy(p, shape, label, c, mccd)
                    r[f'worst error c={c:g} um'] = ('' if acc is None else 'filled' if acc[1]
                                                    else f'{acc[0]:.2f}')
                rows.append(r)
    return pd.DataFrame(rows)


# ----- figures ---------------------------------------------------------------------

def _symlog(ax, lin=1.0):
    ax.set_xscale('symlog', linthresh=lin)
    ax.axvline(0, color=INK2, lw=1, zorder=1)


def _mccd(df, mccd):
    """Rows for a MULTICCD setting; SDF ignores the flag and only runs 'on'."""
    return df[(df.multiccd == mccd) | (df.method == 'sdf')]


def fig_accuracy(p, shape, out, c_ref=100.0, mccd=True):
    lat, tilt, static = p.get('lateral'), p.get('tilt'), p.get('static')
    if lat is None:
        return
    lat = _mccd(lat[lat['shape'] == shape], mccd)
    labels = _rep_order(lat)
    panels = [('Geometry: wall offset, no solver', p.get('geometry')),
              ('S0: contact onset error', static),
              (f'S2: lateral play error, c = {c_ref:g} µm', lat),
              (f'S3: tilt error, c = {c_ref:g} µm', tilt),
              (f'At contacts: true distance, c = {c_ref:g} µm', lat)]
    panels = [(t, df) for t, df in panels if df is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 0.32 * len(labels) + 1.4),
                             sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, (title, df) in zip(np.atleast_1d(axes), panels):
        df = df[df['shape'] == shape]
        if not title.startswith('Geometry'):
            df = _mccd(df, mccd)
        for yi, label in zip(y, labels):
            sub = df[df.label == label]
            if title.startswith('Geometry'):
                if sub.empty:
                    continue
                g = sub.iloc[0]
                vals = np.array([g.wall_min_um, g.wall_max_um])
                m = g.method
                ax.plot(vals, [yi, yi], color=COLOR[m], lw=2, solid_capstyle='round', zorder=2)
                # Dot: mid-wall extreme (interior wall, rim-edge rounding excluded).
                mid = g.wall_mid_min_um if abs(g.wall_mid_min_um) > abs(g.wall_mid_max_um) \
                    else g.wall_mid_max_um
                ax.plot(mid, yi, MARKER[m], color=COLOR[m], ms=6, mec='#fcfcfb', mew=1.5, zorder=3)
                continue
            if title.startswith('At contacts'):
                m = sub.method.iloc[0] if len(sub) else None
                sd = contact_sd(p, shape, label, c_ref, mccd or m == 'sdf')
                if m is None:
                    continue
                if sd is None:
                    ax.text(0.02, yi, 'clearance filled', transform=ax.get_yaxis_transform(),
                            va='center', fontsize=7.5, color=COLOR[m], fontstyle='italic')
                    continue
                # − = contact reported across a gap; + = real overlap (penetration).
                ax.plot([-sd[0], sd[1]], [yi, yi], color=COLOR[m], lw=2,
                        solid_capstyle='round', zorder=2)
                ax.plot(sd[1] if sd[1] >= sd[0] else -sd[0], yi, MARKER[m], color=COLOR[m],
                        ms=6, mec='#fcfcfb', mew=1.5, zorder=3)
                continue
            if title.startswith('S0'):
                vals = onset_error(sub)
            else:
                sub = sub[sub.clearance_um == c_ref]
                vals = sub.error_um.values
            if len(vals) == 0:
                continue
            m = sub.method.iloc[0]
            if not title.startswith('S0') and sub.overlap_at_start.any():
                # The centered peg already intersects the plate: no play to measure.
                ax.text(0.02, yi, 'clearance filled', transform=ax.get_yaxis_transform(),
                        va='center', fontsize=7.5, color=COLOR[m], fontstyle='italic')
                continue
            ax.plot([vals.min(), vals.max()], [yi, yi], color=COLOR[m], lw=2,
                    solid_capstyle='round', zorder=2)
            ax.plot(np.median(vals), yi, MARKER[m], color=COLOR[m], ms=6,
                    mec='#fcfcfb', mew=1.5, zorder=3)
        _symlog(ax)
        ax.set_title(title, loc='left')
        ax.grid(axis='y', visible=False)
    axes = np.atleast_1d(axes)
    axes[0].set_yticks(y, [_short(lbl) for lbl in labels], color=INK2)
    axes[0].set_ylim(-0.7, len(labels) - 0.3)
    fig.suptitle(f'{shape}: error vs exact geometry in µm, MULTICCD {"on" if mccd else "off"} '
                 f'(SDF unaffected); c = {c_ref:g} µm is clearance ratio {ratio(c_ref)}. '
                 f'0 = exact; − = phantom material / contact across a gap; '
                 f'+ = recessed wall / penetration.\nLine = range over directions, rays, or '
                 f'contact states; dot = median (geometry: mid-wall extreme, i.e. without '
                 f'rim-edge rounding). S0 at −1000 = contact at every probed gap.',
                 x=0.01, ha='left', fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / f'accuracy_{shape}{"" if mccd else "_multiccd_off"}.png', dpi=160)
    plt.close(fig)


def fig_drop(p, shape, out):
    drop = p.get('drop')
    if drop is None:
        return
    drop = drop[(drop['shape'] == shape) & drop.multiccd]
    labels = _rep_order(drop)
    cs = sorted(drop.clearance_um.unique(), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.3 * len(labels) + 1.6), sharey=True)
    for ax, fr in zip(axes, (True, False)):
        d = drop[drop.friction == fr]
        for yi, label in enumerate(labels):
            for xi, c in enumerate(cs):
                row = d[(d.label == label) & (d.clearance_um == c)]
                if row.empty:
                    continue
                ok = row.outcome.iloc[0] == row.expected.iloc[0]
                ax.add_patch(plt.Rectangle((xi - 0.46, yi - 0.42), 0.92, 0.84,
                                           color=GOOD if ok else BAD, alpha=0.18 if ok else 0.85,
                                           lw=0))
                ax.text(xi, yi, row.outcome.iloc[0][0].upper(), ha='center', va='center',
                        fontsize=7.5, color=INK if ok else 'white', fontweight='bold')
        ax.axvline(cs.index(min(c for c in cs if c > 0)) + 0.5, color=INK2, lw=1)
        ax.set_xticks(range(len(cs)), [f'{c:+g}' for c in cs], rotation=60, fontsize=7.5)
        ax.set_xlim(-0.6, len(cs) - 0.4)
        ax.set_title(f'S1 drop, friction {"on (µ=1)" if fr else "off"}', loc='left')
        ax.grid(False)
    axes[0].set_yticks(range(len(labels)), [_short(lbl) for lbl in labels], color=INK2)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    fig.suptitle(f'{shape}: P = passes through, B = blocked on top, S = stuck partway. '
                 f'Red = wrong for the true geometry (MULTICCD on).',
                 x=0.01, ha='left', fontsize=9, color=INK2)
    fig.supxlabel('radial clearance c (µm): left of the line the peg must pass, '
                  'right of it the peg must stay on top', fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f'drop_{shape}.png', dpi=160)
    plt.close(fig)


def fig_speed(speed, out, scenario='wall', nworld=GPU_NWORLD):
    """Per representation: CPU µs per step (1 env, 1 thread; ring = collision
    share) and GPU µs per env-step with nworld Warp worlds."""
    cs = speed.get('cpu_single')
    if cs is None:
        return
    shapes = [s for s in ('circle', 'square') if s in set(cs['shape'])]
    fig, axes = plt.subplots(len(shapes), 2, figsize=(11, 4.6 * len(shapes)), squeeze=False)
    for row, shape in zip(axes, shapes):
        d = cs[(cs['shape'] == shape) & (cs.scenario == scenario) & cs.multiccd]
        labels = _rep_order(d)
        y = np.arange(len(labels))[::-1]
        for ax, backend in zip(row, ('cpu', 'gpu')):
            for yi, label in zip(y, labels):
                m = d[d.label == label].method.iloc[0]
                c = COLOR[m]
                v = _cost(speed, shape, label, backend, scenario, nworld=nworld)
                if v is None:
                    continue
                if backend == 'cpu':
                    coll = float(d[d.label == label].collision_us.iloc[0])
                    ax.plot([coll, v], [yi, yi], color=c, lw=1.5, zorder=2)
                    ax.plot(coll, yi, MARKER[m], ms=6, mfc='#fcfcfb', mec=c, mew=1.5, zorder=3)
                ax.plot(v, yi, MARKER[m], ms=7, color=c, mec='#fcfcfb', mew=1.2, zorder=3)
                ax.text(v * 1.12, yi, f'{v:.0f}' if v >= 100 else f'{v:.3g}', va='center',
                        fontsize=7.5, color=INK2)
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.tick_params(which='minor', labelsize=7)
            ax.set_xlim(right=ax.get_xlim()[1] * 1.6)
            ax.set_yticks(y, [_short(lbl) for lbl in labels] if backend == 'cpu' else [],
                          color=INK2)
            ax.grid(axis='y', visible=False)
            if backend == 'cpu':
                ax.set_xlabel('CPU µs per step, 1 env, 1 thread  (ring = collision only)')
                ax.set_title(f'{shape}: CPU, "{scenario}" scenario', loc='left')
            else:
                ax.set_xlabel(f'GPU µs per env-step, {nworld} Warp worlds')
                ax.set_title(f'{shape}: GPU (MuJoCo Warp)', loc='left')
    fig.tight_layout()
    fig.savefig(out / 'speed.png', dpi=160)
    plt.close(fig)


def _accuracy(p, shape, label, c_ref, mccd=True, backend='cpu'):
    """(worst |settled error| over S2-S4 at clearance c_ref, clearance filled?),
    from CPU MuJoCo or, for backend 'gpu', from the same probes on MuJoCo Warp."""
    probes = ('warp_lateral', 'warp_tilt') if backend == 'gpu' else ('lateral', 'tilt', 'yaw')
    parts = [df[(df['shape'] == shape) & (df.label == label) & (df.multiccd == mccd)
                & (df.clearance_um == c_ref)]
             for df in map(p.get, probes) if df is not None]
    parts = [x for x in parts if len(x)]
    # Warp runs that overflowed a buffer dropped contacts: not physics.
    if not parts or any(x.get('warp_invalid', pd.Series(False)).any() for x in parts):
        return None
    if any(x.overlap_at_start.any() for x in parts):
        return c_ref, True  # error >= c; pinned to the clearance line
    return max(max(x.error_um.abs().max() for x in parts), 0.05), False


def _cost(speed, shape, label, backend, scenario='wall', mccd=True, nworld=GPU_NWORLD):
    """CPU: µs per step on 1 thread. GPU: µs per env-step at nworld Warp worlds."""
    if backend == 'cpu':
        df = speed.get('cpu_single')
        col, scale = 'us_per_step', 1.0
    else:
        df = speed.get('warp')
        col, scale = 'env_steps_per_s', None
    if df is None or col not in df:
        return None
    pick = df[(df['shape'] == shape) & (df.label == label) & (df.multiccd == mccd)
              & (df.scenario == scenario)]
    if backend == 'gpu':
        pick = pick[pick.nworld == nworld]
    pick = pick.dropna(subset=[col])
    if pick.empty:
        return None
    v = float(pick[col].iloc[0])
    return v * scale if scale else 1e6 / v


def _pareto_front(pts):
    """Settings not beaten on both time (x) and error (y), lower being better
    for both, ordered by time."""
    front, best = [], np.inf
    for q in sorted(pts, key=lambda q: (q[0], q[1])):
        if q[1] < best:
            front.append(q)
            best = q[1]
    return front


def _setting_tag(label: str, mccd: bool) -> str:
    """Short on-plot label for a front point: the method's knob values, plus
    MULTICCD off."""
    from .representations import circle_sides
    kv = dict(re.findall(r'(\w+)=([\d.e+-]+)', label))
    if 'chord_error_um' in kv:
        tag = f'n={circle_sides(0.008, float(kv["chord_error_um"]) * 1e-6)}'
    elif 'threshold' in kv:
        tag = f'thr {kv["threshold"]}'
    elif 'max_hulls' in kv:
        tag = f'{kv["max_hulls"]} hulls, {float(kv.get("resolution", 4e5)) / 1e6:g}M vox'
    elif 'depth' in kv:
        tag = f'depth {kv["depth"]}' + (f', {kv["initpoints"]} pts' if 'initpoints' in kv else '')
    else:
        tag = ''
    if not tag:
        return '' if mccd else 'MULTICCD off'
    return tag + (', off' if not mccd else '')


def _xval(p, speed, shape, label, xkind, mccd, nworld=GPU_NWORLD):
    """x for a Pareto point: 'cpu' / 'gpu' compute time, or 'memory' (compiled
    model size, MB; identical for both MULTICCD settings)."""
    if xkind != 'memory':
        return _cost(speed, shape, label, xkind, mccd=mccd, nworld=nworld)
    mem = p.get('memory')
    if mem is None:
        return None
    pick = mem[(mem['shape'] == shape) & (mem.label == label)]
    return float(pick.model_MB.iloc[0]) if len(pick) else None


XLABEL = {'cpu': 'CPU µs per step, 1 env, 1 thread',
          'gpu': 'GPU µs per env-step, {nworld} Warp worlds',
          'memory': 'compiled model memory (MB)'}


# Per-method label placement (points, points, alignment) so fronts that
# cross do not print on top of each other.
LABEL_OFFSET = {'ours': (5, -8, 'left'), 'coacd': (5, -9, 'left'),
                'vhacd': (5, 8, 'left'), 'sdf': (-5, 8, 'right')}


def _pareto_points(p, speed, backend, shape, c_ref, method, nworld=GPU_NWORLD):
    """(x, error, clearance filled?, label, multiccd) for each setting of one
    method: its granularity knob x MULTICCD (SDF ignores MULTICCD). On the GPU
    axis the error is measured on Warp too."""
    lat = p['lateral']
    pts = []
    for label in _rep_order(lat[(lat['shape'] == shape) & (lat.method == method)]):
        for mccd in (True, False) if method != 'sdf' else (True,):
            cost = _xval(p, speed, shape, label, backend, mccd, nworld)
            acc = _accuracy(p, shape, label, c_ref, mccd, backend)
            if cost is not None and acc is not None:
                pts.append((cost, *acc, label, mccd))
    return pts


def _pareto_panel(ax, p, speed, backend, shape, c_ref, nworld=GPU_NWORLD):
    for m in METHODS:
        pts = _pareto_points(p, speed, backend, shape, c_ref, m, nworld)
        if not pts:
            continue
        front = _pareto_front(pts)
        ax.plot([q[0] for q in front], [q[1] for q in front], '-', color=COLOR[m], lw=1.5,
                zorder=2)
        for q in pts:
            x, yv, filled, label, mccd = q
            on_front = q in front
            ax.plot(x, yv, MARKER[m], ms=6 if on_front else 4, zorder=3, color=COLOR[m],
                    alpha=1.0 if on_front else 0.3, mfc='#fcfcfb' if filled else COLOR[m],
                    mec=COLOR[m] if filled else '#fcfcfb', mew=1.5 if filled else 1.0)
        # Label front settings, skipping any within 0.15 decades of time of the
        # previous label, or less than 10% more accurate (a flat step), so
        # dense stretches of a front stay legible.
        last_x = last_y = None
        for x, yv, filled, label, mccd in front:
            tag = _setting_tag(label, mccd)
            if filled or not tag or (last_x and (abs(np.log10(x / last_x)) < 0.15
                                                 or yv > 0.9 * last_y)):
                continue
            dx, dy, ha = LABEL_OFFSET[m]
            ax.annotate(tag, (x, yv), xytext=(dx, dy), textcoords='offset points',
                        ha=ha, va='center', fontsize=6.5, color=COLOR[m])
            last_x, last_y = x, yv
    ax.set_xscale('log')
    ax.set_yscale('log')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        axis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 5)))
        axis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.tick_params(which='minor', labelsize=7)
    ax.axhline(c_ref, color=MUTED, lw=1, ls=':')
    ax.text(0.99, c_ref * 1.12, 'error = clearance; hollow = clearance filled',
            transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED,
            va='bottom', ha='right')
    ax.set_xlabel(XLABEL[backend].format(nworld=nworld))
    ax.set_ylabel('worst |error| over S2–S4 (µm)' + (', on Warp' if backend == 'gpu' else ''))
    ax.set_title(f'{shape}, c = {c_ref:g} µm (clearance ratio {ratio(c_ref)})'
                 + ('' if backend == 'memory' else f', {backend.upper()}'), loc='left')


def fig_pareto(p, speed, out, layout, name, nworld=GPU_NWORLD):
    """Compute time vs error: each method is its Pareto front over all its
    settings (lower-left is better). layout: rows of (backend, shape, c_um)."""
    if p.get('lateral') is None:
        return
    layout = [[cell for cell in row
               if any(_pareto_points(p, speed, *cell, m, nworld) for m in METHODS)]
              for row in layout]
    layout = [row for row in layout if row]
    if not layout:
        return
    ncol = max(len(row) for row in layout)
    fig, axes = plt.subplots(len(layout), ncol, squeeze=False,
                             figsize=(6.2 * ncol, 4.8 * len(layout)))
    for r, row in enumerate(layout):
        for c, cell in enumerate(row):
            _pareto_panel(axes[r][c], p, speed, *cell, nworld)
        for c in range(len(row), ncol):
            axes[r][c].set_visible(False)
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], lw=1.5, label=NAMES[m])
               for m in METHODS]
    fig.legend(handles=handles, fontsize=8, loc='upper center', ncol=len(handles),
               bbox_to_anchor=(0.5, 0.955 if len(layout) > 1 else 0.925))
    what = 'Model memory' if all(c[0] == 'memory' for row in layout for c in row) \
        else 'Compute time'
    fig.suptitle(f'{what} vs error; lower-left is better. Each line is the Pareto front of '
                 'one method over its settings (granularity × MULTICCD); faint points are '
                 'dominated settings.', x=0.01, ha='left', fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if len(layout) > 1 else 0.87))
    fig.savefig(out / name, dpi=160)
    plt.close(fig)


def fig_backends(p, out, c_ref=100.0):
    """Same S2 scenes on CPU MuJoCo vs MuJoCo Warp: settled play error, and the
    worst true |sd| at contact-reporting states."""
    cpu, gpu = p.get('lateral'), p.get('warp_lateral')
    if cpu is None or gpu is None:
        return
    metrics = [('worst |S2 settled error|', lambda d: d.error_um.abs().max()),
               ('worst |true sd| at reported contacts',
                lambda d: np.nanmax([d.sd_false_max_um.max(), d.sd_pen_max_um.max()]))]
    if 'sd_false_max_um' not in gpu:
        metrics = metrics[:1]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8), squeeze=False)
    lim = [0.05, 1e3]
    for ax, (name, metric) in zip(axes[0], metrics):
        for shape, fill in (('circle', True), ('square', False)):
            for label in dict.fromkeys(gpu.loc[gpu['shape'] == shape, 'label']):
                for mccd in (True, False):
                    pick = lambda df: df[(df['shape'] == shape) & (df.label == label)  # noqa: E731
                                         & (df.multiccd == mccd) & (df.clearance_um == c_ref)]
                    c, g = pick(cpu), pick(gpu)
                    if c.empty or g.empty or c.overlap_at_start.any() or g.warp_invalid.any():
                        continue
                    x, y = metric(c), metric(g)
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    m = c.method.iloc[0]
                    ax.plot(max(x, 0.05), max(y, 0.05), MARKER[m], ms=6, color=COLOR[m],
                            mfc=COLOR[m] if fill else '#fcfcfb', mec=COLOR[m], mew=1.5)
        ax.plot(lim, lim, color=MUTED, lw=1, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f'CPU MuJoCo (µm)')
        ax.set_ylabel('MuJoCo Warp (µm)')
        ax.set_title(f'{name}, S2, c = {c_ref:g} µm', loc='left')
    handles = [plt.Line2D([], [], ls='', marker=MARKER[m], color=COLOR[m], label=NAMES[m])
               for m in METHODS]
    handles += [plt.Line2D([], [], ls='', marker='o', color=INK2, label='circle (filled)'),
                plt.Line2D([], [], ls='', marker='o', mfc='#fcfcfb', mec=INK2,
                           label='square (hollow)')]
    axes[0][0].legend(handles=handles, fontsize=7.5, loc='upper left')
    fig.suptitle('Same scenes, CPU vs GPU physics: above the dotted line = Warp less accurate '
                 '(clearance-filled settings omitted)', x=0.01, ha='left', fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / 'cpu_vs_warp.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=tables.ROOT,
                    help='raw results root (default: benchmark/results)')
    ap.add_argument('--out', type=Path, default=HERE / 'report')
    ap.add_argument('--gpu-nworld', type=int, default=GPU_NWORLD,
                    help='MuJoCo Warp batch size of the GPU timings in the summaries and '
                         'figures (measured: 256 and 512)')
    ap.add_argument('--no-figures', action='store_true', help='tables only')
    args = ap.parse_args(argv)
    out, nw = args.out, args.gpu_nworld
    out.mkdir(parents=True, exist_ok=True)
    p, speed = load(args.results)
    figures = not args.no_figures
    md = ['# Collision-representation benchmark: summary\n']
    for shape in ('circle', 'square'):
        if p['lateral'] is None or shape not in set(p['lateral']['shape']):
            continue
        table = summarize(p, speed, shape, nworld=nw)
        table.to_csv(out / f'summary_{shape}.csv', index=False)
        md += [f'## {shape}\n', table.to_markdown(index=False), '\n']
        for mccd in (True, False) if figures else ():
            fig_accuracy(p, shape, out, mccd=mccd)
        if figures:
            fig_drop(p, shape, out)
    st = speed_table(p, speed)
    if len(st):
        st.to_csv(out / 'speed_table.csv', index=False)
        (out / 'speed_table.md').write_text(
            '# Real-time factor at the 1 ms step, "wall" scenario\n\n'
            'CPU: one environment on one thread. GPU: summed over the MuJoCo Warp batch. '
            'Worst error: worst settled |error| over S2-S4 on CPU, µm.\n\n'
            + st.to_markdown(index=False) + '\n')
    if figures:
        fig_speed(speed, out, nworld=nw)
        fig_pareto(p, speed, out, [[('cpu', 'circle', 100.0), ('cpu', 'circle', 20.0)]],
                   'pareto_cpu_circle.png', nw)
        fig_pareto(p, speed, out, [[('memory', 'circle', 100.0), ('memory', 'circle', 20.0)]],
                   'pareto_memory_circle.png', nw)
        fig_pareto(p, speed, out, [[('gpu', 'circle', 100.0), ('gpu', 'circle', 20.0)]],
                   'pareto_gpu_circle.png', nw)
        for c in (100.0, 20.0):
            fig_pareto(p, speed, out, [[(b, shape, c) for shape in ('circle', 'square')]
                                       for b in ('cpu', 'gpu')], f'pareto_c{c:g}.png', nw)
        fig_backends(p, out)
    (out / 'summary.md').write_text('\n'.join(md))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
