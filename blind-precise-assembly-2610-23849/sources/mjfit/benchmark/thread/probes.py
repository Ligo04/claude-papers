"""Passive bolt/nut probes with geometric ground truth (CPU MuJoCo).

The bolt is fixed; the nut starts centred in its play. No actuators: the nut
moves only under gravity or a constant applied wrench, on plain joints.

  T1 axial  : axial slide only, frictionless, force along +z and -z. Settled
              travel vs the exact axial play 2c/sqrt3.
  T2 radial : lateral slides only, frictionless, force along theta. Settled
              travel vs the exact radial play c/2 (nut tip on bolt core).
  T3 tilt   : hinge through the nut centre, frictionless, torque. Settled
              tilt vs the exact maximum tilt; reported at the nut's end faces.
  T4 screw  : axial slide + spin hinge, frictionless; a torque turns the nut
              up the thread against a 0.1 N axial load. Exact: the nut follows
              the helix z = P * angle / 2pi, resting on its lower flanks, i.e.
              offset -2c/sqrt3 from the centred helix.
  T5 spin   : free nut on the vertical bolt under gravity. Frictionless: it
              must spin down with the exact acceleration
              g / (1 + I (2pi/P)^2 / m). At friction 0.3 (above the 3 deg lead
              angle) the thread is self-locking: after settling onto its
              flanks it must not move.

The nut on the bolt's tip (initial engagement):

  T6 start  : axial slide + spin hinge, frictionless; the nut rests on the
              bolt's top face under T4's axial load, and a constant torque
              turns it. 'back' unscrews it: the back-spin click, where it drops
              into the thread by its axial play and rides the flank back up.
              'on' screws it: it starts down the thread and follows the helix
              for a turn. Exact: the quasi-static path from geometry.Tip.
  T7 partial: T2 and T3 with the nut threaded only 1/4 to 2 turns onto the
              tip, the hinge half-way along the engaged length. Exact, on the
              sampled exact surfaces (tip faces included): radial, the reach
              of the free sideways translations (the nut slides around a
              contact off the push, as the peg's S2); tilt, the first contact
              about the hinge.

Friction:

  T8 torque : the engaged nut (axial slide + spin hinge, no joint damping)
              pressed down by an axial load, with friction mu; the torque
              ramps slowly until the nut turns 5 deg, against it ('up') or
              with it ('down'). Reference: the power-screw formula
              (geometry.screw_torque), textbook rather than exact.

Errors are micrometres of wall position: positive means the nut went further
than the geometry allows (penetration), negative that it was stopped early
(phantom material). T1-T4 also record the true signed distance at every
reported contact, as in the peg probes.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from mujoco import rollout

from ..geometry import contact_stats
from ..probes import FULL, XFRC, _window_stats
from . import geometry as geo
from .geometry import Pair, ThreadTask
from .representations import ThreadRep
from .scene import build_scene, pair_contacts

UM = 1e6
FORCE, TORQUE, SCREW_LOAD, SCREW_TORQUE = 0.1, 1.5e-3, 0.1, 1.2e-4
SPIN_FRICTION = 0.3
_PAIRS: dict = {}


def _pair(task: ThreadTask, c: float) -> Pair:
    """Coarse sampled surfaces for the contact signed distance."""
    key = (task, c)
    if key not in _PAIRS:
        _PAIRS[key] = Pair(task, c, spacing=1e-4)
    return _PAIRS[key]


def exact_tilt(task: ThreadTask, c: float, phi: float) -> float:
    """Exact maximum tilt, cached on disk (it takes ~20 s): the runner
    computes it once before starting its worker processes."""
    import json
    from ..representations import CACHE_DIR
    key = json.dumps([task.key(), c, round(phi, 9)], sort_keys=True)
    path = CACHE_DIR / 'thread_tilt.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    if key not in cache:
        cache[key] = geo.max_tilt(Pair(task, c, spacing=5e-5), phi)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(cache, indent=1))
        tmp.replace(path)
    return cache[key]


def axial_play(task: ThreadTask, c: float, sign: int = -1) -> float:
    """Exact axial play along sign·z (2c/√3 each way for ISO)."""
    return geo.axial_gap(task, c, sign)


def _base(task, rep: ThreadRep, multiccd: bool) -> dict:
    return {'shape': task.shape, 'method': rep.method, 'label': rep.label,
            'clearance_um': rep.clearance * UM, 'multiccd': multiccd}


def _settle(sc, wrenches: np.ndarray, duration: float):
    """One world per constant wrench on the nut; qpos over the second half
    of the run, and whether the parts overlapped at the start."""
    m = sc.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    overlap0 = any(c[0] < 0 for c in pair_contacts(sc, d))
    state0 = np.empty(mujoco.mj_stateSize(m, FULL))
    mujoco.mj_getState(m, d, state0, FULL)
    nstep = round(duration / m.opt.timestep)
    control = np.zeros((len(wrenches), nstep, 6 * m.nbody))
    control[:, :, 6 * sc.nut_body:6 * sc.nut_body + 6] = wrenches[:, None, :]
    states, _ = rollout.rollout(m, [mujoco.MjData(m)], np.tile(state0, (len(wrenches), 1)),
                                control, control_spec=XFRC, nstep=nstep)
    return states[:, nstep // 2:, 1:1 + m.nq], overlap0


def _contact_sd(sc, task, c, qpos: np.ndarray, every: int = 25) -> list[np.ndarray]:
    """True signed distance at every `every`-th settled state with a
    reported bolt-nut contact, one array per world."""
    m, d = sc.model, mujoco.MjData(sc.model)
    pair = _pair(task, c)
    out = []
    for world in qpos:
        sd = []
        for q in world[::every]:
            d.qpos[:] = q
            mujoco.mj_fwdPosition(m, d)
            if pair_contacts(sc, d):
                sd.append(pair.signed_distance(d.xpos[sc.nut_body].copy(),
                                               d.xmat[sc.nut_body].reshape(3, 3).copy()))
        out.append(np.array(sd))
    return out


def probe_axial(task: ThreadTask, rep: ThreadRep, duration: float = 1.0,
                multiccd: bool = True) -> list[dict]:
    c = rep.clearance
    sc = build_scene(task, rep, joints='z', multiccd=multiccd)
    signs = np.array([1.0, -1.0])
    wrench = np.zeros((2, 6))
    wrench[:, 2] = FORCE * signs
    qpos, overlap0 = _settle(sc, wrench, duration)
    sds = _contact_sd(sc, task, c, qpos)
    return [{**_base(task, rep, multiccd), 'direction': int(s),
             'truth_um': axial_play(task, c, int(s)) * UM,
             **_window_stats((s * qpos[i, :, 0] - axial_play(task, c, int(s))) * UM),
             'overlap_at_start': overlap0, **contact_stats(sds[i])}
            for i, s in enumerate(signs)]


def probe_radial(task: ThreadTask, rep: ThreadRep, n_dir: int = 12, duration: float = 1.0,
                 multiccd: bool = True) -> list[dict]:
    c = rep.clearance
    sc = build_scene(task, rep, joints='xy', multiccd=multiccd)
    a = np.linspace(0, 2 * math.pi, n_dir, endpoint=False)
    u = np.column_stack([np.cos(a), np.sin(a)])
    wrench = np.zeros((n_dir, 6))
    wrench[:, :2] = FORCE * u
    qpos, overlap0 = _settle(sc, wrench, duration)
    reach = np.einsum('itj,ij->it', qpos[:, :, :2], u)
    sds = _contact_sd(sc, task, c, qpos)
    return [{**_base(task, rep, multiccd), 'theta': float(a[i]), 'truth_um': c / 2 * UM,
             **_window_stats((reach[i] - c / 2) * UM), 'overlap_at_start': overlap0,
             **contact_stats(sds[i])}
            for i in range(n_dir)]


def probe_tilt(task: ThreadTask, rep: ThreadRep, phi: float = 0.0, duration: float = 1.0,
               multiccd: bool = True) -> dict:
    c = rep.clearance
    axis = np.array([math.cos(phi), math.sin(phi), 0.0])
    sc = build_scene(task, rep, joints='tilt', tilt_axis=axis, multiccd=multiccd)
    qpos, overlap0 = _settle(sc, np.concatenate([np.zeros(3), TORQUE * axis])[None], duration)
    truth = exact_tilt(task, c, phi)
    theta = qpos[0, :, 0]
    return {**_base(task, rep, multiccd), 'phi_deg': math.degrees(phi),
            'tilt_mrad': float(np.median(theta)) * 1e3, 'truth_mrad': truth * 1e3,
            **_window_stats((theta - truth) * task.nut_height / 2 * UM),
            'overlap_at_start': overlap0,
            **contact_stats(_contact_sd(sc, task, c, qpos)[0])}


def probe_screw(task: ThreadTask, rep: ThreadRep, duration: float = 1.0,
                multiccd: bool = True) -> dict:
    """Torque up the thread against an axial load, frictionless."""
    c = rep.clearance
    sc = build_scene(task, rep, joints='screw', multiccd=multiccd)
    wrench = np.array([[0, 0, -SCREW_LOAD, 0, 0, SCREW_TORQUE]])
    qpos, overlap0 = _settle(sc, wrench, duration)
    z, angle = qpos[0, :, 0], qpos[0, :, 1]
    residual = z - geo.helix_z(task, angle)                 # offset from the centred helix
    turned = angle[-1] - angle[0]
    slope = np.polyfit(angle, z, 1)[0] if abs(turned) > 1e-3 else float('nan')
    # A skipped (or slipped) thread moves the nut off the helix by a sizeable
    # fraction of a pitch; the exact nut stays on one flank.
    skips = residual.max() - residual.min() > task.pitch / 4
    outcome = ('jammed' if abs(turned) < 0.1 else 'skips' if skips else 'follows')
    return {**_base(task, rep, multiccd), 'outcome': outcome,
            'turns_in_window': turned / (2 * math.pi),
            'lead_ratio': slope / (task.hand * task.lead / (2 * math.pi)),
            'truth_um': -axial_play(task, c) * UM,
            # Below its exact seat on the lower flanks is further than the
            # geometry allows (positive), above is phantom material (negative).
            **_window_stats(-(residual + axial_play(task, c)) * UM), 'overlap_at_start': overlap0,
            **contact_stats(_contact_sd(sc, task, c, qpos)[0])}


def probe_spin(task: ThreadTask, rep: ThreadRep, friction: float = 0.0, duration: float = 0.8,
               multiccd: bool = True, start_angle: float = 0.0) -> dict:
    """Free nut on the vertical bolt under gravity, started 2 pitches up
    (and `start_angle` further along the helix)."""
    c = rep.clearance
    sc = build_scene(task, rep, joints='free', gravity=True, friction=friction,
                     nut_offset=(0, 0, 2 * task.pitch), multiccd=multiccd, nut_angle=start_angle)
    m, d = sc.model, mujoco.MjData(sc.model)
    mujoco.mj_forward(m, d)
    overlap0 = any(con[0] < 0 for con in pair_contacts(sc, d))
    nstep = round(duration / m.opt.timestep)
    t = np.empty(nstep)
    z = np.empty(nstep)
    for i in range(nstep):
        mujoco.mj_step(m, d)
        t[i], z[i] = d.time, d.qpos[2]
    z0 = task.nut_home() + 2 * task.pitch + geo.helix_z(task, start_angle)
    a_exact = geo.spin_down_accel(task, sc.mass, sc.inertia_axis)
    tilt = math.degrees(math.acos(min(1.0, abs(1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)))))
    row = {**_base(task, rep, multiccd), 'friction': friction, 'overlap_at_start': overlap0,
           'tilt_deg': tilt, 'drop_mm': (z0 - z[-1]) * 1e3}
    if friction == 0:
        late = t > 0.2 * duration
        a_sim = -2 * np.polyfit(t[late], z[late], 2)[0] if np.isfinite(z).all() else float('nan')
        row.update(accel_mm_s2=a_sim * 1e3, accel_exact_mm_s2=a_exact * 1e3,
                   accel_error_pct=(a_sim / a_exact - 1) * 100)
    else:
        # Self-locking: the exact nut drops onto its lower flanks and stays.
        row.update(slip_um=(z0 - axial_play(task, c) - z[-1]) * UM)
    return row


# ----- T6, T7: the nut on the bolt's tip -------------------------------------------------------

# T6 drives with this multiple of the exact frictionless back-off torque
# (load x rise per radian); 5 is T4's for M10. The spin damping sets the
# speed on the flat top face, where nothing else resists.
START_TORQUE_RATIO, START_SPEED = 5.0, 1.0      # -, rad/s
START_LEAD_IN = math.pi / 2                     # turned on the flat before the event
START_FALL = math.radians(3)                    # after a drop, still falling: not scored
PARTIAL_TURNS = (0.25, 0.5, 1.0, 2.0)
PARTIAL_RADIAL = 12                              # directions
PARTIAL_TILT = 4                                 # hinge axes
_TIPS: dict = {}


def _tip(task: ThreadTask, c: float) -> geo.Tip:
    if (task, c) not in _TIPS:
        _TIPS[(task, c)] = geo.Tip(task, c)
    return _TIPS[(task, c)]


def _tip_pair(task: ThreadTask, c: float) -> Pair:
    """Coarse sampled tip surfaces, for the contact signed distance."""
    key = (task, c, 'tip')
    if key not in _PAIRS:
        reach = 2 * task.lead + task.chamfer + 1e-3
        _PAIRS[key] = Pair(task, c, spacing=1e-4, tip=reach)
    return _PAIRS[key]


def _start_path(task: ThreadTask, c: float, direction: str):
    """Exact T6 path: (start angle, rotation sign, angles, heights, event
    progress, event size). Progress is the angle turned, in rad."""
    tip = _tip(task, c)
    spin = task.hand * (1 if direction == 'back' else -1)
    if direction == 'back':
        event, drop = tip.click()
        # the drop, the ride back up the flank (the play over the rise), then more flat
        after = (drop / (task.lead / (2 * math.pi))) + START_LEAD_IN
    else:
        event, drop = tip.entry(), 0.0
        after = 2 * math.pi                     # a full turn down the thread
    th0 = event - spin * START_LEAD_IN
    progress = np.arange(0.0, START_LEAD_IN + after, math.radians(0.1))
    z, _ = tip.path(th0 + spin * progress)
    return th0, spin, progress, z, START_LEAD_IN, drop


def probe_start(task: ThreadTask, rep: ThreadRep, direction: str = 'back',
                multiccd: bool = True, torque_ratio: float = START_TORQUE_RATIO,
                sample_dt: float = 2e-3, trace: bool = False):
    """T6: turn the nut resting on the bolt's tip ('back': unscrewing, 'on':
    screwing), against the exact quasi-static path. With `trace`, also
    returns the simulated and exact heights (from the flat) against the angle
    turned past the exact event."""
    from .scene import _inertia
    c = rep.clearance
    th0, spin, prog_x, z_x, p_event, drop = _start_path(task, c, direction)
    rise = task.lead / (2 * math.pi)
    load = SCREW_LOAD
    torque = torque_ratio * load * rise
    i_axis = _inertia(task, c)[3]
    sc = build_scene(task, rep, joints='screw', multiccd=multiccd, nut_angle=th0,
                     nut_offset=(0, 0, z_x[0] + 2e-6 - task.nut_home() - geo.helix_z(task, th0)),
                     settle_time=i_axis * START_SPEED / torque)
    m, d = sc.model, mujoco.MjData(sc.model)
    mujoco.mj_forward(m, d)
    overlap0 = any(con[0] < 0 for con in pair_contacts(sc, d))
    # Slowest exact speed: climbing a flank, (ratio - 1) / ratio of the flat's.
    duration = 1.25 * prog_x[-1] / (START_SPEED * (torque_ratio - 1) / torque_ratio)
    every = max(1, round(sample_dt / m.opt.timestep))
    wrench = np.array([0, 0, -load, 0, 0, spin * torque])
    prog, z, qs = [], [], []
    for i in range(round(duration / m.opt.timestep)):
        d.xfrc_applied[sc.nut_body] = wrench
        mujoco.mj_step(m, d)
        if i % every == 0:
            prog.append(spin * d.qpos[1])
            z.append(d.xpos[sc.nut_body][2])
            qs.append(d.qpos.copy())
            if prog[-1] > prog_x[-1] or not np.isfinite(z[-1]):
                break
    prog, z = np.array(prog), np.array(z)
    flat = z_x[0]
    # Event angles (the click, or the start down the thread), located the
    # same way on the exact and simulated curves; also as axial phase along
    # the helix.
    level = flat - (drop / 2 if direction == 'back' else task.pitch / 10)
    below = np.flatnonzero(z < level)
    below_x = np.flatnonzero(z_x < level)
    p_sim = float(prog[below[0]]) if len(below) else math.nan
    p_exact = float(prog_x[below_x[0]])
    # Exact height at each sampled angle; positive error: lower than the
    # geometry allows (into the bolt), negative: held up. Not scored: the
    # settling from the start gap, and the falls: from the earlier of the
    # exact and simulated clicks to the later (the click angle is scored as
    # event_err instead).
    err = np.interp(prog, prog_x, z_x) - z
    keep = (prog >= math.radians(1)) & (prog <= prog_x[-1]) & np.isfinite(z)
    for p in prog_x[1:][np.diff(z_x) < -1e-6]:
        lo, hi = (min(p, p_sim), max(p, p_sim)) if direction == 'back' and len(below) else (p, p)
        keep &= ~((prog >= lo - START_FALL) & (prog < hi + START_FALL))
    row = {**_base(task, rep, multiccd), 'direction': direction, 'torque_ratio': torque_ratio,
           'turned': float(prog[-1] / (2 * math.pi)) if len(prog) else 0.0,
           'overlap_at_start': overlap0,
           **(_window_stats(err[keep] * UM) if keep.any() else {})}
    row['event_err_deg'] = math.degrees(p_sim - p_exact)
    row['event_err_um'] = (p_sim - p_exact) * rise * UM
    ended = prog[-1] if len(prog) else 0.0
    if direction == 'back':
        row['drop_exact_um'] = drop * UM
        after = (prog >= p_sim) & (prog < p_sim + drop / rise) if len(below) else prog < 0
        row['drop_um'] = (flat - z[after].min()) * UM if after.any() else 0.0
        outcome = ('no click' if not len(below) else
                   'falls' if row['drop_um'] > (drop + task.pitch / 4) * UM else
                   'jams' if ended < 0.95 * prog_x[-1] else 'clicks')
    else:
        outcome = ('no entry' if not len(below) else
                   'jams' if ended < 0.95 * prog_x[-1] else
                   'skips' if np.abs(err[keep]).max() > task.pitch / 4 else 'engages')
    row['outcome'] = outcome
    row['stopped_deg'] = math.degrees(ended - p_exact) if len(below_x) else math.nan
    # true signed distance at contact states
    pair, sds = _tip_pair(task, c), []
    for q in qs[::25]:
        d.qpos[:] = q
        mujoco.mj_fwdPosition(m, d)
        if pair_contacts(sc, d):
            sds.append(pair.signed_distance(d.xpos[sc.nut_body].copy(),
                                            d.xmat[sc.nut_body].reshape(3, 3).copy()))
    row = {**row, **contact_stats(np.array(sds))}
    if not trace:
        return row
    return row, {'deg': np.degrees(prog - p_exact), 'z_um': (z - flat) * UM,
                 'deg_exact': np.degrees(prog_x - p_exact), 'z_exact_um': (z_x - flat) * UM}


def partial_jobs(task: ThreadTask):
    """(turns, kind, angle) of every exact T7 answer."""
    out = []
    for turns in PARTIAL_TURNS:
        out += [(turns, 'radial', float(a))
                for a in np.linspace(0, 2 * math.pi, PARTIAL_RADIAL, endpoint=False)]
        out += [(turns, 'tilt', float(a))
                for a in np.linspace(0, 2 * math.pi, PARTIAL_TILT, endpoint=False)]
    return out


def _partial_key(task, c, turns, kind, angle):
    import json
    # radial answers are supports of the free set (the first ones were rays)
    # c rounded: the runner's 100 * 1e-6 is not 1e-4
    return json.dumps([task.key(), round(c, 12), turns,
                       'radial support' if kind == 'radial' else kind, round(angle, 9)],
                      sort_keys=True)


def compute_partial(task: ThreadTask, c: float, turns: float, kind: str, angle: float) -> float:
    """One exact T7 answer (seconds to minutes: the sampled surfaces)."""
    reach = turns * task.lead + task.chamfer + 2e-3
    pair = Pair(task, c, spacing=5e-5, tip=reach)
    if kind == 'radial':
        coarse = Pair(task, c, spacing=1e-4, tip=reach)    # to find the best ray
        return geo.partial_radial_play(coarse, turns, angle, fine=pair)
    return geo.partial_max_tilt(pair, turns, angle, turns * task.lead / 2)


def exact_partial(task: ThreadTask, c: float, turns: float, kind: str, angle: float,
                  store: dict | None = None) -> float:
    """Exact T7 answer, cached on disk (the runner fills the cache in
    parallel before starting its workers); `store` adds computed answers."""
    import json
    from ..representations import CACHE_DIR
    path = CACHE_DIR / 'thread_partial.json'
    key = _partial_key(task, c, turns, kind, angle)
    cache = json.loads(path.read_text()) if path.exists() else {}
    if store:
        cache.update(store)
    elif key in cache:
        return cache[key]
    else:
        cache[key] = compute_partial(task, c, turns, kind, angle)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{id(cache)}.tmp')
    tmp.write_text(json.dumps(cache, indent=1))
    tmp.replace(path)
    return cache.get(key, math.nan)


def probe_partial(task: ThreadTask, rep: ThreadRep, duration: float = 1.0,
                  multiccd: bool = True) -> list[dict]:
    """T7: T2 (radial) and T3 (tilt) with the nut threaded partway onto the tip."""
    c = rep.clearance
    rows = []
    for turns in PARTIAL_TURNS:
        _, a = geo.partial_pose(task, turns)
        Rz = geo.rot([0, 0, 1], a)
        base = {**_base(task, rep, multiccd), 'turns': turns}
        # radial: slides along the nut's own x, y (turned by a)
        sc = build_scene(task, rep, joints='xy', multiccd=multiccd, nut_angle=a)
        th = np.linspace(0, 2 * math.pi, PARTIAL_RADIAL, endpoint=False)
        u = np.column_stack([np.cos(th), np.sin(th)])
        wrench = np.zeros((len(th), 6))
        wrench[:, :2] = FORCE * u
        qpos, overlap0 = _settle(sc, wrench, duration)
        reach = np.einsum('itj,ij->it', qpos[:, :, :2] @ Rz[:2, :2].T, u)
        for i, t in enumerate(th):
            truth = exact_partial(task, c, turns, 'radial', float(t))
            rows.append({**base, 'kind': 'radial', 'angle_deg': math.degrees(t),
                         'truth_um': truth * UM, **_window_stats((reach[i] - truth) * UM),
                         'overlap_at_start': overlap0})
        # tilt: one hinge per axis, half-way along the engaged length
        for phi in np.linspace(0, 2 * math.pi, PARTIAL_TILT, endpoint=False):
            axis = np.array([math.cos(phi), math.sin(phi), 0.0])
            sc = build_scene(task, rep, joints='tilt', tilt_axis=Rz.T @ axis,
                             pivot=turns * task.lead / 2, nut_angle=a, multiccd=multiccd)
            qpos, overlap0 = _settle(sc, np.concatenate([np.zeros(3), TORQUE * axis])[None],
                                     duration)
            truth = exact_partial(task, c, turns, 'tilt', float(phi))
            theta = qpos[0, :, 0]
            # at the thread's major radius: the crest's axial travel
            rows.append({**base, 'kind': 'tilt', 'angle_deg': math.degrees(phi),
                         'truth_mrad': truth * 1e3, 'tilt_mrad': float(np.median(theta)) * 1e3,
                         **_window_stats((theta - truth) * task.r_major * UM),
                         'overlap_at_start': overlap0})
    return rows


# ----- T8: turning torque against friction --------------------------------------------------

TORQUE_MU = (0.1, 0.3, 1.0)
# (axial load N, condim, friction cone, timestep s). condim 4 adds the torsional
# coefficient TORQUE_TORSION (mjfit's default torsional friction).
TORQUE_CASES = ((0.1, 3, 'pyramidal', 1e-3), (0.1, 3, 'elliptic', 1e-3),
                (0.1, 4, 'pyramidal', 1e-3), (1.0, 3, 'pyramidal', 1e-3),
                (1.0, 3, 'elliptic', 1e-3), (1.0, 3, 'elliptic', 2.5e-4))
TORQUE_TORSION = 0.01                            # m
TORQUE_TURN = math.radians(5)                    # a sustained turn: breakaway


def probe_torque(task: ThreadTask, rep: ThreadRep, mu: float, direction: str = 'up',
                 load: float = 0.1, condim: int = 3, cone: str = 'pyramidal',
                 timestep: float = 1e-3, multiccd: bool = True, ramp: float = 3.0) -> dict:
    """T8: the torque that makes the engaged nut turn. The nut sits at home,
    pressed down by `load`; the torque ramps from 0 to 3x the formula's
    raising torque over `ramp` seconds, against the load ('up') or with it
    ('down'), and breakaway is the first time the nut has turned
    TORQUE_TURN that way. Joint damping is off, so the torque only meets
    contact."""
    c = rep.clearance
    sc = build_scene(task, rep, joints='screw', friction=mu, condim=condim, multiccd=multiccd,
                     timestep=timestep)
    m = sc.model
    m.dof_damping[:] = 0.0
    m.geom_friction[:, 1] = TORQUE_TORSION if condim >= 4 else 0.0
    m.opt.cone = (mujoco.mjtCone.mjCONE_ELLIPTIC if cone == 'elliptic'
                  else mujoco.mjtCone.mjCONE_PYRAMIDAL)
    d = mujoco.MjData(m)
    up = direction == 'up'
    sign = task.hand * (1.0 if up else -1.0)          # + spin raises a right-hand nut
    exact = geo.screw_torque(task, load, mu, raising=up, c=c)
    top = 3.0 * geo.screw_torque(task, load, mu, raising=True, c=c)
    settle = 0.2
    broke, free = math.nan, 0
    for i in range(int((settle + ramp) / timestep)):
        t = i * timestep
        tau = 0.0 if t < settle else top * (t - settle) / ramp
        d.xfrc_applied[sc.nut_body] = [0, 0, -load, 0, 0, sign * tau]
        mujoco.mj_step(m, d)
        if t >= settle:
            free += d.ncon == 0        # a step that began with no contact: the nut fell freely
            if sign * d.qpos[1] > TORQUE_TURN:
                broke = tau
                break
    # A self-locking thread holds with no torque; turning down at under 5 % of
    # the lowering torque means it back-drove.
    if math.isnan(broke):
        outcome = 'holds'               # never turned within 3x the raising torque
    elif not up and exact > 0 and broke < 0.05 * exact:
        outcome = 'back-drives'
    else:
        outcome = 'turns'
    return {**_base(task, rep, multiccd), 'mu': mu, 'direction': direction, 'load_N': load,
            'condim': condim, 'cone': cone, 'timestep_ms': timestep * 1e3,
            'formula_mNm': exact * 1e3, 'breakaway_mNm': broke * 1e3,
            'ratio': broke / exact if exact > 0 else math.nan, 'outcome': outcome,
            'free_steps': int(free)}


def memory(task: ThreadTask, rep: ThreadRep) -> dict:
    sc = build_scene(task, rep, joints='screw')
    m = sc.model
    return {'shape': task.shape, 'method': rep.method, 'label': rep.label,
            'clearance_um': rep.clearance * UM, 'multiccd': True,
            'model_MB': m.nbuffer / 2**20, 'n_geoms': sc.n_geoms, 'noct': m.noct,
            'compile_s': sc.compile_s, 'setup_s': rep.setup_s}
