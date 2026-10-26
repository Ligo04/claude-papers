"""Passive physics probes with geometric ground truth (CPU MuJoCo).

No actuators: the peg is moved only by gravity, a constant applied wrench
(xfrc_applied), or placed kinematically, and constrained by plain joints.

  S0 static  : place the aligned peg at poses of known true gap; does the
               representation report contact, and how deep?
  S1 drop    : aligned, centered peg falls onto the hole under gravity.
               c > 0 must pass with zero wall contact; c < 0 must stay on top.
  S2 lateral : rotation locked, frictionless, constant lateral force along
               theta; settled displacement vs. the exact feasible translation.
  S3 tilt    : lateral slides + one tilt hinge, constant torque; settled tilt
               vs. the exact maximum tilt.
  S4 yaw     : lateral slides + z hinge (square only); settled yaw vs. exact.

Errors are reported in micrometers of wall position: positive means the
simulated peg went further than the geometry allows (penetration), negative
means it was stopped early (phantom material).

S1-S4 also record the true signed distance whenever the simulator reports
peg-plate contact (Son & Kim's min-|sd|): sd_false_max_um is the largest gap
the simulator treated as contact, sd_pen_max_um the deepest real overlap.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from mujoco import rollout

from . import geometry as geo
from .geometry import Task
from .representations import PlateRep
from .scene import build_scene, plate_contacts

UM = 1e6
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS
XFRC = mujoco.mjtState.mjSTATE_XFRC_APPLIED

S0_GAPS_UM = np.array([1000, 700, 500, 300, 200, 150, 100, 70, 50, 30, 20, 15, 10, 7, 5,
                       3, 2, 1, 0.5, 0, -0.5, -1, -2, -3, -5, -7, -10, -15, -20, -30,
                       -50, -100, -200], dtype=float)


def directions(n: int) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([np.cos(a), np.sin(a)])


def _base(task: Task, rep: PlateRep, clearance: float) -> dict:
    return {'shape': task.shape, 'method': rep.method, 'label': rep.label,
            'clearance_um': clearance * UM}


# ----- S0: static contact onset ------------------------------------------------

def probe_static(task: Task, rep: PlateRep, clearance: float = 1e-3,
                 n_dir: int = 24, gaps_um: np.ndarray = S0_GAPS_UM,
                 multiccd: bool = True) -> list[dict]:
    """One row per (direction, true gap): reported contact and min distance."""
    sc = build_scene(task, rep, clearance, joints='xy', gravity=False,
                     peg_pos=(0, 0, task.thickness / 2), multiccd=multiccd)
    m, d = sc.model, mujoco.MjData(sc.model)
    peg_pts, rim_pts = geo.peg_samples(task, clearance, 5e-5), geo.rim_samples(task, 2e-5)
    center = np.array([0, 0, task.thickness / 2])
    rows = []
    for u in directions(n_dir):
        for g in gaps_um[gaps_um <= clearance * UM + 1e-9] / UM:
            t = _bisect_ray(lambda s: geo.translation_gap(task, clearance, s * u) >= g,
                            0.0, clearance + 2e-3)
            d.qpos[:2] = t * u
            mujoco.mj_fwdPosition(m, d)
            dists = [c[0] for c in plate_contacts(sc, d)]
            # Per-wall depth, the quantity a contact's dist estimates (at a
            # square's corner it is the feasible-set distance / sqrt(2)).
            pen = geo.penetration(task, clearance, center + np.r_[t * u, 0], np.eye(3),
                                  peg_pts, rim_pts) if g < 0 else 0.0
            rows.append({**_base(task, rep, clearance), 'multiccd': multiccd,
                         'theta': math.atan2(u[1], u[0]), 'gap_true_um': g * UM,
                         'pen_true_um': pen * UM, 'contact': bool(dists),
                         'dist_um': min(dists) * UM if dists else math.nan})
    return rows


def _bisect_ray(ok, lo: float, hi: float, iters: int = 60) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


# ----- S1: gravity drop ------------------------------------------------------------

def probe_drop(task: Task, rep: PlateRep, clearance: float, friction: bool = True,
               drop_height: float = 1e-3, duration: float = 0.3,
               gt_every: int = 5, multiccd: bool = True) -> dict:
    top = task.thickness
    z0 = top + task.peg_length / 2 + drop_height
    sc = build_scene(task, rep, clearance, joints='free', gravity=True,
                     friction=friction, peg_pos=(0, 0, z0), multiccd=multiccd)
    m, d = sc.model, mujoco.MjData(sc.model)
    peg_pts, rim_pts = geo.peg_samples(task, clearance, 2e-4), geo.rim_samples(task)
    impulse, contact_steps, max_pen, sds = 0.0, 0, 0.0, []
    t_pass = math.nan
    min_bottom = math.inf
    for step in range(round(duration / m.opt.timestep)):
        mujoco.mj_step(m, d)
        bottom = d.qpos[2] - task.peg_length / 2
        min_bottom = min(min_bottom, bottom)
        if math.isnan(t_pass) and bottom < 0:
            t_pass = d.time
        contacts = plate_contacts(sc, d)
        if contacts:
            contact_steps += 1
            impulse += sum(c[2] for c in contacts) * m.opt.timestep
        if step % gt_every == 0 or contacts:
            rot = geo.quat_to_mat(d.qpos[3:7])
            if contacts:
                sds.append(geo.true_signed_distance(task, clearance, d.qpos[:3], rot,
                                                    peg_pts, rim_pts))
            if step % gt_every == 0:
                max_pen = max(max_pen, geo.penetration(task, clearance, d.qpos[:3], rot,
                                                       peg_pts, rim_pts))
    bottom = d.qpos[2] - task.peg_length / 2
    final_pen = geo.penetration(task, clearance, d.qpos[:3], geo.quat_to_mat(d.qpos[3:7]),
                                peg_pts, rim_pts)
    passed = bottom < 0
    blocked = bottom > top - 1e-3
    return {**_base(task, rep, clearance), 'multiccd': multiccd, 'friction': friction,
            'outcome': 'pass' if passed else ('block' if blocked else 'stuck'),
            'expected': 'pass' if clearance > 0 else 'block',
            'sink_um': max(0.0, top - bottom) * UM if not passed else math.nan,
            't_pass_ms': t_pass * 1e3,
            't_pass_ballistic_ms': math.sqrt(2 * (top + drop_height) / 9.81) * 1e3,
            'wall_impulse_mNs': impulse * 1e3, 'contact_steps': contact_steps,
            'max_pen_um': max_pen * UM, 'final_pen_um': final_pen * UM,
            **geo.contact_stats(sds),
            'tilt_deg': math.degrees(math.acos(min(1.0, abs(1 - 2 * (d.qpos[4]**2 + d.qpos[5]**2))))),
            'n_plate_geoms': sc.n_plate_geoms, 'compile_s': sc.compile_s}


# ----- S2-S4: constrained equilibria under a constant wrench ------------------------

def _settle(sc, wrenches: np.ndarray, duration: float, nthread: int):
    """Roll out one world per wrench (constant for the whole run).

    Returns the qpos trajectory over the second half of the run (settled
    window; contact chatter shows up as spikes in it) and whether the peg
    overlapped the plate at the start.
    """
    m = sc.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    overlap0 = any(c[0] < 0 for c in plate_contacts(sc, d))
    state0 = np.empty(mujoco.mj_stateSize(m, FULL))
    mujoco.mj_getState(m, d, state0, FULL)
    nstep = round(duration / m.opt.timestep)
    control = np.zeros((len(wrenches), nstep, 6 * m.nbody))
    control[:, :, 6 * sc.peg_body:6 * sc.peg_body + 6] = wrenches[:, None, :]
    datas = [mujoco.MjData(m) for _ in range(max(1, min(nthread, len(wrenches))))]
    states, _ = rollout.rollout(m, datas, np.tile(state0, (len(wrenches), 1)), control,
                                control_spec=XFRC, nstep=nstep)
    qpos_adr = 1  # FULLPHYSICS state layout: time, qpos, qvel, ...
    return states[:, nstep // 2:, qpos_adr:qpos_adr + m.nq], overlap0


def _contact_sd(sc, task: Task, clearance: float, qpos: np.ndarray,
                every: int = 10) -> list[np.ndarray]:
    """True signed distance at every `every`-th settled state in which the
    simulator reports peg-plate contact, one array per world."""
    m, d = sc.model, mujoco.MjData(sc.model)
    peg_pts, rim_pts = geo.peg_samples(task, clearance), geo.rim_samples(task)
    out = []
    for world in qpos:
        sd = []
        for q in world[::every]:
            d.qpos[:] = q
            mujoco.mj_fwdPosition(m, d)
            if plate_contacts(sc, d):
                sd.append(geo.true_signed_distance(
                    task, clearance, d.xpos[sc.peg_body].copy(),
                    d.xmat[sc.peg_body].reshape(3, 3).copy(), peg_pts, rim_pts))
        out.append(np.array(sd))
    return out


def _window_stats(err_um: np.ndarray) -> dict:
    """Settled value, worst excursion, and jitter of an error time series."""
    return {'error_um': float(np.median(err_um)), 'error_max_um': float(err_um.max()),
            'error_min_um': float(err_um.min()), 'jitter_um': float(err_um.std())}


def probe_lateral(task: Task, rep: PlateRep, clearance: float, force: float = 0.1,
                  n_dir: int = 24, duration: float = 1.0, nthread: int = 8,
                  multiccd: bool = True) -> list[dict]:
    sc = build_scene(task, rep, clearance, joints='xy', gravity=False, friction=False,
                     peg_pos=(0, 0, task.thickness / 2), multiccd=multiccd)
    u = directions(n_dir)
    wrench = np.zeros((n_dir, 6))
    wrench[:, :2] = force * u
    qpos, overlap0 = _settle(sc, wrench, duration, nthread)
    support = geo.translation_support(task, clearance, u)
    reach = np.einsum('itj,ij->it', qpos[:, :, :2], u)
    sds = _contact_sd(sc, task, clearance, qpos)
    return [{**_base(task, rep, clearance), 'multiccd': multiccd, 'force_N': force,
             'theta': math.atan2(u[i, 1], u[i, 0]), 'truth_um': support[i] * UM,
             **_window_stats((reach[i] - support[i]) * UM), 'overlap_at_start': overlap0,
             **geo.contact_stats(sds[i])}
            for i in range(n_dir)]


def probe_tilt(task: Task, rep: PlateRep, clearance: float, phi: float = 0.0,
               torque: float = 1.5e-3, duration: float = 1.0,
               multiccd: bool = True) -> dict:
    sc = build_scene(task, rep, clearance, joints='tilt', phi=phi, gravity=False,
                     friction=False, peg_pos=(0, 0, task.thickness / 2), multiccd=multiccd)
    k = np.array([math.cos(phi), math.sin(phi), 0.0])
    qpos, overlap0 = _settle(sc, np.concatenate([np.zeros(3), torque * k])[None], duration, 1)
    truth = geo.max_tilt(task, clearance, phi)
    theta = qpos[0, :, 2]
    # Reported as wall displacement at the plate faces, T/2 from the pivot.
    return {**_base(task, rep, clearance), 'multiccd': multiccd, 'phi_deg': math.degrees(phi),
            'torque_Nm': torque, 'tilt_mrad': float(np.median(theta)) * 1e3,
            'truth_mrad': truth * 1e3,
            **_window_stats((theta - truth) * task.thickness / 2 * UM),
            'overlap_at_start': overlap0, **geo.contact_stats(_contact_sd(sc, task, clearance, qpos)[0])}


def probe_yaw(task: Task, rep: PlateRep, clearance: float, torque: float = 8e-4,
              duration: float = 1.0, multiccd: bool = True) -> dict:
    sc = build_scene(task, rep, clearance, joints='yaw', gravity=False, friction=False,
                     peg_pos=(0, 0, task.thickness / 2), multiccd=multiccd)
    qpos, overlap0 = _settle(sc, np.array([[0, 0, 0, 0, 0, torque]]), duration, 1)
    truth = geo.max_yaw(task, clearance)
    psi = qpos[0, :, 2]
    corner = task.peg_half(clearance) * math.sqrt(2)
    return {**_base(task, rep, clearance), 'multiccd': multiccd, 'torque_Nm': torque,
            'yaw_mrad': float(np.median(psi)) * 1e3, 'truth_mrad': truth * 1e3,
            **_window_stats((psi - truth) * corner * UM), 'overlap_at_start': overlap0,
            **geo.contact_stats(_contact_sd(sc, task, clearance, qpos)[0])}
