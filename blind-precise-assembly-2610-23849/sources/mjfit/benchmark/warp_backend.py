"""MuJoCo Warp (GPU) backend: batched probes and throughput timing.

Every run checks Warp's sticky per-world overflow bitmask. Contact, constraint,
CCD, and EPA overflows silently drop contacts, so such runs are flagged
invalid instead of being reported as physics. Solver iteration-limit flags
are recorded but do not invalidate a run (CPU MuJoCo has the same limit).
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw
from mujoco_warp._src.types import OverflowType

from . import geometry as geo
from . import probes
from .geometry import Task
from .representations import PlateRep
from .scene import Scene, build_scene

wp.config.quiet = True
INVALID = (OverflowType.NEFC | OverflowType.NJMAX_NNZ | OverflowType.BROADPHASE
           | OverflowType.NARROWPHASE | OverflowType.CCD | OverflowType.EPA_HORIZON)


class Batch:
    """nworld copies of one scene with per-world qpos and applied wrenches."""

    def __init__(self, sc: Scene, qpos: np.ndarray, wrench: np.ndarray,
                 nconmax: int = 256, njmax: int | None = None):
        m = sc.model
        self.sc, self.nworld = sc, len(qpos)
        self.mw = mjw.put_model(m)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        # put_data copies d's contacts, so the buffers must hold them (a plate
        # that fills the clearance starts with hundreds).
        nconmax = max(nconmax, d.ncon)
        self.dw = mjw.put_data(m, d, nworld=self.nworld, nconmax=nconmax,
                               njmax=max(njmax or 6 * nconmax, d.nefc))
        self.dw.qpos.assign(qpos.astype(np.float32))
        xfrc = np.zeros((self.nworld, m.nbody, 6), dtype=np.float32)
        xfrc[:, sc.peg_body] = wrench
        self.dw.xfrc_applied.assign(xfrc)
        self.dw.overflow.zero_()
        self._graphs: dict[int, wp.Graph] = {}

    def start_overlap(self) -> np.ndarray:
        """Per world: does the initial pose already report a penetrating
        contact (the representation fills the clearance)?"""
        mjw.forward(self.mw, self.dw)
        n = int(self.dw.nacon.numpy()[0])
        hit = self.dw.contact.dist.numpy()[:n] < 0
        return np.bincount(self.dw.contact.worldid.numpy()[:n][hit],
                           minlength=self.nworld) > 0

    def advance(self, nstep: int):
        """Advance nstep steps as one CUDA graph (captured on first use; the
        capture loads kernel modules itself, so no eager warm-up step)."""
        if nstep not in self._graphs:
            with wp.ScopedCapture() as cap:
                for _ in range(nstep):
                    mjw.step(self.mw, self.dw)
            self._graphs[nstep] = cap.graph
        wp.capture_launch(self._graphs[nstep])

    def run(self, nstep: int, chunk: int, contacts: bool = False):
        """Step nstep, recording qpos every `chunk` steps -> (nrec, nworld, nq),
        and with contacts=True also a (nrec, nworld) any-contact flag."""
        rec, flags = [], []
        for _ in range(nstep // chunk):
            self.advance(chunk)
            rec.append(self.dw.qpos.numpy().copy())
            if contacts:
                flags.append(self.contacts_per_world() > 0)
        return (np.array(rec), np.array(flags)) if contacts else np.array(rec)

    def overflow(self) -> np.ndarray:
        return self.dw.overflow.numpy()

    def release(self):
        """Drop captured graphs and device arrays now rather than at GC."""
        import gc
        wp.synchronize()
        self._graphs.clear()
        self.mw = self.dw = None
        gc.collect()

    def contacts_per_world(self) -> np.ndarray:
        n = int(self.dw.nacon.numpy()[0])
        return np.bincount(self.dw.contact.worldid.numpy()[:n], minlength=self.nworld)


def _flags(ov: np.ndarray) -> dict:
    return {'warp_invalid': bool((ov & int(INVALID)).any()),
            'warp_iter_limit': bool((ov & int(OverflowType.ITERATIONS)).any())}


def run_checked(sc: Scene, qpos: np.ndarray, wrench: np.ndarray, nstep: int, chunk: int,
                tries: int = 4):
    """Run a batch from its initial state; on any contact/constraint overflow
    (Warp drops contacts silently) double the buffers and rerun from scratch.
    Returns ((qpos, touching), overlap at start per world, flags).

    Warp's broadphase stores candidate pairs in the contact buffer, so the
    starting capacity covers one pair per plate geom, not just real contacts.
    """
    nconmax = max(256, sc.n_plate_geoms + 32)
    for _ in range(tries):
        batch = Batch(sc, qpos, wrench, nconmax=nconmax)
        overlap0 = batch.start_overlap()
        traj = batch.run(nstep, chunk, contacts=True)
        flags = _flags(batch.overflow())
        batch.release()
        if not flags['warp_invalid']:
            break
        nconmax *= 2
    return traj, overlap0, {**flags, 'nconmax': nconmax}


def _sd_at_contacts(sc: Scene, task: Task, clearance: float, qpos: np.ndarray,
                    touching: np.ndarray) -> list[float]:
    """True signed distance at the recorded states where Warp reported
    contact; the peg pose comes from CPU forward kinematics of Warp's qpos."""
    m, d = sc.model, mujoco.MjData(sc.model)
    peg_pts, rim_pts = geo.peg_samples(task, clearance), geo.rim_samples(task)
    out = []
    for q in qpos[touching]:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        out.append(geo.true_signed_distance(task, clearance, d.xpos[sc.peg_body].copy(),
                                            d.xmat[sc.peg_body].reshape(3, 3).copy(),
                                            peg_pts, rim_pts))
    return out


def warp_lateral(task: Task, rep: PlateRep, clearance: float, force: float = 0.1,
                 n_dir: int = 24, duration: float = 1.0, multiccd: bool = True) -> list[dict]:
    """S2 on the GPU: one world per push direction, same scene as the CPU probe."""
    sc = build_scene(task, rep, clearance, joints='xy', gravity=False, friction=False,
                     peg_pos=(0, 0, task.thickness / 2), multiccd=multiccd)
    u = probes.directions(n_dir)
    wrench = np.zeros((n_dir, 6))
    wrench[:, :2] = force * u
    nstep = round(duration / sc.model.opt.timestep)
    (traj, touching), overlap0, flags = run_checked(
        sc, np.tile(sc.model.qpos0, (n_dir, 1)), wrench, nstep, chunk=10)
    half = len(traj) // 2  # settled half, as on CPU
    traj, touching = traj[half:].astype(np.float64), touching[half:]
    support = geo.translation_support(task, clearance, u)
    reach = np.einsum('twj,wj->wt', traj[:, :, :2], u)
    return [{**probes._base(task, rep, clearance), 'backend': 'warp', 'multiccd': multiccd,
             'force_N': force, 'theta': math.atan2(u[i, 1], u[i, 0]),
             'truth_um': support[i] * probes.UM,
             **probes._window_stats((reach[i] - support[i]) * probes.UM),
             'overlap_at_start': bool(overlap0[i]), **flags,
             **geo.contact_stats(_sd_at_contacts(sc, task, clearance, traj[:, i],
                                                 touching[:, i]))}
            for i in range(n_dir)]


def warp_tilt(task: Task, rep: PlateRep, clearance: float, phi: float = 0.0,
              torque: float = 1.5e-3, duration: float = 1.0, multiccd: bool = True) -> dict:
    """S3 on the GPU (one world), same scene and wrench as the CPU probe."""
    sc = build_scene(task, rep, clearance, joints='tilt', phi=phi, gravity=False,
                     friction=False, peg_pos=(0, 0, task.thickness / 2), multiccd=multiccd)
    k = np.array([math.cos(phi), math.sin(phi), 0.0])
    wrench = np.concatenate([np.zeros(3), torque * k])[None]
    nstep = round(duration / sc.model.opt.timestep)
    (traj, touching), overlap0, flags = run_checked(sc, sc.model.qpos0[None].copy(), wrench,
                                                    nstep, chunk=10)
    half = len(traj) // 2
    traj, touching = traj[half:, 0].astype(np.float64), touching[half:, 0]
    truth = geo.max_tilt(task, clearance, phi)
    theta = traj[:, 2]
    # Reported as wall displacement at the plate faces, T/2 from the pivot.
    return {**probes._base(task, rep, clearance), 'backend': 'warp', 'multiccd': multiccd,
            'phi_deg': math.degrees(phi), 'torque_Nm': torque,
            'tilt_mrad': float(np.median(theta)) * 1e3, 'truth_mrad': truth * 1e3,
            **probes._window_stats((theta - truth) * task.thickness / 2 * probes.UM),
            'overlap_at_start': bool(overlap0[0]), **flags,
            **geo.contact_stats(_sd_at_contacts(sc, task, clearance, traj, touching))}


def warp_drop(task: Task, rep: PlateRep, clearances_um, friction: bool = True,
              drop_height: float = 1e-3, duration: float = 0.3,
              multiccd: bool = True) -> list[dict]:
    """S1 outcome on the GPU (one single-world batch per clearance)."""
    rows = []
    top = task.thickness
    for c_um in clearances_um:
        c = c_um * 1e-6
        sc = build_scene(task, rep, c, joints='free', gravity=True, friction=friction,
                         peg_pos=(0, 0, top + task.peg_length / 2 + drop_height),
                         multiccd=multiccd)
        m = sc.model
        (traj, _), _, flags = run_checked(sc, m.qpos0[None].copy(), np.zeros((1, 6)),
                                       round(duration / m.opt.timestep), chunk=50)
        bottom = float(traj[-1, 0, 2]) - task.peg_length / 2
        passed, blocked = bottom < 0, bottom > top - 1e-3
        rows.append({**probes._base(task, rep, c), 'backend': 'warp', 'multiccd': multiccd,
                     'friction': friction,
                     'outcome': 'pass' if passed else ('block' if blocked else 'stuck'),
                     'expected': 'pass' if c > 0 else 'block',
                     'sink_um': max(0.0, top - bottom) * probes.UM if not passed else math.nan,
                     **flags})
    return rows
