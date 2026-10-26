"""Videos of bolt/nut settings under three thread behaviors.

screw and spin: two settings per method (coarse / fine, from each method's CPU
Pareto front at c = 100 µm), each re-simulated exactly as its probe runs it:

  screw   : T4. Axial slide + spin hinge, frictionless; a torque turns the nut
            up the thread against a 0.1 N load. Exact: it follows the helix,
            sitting on its lower flanks (2c/sqrt3 below the centred helix).
  spin    : T5. Free nut on the vertical bolt, frictionless, under gravity.
            Exact: it spins down with constant acceleration g/(1 + I(2pi/P)^2/m).

rundown and compare: the same run, at the 1 ms step. rundown puts mjfit's
wedges (with seam relief) against the two SDFs, with traces; compare is plain
side-by-side renders of each method's pick with its real-time factor:

  rundown : Free nut on the vertical bolt under gravity, sliding friction 0.3
            (condim 3). A nut runner, a speed servo about the nut axis, turns
            it 4 turns down at 1 rev/s. Exact: it rides down on its lower
            flanks, and the runner needs the power-screw lowering torque.

wacky: the run-down on the wacky screws of wacky.py, 3 x 3, each with its
teeth drawn beside it; undercut: the same for its dovetail and hooked
(non-convex) teeth, 2 x 2.

forms: the run-down on every thread form of forms.py (ISO, trapezoidal,
square, buttress, round wire, the FurnitureBench leg), ours only, each with
its own exact seat and screw-formula torque. compare --forms: the compare
video for each form (CoACD at 0.02, built for every form), then all of them
back to back.

The render is the simulation model itself (its collision geometry: hulls,
the mesh an SDF was built from, or MuJoCo's surface mesh of the analytic SDF),
with a stripe on the nut to show its rotation. Black arrows: applied load
(torque drawn as a couple at the nut rim). Magenta: the net contact force of
the bolt on the nut.

    MUJOCO_GL=egl python -m benchmark.thread.render
    MUJOCO_GL=egl python -m benchmark.thread.render --behaviors rundown --clearance 20
    MUJOCO_GL=egl python -m benchmark.thread.render --behaviors compare --workers 1
    MUJOCO_GL=egl python -m benchmark.thread.render --behaviors forms --workers 1
    MUJOCO_GL=egl python -m benchmark.thread.render --behaviors compare --forms all --workers 1
    MUJOCO_GL=egl python -m benchmark.thread.render --behaviors wacky --workers 1
    MUJOCO_GL=egl python -m benchmark.thread.render --behaviors undercut --workers 1
"""
from __future__ import annotations

import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .. import report as R  # noqa: E402
from ..render import _arrow, _px  # noqa: E402
from . import geometry as geo  # noqa: E402
from . import probes as T  # noqa: E402
from . import forms as F  # noqa: E402
from . import report as TR  # noqa: E402
from . import wacky as W  # noqa: E402
from .geometry import ThreadTask  # noqa: E402
from .representations import build_thread  # noqa: E402
from .scene import _inertia, build_scene, pair_contacts  # noqa: E402

HERE = Path(__file__).resolve().parents[1]

# (method, params, MULTICCD): coarse and fine front settings at c = 100 µm.
POINTS = [
    ('ours', {'segments': 24}, False), ('ours', {'segments': 96}, False),
    ('coacd', {'threshold': 0.05}, False), ('coacd', {'threshold': 0.02}, False),
    ('vhacd', {'max_hulls': 256, 'resolution': 4000000, 'min_error': 0.1, 'max_depth': 12},
     False),
    ('vhacd', {'max_hulls': 4096, 'resolution': 60000000, 'min_error': 0.1, 'max_depth': 14},
     False),
    ('sdf', {'depth': 7}, True), ('sdf', {'depth': 9}, True),
    ('sdf_analytic', {'initpoints': 40, 'iterations': 10}, True),
    ('sdf_analytic', {'initpoints': 300, 'iterations': 60}, True),
]
# rundown: mine on the top row, the SDFs below.
RUNDOWN = [
    ('ours', {'segments': 24, 'relief_um': 5}, False),
    ('ours', {'segments': 96, 'relief_um': 5}, False),
    ('sdf', {'depth': 9}, True),
    ('sdf_analytic', {'initpoints': 100, 'iterations': 30}, True),
]
# compare: each method's pick. ours: the coarse setting, clean at both
# clearances; CoACD: its finest decomposition (every threshold jams); SDF: the
# analytic (exact-shape) one, as the mesh SDF jams.
COMPARE = [
    ('ours', {'segments': 24, 'relief_um': 5}, False),
    ('coacd', {'threshold': 0.01}, False),
    ('sdf_analytic', {'initpoints': 100, 'iterations': 30}, True),
]
# forms: ours on every thread form; compare --forms: CoACD at 0.02, the
# finest threshold built for all of them.
FORMS_OURS = ('ours', {'segments': 24, 'relief_um': 5}, False)
COMPARE_FORMS = [FORMS_OURS, ('coacd', {'threshold': 0.02}, False),
                 ('sdf_analytic', {'initpoints': 100, 'iterations': 30}, True)]
RUN_MU = 0.3                   # sliding friction (condim 3: a flank contact has no spin friction)
RUN_SPEED = 1.0                # rev/s, reached over RUN_RAMP s
RUN_RAMP = 0.2
RUN_TMAX = 5e-3                # N·m, the runner's torque limit (~50x the run-down torque)
RUN_KV = 2e-3                  # N·m·s/rad, the speed servo's gain
# (both for the ISO M10 nut; other forms scale them with their own run-down
# torque, see runner_gains)
BEHAVIORS = {
    # title, blurb, duration (s), simulated seconds per frame, fps
    'screw': ('Screwing (T4)', 'Nut on an axial slide + spin hinge, frictionless. A 0.12 mN·m '
              'torque turns it up the thread against a 0.1 N load. Exact: it follows the helix '
              'on its lower flanks.', 2.0, 0.02, 25),
    'spin': ('Spin-down (T5)', 'Free nut on the vertical bolt, frictionless, under gravity. '
             'Exact: it spins down the helix at a constant acceleration.', 0.8, 0.008, 25),
    'rundown': ('Run-down', f'Free nut on the vertical bolt, gravity, sliding friction μ = {RUN_MU:g}. '
                f'A nut runner (speed servo about the nut axis: {RUN_SPEED:g} rev/s, limit '
                f'{RUN_TMAX * 1e3:g} mN·m) turns it 4 turns down. Exact: it rides its lower flanks '
                '(2c/√3 below centred) at the power-screw lowering torque.', 4.3, 0.04, 25),
    'compare': ('Run-down', '', 4.3, 0.04, 25),
    'forms': ('Thread forms', 'Ours (mjfit, 24 segments per turn, 5 µm seam relief) in the '
              f'run-down: free nut, gravity, sliding friction μ = {RUN_MU:g}. A nut runner turns '
              f'it 4 turns down at {RUN_SPEED:g} rev/s, with a torque limit of about 50× the '
              "form's screw-formula torque. Exact: it rides its loaded flanks at the exact seat.",
              4.3, 0.04, 25),
    'wacky': ('Wacky screws', 'Ours (mjfit, 24 segments per turn, 5 µm seam relief) in the '
              f'run-down: free nut, gravity, sliding friction μ = {RUN_MU:g}, 1 ms step. A nut '
              'runner drives each nut down its thread. Insets: the two teeth in the meridian '
              'plane, bolt grey, nut orange.', 4.3, 0.04, 25),
    'undercut': ('Undercut teeth', 'A dovetail, and non-convex hooks that mjfit cuts into '
                 'convex pieces (white lines in the insets; bolt grey, nut orange). Ours, 24 '
                 f'segments per turn, 5 µm seam relief; free nut, μ = {RUN_MU:g}, driven down by '
                 'a nut runner.', 4.3, 0.04, 25),
}
FW, FH, TOP = 1920, 1080, 76
NCOL = 5
TW, TH = FW // NCOL, (FH - TOP) // 2
RS = 236                       # render box (square, drawn at 2x)
RS_RUN = 420                   # rundown: 2 x 2 tiles, picture left of three plots
RS_FORMS = 300                 # forms: 3 x 2 tiles, picture left of three plots
RS_CMP = 610                   # compare: three pictures side by side
FH_CMP = 840
LOAD_RGBA = np.array([0.05, 0.05, 0.05, 1.0])
CONTACT_RGBA = np.array([0.88, 0.05, 0.55, 1.0])
M_PER_N = 0.05                 # arrow length: 0.1 N draws as 5 mm


def _scale(task: ThreadTask) -> float:
    """Size of the part relative to the M10 benchmark part (camera, arrows)."""
    return max(task.bolt_length / 0.016, task.nut_outer_radius / 0.008)


def _decorate(spec, bolt, nut, task: ThreadTask):
    k = _scale(task)
    spec.visual.global_.offwidth = spec.visual.global_.offheight = 2 * RS_CMP
    spec.visual.headlight.ambient = [0.45] * 3
    spec.visual.headlight.diffuse = [0.45] * 3
    spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.98, 0.98, 0.975],
                     rgb2=[0.86, 0.86, 0.85], width=64, height=64)
    light = spec.worldbody.add_light(pos=[0.05 * k, -0.08 * k, 0.12 * k],
                                     dir=[-0.3, 0.55, -0.78])
    light.diffuse = [0.35] * 3
    # Stripe across the nut's top face: its rotation is the thread's motion.
    nut.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[0.95 * task.nut_outer_radius, 6e-4 * k, 1e-4 * k],
                 pos=[0, 0, task.nut_height + 1e-4 * k], rgba=[0.1, 0.1, 0.1, 1], contype=0,
                 conaffinity=0)
    elev = math.radians(25)
    z_cam = np.array([0, -math.cos(elev), math.sin(elev)])
    x_cam = np.array([1.0, 0, 0])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x_cam, np.cross(z_cam, x_cam), z_cam]).ravel())
    target = np.array([0, 0, 0.0085 * k])
    spec.worldbody.add_camera(name='view', pos=target + 0.075 * k * z_cam, quat=quat, fovy=19)


def _runner(spec, kv: float, tmax: float):
    """Nut runner: a velocity servo on the free joint's spin about the nut axis."""
    spec.joints[0].name = 'nut_free'
    a = spec.add_actuator(name='runner', target='nut_free', trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.gear = [0, 0, 0, 0, 0, 1]
    a.set_to_velocity(kv=kv)
    a.forcelimited = True
    a.forcerange = [-tmax, tmax]


def _friction_torque(task: ThreadTask, c: float) -> float:
    """m g r mu': the friction part of the run-down torque. Unlike the net
    torque it stays positive on a thread steep enough to back-drive."""
    r_contact, cos_beta = geo.seat_contact(task, c)
    r = geo.pitch_radius(task) or r_contact
    return 9.81 * _inertia(task, c)[0] * r * RUN_MU / cos_beta


@lru_cache(maxsize=1)
def _iso_friction_torque() -> float:
    """For the ISO M10 nut, which RUN_KV and RUN_TMAX were set for."""
    return _friction_torque(ThreadTask(), 1e-4)


def runner_gains(task: ThreadTask, c: float):
    """(kv, torque limit, screw-formula torque): the ISO gains scaled by the
    thread's friction torque (ISO at c = 100 µm keeps them exactly). The
    formula torque is negative on a back-drivable thread (the runner brakes)."""
    t_exact = geo.rundown_torque(task, _inertia(task, c)[0], RUN_MU, c)
    k = _friction_torque(task, c) / _iso_friction_torque()
    return RUN_KV * k, RUN_TMAX * k, t_exact


def run_speed(t: float, speed: float = RUN_SPEED, hand: int = 1) -> float:
    """Runner speed command (rad/s about the nut axis) that turns the nut down
    the thread at `speed` rev/s: clockwise from above for a right-hand
    thread, anticlockwise for a left-hand one."""
    return -hand * 2 * math.pi * speed * min(1.0, t / RUN_RAMP)


def render_frames(task, c, behavior, point, qpos, rs, overlay=None) -> np.ndarray:
    """Render the simulation model at each qpos (rs x rs, drawn at 2x).
    overlay(scene, frame) may add geoms (arrows) to each frame."""
    method, params, mccd = point
    rep = build_thread(task, method, params, c)
    joints = 'screw' if behavior == 'screw' else 'free'
    sc = build_scene(task, rep, joints=joints, multiccd=mccd,
                     decorate=lambda s, b, n: _decorate(s, b, n, task))
    m, d = sc.model, mujoco.MjData(sc.model)
    m.geom_rgba[sc.bolt_geoms] = [0.62, 0.64, 0.68, 1]
    base = np.array(matplotlib.colors.to_rgb(TR.COLOR[method]))
    m.geom_rgba[sc.nut_geoms] = [*(base + (1 - base) * 0.25), 1]
    renderer = mujoco.Renderer(m, 2 * rs, 2 * rs, max_geom=20000)
    frames = np.empty((len(qpos), rs, rs, 3), np.uint8)
    for f, q in enumerate(qpos):
        if np.isfinite(q).all():
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
        renderer.update_scene(d, camera='view')
        if overlay is not None:
            overlay(renderer.scene, f)
        img = renderer.render().astype(np.uint16)
        frames[f] = img.reshape(rs, 2, rs, 2, 3).sum(axis=(1, 3)) // 4
    renderer.close()
    return frames


def simulate(task, method, params, mccd, behavior, clearance, speed=RUN_SPEED):
    """Re-run the probe's scene, one row per video frame. `speed`: the nut
    runner's rev/s."""
    rep = build_thread(task, method, params, clearance)
    _, _, duration, frame_dt, _ = BEHAVIORS[behavior]
    look = lambda s, b, n: _decorate(s, b, n, task)  # noqa: E731
    if behavior == 'screw':
        sc = build_scene(task, rep, joints='screw', multiccd=mccd, decorate=look)
        wrench = np.array([0, 0, -T.SCREW_LOAD, 0, 0, T.SCREW_TORQUE])
    elif behavior == 'spin':
        sc = build_scene(task, rep, joints='free', gravity=True, friction=0.0,
                         nut_offset=(0, 0, 2 * task.pitch), multiccd=mccd, decorate=look)
        wrench = np.zeros(6)
    else:
        kv, tmax, _ = runner_gains(task, clearance)
        sc = build_scene(task, rep, joints='free', gravity=True, friction=RUN_MU, condim=3,
                         nut_offset=(0, 0, 2 * task.pitch), multiccd=mccd,
                         decorate=lambda s, b, n: (look(s, b, n), _runner(s, kv, tmax)))
        wrench = np.zeros(6)
    m, d = sc.model, mujoco.MjData(sc.model)
    every = round(frame_dt / m.opt.timestep)
    nframe = round(duration / frame_dt) + 1
    qpos, xpos, xmat, net, turn, command, torque = [], [], [], [], [], [], []
    phi = prev = cmd = wall = 0.0
    mujoco.mj_forward(m, d)
    for f in range(nframe):
        tau = 0.0
        for _ in range(every if f else 0):
            d.xfrc_applied[sc.nut_body] = wrench
            if m.nu:
                d.ctrl[0] = run_speed(d.time, speed, task.hand)
                cmd += d.ctrl[0] * m.opt.timestep
            t0 = time.perf_counter()
            mujoco.mj_step(m, d)
            wall += time.perf_counter() - t0
            if m.nu:
                tau += d.actuator_force[0] / every
            # Nut rotation about z, unwrapped step by step.
            a = math.atan2(d.xmat[sc.nut_body][3], d.xmat[sc.nut_body][0])
            phi += (a - prev + math.pi) % (2 * math.pi) - math.pi
            prev = a
        qpos.append(d.qpos.copy())
        xpos.append(d.xpos[sc.nut_body].copy())
        xmat.append(d.xmat[sc.nut_body].copy())
        net.append(sum((c[2] for c in pair_contacts(sc, d)), np.zeros(3)))
        turn.append(phi)
        command.append(cmd)
        torque.append(tau)
    home = task.nut_home() + (0.0 if behavior == 'screw' else 2 * task.pitch)
    return {'qpos': np.array(qpos), 'xpos': np.array(xpos), 'xmat': np.array(xmat),
            'net': np.array(net), 't': np.arange(nframe) * frame_dt, 'home': home,
            'turn': np.array(turn), 'command': np.array(command), 'torque': np.array(torque),
            'step_us': wall / ((nframe - 1) * every) * 1e6,
            'a_exact': geo.spin_down_accel(task, sc.mass, sc.inertia_axis), 'mass': sc.mass,
            't_exact': runner_gains(task, clearance)[2], 'tmax': runner_gains(task, clearance)[1],
            # resting on the flanks it is pushed onto (by the load or its weight)
            'seat': -geo.axial_gap(task, clearance, -1), 'n_geoms': sc.n_geoms,
            'hand': task.hand, 'lead': task.lead}


class Tile:
    def __init__(self, fig, col, row, task, behavior, c, point, run, metrics, title=None):
        method, params, mccd = point
        self.task, self.behavior, self.run, self.c = task, behavior, run, c
        self.series = []   # (line, dot, x, y) traced frame by frame
        rundown = behavior in ('rundown', 'forms')
        self.rs = {'rundown': RS_RUN, 'forms': RS_FORMS}.get(behavior, RS)
        tw, th = {'rundown': (FW // 2, (FH - TOP) // 2),
                  'forms': (FW // 3, (FH - TOP) // 2)}.get(behavior, (TW, TH))
        x0, y0 = col * tw, TOP + row * th
        color = TR.COLOR[method]
        label = method + '(' + ','.join(f'{k}={v}' for k, v in params.items()) + ')'
        fig.text((x0 + 12) / FW, 1 - (y0 + 8) / FH, title or TR.NAMES[method], color=color,
                 fontsize=11, fontweight='bold', va='top')
        sub = TR.tag(label, mccd)
        if behavior == 'forms':
            sub = (f'P {task.pitch * 1e3:g} mm, ' + (f'c = {c * 1e6:g} µm' if c else
                                                    'as measured') + f' · ours {sub}')
        fig.text((x0 + 12) / FW, 1 - (y0 + 28) / FH, sub, color=R.INK, fontsize=8.5, va='top')
        if rundown:
            info = f'{run["step_us"]:,.0f} µs/step in this run · {run["n_geoms"]} collision geoms'
        else:
            cost, err = metrics.get((label, mccd), (None, None))
            cost_s = f'{cost:,.0f} µs/step' if cost else 'µs/step n/a'
            err_s = ('fills the clearance' if err and err[1] else
                     f'T1–T4 error {err[0]:.3g} µm' if err else '')
            info = f'{cost_s} · {err_s}'
        fig.text((x0 + 12) / FW, 1 - (y0 + 44) / FH, info, color=R.INK2, fontsize=7.5, va='top')
        self.frames = self._render(point)
        ax = _px(fig, x0 + 6, y0 + 62, self.rs, self.rs)
        ax.set_axis_off()
        self.image = ax.imshow(self.frames[0], interpolation='nearest')
        if rundown:
            narrow = behavior == 'forms'
            self.readout = fig.text((x0 + self.rs + (20 if narrow else 50)) / FW,
                                    1 - (y0 + (52 if narrow else 8)) / FH, '',
                                    color=R.INK2, fontsize=7.5 if narrow else 8, va='top',
                                    family='monospace', linespacing=1.5)
            px = x0 + self.rs + (70 if narrow else 110)
            pw, ph, gap = tw - (px - x0) - (20 if narrow else 30), 96 if not narrow else 88, 22
            top = y0 + (128 if narrow else 104)
            axes = [_px(fig, px, top + i * (ph + gap), pw, ph) for i in range(3)]
            self._init_rundown(axes, color)
        else:
            self.readout = fig.text((x0 + RS + 14) / FW, 1 - (y0 + 80) / FH, '', color=R.INK2,
                                    fontsize=7.5, va='top', family='monospace', linespacing=1.6)
            pax = _px(fig, x0 + 50, y0 + RS + 78, TW - 66, TH - RS - 118)
            getattr(self, f'_init_{behavior}')(pax, color)

    def _trace(self, ax, x, y, color):
        line, = ax.plot([], [], color=color, lw=1.4)
        dot, = ax.plot([], [], 'o', color=color, ms=4)
        self.series.append((line, dot, x, y))

    def _render(self, point) -> np.ndarray:
        return render_frames(self.task, self.c, self.behavior, point, self.run['qpos'], self.rs,
                             self._arrows)

    def _arrows(self, scn, f):
        """Net forces are drawn on the nut's top face, between the bolt (which
        runs up through the nut) and the rim, on the two sides facing the
        camera: a force pointing down ends at the face, one pointing up
        starts from it."""
        t = self.task
        k = _scale(t)                           # arrows grow with the part
        m_per_n = M_PER_N * k
        pos = self.run['xpos'][f]
        R_ = self.run['xmat'][f].reshape(3, 3)
        z_top = (pos + R_ @ np.array([0, 0, t.nut_height]))[2]
        face = (t.r_major + t.nut_outer_radius) / 2

        def on_face(x, force, rgba, width):
            anchor = np.array([pos[0] + x, pos[1], z_top])
            n = np.linalg.norm(force)
            if n > 0 and force[2] < 0:   # pushing down onto the face
                anchor = anchor - force / n * min(n, 0.15) * m_per_n
            _arrow(scn, anchor, force, rgba, width * k, m_per_n)

        def couple(torque):                     # torque as a couple at the rim, x20
            rim = t.nut_outer_radius
            F = torque / (2 * rim)
            for s in (1, -1):
                p = pos + R_ @ np.array([s * rim, 0, t.nut_height])
                _arrow(scn, p, R_ @ np.array([0, s * F, 0]) * 20, LOAD_RGBA, 6e-4 * k, m_per_n)

        if self.behavior == 'screw':
            on_face(-face, np.array([0, 0, -T.SCREW_LOAD]), LOAD_RGBA, 6e-4)
            couple(T.SCREW_TORQUE)
        else:
            on_face(-face, np.array([0, 0, -9.81 * self.run['mass']]), LOAD_RGBA, 6e-4)
            if self.behavior in ('rundown', 'forms'):
                couple(self.run['torque'][f])
        on_face(face, self.run['net'][f], CONTACT_RGBA, 7e-4)

    # -- screw: offset from the centred helix against the exact seat
    def _init_screw(self, ax, color):
        q, t = self.run['qpos'], self.run['t']
        self.y = (q[:, 0] - self.task.pitch * q[:, 1] / (2 * math.pi)) * 1e6
        seat = self.run['seat'] * 1e6
        ax.axhline(seat, ls='--', color=R.MUTED, lw=1)
        ax.text(t[-1], seat, 'exact seat', ha='right', va='bottom', fontsize=6.5, color=R.MUTED)
        self._trace(ax, t, self.y, color)
        lim = 3 * self.c * 1e6
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('time (s)', fontsize=7)
        ax.set_ylabel('nut z − helix (µm)', fontsize=7)
        ax.tick_params(labelsize=6)

    def _update_screw(self, f):
        q = self.run['qpos'][f]
        return (f'turns {q[1] / (2 * math.pi):6.2f}\n'
                f'z − helix {self.y[f]:+7.1f} µm\n'
                f'exact     {self.run["seat"] * 1e6:+7.1f} µm')

    # -- spin: descent against the exact parabola
    def _init_spin(self, ax, color):
        t = self.run['t']
        self.y = (self.run['home'] - self.run['qpos'][:, 2]) * 1e3
        seat = -self.run['seat'] * 1e3
        exact = seat + 0.5 * self.run['a_exact'] * t ** 2 * 1e3
        ax.plot(t, exact, '--', color=R.MUTED, lw=1, label='exact')
        self._trace(ax, t, self.y, color)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-0.2, exact[-1] * 1.1)
        ax.set_xlabel('time (s)', fontsize=7)
        ax.set_ylabel('descent (mm)', fontsize=7)
        ax.legend(fontsize=6, loc='upper left')
        ax.tick_params(labelsize=6)
        self.exact = exact

    def _update_spin(self, f):
        return (f'descent {self.y[f]:6.2f} mm\n'
                f'exact   {self.exact[f]:6.2f} mm\n'
                f'({self.y[f] / max(self.exact[f], 1e-9) * 100:5.0f} %)')

    # -- rundown: turns against the runner, height against the seat, torque
    def _init_rundown(self, axes, color):
        r, t, task = self.run, self.run['t'], self.task
        H = task.nut_height
        # turns down the thread (the runner's direction)
        self.turns = -r['hand'] * r['turn'] / (2 * math.pi)
        self.cmd = -r['hand'] * r['command'] / (2 * math.pi)
        # Nut centre (on its axis, mid-height) against the centred helix.
        centre = r['xpos'][:, 2] + r['xmat'][:, 8] * H / 2
        self.dz = (centre - (r['home'] + H / 2 - r['lead'] * self.turns)) * 1e6
        self.tilt = np.degrees(np.arccos(np.clip(r['xmat'][:, 8], -1, 1)))
        self.tau = -r['hand'] * r['torque'] * 1e3      # tightening positive
        seat = r['seat'] * 1e6
        ax_turn, ax_z, ax_tau = axes
        ax_turn.plot(t, self.cmd, '--', color=R.MUTED, lw=1, label='runner command')
        self._trace(ax_turn, t, self.turns, color)
        ax_turn.set_ylim(-0.2, self.cmd[-1] * 1.1)
        ax_turn.set_ylabel('turns', fontsize=7)
        ax_turn.legend(fontsize=6, loc='upper left', frameon=False)
        ax_z.axhline(seat, ls='--', color=R.MUTED, lw=1)
        ax_z.text(t[-1], seat, 'exact seat', ha='right', va='bottom', fontsize=6.5, color=R.MUTED)
        self._trace(ax_z, t, self.dz, color)
        lim = max(3 * self.c * 1e6, 1.6 * abs(seat))
        ax_z.set_ylim(-lim, lim)
        ax_z.set_ylabel('nut z − helix (µm)', fontsize=7)
        ax_tau.axhline(r['t_exact'] * 1e3, ls='--', color=R.MUTED, lw=1)
        ax_tau.text(t[-1], r['t_exact'] * 1e3, 'screw formula', ha='right', va='bottom',
                    fontsize=6.5, color=R.MUTED)
        ax_tau.axhline(r['tmax'] * 1e3, ls=':', color=R.MUTED, lw=1)
        ax_tau.text(t[-1], r['tmax'] * 1e3, 'runner limit', ha='right', va='top', fontsize=6.5,
                    color=R.MUTED)
        lo = r['t_exact'] * 1e3 / 10.63     # ISO: 0.01 to 10 mN·m
        self._trace(ax_tau, t, np.maximum(self.tau, 1.1 * lo), color)
        ax_tau.set_yscale('log')
        ax_tau.set_ylim(lo, 2 * r['tmax'] * 1e3)
        ax_tau.set_ylabel('torque (mN·m)\n40 ms mean', fontsize=7)
        ax_tau.set_xlabel('time (s)', fontsize=7)
        for ax in axes:
            ax.set_xlim(0, t[-1])
            ax.tick_params(labelsize=6)
        for ax in axes[:2]:
            ax.tick_params(labelbottom=False)

    def _update_rundown(self, f):
        return (f'turns   {self.turns[f]:5.2f}  (runner {self.cmd[f]:4.2f})\n'
                f'z−helix {self.dz[f]:+7.1f} µm (exact {self.run["seat"] * 1e6:+.1f})\n'
                f'tilt    {self.tilt[f]:5.2f}°\n'
                f'torque  {self.tau[f]:6.3f} mN·m (formula {self.run["t_exact"] * 1e3:.3f})')

    _update_forms = _update_rundown

    def update(self, f):
        self.image.set_data(self.frames[f])
        for line, dot, x, y in self.series:
            line.set_data(x[:f + 1], y[:f + 1])
            dot.set_data([x[f]], [y[f]])
        self.readout.set_text(getattr(self, f'_update_{self.behavior}')(f))


def _metrics(c_um: float) -> dict:
    p, speed = TR.load()
    out = {}
    for method, params, mccd in POINTS:
        label = method + '(' + ','.join(f'{k}={v}' for k, v in params.items()) + ')'
        out[(label, mccd)] = (TR.cost(p, speed, label, c_um, mccd, 'cpu'),
                              TR.accuracy(p, label, c_um, mccd))
    return out


def write_video(path, task, behavior, c, points, runs, metrics, tasks=None, cs=None,
                titles=None):
    """tasks, cs, titles: per tile (the forms video); else `task` and `c`."""
    import imageio_ffmpeg
    title, blurb, duration, frame_dt, fps = BEHAVIORS[behavior]
    rundown = behavior == 'rundown'
    forms = behavior == 'forms'
    tasks = tasks or [task] * len(points)
    cs = cs or [c] * len(points)
    titles = titles or [None] * len(points)
    fig = plt.figure(figsize=(FW / 100, FH / 100), dpi=100)
    head = (f'{title}   nut run-down, 1 ms step' if forms else
            f'{title}   M10×1.5, c = {c * 1e6:g} µm flank clearance '
            f'(axial play ±{-runs[0]["seat"] * 1e6:.3g} µm)' + (', 1 ms step' if rundown else ''))
    fig.text(16 / FW, 1 - 12 / FH, head, fontsize=14, fontweight='bold', color=R.INK, va='top')
    rows = ('Top: ours; bottom: the SDFs.' if rundown else '' if forms else
            'Top row: coarse setting; bottom row: fine.')
    fig.text(16 / FW, 1 - 36 / FH, blurb + '  ' + rows, fontsize=9, color=R.INK2, va='top')
    load = ('0.1 N axial load, and the torque drawn as a couple at the rim (×20)'
            if behavior == 'screw' else
            'weight, and the runner torque drawn as a couple at the rim (×20)'
            if rundown or forms else 'weight')
    scale = 'longer on the larger FurnitureBench part' if forms else '0.1 N = 5 mm'
    fig.text(16 / FW, 1 - 54 / FH, f'Arrows from the nut top ({scale}): black = {load}; '
             'magenta = net contact force of the bolt on the nut. The stripe shows the nut turning.',
             fontsize=9, color=R.INK2, va='top')
    clock = fig.text(1 - 16 / FW, 1 - 14 / FH, '', fontsize=12, color=R.INK, va='top',
                     ha='right', family='monospace')
    if rundown or forms:
        ncol = 3 if forms else 2
        for i in range(1, ncol):
            fig.add_artist(plt.Line2D([i / ncol] * 2, [0.02, 1 - (TOP + 6) / FH],
                                      color=R.GRID, lw=1))
        y = 1 - (TOP + (FH - TOP) // 2) / FH
        fig.add_artist(plt.Line2D([0.01, 0.99], [y] * 2, color=R.GRID, lw=1))
        place = [(i % ncol, i // ncol) for i in range(len(points))]
    else:
        for i in range(1, NCOL):
            fig.add_artist(plt.Line2D([i * TW / FW] * 2, [0.02, 1 - (TOP + 6) / FH],
                                      color=R.GRID, lw=1))
        place = [(i // 2, i % 2) for i in range(len(points))]
    tiles = [Tile(fig, *place[i], tasks[i], behavior, cs[i], points[i], runs[i], metrics,
                  titles[i]) for i in range(len(points))]
    slow = 1 / (fps * frame_dt)
    pace = 'real time' if abs(slow - 1) < 1e-9 else f'{slow:g}x slow motion'
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{FW}x{FH}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p', '-crf', '20', str(path)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = len(runs[0]['t'])
    for f in range(n):
        for tile in tiles:
            tile.update(f)
        clock.set_text(f't = {runs[0]["t"][f]:4.2f} s  ({pace})')
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes()
        for _ in range(1 + (round(1.5 * fps) if f == n - 1 else 0)):
            ff.stdin.write(frame)
    ff.stdin.close()
    ff.wait()
    plt.close(fig)


def _setting(method: str, params: dict, n_geoms: int) -> str:
    if method == 'ours':
        relief = params.get('relief_um')
        return (f'{params["segments"]} segments per turn'
                + (f', {relief:g} µm seam relief' if relief else '') + f' · {n_geoms:,} geoms')
    if method == 'coacd':
        return f'threshold {params["threshold"]:g} · {n_geoms:,} convex hulls'
    if method == 'vhacd':
        return f'{params["max_hulls"]} max hulls · {n_geoms:,} convex hulls'
    if method == 'sdf':
        return f'mesh SDF, octree depth {params["depth"]}'
    return f'contact search {params["initpoints"]} points × {params["iterations"]} iterations'


def _play(gap: float) -> str:
    return f'{gap * 1e3:.2g} mm' if gap >= 1e-3 else f'{gap * 1e6:.3g} µm'


def write_compare_video(path, task, c, points, runs, title='M10×1.5'):
    """Plain side-by-side renders with each method's real-time factor."""
    import imageio_ffmpeg
    _, _, _, frame_dt, fps = BEHAVIORS['compare']
    fh, tw = FH_CMP, FW // len(points)
    fig = plt.figure(figsize=(FW / 100, fh / 100), dpi=100)

    def text(x, y, s, **kw):
        return fig.text(x / FW, 1 - y / fh, s, va='top', **kw)

    fit = (f'c = {c * 1e6:g} µm flank clearance' if c else 'as measured') + \
        f' (axial play ±{_play(-runs[0]["seat"])})'
    text(16, 12, f'Screwing a nut down: {title}, {fit}', fontsize=15, fontweight='bold',
         color=R.INK)
    text(16, 40, f'Free nut, gravity, friction μ = {RUN_MU:g}, 1 ms step. A nut runner turns it at '
         f'{RUN_SPEED:g} rev/s (torque limit {runs[0]["tmax"] * 1e3:.2g} mN·m). Real time = '
         'simulated seconds per wall-clock second, one CPU thread. The video plays in real time.',
         fontsize=10, color=R.INK2)
    clock = text(FW - 16, 14, '', fontsize=12, color=R.INK, ha='right', family='monospace')
    status, images = [], []
    for i, ((method, params, _), run) in enumerate(zip(points, runs)):
        x0 = i * tw
        if i:
            fig.add_artist(plt.Line2D([x0 / FW] * 2, [0.01, 1 - 70 / fh], color=R.GRID, lw=1))
        rtf = 1e3 / run['step_us']   # 1 ms step
        text(x0 + 16, 78, TR.NAMES[method], fontsize=16, fontweight='bold',
             color=TR.COLOR[method])
        text(x0 + 16, 106, _setting(method, params, run['n_geoms']), fontsize=10, color=R.INK)
        text(x0 + 16, 126, f'{rtf:.2g}× real time', fontsize=15, fontweight='bold', color=R.INK)
        text(x0 + 16, 152, f'{run["step_us"]:,.0f} µs per 1 ms step', fontsize=9, color=R.INK2)
        rs = min(RS_CMP, tw - 20)
        top = 176
        ax = fig.add_axes([(x0 + (tw - rs) // 2) / FW, 1 - (top + rs) / fh, rs / FW, rs / fh])
        ax.set_axis_off()
        frames = render_frames(task, c, 'compare', points[i], run['qpos'], rs)
        images.append((ax.imshow(frames[0], interpolation='nearest'), frames))
        status.append((text(x0 + 16, top + rs + 8, '', fontsize=11, color=R.INK,
                            family='monospace'), run))
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{FW}x{fh}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p', '-crf', '20', str(path)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = len(runs[0]['t'])
    for f in range(n):
        for image, frames in images:
            image.set_data(frames[f])
        for label, run in status:
            k = -run['hand'] / (2 * math.pi)
            label.set_text(f'nut turns {k * run["turn"][f]:4.2f}  '
                           f'(runner {k * run["command"][f]:4.2f})')
        clock.set_text(f't = {runs[0]["t"][f]:4.2f} s')
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes()
        for _ in range(1 + (round(1.5 * fps) if f == n - 1 else 0)):
            ff.stdin.write(frame)
    ff.stdin.close()
    ff.wait()
    plt.close(fig)


def _teeth_inset(ax, task, c):
    """The two teeth in the meridian plane over two pitches, to scale, with
    the cuts of a non-convex tooth's convex pieces."""
    prof = geo._profile(task)
    bolt, nut = prof.teeth(c)
    P = task.pitch
    for k in range(-1, 3):
        for tooth, color in ((bolt, '#9aa3b2'), (nut, '#e0a458')):
            ax.fill((tooth[:, 1] + k * P) * 1e3, tooth[:, 0] * 1e3, color=color, lw=0)
            pieces = geo._pieces(tooth)
            if len(pieces) > 1:
                for q in pieces:
                    q = np.vstack([q, q[:1]])
                    ax.plot((q[:, 1] + k * P) * 1e3, q[:, 0] * 1e3, color='white', lw=0.6)
    lo, hi = prof.r_core * 1e3, prof.r_root * 1e3 + c * 1e3 / 2
    ax.axhspan(lo - 50, lo, color='#9aa3b2', lw=0)
    ax.axhspan(hi, hi + 50, color='#e0a458', lw=0)
    pad = 0.25 * (hi - lo)
    ax.set_xlim(0, 2 * P * 1e3)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.set_xlabel('axial (mm)', fontsize=6, labelpad=1)
    ax.set_ylabel('radius (mm)', fontsize=6, labelpad=1)


def write_grid_video(path, names, runs, designs=None, behavior='wacky', ncol=3):
    """Wacky screws in a grid: render, description, teeth, live readout."""
    import imageio_ffmpeg
    import textwrap
    designs = designs or W.DESIGNS
    title, blurb, _, frame_dt, fps = BEHAVIORS[behavior]
    top = 64
    nrow = math.ceil(len(names) / ncol)
    tw, th = FW // ncol, (FH - top) // nrow
    rs = min(th - 20, tw // 2)
    big = rs > 400
    fig = plt.figure(figsize=(FW / 100, FH / 100), dpi=100)

    def text(x, y, s, **kw):
        return fig.text(x / FW, 1 - y / FH, s, va='top', **kw)

    text(16, 10, f'{title}   nut run-down, 1 ms step', fontsize=14, fontweight='bold',
         color=R.INK)
    text(16, 36, blurb, fontsize=9, color=R.INK2)
    clock = text(FW - 16, 12, '', fontsize=12, color=R.INK, ha='right', family='monospace')
    for i in range(1, ncol):
        fig.add_artist(plt.Line2D([i / ncol] * 2, [0.005, 1 - (top - 4) / FH], color=R.GRID,
                                  lw=1))
    for i in range(1, nrow):
        y = 1 - (top + i * th) / FH
        fig.add_artist(plt.Line2D([0.005, 0.995], [y] * 2, color=R.GRID, lw=1))
    tiles = []
    for i, (name, run) in enumerate(zip(names, runs)):
        d = designs[name]
        c = d.clearance_um * 1e-6
        x0, y0 = (i % ncol) * tw, top + (i // ncol) * th
        ax = fig.add_axes([(x0 + 6) / FW, 1 - (y0 + 10 + rs) / FH, rs / FW, rs / FH])
        ax.set_axis_off()
        frames = render_frames(d.task, c, 'wacky', FORMS_OURS, run['qpos'], rs)
        image = ax.imshow(frames[0], interpolation='nearest')
        xr = x0 + rs + 18
        text(xr, y0 + 10, d.title, fontsize=14 if big else 11, fontweight='bold',
             color=TR.COLOR['ours'])
        spec = (f'D {d.task.major_diameter * 1e3:g} mm, P {d.task.pitch * 1e3:g} mm'
                + (f', {d.task.starts} starts' if d.task.starts > 1 else '')
                + (', left-hand' if d.task.hand < 0 else '')
                + (f', c = {d.clearance_um:g} µm' if c else ', loose by design'))
        f = 1.25 if big else 1.0
        text(xr, y0 + 30 * f + (4 if big else 0), spec, fontsize=7.5 * f, color=R.INK)
        text(xr, y0 + 45 * f + (6 if big else 0), textwrap.fill(d.blurb, 50), fontsize=7.5 * f,
             color=R.INK2, linespacing=1.3)
        ih = 190 if big else 104
        iy = y0 + (110 if big else 86)
        inset = fig.add_axes([(xr + 40) / FW, 1 - (iy + ih) / FH, (tw - rs - 76) / FW, ih / FH])
        _teeth_inset(inset, d.task, c)
        readout = text(xr, iy + ih + (40 if big else 34), '', fontsize=7.5 * f, color=R.INK,
                       family='monospace', linespacing=1.45)
        tiles.append((image, frames, readout, run, d))
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{FW}x{FH}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p', '-crf', '20', str(path)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = len(runs[0]['t'])
    for f in range(n):
        for image, frames, readout, run, d in tiles:
            image.set_data(frames[f])
            k = -run['hand'] / (2 * math.pi)
            turns, cmd_turns = k * run['turn'][f], k * run['command'][f]
            H = d.task.nut_height
            centre = run['xpos'][f][2] + run['xmat'][f][8] * H / 2
            dz = (centre - (run['home'] + H / 2 - run['lead'] * turns)) * 1e6
            tau = -run['hand'] * run['torque'][f] * 1e3
            readout.set_text(
                f'turns   {turns:5.2f}  (runner {cmd_turns:4.2f})\n'
                f'seat    {dz:+7.0f} µm (exact {run["seat"] * 1e6:+.0f})\n'
                f'runner  {tau:+6.3f} mN·m ({"braking" if tau < 0 else "driving"})\n'
                f'{run["step_us"]:,.0f} µs/step · {run["n_geoms"]} geoms')
        clock.set_text(f't = {runs[0]["t"][f]:4.2f} s  (real time)')
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes()
        for _ in range(1 + (round(1.5 * fps) if f == n - 1 else 0)):
            ff.stdin.write(frame)
    ff.stdin.close()
    ff.wait()
    plt.close(fig)


def _concat(parts, path):
    """Videos of one size, back to back (no re-encode)."""
    import imageio_ffmpeg
    lst = path.with_suffix('.txt')
    lst.write_text(''.join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat',
                    '-safe', '0', '-i', str(lst), '-c', 'copy', str(path)], check=True)
    lst.unlink()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--behaviors', nargs='*', choices=list(BEHAVIORS), default=list(BEHAVIORS))
    ap.add_argument('--clearance', type=float, default=100.0, help='µm')
    ap.add_argument('--out', type=Path, default=HERE / 'results' / 'render')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--forms', nargs='*', default=None,
                    help=f"thread forms for 'forms' and 'compare' ({', '.join(F.FORMS)} or all); "
                         "compare with --forms writes one video per form, then all back to back")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    forms = list(F.FORMS) if args.forms in (None, ['all']) else args.forms
    task, c = ThreadTask(), args.clearance * 1e-6
    metrics = _metrics(args.clearance)
    with ProcessPoolExecutor(args.workers) as pool:
        for b in args.behaviors:
            if b == 'forms':
                tasks = [F.FORMS[n] for n in forms]
                cs = [F.CLEARANCE_UM[n] * 1e-6 for n in forms]
                runs = list(pool.map(simulate, tasks, *zip(*[FORMS_OURS] * len(forms)),
                                     [b] * len(forms), cs))
                path = args.out / 'thread_forms.mp4'
                write_video(path, task, b, c, [FORMS_OURS] * len(forms), runs, metrics,
                            tasks=tasks, cs=cs, titles=[F.TITLES[n] for n in forms])
                print(f'wrote {path}', flush=True)
                continue
            if b in ('wacky', 'undercut'):
                designs = W.DESIGNS if b == 'wacky' else W.UNDERCUT
                names = list(designs)
                ds = [designs[n] for n in names]
                runs = list(pool.map(simulate, [d.task for d in ds],
                                     *zip(*[FORMS_OURS] * len(ds)), [b] * len(ds),
                                     [d.clearance_um * 1e-6 for d in ds], [d.speed for d in ds]))
                path = args.out / f'thread_{b}.mp4'
                write_grid_video(path, names, runs, designs, b, ncol=3 if b == 'wacky' else 2)
                print(f'wrote {path}', flush=True)
                continue
            if b == 'compare' and args.forms is not None:
                parts = []
                for n in forms:
                    ft, fc = F.FORMS[n], F.CLEARANCE_UM[n] * 1e-6
                    for method, params, _ in COMPARE_FORMS:
                        build_thread(ft, method, params, fc)
                    runs = list(pool.map(simulate, [ft] * 3, *zip(*COMPARE_FORMS), [b] * 3,
                                         [fc] * 3))
                    parts.append(args.out / f'thread_compare_{n}.mp4')
                    write_compare_video(parts[-1], ft, fc, COMPARE_FORMS, runs, F.TITLES[n])
                    print(f'wrote {parts[-1]}', flush=True)
                _concat(parts, args.out / 'thread_compare_forms.mp4')
                print(f"wrote {args.out / 'thread_compare_forms.mp4'}", flush=True)
                continue
            points = {'rundown': RUNDOWN, 'compare': COMPARE}.get(b, POINTS)
            for method, params, _ in points:   # decompose once, before the workers
                build_thread(task, method, params, c)
            runs = list(pool.map(simulate, [task] * len(points), *zip(*points),
                                 [b] * len(points), [c] * len(points)))
            path = args.out / f'thread_{b}_c{args.clearance:g}.mp4'
            if b == 'compare':
                write_compare_video(path, task, c, points, runs)
            else:
                write_video(path, task, b, c, points, runs, metrics)
            print(f'wrote {path}', flush=True)


if __name__ == '__main__':
    main()
