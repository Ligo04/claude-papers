"""Videos of Pareto-optimal settings under passive disturbances.

Two settings per method, both on that method's CPU Pareto front for the
circle at c = 100 µm (report/pareto_cpu_circle.png), each run through passive
behaviors whose exact answer is drawn, magnified, beside the render:

  drop : S1. The aligned, centred peg is released 1 mm above the hole under
         gravity. Exact: free fall, never touching the wall (c > 0).
  push : S2 made continuous. Free peg, no gravity, a 0.1 N lateral force at
         its centre that turns once every 2 s. Exact: the centre rides the
         circle of radius c.
  tilt : S3 made continuous. Free peg, no gravity, a 1.5 mN·m torque about a
         horizontal axis that turns once every 2 s. Exact: the peg axis sits
         at the hole's maximum tilt.

The render cuts the plate at y = 0 to show the collision geometry the
simulator uses: each convex piece in its own shade, or the zero level set of
the SDF octree. The peg is checkered and translucent. Black arrows are the
applied load: gravity, the push at the centre, or the torque drawn as its
equivalent couple at the two plate faces. Magenta arrows are the contact
forces the plate applies to the peg, at their contact points. Frames are
composed with matplotlib and piped to ffmpeg.

    MUJOCO_GL=egl python -m benchmark.render                        # all, c = 100 and 20 µm
    MUJOCO_GL=egl python -m benchmark.render --behaviors drop --clearance 20
"""
from __future__ import annotations

import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from . import geometry as geo  # noqa: E402
from . import report as R  # noqa: E402
from .geometry import Task  # noqa: E402
from .representations import PlateRep, build_plate, circle_sides  # noqa: E402
from .scene import build_scene  # noqa: E402
from .wall import octree_sdf  # noqa: E402

HERE = Path(__file__).resolve().parent

# (method, params, MULTICCD): a coarse and a fine setting from each method's
# CPU Pareto front on the circle at c = 100 µm.
POINTS = [
    ('ours', {'chord_error_um': 500}, False),
    ('ours', {'chord_error_um': 20}, False),
    ('coacd', {'threshold': 0.01}, False),
    ('coacd', {'threshold': 0.005}, False),
    ('vhacd', {'max_hulls': 1024, 'resolution': 20_000_000, 'min_error': 0.1,
               'max_depth': 12}, False),
    ('vhacd', {'max_hulls': 4096, 'resolution': 120_000_000, 'min_error': 0.1,
               'max_depth': 14}, False),
    ('sdf', {'depth': 5}, True),
    ('sdf', {'depth': 6}, True),
]


@dataclass(frozen=True)
class Behavior:
    title: str
    blurb: str
    duration: float   # simulated seconds
    frame_dt: float   # simulated seconds per video frame
    fps: int


BEHAVIORS = {
    'drop': Behavior('Drop (S1)',
                     'Aligned peg released 1 mm above the hole under gravity. '
                     'Exact: free fall, no wall contact.', 0.2, 2e-3, 20),
    'push': Behavior('Push (S2, continuous)',
                     'Free peg, no gravity. A 0.1 N lateral force at its centre turns '
                     'once every 2 s. Exact: the centre rides the circle of radius c.',
                     4.0, 0.02, 25),
    'tilt': Behavior('Tilt (S3, continuous)',
                     'Free peg, no gravity. A 1.5 mN·m torque about a horizontal axis '
                     'turns once every 2 s. Exact: the axis sits at the maximum tilt.',
                     4.0, 0.02, 25),
}
FORCE, TORQUE, PERIOD, DROP_HEIGHT = 0.1, 1.5e-3, 2.0, 1e-3
HOLD_S = 1.5  # final frame is held this long

# Pixels: 1920x1080 frame, 4 method columns x 2 setting rows.
FW, FH, TOP = 1920, 1080, 76
TW, TH = FW // 4, (FH - TOP) // 2
RW, RH = 205, 430            # render box inside a tile (drawn at 2x)
PEG_ALPHA = 0.3
LOAD_RGBA = np.array([0.05, 0.05, 0.05, 1.0])     # applied load
CONTACT_RGBA = np.array([0.88, 0.05, 0.55, 1.0])  # contact force on the peg
# Arrow length per newton and shaft width (m), per view: 0.1 N is 20 mm in the
# wide drop view and 10 mm in the close push / tilt views.
ARROW = {'drop': (0.2, 1.4e-3), 'push': (0.1, 0.8e-3), 'tilt': (0.1, 0.8e-3)}


# ----- simulation --------------------------------------------------------------------

def simulate(task: Task, method: str, params: dict, multiccd: bool, behavior: str,
             clearance: float) -> dict:
    """Run one behavior on the benchmark scene; one row per video frame."""
    rep = build_plate(task, method, params)
    b = BEHAVIORS[behavior]
    drop = behavior == 'drop'
    z0 = task.thickness + task.peg_length / 2 + DROP_HEIGHT if drop else task.thickness / 2
    sc = build_scene(task, rep, clearance, joints='free', gravity=drop, friction=True,
                     peg_pos=(0, 0, z0), multiccd=multiccd)
    m, d = sc.model, mujoco.MjData(sc.model)
    every = round(b.frame_dt / m.opt.timestep)
    nframe = round(b.duration / b.frame_dt) + 1
    peg_pts, rim_pts = geo.peg_samples(task, clearance, 2e-4), geo.rim_samples(task)
    qpos = np.empty((nframe, m.nq))
    touch = np.zeros(nframe, dtype=bool)
    pen = np.full(nframe, np.nan)
    contacts = []
    omega = 2 * math.pi / PERIOD

    def touching() -> bool:
        g = d.contact.geom[:d.ncon]
        return bool((sc.plate_geoms[g[:, 0]] | sc.plate_geoms[g[:, 1]]).any()) if d.ncon else False

    def contact_forces() -> np.ndarray:
        """(k, 6): world position and world force on the peg of each
        peg-plate contact."""
        out, f6 = [], np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if not (sc.plate_geoms[c.geom1] or sc.plate_geoms[c.geom2]):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            # Contact frame rows are (normal, t1, t2); the force acts on geom2
            # along +normal, on geom1 along -normal.
            force = c.frame.reshape(3, 3).T @ f6[:3]
            out.append(np.r_[c.pos, force if c.geom2 == sc.peg_geom else -force])
        return np.array(out).reshape(-1, 6)

    mujoco.mj_forward(m, d)
    for f in range(nframe):
        if f == 0:
            touch[0] = touching()
        for _ in range(every if f else 0):
            u = np.array([math.cos(omega * d.time), math.sin(omega * d.time), 0.0])
            if behavior == 'push':
                d.xfrc_applied[sc.peg_body, :3] = FORCE * u
            elif behavior == 'tilt':
                d.xfrc_applied[sc.peg_body, 3:] = TORQUE * u
            mujoco.mj_step(m, d)
            touch[f] |= touching()
        qpos[f] = d.qpos
        contacts.append(contact_forces())
        if np.isfinite(d.qpos).all():
            pen[f] = geo.penetration(task, clearance, d.qpos[:3].copy(),
                                     geo.quat_to_mat(d.qpos[3:7]), peg_pts, rim_pts)
    return {'qpos': qpos, 'touch': touch, 'pen_um': pen * 1e6, 'contacts': contacts,
            't': np.arange(nframe) * b.frame_dt, 'z0': z0,
            'weight': float(m.body_subtreemass[sc.peg_body] * 9.81)}


# ----- visual model: the plate as the collider sees it, cut at y = 0 --------------------

def _cut_hull(points: np.ndarray):
    """Convex piece clipped to y >= 0 (capped), or None if nothing is left."""
    import trimesh
    mesh = trimesh.convex.convex_hull(points)
    y = mesh.vertices[:, 1]
    if y.max() <= 1e-9:
        return None
    if y.min() < 0:
        mesh = mesh.slice_plane([0, 0, 0], [0, 1, 0], cap=True)
        if mesh is None or len(mesh.faces) < 4 or abs(mesh.volume) < 1e-13:
            return None
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _sdf_surface(task: Task, rep: PlateRep, h: float = 2e-4):
    """Zero level set of the compiled SDF octree over y >= 0 (world frame)."""
    import trimesh
    from skimage.measure import marching_cubes
    sc = build_scene(task, rep, 0.0, joints='xy', gravity=False,
                     peg_pos=(0, 0, 10 * task.peg_length))
    m, d = sc.model, mujoco.MjData(sc.model)
    mujoco.mj_kinematics(m, d)
    g = int(np.flatnonzero(sc.plate_geoms)[0])
    pos, rot = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
    P, T = task.plate_half, task.thickness
    xs = np.arange(-P - 1e-3, P + 1e-3, h)
    ys = np.arange(-3 * h, P + 1e-3, h)
    zs = np.arange(-1e-3, T + 1e-3, h)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)
    f = np.concatenate([octree_sdf(m, m.geom_dataid[g], (chunk - pos) @ rot)
                        for chunk in np.array_split(grid, max(1, len(grid) // 400_000))])
    f = np.maximum(f, -grid[:, 1]).reshape(len(xs), len(ys), len(zs))  # cut at y = 0
    verts, faces, _, _ = marching_cubes(f, 0.0, spacing=(h, h, h))
    verts += [xs[0], ys[0], zs[0]]
    if trimesh.Trimesh(verts, faces, process=False).volume < 0:
        faces = faces[:, ::-1]
    return verts, faces


def plate_visuals(task: Task, method: str, params: dict) -> list:
    """(verts, faces, rgba) per visual piece, world frame."""
    rep = build_plate(task, method, params)
    base = np.array(matplotlib.colors.to_rgb(R.COLOR[method]))
    if rep.sdf_mesh is not None:
        v, f = _sdf_surface(task, rep)
        return [(v, f, (*(base + (1 - base) * 0.35), 1.0))]
    rng = np.random.default_rng(0)
    out = []
    for hull in rep.hulls:
        cut = _cut_hull(hull)
        if cut is not None:
            v, f = cut
            k = rng.uniform(0.1, 0.65)  # each piece its own shade of the method color
            out.append((v + [0, 0, task.thickness / 2], f, (*(base + (1 - base) * k), 1.0)))
    return out


def _peg_mesh(r: float, length: float, n: int = 96, checks=(12, 8)):
    """Cylinder with texture coordinates: `checks` squares around and along
    the side, a 4 x 4 checker across each end face."""
    ang = np.linspace(0, 2 * np.pi, n + 1)  # seam duplicated so u wraps
    v, uv, f = [], [], []
    for k, z in enumerate((-length / 2, length / 2)):
        for j, a in enumerate(ang):
            v.append((r * np.cos(a), r * np.sin(a), z))
            uv.append((checks[0] / 2 * j / n, checks[1] / 2 * k))  # builtin checker = 2x2
    for j in range(n):
        a, b, c, e = j, j + 1, n + 1 + j, n + 2 + j
        f += [(a, b, e), (a, e, c)]
    for z, up in ((-length / 2, False), (length / 2, True)):
        c0 = len(v)
        v.append((0.0, 0.0, z))
        uv.append((1.0, 1.0))
        for a in ang[:-1]:
            v.append((r * np.cos(a), r * np.sin(a), z))
            uv.append((1 + np.cos(a), 1 + np.sin(a)))
        for j in range(n):
            i0, i1 = c0 + 1 + j, c0 + 1 + (j + 1) % n
            f.append((c0, i0, i1) if up else (c0, i1, i0))
    return np.array(v), np.array(uv), np.array(f)


# Camera per behavior: world z at the frame center, vertical extent (m), and
# elevation (deg). The close views look down more so that horizontal force
# arrows pointing toward or away from the camera still show their length.
VIEWS = {'drop': (0.005, 0.11, 12), 'push': (0.0075, 0.05, 30), 'tilt': (0.0075, 0.05, 30)}


def visual_model(task: Task, pieces: list, clearance: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 2 * RW, 2 * RH
    spec.visual.headlight.ambient = [0.45] * 3
    spec.visual.headlight.diffuse = [0.45] * 3
    spec.visual.headlight.specular = [0.05] * 3
    spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.98, 0.98, 0.975],
                     rgb2=[0.86, 0.86, 0.85], width=64, height=64)
    spec.add_texture(name='checker', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.96, 0.96, 0.97],
                     rgb2=[0.45, 0.47, 0.52], width=64, height=64)
    mat = spec.add_material(name='peg', rgba=[1, 1, 1, PEG_ALPHA])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'checker'
    world = spec.worldbody
    light = world.add_light(pos=[0.15, -0.25, 0.35], dir=[-0.3, 0.55, -0.78])
    light.diffuse = [0.35] * 3
    light.castshadow = False
    world.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.1, 0.1, 0.01],
                   pos=[0, 0, -task.peg_length - 0.005], rgba=[0.9, 0.9, 0.88, 1],
                   contype=0, conaffinity=0)
    for i, (v, f, rgba) in enumerate(pieces):
        mesh = spec.add_mesh(name=f'piece_{i}')
        mesh.uservert = np.asarray(v, dtype=np.float32).ravel()
        mesh.userface = np.asarray(f, dtype=np.int32).ravel()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'piece_{i}', rgba=rgba,
                       contype=0, conaffinity=0)
    peg = world.add_body(name='peg')
    peg.add_freejoint()
    pv, puv, pf = _peg_mesh(task.peg_half(clearance), task.peg_length)
    mesh = spec.add_mesh(name='peg')
    mesh.uservert = pv.ravel()
    mesh.usertexcoord = puv.ravel()
    mesh.userface = pf.ravel()
    peg.add_geom(name='peg', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='peg', material='peg',
                 contype=0, conaffinity=0)
    fovy = 20.0
    for name, (zc, extent, elev_deg) in VIEWS.items():
        elev = math.radians(elev_deg)
        z_cam = np.array([0, -math.cos(elev), math.sin(elev)])
        x_cam = np.array([1.0, 0, 0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.column_stack([x_cam, np.cross(z_cam, x_cam), z_cam]).ravel())
        dist = extent / 2 / math.tan(math.radians(fovy) / 2)
        world.add_camera(name=name, pos=np.array([0, 0, zc]) + dist * z_cam, quat=quat,
                         fovy=fovy)
    return spec.compile()


# ----- frame composition --------------------------------------------------------------

def _tag(method: str, params: dict, multiccd: bool, n_pieces: int) -> str:
    if method == 'ours':
        n = circle_sides(0.008, params['chord_error_um'] * 1e-6)
        s = f"{n}-gon (ε = {params['chord_error_um']:g} µm)"
    elif method == 'coacd':
        s = f"threshold {params['threshold']:g}, {n_pieces} hulls"
    elif method == 'vhacd':
        s = f"{params['resolution'] / 1e6:g}M voxels, {n_pieces} hulls"
    else:
        s = f"octree depth {params['depth']}"
    return s + ('' if multiccd else ', MULTICCD off')


def _metrics() -> dict:
    """label -> (CPU µs/step, model MB, {c: (error µm, filled)}) from the raw
    results (benchmark/results/peg/)."""
    p, speed = R.load()
    out = {}
    for method, params, mccd in POINTS:
        label = PlateRep(method, params).label
        out[label] = (R._cost(speed, 'circle', label, 'cpu', mccd=mccd),
                      R._xval(p, speed, 'circle', label, 'memory', mccd),
                      {c: R._accuracy(p, 'circle', label, c, mccd) for c in (100.0, 20.0)})
    return out


def _px(fig, x, y, w, h):
    """Axes at pixel box (x, y from top-left)."""
    return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])


def _arrow(scn: mujoco.MjvScene, pos, force, rgba, width: float, m_per_n: float):
    """Arrow from pos along force, m_per_n long per newton, capped at 0.15 N
    (impacts); forces under 2.5 mN are skipped."""
    newtons = np.linalg.norm(force)
    if newtons < 2.5e-3 or scn.ngeom >= scn.maxgeom:
        return
    tip = np.asarray(pos) + force / newtons * min(newtons, 0.15) * m_per_n
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(pos, float), tip)
    scn.ngeom += 1


class Tile:
    def __init__(self, fig, col, row, task, behavior, clearance, point, run, pieces, metrics):
        method, params, mccd = point
        self.method, self.behavior, self.task, self.run = method, behavior, task, run
        self.c_um = clearance * 1e6
        x0, y0 = col * TW, TOP + row * TH
        color = R.COLOR[method]
        cost, mem, acc = metrics[PlateRep(method, params).label]
        err = acc.get(round(self.c_um, 6)) if acc else None
        fig.text((x0 + 16) / FW, 1 - (y0 + 10) / FH, R.NAMES[method], color=color,
                 fontsize=12, fontweight='bold', va='top')
        fig.text((x0 + 16) / FW, 1 - (y0 + 32) / FH, _tag(method, params, mccd, len(pieces)),
                 color=R.INK, fontsize=9, va='top')
        err_s = ('fills the clearance' if err and err[1]
                 else f'S2–S4 error {err[0]:.3g} µm' if err else 'error n/a')
        fig.text((x0 + 16) / FW, 1 - (y0 + 50) / FH,
                 f'{cost:,.0f} µs/step · {mem:.2g} MB · {err_s} at c = {self.c_um:g} µm',
                 color=R.INK2, fontsize=8, va='top')

        self.frames = self._render(visual_model(task, pieces, clearance))
        ax = _px(fig, x0 + 8, y0 + 70, RW, RH)
        ax.set_axis_off()
        self.image = ax.imshow(self.frames[0], interpolation='nearest')

        px0, pw = x0 + RW + 58, TW - RW - 78
        self.readout = fig.text((px0 - 36) / FW, 1 - (y0 + 352) / FH, '', color=R.INK2,
                                fontsize=8, va='top', linespacing=1.6, family='monospace')
        getattr(self, f'_init_{behavior}')(_px(fig, px0, y0 + 80, pw, 220), color)

    def _render(self, model: mujoco.MjModel) -> np.ndarray:
        """Every frame of this tile, rendered at 2x and box-filtered down.
        Tiles render one after another: several live EGL contexts in one
        process read back each other's buffers."""
        data = mujoco.MjData(model)
        floor = model.geom('floor').id
        if self.behavior != 'drop':
            model.geom_rgba[floor, 3] = 0.0  # no gravity, no floor
        renderer = mujoco.Renderer(model, 2 * RH, 2 * RW, max_geom=20000)
        frames = np.empty((len(self.run['qpos']), RH, RW, 3), np.uint8)
        for f, q in enumerate(self.run['qpos']):
            if np.isfinite(q).all():
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)  # camera poses
            renderer.update_scene(data, camera=self.behavior)
            self._arrows(renderer.scene, f)
            img = renderer.render().astype(np.uint16)
            frames[f] = img.reshape(RH, 2, RW, 2, 3).sum(axis=(1, 3)) // 4
        renderer.close()
        return frames

    def _arrows(self, scn: mujoco.MjvScene, f: int):
        """Applied load (black) and contact forces on the peg (magenta)."""
        q = self.run['qpos'][f]
        if not np.isfinite(q).all():
            return
        com, axis = q[:3], geo.quat_to_mat(q[3:7])[:, 2]
        a = 2 * math.pi / PERIOD * self.run['t'][f]
        if self.behavior == 'drop':
            loads = [(com, np.array([0, 0, -self.run['weight']]))]
        elif self.behavior == 'push':
            loads = [(com, FORCE * np.array([math.cos(a), math.sin(a), 0.0]))]
        else:  # torque about (cos a, sin a, 0) as a couple at the plate faces
            F = TORQUE / self.task.thickness * np.array([math.sin(a), -math.cos(a), 0.0])
            half = self.task.thickness / 2 * axis
            loads = [(com + half, F), (com - half, -F)]
        scale, width = ARROW[self.behavior]
        for pos, force in loads:
            _arrow(scn, pos, force, LOAD_RGBA, width, scale)
        for row in self.run['contacts'][f]:
            _arrow(scn, row[:3], row[3:], CONTACT_RGBA, 0.8 * width, scale)

    # -- drop: bottom height vs time against free fall
    def _init_drop(self, ax, color):
        t_ms = self.run['t'] * 1e3
        T_mm, floor = self.task.thickness * 1e3, -(self.task.peg_length + 0.005) * 1e3
        b0 = (self.run['z0'] - self.task.peg_length / 2) * 1e3
        ax.axhspan(0, T_mm, color=R.GRID, lw=0)
        ax.text(t_ms[-1] * 0.97, T_mm / 2, 'plate', ha='right', va='center', fontsize=7,
                color=R.MUTED)
        ax.axhline(floor, color=R.BASE, lw=1)
        tt = np.linspace(0, t_ms[-1], 400)
        ax.plot(tt, np.maximum(b0 - 0.5 * 9.81 * (tt / 1e3) ** 2 * 1e3, floor), '--',
                color=R.MUTED, lw=1, label='exact (free fall)')
        self.line, = ax.plot([], [], color=color, lw=1.8, label='simulated')
        self.dot, = ax.plot([], [], 'o', color=color, ms=5)
        ax.set_xlim(0, t_ms[-1])
        ax.set_ylim(floor - 5, b0 + 4)
        ax.set_xlabel('time (ms)', fontsize=8)
        ax.set_ylabel('peg bottom height (mm)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='lower left')
        self.xy = np.column_stack([t_ms, (self.run['qpos'][:, 2] - self.task.peg_length / 2) * 1e3])

    def _update_drop(self, f):
        self.line.set_data(*self.xy[:f + 1].T)
        self.dot.set_data([self.xy[f, 0]], [self.xy[f, 1]])
        touch = self.run['touch'][:f + 1]
        bottom, T_mm = self.xy[f, 1], self.task.thickness * 1e3
        # Faster than 20 mm/s counts as moving; slower is a stuck peg creeping.
        speed = abs(bottom - self.xy[f - 1, 1]) / BEHAVIORS['drop'].frame_dt if f else 0.0
        sink = T_mm - bottom
        where = ('through the plate' if bottom < 0 else
                 'above the hole' if sink < -0.05 else
                 'sliding in the hole' if speed > 20 else
                 'stuck at the rim' if sink < 1 else f'stuck {sink:.1f} mm deep')
        return (f'wall contact  {touch.sum() * BEHAVIORS["drop"].frame_dt * 1e3:5.0f} ms\n'
                f'max true pen. {np.nanmax(self.run["pen_um"][:f + 1]):7.1f} µm\n'
                f'peg           {where}')

    # -- push: centre trace against the circle of radius c
    def _init_push(self, ax, color):
        lim = 1.7 * self.c_um
        ax.add_patch(plt.Circle((0, 0), self.c_um, fill=False, ls='--', ec=R.MUTED, lw=1))
        ax.text(0, -self.c_um * 1.08, 'exact: radius c', ha='center', va='top', fontsize=7,
                color=R.MUTED)
        self.line, = ax.plot([], [], color=color, lw=1.0)
        self.dot, = ax.plot([], [], 'o', color=color, ms=5)
        self.arrow, = ax.plot([], [], color=R.INK2, lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('peg centre x (µm)', fontsize=8)
        ax.set_ylabel('y (µm)', fontsize=8)
        ax.tick_params(labelsize=7)
        self.xy = (self.run['qpos'][:, :2] - self.run['qpos'][0, :2]) * 1e6

    def _update_push(self, f):
        q = self.run['qpos'][f]
        self.line.set_data(*self.xy[:f + 1].T)
        self.dot.set_data([self.xy[f, 0]], [self.xy[f, 1]])
        a = 2 * math.pi / PERIOD * self.run['t'][f]
        self.arrow.set_data([0, 0.45 * self.c_um * math.cos(a)], [0, 0.45 * self.c_um * math.sin(a)])
        tilt = math.acos(min(1.0, geo.quat_to_mat(q[3:7])[2, 2])) * 1e3
        return (f'|centre| - c  {np.hypot(*self.xy[f]) - self.c_um:+8.1f} µm\n'
                f'axial drift   {(q[2] - self.run["z0"]) * 1e6:+8.1f} µm\n'
                f'tilt          {tilt:8.2f} mrad\n'
                f'max true pen. {np.nanmax(self.run["pen_um"][:f + 1]):8.1f} µm')

    # -- tilt: axis direction against the maximum tilt
    def _init_tilt(self, ax, color):
        self.theta_max = geo.max_tilt(self.task, self.c_um * 1e-6) * 1e3
        lim = 1.7 * self.theta_max
        ax.add_patch(plt.Circle((0, 0), self.theta_max, fill=False, ls='--', ec=R.MUTED, lw=1))
        ax.text(0, -self.theta_max * 1.08, 'exact: max tilt', ha='center', va='top', fontsize=7,
                color=R.MUTED)
        self.line, = ax.plot([], [], color=color, lw=1.0)
        self.dot, = ax.plot([], [], 'o', color=color, ms=5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('peg axis tilt x (mrad)', fontsize=8)
        ax.set_ylabel('y (mrad)', fontsize=8)
        ax.tick_params(labelsize=7)
        rot = [geo.quat_to_mat(q[3:7]) for q in self.run['qpos']]
        self.xy = np.array([r[:2, 2] for r in rot]) * 1e3

    def _update_tilt(self, f):
        q = self.run['qpos'][f]
        self.line.set_data(*self.xy[:f + 1].T)
        self.dot.set_data([self.xy[f, 0]], [self.xy[f, 1]])
        err = np.hypot(*self.xy[f]) - self.theta_max
        return (f'tilt - max    {err:+8.2f} mrad\n'
                f'  at faces    {err * self.task.thickness / 2 * 1e3:+8.1f} µm\n'
                f'centre drift  {np.hypot(*(q[:2] - self.run["qpos"][0, :2])) * 1e6:8.1f} µm\n'
                f'max true pen. {np.nanmax(self.run["pen_um"][:f + 1]):8.1f} µm')

    def update(self, f):
        self.image.set_data(self.frames[f])
        self.readout.set_text(getattr(self, f'_update_{self.behavior}')(f))


def write_video(path: Path, task: Task, behavior: str, clearance: float, runs: list,
                pieces: list, metrics: dict):
    import imageio_ffmpeg
    b = BEHAVIORS[behavior]
    fig = plt.figure(figsize=(FW / 100, FH / 100), dpi=100)
    c_um = clearance * 1e6
    slow = 1 / (b.fps * b.frame_dt)
    fig.text(16 / FW, 1 - 12 / FH, f'{b.title}   c = {c_um:g} µm radial clearance '
             f'(clearance ratio {R.ratio(c_um)})', fontsize=14, fontweight='bold',
             color=R.INK, va='top')
    fig.text(16 / FW, 1 - 36 / FH, b.blurb + '  Cutaway at y = 0: the collision geometry '
             'the simulator uses. Plots are magnified; dashed = exact geometry.',
             fontsize=9, color=R.INK2, va='top')
    load = {'drop': 'weight', 'push': 'applied push (acts at the centre)',
            'tilt': 'applied torque, drawn as its equivalent couple at the two plate faces'}
    fig.text(16 / FW, 1 - 54 / FH, f'Arrows, 0.1 N = {ARROW[behavior][0] * 100:g} mm '
             f'(capped at 0.15 N):   black = {load[behavior]};   '
             'magenta = contact forces the plate applies to the peg.',
             fontsize=9, color=R.INK2, va='top')
    clock = fig.text(1 - 16 / FW, 1 - 14 / FH, '', fontsize=12, color=R.INK, va='top',
                     ha='right', family='monospace')
    for i in range(1, 4):
        fig.add_artist(plt.Line2D([i * TW / FW] * 2, [0.02, 1 - (TOP + 6) / FH],
                                  color=R.GRID, lw=1))
    tiles = [Tile(fig, i // 2, i % 2, task, behavior, clearance, POINTS[i], runs[i], pieces[i],
                  metrics) for i in range(len(POINTS))]
    nframe = len(runs[0]['t'])
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{FW}x{FH}', '-r', str(b.fps), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(path)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(nframe):
        for tile in tiles:
            tile.update(f)
        t = runs[0]['t'][f]
        clock.set_text(f't = {t * 1e3:4.0f} ms  ({slow:g}x slow motion)' if slow > 1.01
                       else f't = {t:4.2f} s')
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes()
        for _ in range(1 + (round(HOLD_S * b.fps) if f == nframe - 1 else 0)):
            ff.stdin.write(frame)
    ff.stdin.close()
    ff.wait()
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--behaviors', nargs='*', choices=list(BEHAVIORS), default=list(BEHAVIORS))
    ap.add_argument('--clearance', nargs='*', type=float, default=[100.0, 20.0],
                    help='radial clearances in µm')
    ap.add_argument('--out', type=Path, default=HERE / 'results' / 'render')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    task = Task('circle')
    metrics = _metrics()
    videos = [(b, c) for c in args.clearance for b in args.behaviors]
    with ProcessPoolExecutor(args.workers) as pool:
        sims = {(b, c, i): pool.submit(simulate, task, *POINTS[i], b, c * 1e-6)
                for b, c in videos for i in range(len(POINTS))}
        pieces = list(pool.map(plate_visuals, [task] * len(POINTS),
                               [p[0] for p in POINTS], [p[1] for p in POINTS]))
        for b, c in videos:
            runs = [sims[b, c, i].result() for i in range(len(POINTS))]
            path = args.out / f'{b}_c{c:g}.mp4'
            write_video(path, task, b, c * 1e-6, runs, pieces, metrics)
            print(f'wrote {path}', flush=True)


if __name__ == '__main__':
    main()
