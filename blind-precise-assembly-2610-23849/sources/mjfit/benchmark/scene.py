"""Common MjSpec scaffold shared by every representation and probe.

Everything except the plate's collision geometry is identical across methods:
options, contact parameters (mjfit.peg's defaults), the exact primitive peg, and the
probe's joints. Scenes are built with MjSpec because the SDF octree depth only
exists on the spec, and meshes are passed as arrays (not files) so MuJoCo's
file-keyed mesh cache cannot hand back an octree built at another depth.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .geometry import Task
from .representations import PlateRep

# mjfit.peg's default contact parameters.
SOLREF = (0.002, 1.0)
SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
FRICTION = (1.0, 0.01, 0.001)

JOINTS = ('free', 'xy', 'tilt', 'yaw')


@dataclass
class Scene:
    model: mujoco.MjModel
    peg_body: int
    peg_geom: int
    plate_geoms: np.ndarray   # bool mask over geoms
    compile_s: float
    n_plate_geoms: int


def build_scene(task: Task, rep: PlateRep, clearance: float, *,
                joints: str = 'free', phi: float = 0.0, gravity: bool = True,
                friction: bool = True, peg_pos=(0.0, 0.0, 0.0),
                timestep: float = 1e-3, multiccd: bool = True,
                settle_time: float = 0.02) -> Scene:
    """Compile a probe scene.

    joints: 'free' (freejoint), 'xy' (x/y slides, rotation locked), 'tilt'
    (slides + hinge about (cos phi, sin phi, 0)), 'yaw' (slides + hinge z).
    Constrained probes get passive joint damping with time constant
    `settle_time` so frictionless equilibria settle; it is not actuation.
    """
    if joints not in JOINTS:
        raise ValueError(f"joints must be one of {JOINTS}")
    spec = mujoco.MjSpec()
    spec.modelname = f'{task.shape}_{rep.method}'
    opt = spec.option
    opt.timestep = timestep
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.gravity = [0, 0, -9.81 if gravity else 0.0]
    opt.sdf_initpoints = int(rep.params.get('initpoints', 40))
    opt.sdf_iterations = int(rep.params.get('iterations', 10))
    if not multiccd:
        opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD

    g = spec.default.geom
    g.solref = SOLREF
    g.solimp = SOLIMP
    g.friction = FRICTION
    g.condim = 4 if friction else 1

    if gravity:  # catches a peg that falls through; peg top then sits below the plate
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01],
                                pos=[0, 0, -task.peg_length - 0.005])

    plate = spec.worldbody.add_body(name='plate', pos=[0, 0, task.thickness / 2])
    if rep.sdf_mesh is not None:
        verts, faces = rep.sdf_mesh
        mesh = spec.add_mesh(name='plate_sdf')
        mesh.uservert = verts.ravel()
        mesh.userface = faces.ravel()
        mesh.octree_maxdepth = int(rep.params.get('depth', 6))
        plate.add_geom(type=mujoco.mjtGeom.mjGEOM_SDF, meshname='plate_sdf')
        n_plate = 1
    else:
        for i, hull in enumerate(rep.hulls):
            mesh = spec.add_mesh(name=f'plate_{i}')
            mesh.uservert = hull.ravel()
            plate.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'plate_{i}')
        n_plate = len(rep.hulls)

    peg = spec.worldbody.add_body(name='peg', pos=list(peg_pos))
    if joints == 'free':
        peg.add_freejoint()
    else:
        for axis in ([1, 0, 0], [0, 1, 0]):
            peg.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
        if joints == 'tilt':
            peg.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE,
                          axis=[math.cos(phi), math.sin(phi), 0])
        elif joints == 'yaw':
            peg.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    h, half_len = task.peg_half(clearance), task.peg_length / 2
    if task.shape == 'circle':
        peg.add_geom(name='peg', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[h, half_len, 0])
    else:
        peg.add_geom(name='peg', type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, half_len])

    t0 = time.perf_counter()
    model = spec.compile()
    compile_s = time.perf_counter() - t0

    peg_body = model.body('peg').id
    if joints != 'free':
        # Damping per dof: mass for slides, inertia about the hinge axis for hinges.
        for j in range(model.njnt):
            dof = model.jnt_dofadr[j]
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                inertia = model.body_mass[peg_body]
            else:
                inertia = model.body_inertia[peg_body] @ np.abs(model.jnt_axis[j])
            model.dof_damping[dof] = inertia / settle_time
    plate_geoms = model.geom_bodyid == model.body('plate').id
    return Scene(model, peg_body, model.geom('peg').id, plate_geoms, compile_s, n_plate)


def plate_contacts(scene: Scene, data: mujoco.MjData):
    """(dist, pos, normal force) of every peg-plate contact."""
    out = []
    f = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if scene.plate_geoms[c.geom1] or scene.plate_geoms[c.geom2]:
            mujoco.mj_contactForce(scene.model, data, i, f)
            out.append((c.dist, c.pos.copy(), f[0]))
    return out
