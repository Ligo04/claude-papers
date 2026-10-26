"""Common MjSpec scene for the bolt/nut probes.

The bolt is fixed in the world with its thread axis along z. The nut is the
moving part: it starts centred in its play (frames on the pitch grid) and
moves on the probe's joints. Options and contact parameters are the peg
benchmark's (mjfit's default solref and solimp). The nut's mass and inertia are those of
the exact nut, set explicitly, so representations differ only in contact.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from ..scene import FRICTION, SOLIMP, SOLREF
from . import geometry as geo
from .geometry import ThreadTask
from .representations import PartRep, ThreadRep, exact_meshes

JOINTS = ('free', 'z', 'xy', 'tilt', 'screw')


@dataclass
class Scene:
    model: mujoco.MjModel
    nut_body: int
    bolt_geoms: np.ndarray   # bool masks over geoms
    nut_geoms: np.ndarray
    compile_s: float
    n_geoms: int             # bolt + nut collision geoms
    mass: float
    inertia_axis: float      # nut moment about the thread axis


@lru_cache(maxsize=8)
def _inertia(task: ThreadTask, clearance: float):
    return geo.nut_inertia(task, clearance, exact_meshes(task, clearance)[1])


def _add_part(spec, body, part: PartRep, name: str, depth: int):
    if part.analytic is not None:
        from . import plugin
        pname = plugin.load()
        spec.activate_plugin(pname)
        inst = spec.add_plugin(name=f'{name}_iso', plugin_name=pname, active=True)
        inst.config = {k: repr(float(v)) for k, v in part.analytic.items()}
        mesh = spec.add_mesh(name=f'{name}_iso')
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_SDF, meshname=f'{name}_iso')
        for ref in (mesh.plugin, geom.plugin):
            ref.plugin_name, ref.name, ref.active = pname, f'{name}_iso', True
    if part.sdf_mesh is not None:
        v, f = part.sdf_mesh
        mesh = spec.add_mesh(name=f'{name}_sdf')
        mesh.uservert = np.asarray(v).ravel()
        mesh.userface = np.asarray(f).ravel()
        mesh.octree_maxdepth = depth
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SDF, meshname=f'{name}_sdf')
    for i, hull in enumerate(part.hulls):
        mesh = spec.add_mesh(name=f'{name}_{i}')
        mesh.uservert = np.asarray(hull).ravel()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'{name}_{i}')
    for r, z0, z1 in part.cylinders:
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, (z1 - z0) / 2, 0],
                      pos=[0, 0, (z0 + z1) / 2])


def build_scene(task: ThreadTask, rep: ThreadRep, *, joints: str = 'free',
                gravity: bool = False, friction: float = 0.0, nut_offset=(0.0, 0.0, 0.0),
                tilt_axis=(1.0, 0.0, 0.0), timestep: float = 1e-3, multiccd: bool = True,
                settle_time: float = 0.02, decorate=None, nut_angle: float = 0.0,
                condim: int | None = None, pivot: float | None = None) -> Scene:
    """Compile a probe scene.

    joints: 'free', 'z' (axial slide), 'xy' (lateral slides), 'tilt' (hinge
    about `tilt_axis` through the point `pivot` above the nut's base on its
    axis, default its centre), 'screw' (axial slide + hinge about the thread
    axis). Joint axes are in the nut's frame, which `nut_angle` turns.
    friction = 0 uses frictionless contacts (condim 1);
    friction > 0 uses condim 4 unless `condim` says otherwise.
    Constrained probes get passive joint damping with time constant
    `settle_time` so equilibria settle; it is not actuation. `decorate(spec,
    bolt_body, nut_body)` may add visual-only elements (cameras, lights,
    contype-0 markers) before compiling; it must not change the physics.
    `nut_angle` turns the nut's start pose along the helix (rotation about the
    axis plus the matching lead), which keeps it centred in its play.
    """
    if joints not in JOINTS:
        raise ValueError(f"joints must be one of {JOINTS}")
    c = rep.clearance
    spec = mujoco.MjSpec()
    spec.modelname = f'thread_{rep.method}'
    # Fine decompositions with MULTICCD report thousands of contacts; the
    # default arena (sized from the model) overflows.
    spec.memory = 512 * 2**20
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
    g.friction = (friction, FRICTION[1], FRICTION[2]) if friction > 0 else FRICTION
    g.condim = condim if condim is not None else 4 if friction > 0 else 1

    depth = int(rep.params.get('depth', 6))
    bolt = spec.worldbody.add_body(name='bolt')
    _add_part(spec, bolt, rep.bolt, 'bolt', depth)

    mass, com, inertia, i_axis = _inertia(task, c)
    nut = spec.worldbody.add_body(
        name='nut', pos=[nut_offset[0], nut_offset[1],
                         task.nut_home() + nut_offset[2] + geo.helix_z(task, nut_angle)],
        quat=[math.cos(nut_angle / 2), 0, 0, math.sin(nut_angle / 2)])
    nut.explicitinertial = True
    nut.mass = mass
    nut.ipos = com
    nut.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                       inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    centre = [0, 0, task.nut_height / 2 if pivot is None else pivot]
    if joints == 'free':
        nut.add_freejoint()
    if joints in ('z', 'screw'):
        nut.add_joint(name='z', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    if joints == 'xy':
        nut.add_joint(name='x', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
        nut.add_joint(name='y', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    if joints == 'tilt':
        nut.add_joint(name='tilt', type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(tilt_axis),
                      pos=centre)
    if joints == 'screw':
        nut.add_joint(name='spin', type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    _add_part(spec, nut, rep.nut, 'nut', depth)
    if decorate is not None:
        decorate(spec, bolt, nut)

    t0 = time.perf_counter()
    model = spec.compile()
    compile_s = time.perf_counter() - t0

    nut_id = model.body('nut').id
    if joints != 'free':
        for j in range(model.njnt):
            dof = model.jnt_dofadr[j]
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                inert = mass
            else:
                # Moment of the nut about the joint axis through the joint anchor.
                axis = model.jnt_axis[j]
                anchor = model.jnt_pos[j]
                d = com - anchor
                d_perp = d - axis * (d @ axis)
                inert = axis @ inertia @ axis + mass * (d_perp @ d_perp)
            model.dof_damping[dof] = inert / settle_time
    colliding = (model.geom_contype | model.geom_conaffinity) > 0
    bolt_geoms = (model.geom_bodyid == model.body('bolt').id) & colliding
    nut_geoms = (model.geom_bodyid == nut_id) & colliding
    return Scene(model, nut_id, bolt_geoms, nut_geoms, compile_s,
                 int(bolt_geoms.sum() + nut_geoms.sum()), mass, i_axis)


def pair_contacts(scene: Scene, data: mujoco.MjData):
    """(dist, pos, force on the nut in world) of every bolt-nut contact."""
    out = []
    f6 = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        if scene.nut_geoms[con.geom1] == scene.nut_geoms[con.geom2]:
            continue
        mujoco.mj_contactForce(scene.model, data, i, f6)
        force = con.frame.reshape(3, 3).T @ f6[:3]
        out.append((con.dist, con.pos.copy(), force if scene.nut_geoms[con.geom2] else -force))
    return out
