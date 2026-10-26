"""Putting several generated assemblies into one MuJoCo model."""
from __future__ import annotations

import math

import mujoco

from ._mjcf import SOLIMP, SOLREF


def grid(parts, spacing: float, *, floor: bool = True, timestep: float = 0.001) -> mujoco.MjSpec:
    """Lay out MJCF models on a square grid in one :class:`mujoco.MjSpec`.

    Every element of part ``k`` is renamed with the prefix ``"p{k}_"`` (bodies,
    joints, geoms, meshes and default classes), so the copies never clash.
    Generate the parts with ``standalone=False``; the grid adds one floor, a
    light, and the solver options.

    Args:
        parts: MJCF strings.
        spacing: Distance between neighboring grid cells, in meters.
        floor: Add a ground plane.
        timestep: Simulation time step in seconds.

    Returns:
        The combined model; ``spec.compile()`` gives the MjModel, and then
        ``spec.to_xml()`` the MJCF.
    """
    parts = list(parts)
    spec = mujoco.MjSpec()
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    cols = max(1, math.ceil(math.sqrt(len(parts))))
    extent = spacing * cols
    spec.worldbody.add_light(pos=[0, 0, 2 * extent + 1], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    if floor:
        spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[extent, extent, 0.01], rgba=[0.92, 0.92, 0.92, 1],
                                solref=list(SOLREF), solimp=list(SOLIMP))
    for k, xml in enumerate(parts):
        row, col = divmod(k, cols)
        frame = spec.worldbody.add_frame(pos=[(col - (cols - 1) / 2) * spacing,
                                              ((cols - 1) / 2 - row) * spacing, 0.0])
        spec.attach(mujoco.MjSpec.from_string(xml), prefix=f'p{k}_', frame=frame)
    return spec
