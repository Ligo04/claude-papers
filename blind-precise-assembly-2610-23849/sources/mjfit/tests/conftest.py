import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def inline_meshes(xml: str) -> dict[str, np.ndarray]:
    """Vertices of every inline mesh, by name."""
    return {
        m.get('name'): np.array(m.get('vertex').split(), float).reshape(-1, 3)
        for m in ET.fromstring(xml).iter('mesh')
        if m.get('vertex')
    }


def hull_volume(model: mujoco.MjModel, mesh: int) -> float:
    """Volume of a compiled mesh (MuJoCo's convex hull of the given points)."""
    v = model.mesh_vert[
        model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]
    ]
    f = model.mesh_face[
        model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]
    ]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()) / 6


def prism_volume(vertices: np.ndarray) -> float:
    """Volume of a prism given as a bottom ring followed by a top ring."""
    n = len(vertices) // 2
    ring = vertices[:n, :2]
    area = 0.5 * abs(
        np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])
    )
    return area * (vertices[n:, 2].mean() - vertices[:n, 2].mean())
