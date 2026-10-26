#!/usr/bin/env python3
"""用固定版本 mjfit 生成星形销孔，核验凸片、间隙与 MuJoCo 编译几何。"""

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mjfit
import mujoco
import numpy as np
from mjfit import _polygon as pg
from mjfit import shapes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources" / "mjfit"
EXPECTED = "eee986cc087d14458337fdc195d5cf03abc6aa6d"


def compiled_volume(model, mesh_id):
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start:start + count].astype(np.float64)
    start, count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    faces = model.mesh_face[start:start + count]
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6


def main():
    sha = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    assert sha == EXPECTED
    assert Path(mjfit.__file__).resolve().is_relative_to(SOURCE)
    hole = shapes.star(5, 0.012, 0.0054)
    clearance = 0.0005
    plate_size, thickness, peg_length = 0.048, 0.008, 0.020
    peg_outline = pg.offset_inward(hole, clearance)
    xml = mjfit.peg(
        hole=hole,
        clearance=clearance,
        plate_size=plate_size,
        plate_thickness=thickness,
        peg_length=peg_length,
        center=False,
        standalone=False,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    groups = {"plate": [], "peg": []}
    relative_errors = []
    areas = {"plate": 0.0, "peg": 0.0}
    for mesh in ET.fromstring(xml).iter("mesh"):
        if "vertex" not in mesh.attrib:
            continue
        name = mesh.attrib["name"]
        kind = "plate" if name.startswith("plate_") else "peg"
        vertices = np.fromstring(mesh.attrib["vertex"], sep=" ").reshape(-1, 3)
        ring = vertices[:len(vertices) // 2, :2]
        assert pg.is_convex(ring)
        area = abs(pg.signed_area(ring))
        areas[kind] += area
        expected_volume = area * np.ptp(vertices[:, 2])
        actual_volume = compiled_volume(model, model.mesh(name).id)
        relative_errors.append(abs(actual_volume - expected_volume) / expected_volume)
        groups[kind].append({
            "name": name,
            "polygon_mm": np.round(ring * 1000, 8).tolist(),
            "z0_mm": float(vertices[:, 2].min() * 1000),
            "z1_mm": float(vertices[:, 2].max() * 1000),
        })
    np.testing.assert_allclose(
        areas["plate"], plate_size**2 - pg.signed_area(hole), rtol=1e-8
    )
    np.testing.assert_allclose(areas["peg"], pg.signed_area(peg_outline), rtol=1e-8)
    edges = np.roll(hole, -1, axis=0) - hole
    inward = np.column_stack([-edges[:, 1], edges[:, 0]])
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    edge_clearance = np.einsum("ij,ij->i", inward, peg_outline - hole)
    np.testing.assert_allclose(edge_clearance, clearance, atol=1e-14)
    assert max(relative_errors) < 1e-6
    assert len(groups["plate"]) == 10 and len(groups["peg"]) == 10
    report = {
        "source_sha": sha,
        "mjfit_version": mjfit.__version__,
        "mujoco_version": mujoco.__version__,
        "example_not_paper_trial": True,
        "parameters_mm": {
            "star_outer_radius": 12, "star_inner_radius": 5.4,
            "clearance": clearance * 1000, "plate_size": plate_size * 1000,
            "plate_thickness": thickness * 1000, "peg_length": peg_length * 1000,
        },
        "hole_mm": np.round(hole * 1000, 8).tolist(),
        "peg_outline_mm": np.round(peg_outline * 1000, 8).tolist(),
        "groups": groups,
        "checks": {
            "plate_pieces": len(groups["plate"]),
            "peg_pieces": len(groups["peg"]),
            "mujoco_meshes": model.nmesh,
            "mujoco_bodies_excluding_world": model.nbody - 1,
            "all_pieces_convex": True,
            "plate_area_matches_plate_minus_hole": True,
            "peg_area_matches_offset_polygon": True,
            "edge_clearance_min_mm": float(edge_clearance.min() * 1000),
            "edge_clearance_max_mm": float(edge_clearance.max() * 1000),
            "max_compiled_hull_volume_relative_error": max(relative_errors),
            "dynamic_simulation_executed": False,
        },
    }
    (ROOT / "outputs" / "mjfit-star.xml").write_text(xml)
    (ROOT / "outputs" / "mjfit-geometry-inspection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    print("以上是生成与 MuJoCo 编译检查，没有推进动力学。")


if __name__ == "__main__":
    main()
