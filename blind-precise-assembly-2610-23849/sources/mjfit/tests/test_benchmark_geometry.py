"""Ground-truth oracles and plate builders of the collision benchmark."""

import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from benchmark import geometry as geo
from benchmark.geometry import Task
from benchmark.representations import build_plate


def rotation(axis, angle):
    k = np.asarray(axis, dtype=float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def depth(task, c, pos, rot):
    return geo.penetration(task, c, pos, rot, geo.peg_samples(task, c, 5e-5),
                           geo.rim_samples(task, 2e-5))


@pytest.mark.parametrize('shape', ['circle', 'square'])
@pytest.mark.parametrize('c', [2e-4, 5e-5])
def test_oracles_touch_exactly_at_their_limit(shape, c):
    """Each exact answer is where true penetration starts: zero at the limit,
    positive 2% beyond it."""
    task = Task(shape)
    center = np.array([0, 0, task.thickness / 2])
    phis = [0.0] if shape == 'circle' else [0.0, math.pi / 8, math.pi / 4]
    for phi in phis:
        axis = [math.cos(phi), math.sin(phi), 0]
        theta = geo.max_tilt(task, c, phi)
        assert depth(task, c, center, rotation(axis, theta)) < 1e-7
        assert depth(task, c, center, rotation(axis, 1.02 * theta)) > 5e-7
    if shape == 'square':
        psi = geo.max_yaw(task, c)
        assert depth(task, c, center, rotation([0, 0, 1], psi)) < 1e-7
        assert depth(task, c, center, rotation([0, 0, 1], 1.02 * psi)) > 5e-7
    # Lateral support: along an axis the reach is exactly the translation limit.
    reach = geo.translation_support(task, c, np.array([1.0, 0.0]))
    assert depth(task, c, center + [reach, 0, 0], np.eye(3)) < 1e-7
    assert depth(task, c, center + [reach + 2e-6, 0, 0], np.eye(3)) > 1e-6


def test_circle_tilt_matches_small_angle_limit():
    task = Task('circle')
    assert geo.max_tilt(task, 1e-5) == pytest.approx(2e-5 / task.thickness, rel=5e-3)


@pytest.mark.parametrize('shape', ['circle', 'square'])
def test_exact_plates_have_the_true_volume(shape):
    """ours (circumscribed) and the baseline input mesh both keep the hole
    open; nothing fills the clearance."""
    task = Task(shape)
    plate = (2 * task.plate_half) ** 2 * task.thickness
    hole = (math.pi if shape == 'circle' else 4) * task.size ** 2 * task.thickness
    params = {'chord_error_um': 2.5} if shape == 'circle' else {}
    rep = build_plate(task, 'ours', params, use_cache=False)
    volume = sum(ConvexHull(h).volume for h in rep.hulls)
    # Circumscribed n-gon removes slightly more than the circle.
    assert plate - hole - volume == pytest.approx(0, abs=hole * 1e-3)
    import trimesh
    verts, faces = geo.plate_mesh(task)
    assert trimesh.Trimesh(verts, faces).volume == pytest.approx(plate - hole, rel=1e-5)


def test_vhacd_hulls_cover_the_plate():
    """Guards the VTK face layout: plain (n, 3) faces yield flat hulls."""
    rep = build_plate(Task('square'), 'vhacd', {'max_hulls': 16}, use_cache=False)
    task = Task('square')
    solid = ((2 * task.plate_half) ** 2 - 4 * task.size ** 2) * task.thickness
    assert sum(ConvexHull(h).volume for h in rep.hulls) > 0.95 * solid


@pytest.mark.parametrize('shape', ['circle', 'square'])
def test_true_signed_distance_matches_translation_gap(shape):
    """Separated: the true gap; overlapping: minus the per-wall penetration."""
    task, c = Task(shape), 1e-4
    pp, rp = geo.peg_samples(task, c, 5e-5), geo.rim_samples(task, 2e-5)
    center = np.array([0, 0, task.thickness / 2])
    for t, expected in [(0.4 * c, 0.6 * c), (c + 2e-5, -2e-5)]:
        sd = geo.true_signed_distance(task, c, center + [t, 0, 0], np.eye(3), pp, rp)
        assert sd == pytest.approx(expected, abs=5e-7)


def test_wall_offset_exact_and_one_sided_for_ours():
    from benchmark.wall import wall_offset
    square = wall_offset(Task('square'), build_plate(Task('square'), 'ours', {}, use_cache=False))
    assert abs(square['wall_min_um']) < 1e-6 and abs(square['wall_max_um']) < 1e-6
    # Circumscribed n-gon: never phantom, recessed by at most the chord error,
    # including rays running exactly along the seams between wedges.
    task = Task('circle')
    circle = wall_offset(task, build_plate(task, 'ours', {'chord_error_um': 10}, use_cache=False))
    assert circle['wall_min_um'] > -1e-6
    assert circle['wall_max_um'] <= 10 + 1e-6


def test_octree_sdf_sign():
    """MuJoCo's octree: negative inside the plate, positive in the hole."""
    import mujoco

    from benchmark.scene import build_scene
    from benchmark.wall import octree_sdf
    task = Task('circle')
    sc = build_scene(task, build_plate(task, 'sdf', {'depth': 6}), 0.0, joints='xy',
                     gravity=False, peg_pos=(0, 0, 1.0))
    m, d = sc.model, mujoco.MjData(sc.model)
    mujoco.mj_kinematics(m, d)
    g = int(np.flatnonzero(sc.plate_geoms)[0])
    world = np.array([[0, 0, task.thickness / 2], [0.016, 0, task.thickness / 2]])
    local = (world - d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3, 3)
    sdf = octree_sdf(m, m.geom_dataid[g], local)
    assert sdf[0] == pytest.approx(task.size, rel=0.05)      # hole axis: ~R from the wall
    assert sdf[1] == pytest.approx(-task.thickness / 2, rel=0.1)  # mid-plate: T/2 inside
