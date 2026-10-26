"""Exact bolt/nut geometry and oracles of the thread benchmark."""

import math

import numpy as np
import pytest

import mjfit
from benchmark.thread import geometry as G
from mjfit import _polygon, _thread

TASK = G.ThreadTask()


@pytest.mark.parametrize('c', [1e-4, 2e-5])
def test_sampled_surfaces_lie_on_the_exact_boundary(c):
    bolt = G.bolt_surface(TASK, c, 0.004, 0.012, spacing=2e-4)
    nut = G.nut_surface(TASK, c, spacing=2e-4)
    assert np.abs(G.bolt_sdf(TASK, c, bolt)).max() < 1e-9
    assert np.abs(G.nut_sdf(TASK, c, nut)).max() < 1e-9


@pytest.mark.parametrize('c', [1e-4, 2e-5])
def test_centred_nut_play(c):
    """Centred gap c/2 (nut tip over the bolt core), axial play 2c/sqrt3 each
    way (flanks c apart along their normal), radial play c/2."""
    pair = G.Pair(TASK, c, spacing=1e-4)
    home = np.array([0, 0, TASK.nut_home()])
    assert pair.signed_distance(home, np.eye(3)) == pytest.approx(c / 2, abs=1e-9)
    for sign in (1, -1):
        assert G.axial_play(pair, sign) == pytest.approx(2 * c / math.sqrt(3), abs=5e-8)
    for theta in (0.0, 2.0):
        assert G.radial_play(pair, theta) == pytest.approx(c / 2, abs=5e-8)


def test_meshes_are_watertight_and_match_the_exact_parts():
    import trimesh
    c = 1e-4
    for build, sdf in ((G.bolt_mesh, G.bolt_sdf), (G.nut_mesh, G.nut_sdf)):
        v, f = build(TASK, c)
        assert trimesh.Trimesh(v, f, process=False).is_watertight
        # Every vertex on the exact surface, to float32 and 1-degree chord precision.
        assert np.abs(sdf(TASK, c, v)).max() < 1e-6


def test_spin_down_acceleration():
    assert G.spin_down_accel(TASK, 1.0, 0.0) == pytest.approx(9.81)
    m, _, _, i_axis = G.nut_inertia(TASK, 1e-4)
    k = 2 * math.pi / TASK.pitch
    assert G.spin_down_accel(TASK, m, i_axis) == pytest.approx(9.81 * m / (m + i_axis * k * k))


# ----- thread forms ----------------------------------------------------------------

FORM_TASKS = {'trapezoidal': G.ThreadTask(form='trapezoidal'),
              'square': G.ThreadTask(form='square'),
              'buttress': G.ThreadTask(form='buttress'),
              'wire': G.ThreadTask(form='wire')}


def test_iso_task_keeps_its_cache_key_and_references():
    assert 'form' not in TASK.key()
    assert G.ThreadTask(form='square').key()['form'] == 'square'
    # the textbook ISO run-down torque: pitch radius (d - 0.6495 P) / 2, mu / cos 30
    r2 = (TASK.major_diameter - 0.6495 * TASK.pitch) / 2
    tl, mf = TASK.pitch / (2 * math.pi * r2), 0.3 / math.cos(math.radians(30))
    assert G.rundown_torque(TASK, 0.01, 0.3, 1e-4) == pytest.approx(
        9.81 * 0.01 * r2 * (mf - tl) / (1 + mf * tl), rel=1e-5)


@pytest.mark.parametrize('form, down, up', [
    ('trapezoidal', 1 / math.cos(math.radians(15)), 1 / math.cos(math.radians(15))),
    ('square', 1.0, 1.0),
    # the nut rests on the steep 7° flank; lifted, it meets the 45° one
    ('buttress', 1 / math.cos(math.radians(7)), 1 / math.cos(math.radians(45)))])
def test_axial_play_of_flanked_forms(form, down, up):
    """A flank at angle a from the radial plane, c apart along its normal,
    closes after an axial slide of c / cos a."""
    c = 1e-4
    task = FORM_TASKS[form]
    assert G.axial_gap(task, c, -1) == pytest.approx(c * down, rel=1e-9)
    assert G.axial_gap(task, c, 1) == pytest.approx(c * up, rel=1e-9)


@pytest.mark.parametrize('form', list(FORM_TASKS))
def test_form_surfaces_lie_on_the_exact_boundary(form):
    task, c = FORM_TASKS[form], 1e-4
    bolt = G.bolt_surface(task, c, 0.004, 0.012, spacing=2e-4)
    nut = G.nut_surface(task, c, spacing=2e-4)
    assert np.abs(G.bolt_sdf(task, c, bolt)).max() < 1e-9
    assert np.abs(G.nut_sdf(task, c, nut)).max() < 1e-9


def test_analytic_sdf_config_carries_the_tooth():
    from benchmark.thread.representations import build_thread
    task, c = FORM_TASKS['square'], 1e-4
    rep = build_thread(task, 'sdf_analytic', {'initpoints': 40, 'iterations': 10}, c)
    for part, tooth in ((rep.bolt, G.bolt_tooth(task, c)), (rep.nut, G.nut_tooth(task, c))):
        a = part.analytic
        assert a['nvert'] == len(tooth)
        got = np.array([[a[f'v{i}r'], a[f'v{i}u']] for i in range(len(tooth))])
        np.testing.assert_array_equal(got, tooth)
    # ISO keeps the plugin's built-in tooth
    assert 'nvert' not in build_thread(TASK, 'sdf_analytic', {}, c).bolt.analytic


@pytest.mark.parametrize('name', ['sawtooth', 'six_start', 'lefty_square', 'coil',
                                  'house_dart', 'triple_wire',
                                  'dovetail', 'hook', 't_slot', 'barb'])
def test_wacky_screws_stay_inside_and_start_interlocked(name):
    """Multi-start, left-hand, custom and undercut (non-convex) teeth end to
    end: mjfit's wedges inside the exact parts, the exact surfaces on their
    boundary, and the scene's nut on the thread (no contact centred, contact
    when shifted)."""
    import mujoco

    from benchmark.thread.representations import build_thread
    from benchmark.thread.scene import build_scene, pair_contacts
    from benchmark.thread.wacky import DESIGNS, UNDERCUT
    from tests.test_thread import _hull_surface
    d = {**DESIGNS, **UNDERCUT}[name]
    task, c = d.task, d.clearance_um * 1e-6
    rep = build_thread(task, 'ours', {'segments': 24, 'relief_um': 5}, c, use_cache=False)
    rng = np.random.default_rng(0)
    for hulls, sdf in ((rep.bolt.hulls, G.bolt_sdf), (rep.nut.hulls, G.nut_sdf)):
        pts = np.concatenate([_hull_surface(v, rng, 4) for v in hulls])
        assert sdf(task, c, pts).max() < 1e-9
    home = task.nut_home()
    bolt = G.bolt_surface(task, c, home, home + task.nut_height, spacing=3e-4)
    assert np.abs(G.bolt_sdf(task, c, bolt)).max() < 1e-9
    assert np.abs(G.nut_sdf(task, c, G.nut_surface(task, c, spacing=3e-4))).max() < 1e-9
    sc = build_scene(task, rep, joints='free')

    def contacts(dz):
        data = mujoco.MjData(sc.model)
        data.qpos[2] += dz
        mujoco.mj_forward(sc.model, data)
        return pair_contacts(sc, data)
    assert contacts(0.0) == []
    assert contacts(task.pitch / 3) + contacts(-task.pitch / 3)


@pytest.mark.parametrize('name', ['hook', 't_slot', 'barb'])
def test_undercut_teeth_are_cut_into_convex_pieces_that_tile_them(name):
    from benchmark.thread.wacky import UNDERCUT
    task = UNDERCUT[name].task
    for tooth in G._profile(task).teeth(1e-4):
        assert not _polygon.is_convex(tooth)
        pieces = G._pieces(tooth)
        assert len(pieces) > 1 and all(_polygon.is_convex(q) for q in pieces)
        area = sum(abs(_polygon.signed_area(q)) for q in pieces)
        assert area == pytest.approx(abs(_polygon.signed_area(tooth)), rel=1e-12)


@pytest.mark.parametrize('name', ['sawtooth', 'dovetail', 'hook', 't_slot', 'barb'])
def test_wedges_cover_the_teeth(name):
    """No material goes missing: points well inside the exact teeth (in from
    every face by more than the discretization takes off) lie inside some
    wedge. Protrusion tests cannot see a dropped piece."""
    from scipy.spatial import ConvexHull

    from benchmark.thread.representations import build_thread
    from benchmark.thread.wacky import DESIGNS, UNDERCUT
    d = {**DESIGNS, **UNDERCUT}[name]
    task, c = d.task, d.clearance_um * 1e-6
    rep = build_thread(task, 'ours', {'segments': 24, 'relief_um': 5}, c, use_cache=False)
    prof = G._profile(task)
    rng = np.random.default_rng(1)
    for tooth, hulls, lo, hi in ((prof.teeth(c)[0], rep.bolt.hulls, 0.0, task.bolt_length),
                                 (prof.teeth(c)[1], rep.nut.hulls, 0.0, task.nut_height)):
        margin = 0.1 * task.pitch if task.pitch < 3e-3 else 1.5e-4
        inner = (_thread._offset_polygon(tooth, margin) if _polygon.is_convex(tooth)
                 else _thread._offset_mitre(tooth, margin))
        assert inner is not None
        box_lo, box_hi = inner.min(0), inner.max(0)
        q = box_lo + rng.random((20000, 2)) * (box_hi - box_lo)
        q = q[G._polygon_sd(q, inner) < 0][:400]
        phi = rng.uniform(0, 2 * math.pi, len(q))
        z = q[:, 1] + task.hand * task.lead * phi / (2 * math.pi)
        z += task.pitch * np.round((0.5 * (lo + hi) - z) / task.pitch)   # into the part
        pts = np.column_stack([q[:, 0] * np.cos(phi), q[:, 0] * np.sin(phi), z])
        inside = np.zeros(len(pts), bool)
        for h in hulls:
            eq = ConvexHull(h).equations
            inside |= (pts @ eq[:, :3].T + eq[:, 3] <= 1e-12).all(axis=1)
        assert inside.all(), f'{(~inside).sum()} of {len(pts)} points uncovered'


def _end_blades(task, c, rep):
    """Small bolt wedges at the rod's ends that do not reach the core."""
    from scipy.spatial import ConvexHull
    L, edge = task.bolt_length, max(task.chamfer, task.pitch)
    return [h for h in rep.bolt.hulls
            if (h[:, 2].max() > L - edge or h[:, 2].min() < edge)
            and np.hypot(h[:, 0], h[:, 1]).min() > task.r_minor + 0.2e-3
            and ConvexHull(h).volume < 0.05e-9]


def test_chamfer_removes_the_dovetail_blade():
    """Cut off flat, a dovetail's crest runs on past its root and leaves a
    thin blade off the core at each end; a 45° chamfer removes it."""
    import dataclasses

    from benchmark.thread.representations import build_thread
    from benchmark.thread.wacky import UNDERCUT
    task, c = UNDERCUT['dovetail'].task, 1e-4
    flat = dataclasses.replace(task, chamfer=0.0)
    assert _end_blades(flat, c, build_thread(flat, 'ours', {'segments': 24}, c, use_cache=False))
    assert not _end_blades(task, c, build_thread(task, 'ours', {'segments': 24}, c,
                                                 use_cache=False))


@pytest.mark.parametrize('nut_driven', [False, True])
def test_chamfered_iso_stays_inside_and_interlocked(nut_driven):
    import mujoco

    from tests.test_thread import _hull_surface, _meshes
    task, c = G.ThreadTask(chamfer=0.001), 1e-4
    xml = mjfit.thread(clearance=c, chamfer=task.chamfer,
                       thread_length=task.bolt_length, nut_height=task.nut_height,
                       nut_size=2 * task.nut_outer_radius, nut_shape='cylinder',
                       bolt_head_height=0, floor=False, driven='nut' if nut_driven else 'bolt')
    if not nut_driven:
        rng = np.random.default_rng(0)
        for part, sdf in (('bolt', G.bolt_sdf), ('nut', G.nut_sdf)):
            pts = np.concatenate([_hull_surface(v, rng, 6) for k, v in _meshes(xml).items()
                                  if k.startswith(part + '_thread')])
            assert sdf(task, c, pts).max() < 1e-9, part
    m = mujoco.MjModel.from_xml_string(xml)
    bolt, nut = m.body('bolt').id, m.body('nut').id

    def pair(shift):
        d = mujoco.MjData(m)
        d.qpos[m.jnt_qposadr[m.body_jntadr[nut if nut_driven else bolt]] + 2] += shift
        mujoco.mj_forward(m, d)
        return [d.contact[i].dist for i in range(d.ncon)
                if {m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]}
                == {bolt, nut}]
    assert pair(0.0) == []
    assert pair(task.pitch / 3) + pair(-task.pitch / 3)


# ----- the nut on the bolt's tip (T6, T7) ----------------------------------------------

@pytest.mark.parametrize('c', [1e-4, 2e-5])
def test_flat_cut_tip_clicks_by_the_axial_play(c):
    """With flat-cut ends the coaxial nut rests on the top face at every phase
    but its play window: there it drops by the whole axial play (2 x 2c/sqrt3)
    and, turned on the unscrewing way, rides the flank back up at the helix's
    rise."""
    tip = G.Tip(TASK, c)
    lo, hi = G.tip_window(TASK, c)
    assert G.tip_overlap(TASK, c, 0.0) == math.inf
    assert G.tip_overlap(TASK, c, hi + 1e-6) == 0.0 == G.tip_overlap(TASK, c, lo - 1e-6)
    th, drop = tip.click()
    assert drop == pytest.approx(4 * c / math.sqrt(3), rel=1e-6)
    rise = TASK.lead / (2 * math.pi)
    # screwing on starts where the window's other edge reaches the top face
    assert (tip.entry() - th) % (2 * math.pi) == pytest.approx(drop / rise, rel=1e-6)
    thetas = th + np.radians(np.arange(-20, 90, 0.1))
    z, steep = tip.path(thetas)
    L = TASK.bolt_length
    np.testing.assert_allclose(z[thetas < th - 1e-3], L, atol=1e-12)
    ride = (thetas > th + 1e-3) & (thetas < th + drop / rise - 1e-3)
    np.testing.assert_allclose(z[ride], L - drop + (thetas[ride] - th) * rise, atol=1e-9)
    np.testing.assert_allclose(z[thetas > th + drop / rise + 1e-3], L, atol=1e-12)
    assert steep == pytest.approx(1.0, abs=1e-3)


def test_screwing_on_follows_the_helix_from_the_entry():
    task, c = G.ThreadTask(chamfer=1e-3), 1e-4
    tip = G.Tip(task, c)
    rise = task.lead / (2 * math.pi)
    th = tip.entry()
    thetas = th - np.radians(np.arange(-30, 400, 0.1))
    z, _ = tip.path(thetas)
    flat = tip.rest(th + 0.3)
    np.testing.assert_allclose(z[thetas > th + 1e-3], flat, atol=1e-10)
    on = thetas < th - 1e-3
    np.testing.assert_allclose(z[on], flat - (th - thetas[on]) * rise, atol=1e-9)


@pytest.mark.parametrize('chamfer', [0.0, 1e-3])
def test_tip_rest_matches_the_sampled_exact_surfaces(chamfer):
    """Independent of the meridian argument: at the rest height the sampled
    exact surfaces (end faces and chamfer cones included) just touch; 3 um
    lower they overlap, 3 um higher they part."""
    task, c = G.ThreadTask(chamfer=chamfer), 1e-4
    tip = G.Tip(task, c)
    pair = G.Pair(task, c, spacing=5e-5, tip=2.5e-3)
    th, _ = tip.click()
    for theta in (th - 1.0, th + 0.01, th + 0.3, tip.entry() + 0.2):
        z = tip.rest(theta)
        R = G.rot([0, 0, 1], theta)
        assert abs(pair.signed_distance([0, 0, z], R)) < 5e-7
        assert pair.signed_distance([0, 0, z - 3e-6], R) < -1e-6
        assert pair.signed_distance([0, 0, z + 3e-6], R) > 1e-6


@pytest.mark.parametrize('chamfer', [0.0, 1e-3])
def test_tip_faces_lie_on_the_exact_boundary(chamfer):
    task, c = G.ThreadTask(chamfer=chamfer), 1e-4
    bolt, nut = G.tip_faces(task, c, spacing=1e-4)
    assert len(bolt) > 1000 and len(nut) > 1000
    assert np.abs(G.bolt_sdf(task, c, bolt)).max() < 1e-9
    assert np.abs(G.nut_sdf(task, c, nut)).max() < 1e-9


def test_partial_engagement_play():
    """Threaded on, the nut starts centred in its play (c/2 from the bolt).
    Two turns on, its tips are over the core all round: radial play c/2, as
    at full engagement. A quarter turn on, it is looser: straight along
    pi/2 it meets the core's rim where the tip arc ends, 40 deg off the push,
    and free to slide around that contact it reaches further still."""
    c = 1e-4
    pair = G.Pair(TASK, c, spacing=1e-4, tip=2 * TASK.lead + 2e-3)
    for turns in (0.25, 2.0):
        z, a = G.partial_pose(TASK, turns)
        assert pair.signed_distance([0, 0, z], G.rot([0, 0, 1], a)) == pytest.approx(c / 2,
                                                                                      abs=1e-8)
    for theta in (0.0, math.pi):
        assert G.partial_ray(pair, 2.0, theta) == pytest.approx(c / 2, abs=5e-8)
    ray = G.partial_ray(pair, 0.25, math.pi / 2)
    assert ray > 1.2 * c / 2
    support = G.partial_radial_play(pair, 0.25, math.pi / 2, spread=math.radians(30),
                                    step=math.radians(10))
    assert support > ray + 2e-5


# ----- turning torque against friction (T8) ---------------------------------------------

def test_screw_torque_formula():
    """Frictionless, raising takes F r tan(lam) and lowering gives it back; the
    thread self-locks once mu' = mu / cos(beta) passes tan(lam); the run-down
    torque is lowering the nut's weight."""
    c, F = 1e-4, 1.0
    r = G.pitch_radius(TASK)
    tan_lam = TASK.lead / (2 * math.pi * r)
    _, cos_beta = G.seat_contact(TASK, c)
    assert G.screw_torque(TASK, F, 0.0, True, c) == pytest.approx(F * r * tan_lam)
    assert G.screw_torque(TASK, F, 0.0, False, c) == pytest.approx(-F * r * tan_lam)
    assert G.screw_torque(TASK, F, tan_lam * cos_beta, False, c) == pytest.approx(0, abs=1e-15)
    up, down = (G.screw_torque(TASK, F, 0.3, raising, c) for raising in (True, False))
    assert up > down > 0
    assert G.rundown_torque(TASK, 0.01, 0.3, c) == pytest.approx(
        G.screw_torque(TASK, 9.81 * 0.01, 0.3, False, c))


def test_torque_probe_tracks_the_formula():
    """At T4's light load with condim 3, the engaged nut turns at a little
    over the formula; condim 4 adds torsional friction."""
    from benchmark.thread import probes as T
    from benchmark.thread.representations import build_thread
    rep = build_thread(TASK, 'ours', {'segments': 24}, 1e-4)
    c3 = T.probe_torque(TASK, rep, 0.3, 'up', 0.1, 3, multiccd=False)
    c4 = T.probe_torque(TASK, rep, 0.3, 'up', 0.1, 4, multiccd=False)
    assert c3['outcome'] == 'turns' and 1.0 < c3['ratio'] < 1.6
    assert c4['ratio'] > 1.5 * c3['ratio']
