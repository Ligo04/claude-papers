"""Thread geometry against the exact helical parts.

The exact bolt and nut (benchmark/thread/geometry.py) are the ISO profiles
swept along the helix; membership in them is exact, so any wedge surface point
outside them is phantom material that would eat the clearance.
"""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

import mjfit
from benchmark.thread import geometry as G
from mjfit import THREAD_FORMS, thread_profile

P = 0.0015


# Independent references: the ISO 68-1 basic profiles at clearance c, written
# out by hand (the generator derives them from its generic trapezoid form).

def iso_external(pitch: float, clearance: float) -> np.ndarray:
    """Bolt thread tooth in the meridian (dr, z) plane.

    Returns a 4-vertex trapezoid (CCW in the (z, dr) convention used by the
    sweep generator — see `_thread_wedge_vertices`). `dr` is the radial offset
    outward from the minor radius (the bolt core surface). `z` is axial
    position within one pitch (0 ≤ z ≤ pitch).

    ISO 68-1 basic profile, external thread:
      - sharp-V height      H = P·√3/2
      - thread depth        h = 5H/8 = 0.5413·P
      - root flat (axial)   P/4   (at the minor radius)
      - flank span (axial)  5P/16 (60° included, 30° off the radial plane)
      - crest flat (axial)  P/8   (at the major radius)

    Clearance c moves each flank back into the tooth by c/2 along its normal
    (an axial shift of c/√3 at every radius) and lowers the crest by c/2. The
    nut tooth gets the same treatment, so when the nut sits on the pitch grid
    facing flanks are exactly c apart on both sides, and the crest clears the
    nut root by c. Axial play is then ±c/cos 30° = ±2c/√3.
    """
    P = pitch
    h = 5 * P * np.sqrt(3) / 16          # thread depth
    s = clearance / np.sqrt(3)           # axial shift of a flank moved c/2 along its normal
    dr_crest = h - clearance / 2
    crest_start = 9 * P / 16 + s / 2     # flank u = P/4 + s + dr/√3 at dr = h - c/2
    crest_end = 11 * P / 16 - s / 2
    if dr_crest <= 0 or crest_end <= crest_start:
        raise ValueError(
            f"clearance ({clearance}) leaves no bolt crest for pitch {P} "
            f"(needs clearance < √3·P/8 = {np.sqrt(3) * P / 8:.6g}).")
    # Order: two base vertices, then the two crest vertices (the sweep
    # generator's face topology relies on it).
    return np.array([
        [0.0, P / 4 + s],          # base-left
        [0.0, P - s],              # base-right
        [dr_crest, crest_end],     # crest-right
        [dr_crest, crest_start],   # crest-left
    ])


def iso_internal(pitch: float, clearance: float) -> np.ndarray:
    """Nut thread tooth, dr measured from the major radius (negative = toward
    the axis), z in the nut's thread phase .

    The ISO internal tooth is the space between basic external teeth, not a
    mirror image of one: its tip (at the minor radius) is P/4 wide and its
    base (at the major radius) 7P/8, leaving a P/8 root. Clearance moves both
    flanks back c/2 along their normals and lifts the tip by c/2, as for the
    bolt.
    """
    P = pitch
    h = 5 * P * np.sqrt(3) / 16
    s = clearance / np.sqrt(3)
    dr_tip = -(h - clearance / 2)
    tip_start = 7 * P / 16 + s / 2       # flank u = P/8 + s - dr/√3 at dr = -(h - c/2)
    tip_end = 11 * P / 16 - s / 2
    if dr_tip >= 0 or tip_end <= tip_start:
        raise ValueError(
            f"clearance ({clearance}) leaves no nut tooth tip for pitch {P}.")
    return np.array([
        [0.0, P / 8 + s],          # base-left
        [0.0, P - s],              # base-right
        [dr_tip, tip_end],         # tip-right
        [dr_tip, tip_start],       # tip-left
    ])


def screw(**kw):
    """mjfit.thread with an M10's round nut, as the exact parts in G assume."""
    return mjfit.thread(**{'nut_shape': 'cylinder', 'bolt_head': 'cylinder', **kw})


def _meshes(xml):
    return {m.attrib['name']: np.fromstring(m.attrib['vertex'], sep=' ').reshape(-1, 3)
            for m in ET.fromstring(xml).findall('./asset/mesh')}


def _hull_surface(v, rng, n=40):
    """Random points on every hull facet plus every hull edge midpoint (where
    chord bulges peak)."""
    hull = ConvexHull(v)
    out = []
    for simp in hull.simplices:
        a, b, c = v[simp]
        u = rng.random((n, 2))
        flip = u.sum(1) > 1
        u[flip] = 1 - u[flip]
        out += [a + u[:, :1] * (b - a) + u[:, 1:] * (c - a), (v[simp] + v[np.roll(simp, 1)]) / 2]
    return np.concatenate(out)


@pytest.mark.parametrize('c', [1e-4, 2e-5])
def test_flanks_are_c_apart_and_basic_teeth_interlock(c):
    """In the shared (bolt) phase u, each flank is a line u = u0 +/- dr/sqrt3.
    Facing flanks must be 2c/sqrt3 apart axially (c along the normal), and at
    c = 0 the nut tooth must be exactly the space between bolt teeth."""
    bolt = iso_external(P, c)       # base-l, base-r, crest-r, crest-l
    nut = iso_internal(P, c)
    h = 5 * P * math.sqrt(3) / 16
    shift = -7 * P / 16                                     # nut phase -> bolt phase
    gap = 2 * c / math.sqrt(3)
    bolt_rise, bolt_fall = bolt[0, 1], bolt[1, 1]
    # The nut tip sits in the bolt root gap, c/(2 sqrt3) in from each side.
    tip = np.sort(nut[2:, 1]) + shift
    np.testing.assert_allclose(tip, [c / (2 * math.sqrt(3)),
                                     P / 4 - c / (2 * math.sqrt(3))], atol=1e-12)
    # Each facing flank pair, compared at the same radius (dr from r_min).
    for dr in (0.0, h / 2, h - c / 2):
        rise = bolt_rise + dr / math.sqrt(3)                # bolt rising flank
        fall = bolt_fall - dr / math.sqrt(3)                # bolt falling flank
        nut_r = nut[1, 1] + shift + (dr - h) / math.sqrt(3)  # nut right flank, below `rise`
        nut_l = nut[0, 1] + shift - (dr - h) / math.sqrt(3) + P  # next nut's left flank
        assert rise - nut_r == pytest.approx(gap, abs=1e-12)
        assert nut_l - fall == pytest.approx(gap, abs=1e-12)
    # Crest clears the nut root by c, tip clears the core by c/2.
    assert bolt[2, 0] == pytest.approx(h - c / 2)
    assert -nut[2, 0] == pytest.approx(h - c / 2)


@pytest.mark.parametrize('relief', [0.0, 5e-6])
@pytest.mark.parametrize('n', [12, 24, 96])
@pytest.mark.parametrize('c', [1e-4, 2e-5])
def test_wedges_stay_inside_exact_parts(c, n, relief):
    task = G.ThreadTask()
    xml = screw(clearance=c, segments_per_turn=n, seam_relief=relief,
                                     thread_length=task.bolt_length, nut_height=task.nut_height,
                                     nut_size=2 * task.nut_outer_radius,
                                     bolt_head_height=0, floor=False)
    rng = np.random.default_rng(0)
    for part, sdf in (('bolt', G.bolt_sdf), ('nut', G.nut_sdf)):
        pts = np.concatenate([_hull_surface(v, rng) for k, v in _meshes(xml).items()
                              if k.startswith(part)])
        assert sdf(task, c, pts).max() < 1e-9, part


@pytest.mark.parametrize('nut_driven', [False, True])
@pytest.mark.parametrize('nut_height', [0.008, 0.0075])
def test_scene_layouts_start_interlocked(nut_driven, nut_height):
    m = mujoco.MjModel.from_xml_string(screw(
        driven='nut' if nut_driven else 'bolt', nut_height=nut_height))
    bolt, nut = m.body('bolt').id, m.body('nut').id

    def pair_contacts(shift):
        d = mujoco.MjData(m)
        d.qpos[m.jnt_qposadr[m.body_jntadr[nut if nut_driven else bolt]] + 2] += shift
        mujoco.mj_forward(m, d)
        return [d.contact[i].dist for i in range(d.ncon)
                if {m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]}
                == {bolt, nut}]
    assert pair_contacts(0.0) == []
    assert min(pair_contacts(P / 3)) < -1e-4   # the check can see a wrong phase


def test_threads_collide_with_everything_on_the_other_part():
    m = mujoco.MjModel.from_xml_string(screw())
    bolt, nut = m.body('bolt').id, m.body('nut').id
    for g1 in np.flatnonzero(m.geom_bodyid == bolt):
        for g2 in np.flatnonzero(m.geom_bodyid == nut):
            assert (m.geom_contype[g1] & m.geom_conaffinity[g2]) or \
                   (m.geom_contype[g2] & m.geom_conaffinity[g1])


def test_wedges_clipped_to_thread_extent():
    L, Hn = 0.0137, 0.0061
    meshes = _meshes(screw(thread_length=L, nut_height=Hn))
    for name, v in meshes.items():
        if name.startswith('bolt_thread'):
            assert v[:, 2].min() >= -1e-12 and v[:, 2].max() <= L + 1e-12
        elif name.startswith('nut_thread'):
            assert v[:, 2].min() >= -1e-12 and v[:, 2].max() <= Hn + 1e-12


# ----- thread forms ------------------------------------------------------------

FORMS = [f for f in THREAD_FORMS if f != 'custom']
# The FurnitureBench square-table leg (configs/screws/furniture_bench_leg.yaml).
FURNITURE_BENCH = dict(major_diameter=0.02436, pitch=0.009375, form='wire',
                       form_params=dict(wire_diameter=0.00436, nut_wire_offset=0.0031,
                                          bore_diameter=0.025),
                       clearance=0.0, thread_length=0.019, nut_height=0.017,
                       nut_size=0.04)
FORM_CASES = ([dict(form=f) for f in FORMS] + [FURNITURE_BENCH]
              + [dict(form='buttress', n_starts=2, handedness='left'),
                 dict(form='square', n_starts=3)])
FORM_IDS = FORMS + ['furniture_bench', 'buttress_2start_left', 'square_3start']


def _case(case, **kw):
    """mjfit.thread arguments for a form case, on an M10-sized part."""
    return {'clearance': 5e-5, 'thread_length': 0.016, 'nut_height': 0.008,
            'bolt_head_height': 0, 'floor': False, **case, **kw}


def _form(kw):
    return thread_profile(kw.get('form', 'iso'), kw.get('major_diameter', 0.010),
                          kw.get('pitch', P), kw.get('form_params'))


def _swept_sd(p, tooth, line, pitch, lead, sign, bolt, extent):
    """Signed distance (in the meridian plane) to a tooth swept along the
    helix, with the bolt's core (r < line) or the nut's shell (r > line)."""
    if not bolt:   # the nut's tooth reaches on into its shell
        b = tooth[np.abs(tooth[:, 0] - tooth[:, 0].max()) < 1e-12]
        tooth = np.vstack([tooth, [[line + 2e-4, b[:, 1].min()], [line + 2e-4, b[:, 1].max()]]])
        tooth = tooth[ConvexHull(tooth).vertices]
    r = np.hypot(p[:, 0], p[:, 1])
    u = p[:, 2] - sign * lead * np.arctan2(p[:, 1], p[:, 0]) / (2 * math.pi)
    sd = np.minimum(G._periodic_sd(r, u, tooth, pitch), r - line if bolt else line - r)
    return np.maximum(sd, np.maximum(-p[:, 2], p[:, 2] - extent))


@pytest.mark.parametrize('c', [1e-4, 2e-5])
def test_generic_iso_teeth_are_the_iso_profiles(c):
    form = thread_profile('iso', 0.010, P)
    bolt, nut = form.teeth(c)
    np.testing.assert_allclose(bolt - [form.r_core, 0],
                               iso_external(P, c), atol=1e-15)
    np.testing.assert_allclose(nut - [form.r_root, 0],
                               iso_internal(P, c) - [0, 7 * P / 16],
                               atol=1e-15)


@pytest.mark.parametrize('form', FORMS)
def test_facing_teeth_are_c_apart(form):
    """Clearance moves every exposed edge back c/2, so facing teeth are c
    apart at their closest. The round wire's teeth are inscribed polygons,
    which add up to their sag, rho (1 - cos(pi/n)), per wire."""
    c = 5e-5
    bolt, nut = thread_profile(form, 0.010, P).teeth(c)
    gap = min(min(G._polygon_sd(bolt, nut + [0, k * P]).min(),
                  G._polygon_sd(nut + [0, k * P], bolt).min()) for k in (-1, 0, 1))
    if form == 'wire':
        rho = P / math.sqrt(3) / 2                       # the default (tight) wire
        assert c <= gap <= c + 2 * rho * (1 - math.cos(math.pi / 16))
    else:
        assert gap == pytest.approx(c, abs=1e-12)


@pytest.mark.parametrize('relief', [0.0, 5e-6])
@pytest.mark.parametrize('n', [12, 24])
@pytest.mark.parametrize('case', FORM_CASES, ids=FORM_IDS)
def test_every_form_stays_inside_its_teeth(case, n, relief):
    kw = _case(case, segments_per_turn=n, seam_relief=relief)
    form = _form(kw)
    bolt, nut = form.teeth(kw['clearance'])
    pitch = kw.get('pitch', P)
    lead = kw.get('n_starts', 1) * pitch
    sign = 1 if kw.get('handedness', 'right') == 'right' else -1
    meshes = _meshes(screw(**kw))
    rng = np.random.default_rng(0)
    for part, tooth, line, extent in (
            ('bolt', bolt, form.r_core, kw['thread_length']),
            ('nut', nut, form.r_root + kw['clearance'] / 2, kw['nut_height'])):
        pts = np.concatenate([_hull_surface(v, rng, 8) for k, v in meshes.items()
                              if k.startswith(part + '_thread')])
        sd = _swept_sd(pts, tooth, line, pitch, lead, sign, part == 'bolt', extent)
        assert sd.max() < 1e-9, part


@pytest.mark.parametrize('case', FORM_CASES, ids=FORM_IDS)
def test_every_form_fills_its_teeth_to_both_ends(case):
    """No thread goes missing at an end face: points well inside the exact
    teeth (in from every face by more than the discretization takes off)
    within two pitches of either end lie inside some wedge. The turn below
    turn 0 used to be left out, which dropped most of the partial first turn
    at z = 0 of both parts: the bolt's tip and the nut face that goes onto it
    in the nut-driven layout."""
    kw = _case(case, segments_per_turn=24)
    form = _form(kw)
    bolt, nut = form.teeth(kw['clearance'])
    pitch = kw.get('pitch', P)
    lead = kw.get('n_starts', 1) * pitch
    sign = 1 if kw.get('handedness', 'right') == 'right' else -1
    margin = 0.1 * pitch if pitch < 3e-3 else 3e-4
    meshes = _meshes(screw(**kw))
    rng = np.random.default_rng(0)
    for part, tooth, line, extent, r0, r1 in (
            ('bolt', bolt, form.r_core, kw['thread_length'], form.r_core, bolt[:, 0].max()),
            ('nut', nut, form.r_root + kw['clearance'] / 2, kw['nut_height'],
             nut[:, 0].min(), form.r_root)):
        eqs = [ConvexHull(v).equations for k, v in meshes.items()
               if k.startswith(part + '_thread')]
        band = min(2 * pitch, extent / 2)
        for z0 in (0.0, extent - band):
            n = 20000
            r = rng.uniform(r0, r1, n)
            phi = rng.uniform(0, 2 * math.pi, n)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), rng.uniform(z0, z0 + band, n)])
            sd = _swept_sd(pts, tooth, line, pitch, lead, sign, part == 'bolt', extent)
            # well inside the tooth, not the core or shell the wedges join
            pts = pts[(sd < -margin) & (r > r0 + margin) & (r < r1 - margin)]
            inside = np.zeros(len(pts), bool)
            for eq in eqs:
                inside |= (pts @ eq[:, :3].T + eq[:, 3] <= 1e-12).all(axis=1)
            assert len(pts) > 100 and inside.all(), \
                f'{part} end z={z0:g}: {(~inside).sum()} of {len(pts)} uncovered'


@pytest.mark.parametrize('nut_driven', [False, True])
@pytest.mark.parametrize('case', FORM_CASES, ids=FORM_IDS)
def test_every_form_starts_interlocked(case, nut_driven):
    """Both layouts, including the flipped bolt of the nut-driven one (an
    asymmetric tooth gets a mirrored nut, and the phase follows the lead)."""
    m = mujoco.MjModel.from_xml_string(screw(
        **_case(case, driven='nut' if nut_driven else 'bolt')))
    bolt, nut = m.body('bolt').id, m.body('nut').id

    def pair_contacts(shift):
        d = mujoco.MjData(m)
        d.qpos[m.jnt_qposadr[m.body_jntadr[nut if nut_driven else bolt]] + 2] += shift
        mujoco.mj_forward(m, d)
        return [d.contact[i].dist for i in range(d.ncon)
                if {m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]}
                == {bolt, nut}]
    pitch = case.get('pitch', P)
    assert pair_contacts(0.0) == []
    assert min(pair_contacts(pitch / 3) + pair_contacts(-pitch / 3)) < -1e-4


def test_custom_teeth_reproduce_a_named_form():
    form = thread_profile('acme', 0.010, P)
    custom = dict(bolt_tooth=form.bolt_tooth.tolist(), nut_tooth=form.nut_tooth.tolist(),
                  core_diameter=2 * form.r_core, root_diameter=2 * form.r_root)
    a = _meshes(screw(**_case({'form': 'acme'})))
    b = _meshes(screw(**_case({'form': 'custom', 'form_params': custom})))
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-15)


@pytest.mark.parametrize('form', ['iso', 'trapezoidal', 'square', 'wire'])
def test_exported_meshes_load(form, tmp_path):
    """Written OBJs (explicit faces for plain four-sided teeth) compile."""
    xml = screw(form=form, mesh_dir=tmp_path / 'scene_meshes')
    (tmp_path / 'scene.xml').write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / 'scene.xml'))
    assert (m.body_mass[1:] > 0).all()
