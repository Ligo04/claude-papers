"""Random assemblies: reproducible, valid, and placeable side by side."""

import mujoco
import numpy as np
import pytest

import mjfit
from mjfit import _polygon as pg
from mjfit import _sample, shapes


@pytest.mark.parametrize('family', ['primitive', 'convex'])
def test_samples_build_and_compile(family):
    for cfg in mjfit.sample_pegs(40, seed=1, family=family):
        model = mujoco.MjModel.from_xml_string(mjfit.build(cfg))
        assert model.body('peg').mass[0] > 0


def test_samples_are_reproducible_and_independent_of_count():
    assert mjfit.sample_pegs(6, seed=4)[5] == mjfit.sample_peg(4, 5)
    assert mjfit.sample_peg(4, 0) != mjfit.sample_peg(5, 0)
    assert mjfit.sample_peg(4, 0, family='convex') == mjfit.sample_peg(4, 0, family='convex')


def _gap(p, q):
    """Smallest distance between two polygons' boundaries (vertex to edge)."""

    def one(a, b):
        e = np.roll(b, -1, axis=0) - b
        t = np.clip(np.einsum('pij,ij->pi', a[:, None] - b[None], e) / (e * e).sum(1), 0, 1)
        return np.linalg.norm(b[None] + t[..., None] * e[None] - a[:, None], axis=2).min()

    return min(one(p, q), one(q, p))


def test_primitive_islands_leave_the_minimum_wall():
    for cfg in mjfit.sample_pegs(150, seed=2, min_wall=0.001):
        if not cfg.get('islands'):
            continue
        c = cfg['clearance']
        peg = pg.offset_inward(shapes.from_spec(cfg['hole']), c)
        pockets = [pg.offset_outward(shapes.from_spec(s), c) for s in cfg['islands']]
        walls = [_gap(p, peg) for p in pockets]
        walls += [_gap(p, q) for i, p in enumerate(pockets) for q in pockets[i + 1 :]]
        assert min(walls) >= 0.001 - 1e-9


def test_convex_family_keeps_rims_in_bounds():
    for cfg in mjfit.sample_pegs(60, seed=3, family='convex'):
        hole, plate = np.asarray(cfg['hole']), np.asarray(cfg['plate'])
        covered = hole
        assert pg.is_convex(plate)
        if 'handle' in cfg:
            handle = np.asarray(cfg['handle'])
            peg = pg.offset_inward(hole, cfg['clearance'])
            assert pg.min_rim(handle, peg) >= 0.001 - 1e-9
            assert _sample._farthest(handle, pg.convex_hull(peg)) <= 0.01 + 1e-9
            assert cfg['peg_length'] >= 0.022 - 1e-12
            covered = np.vstack([hole, handle])
        assert pg.min_rim(plate, covered) >= 0.005 - 1e-9
        assert _sample._farthest(plate, pg.convex_hull(covered)) <= 0.03 + 1e-9


def test_ranges_and_fixed_values():
    cfgs = mjfit.sample_pegs(30, clearance=2e-4, islands=0, handle_probability=0.0)
    assert all(
        c['clearance'] == 2e-4 and 'islands' not in c and 'handle_height' not in c for c in cfgs
    )
    sizes = [c['plate_thickness'] for c in mjfit.sample_pegs(30, plate_thickness=(0.004, 0.005))]
    assert 0.004 <= min(sizes) and max(sizes) <= 0.005


def test_grid_places_copies_without_clashing():
    parts = [mjfit.build(c, standalone=False) for c in mjfit.sample_pegs(5, family='convex')]
    spec = mjfit.grid(parts, spacing=0.3)
    model = spec.compile()
    names = [model.body(i).name for i in range(model.nbody)]
    assert names[1:3] == ['p0_plate', 'p0_peg'] and len(names) == 11
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    mujoco.MjModel.from_xml_string(spec.to_xml())


def test_unknown_family_and_option():
    with pytest.raises(ValueError, match='family'):
        mjfit.sample_peg(family='round')
    with pytest.raises(TypeError):
        mjfit.sample_peg(sizes=0.01)
