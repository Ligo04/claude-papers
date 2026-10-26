"""Outline constructors and shape specs."""

import numpy as np
import pytest

from mjfit import _polygon as pg
from mjfit import shapes

CONTROL = np.array([[-0.022, -0.013], [0.024, -0.010], [0.019, 0.017], [-0.017, 0.022]])


def _flat_distances(poly):
    edges = np.roll(poly, -1, axis=0) - poly
    mid = poly + edges / 2
    return np.linalg.norm(mid, axis=1)


@pytest.mark.parametrize('error', [1e-4, 1e-5, 1e-6])
def test_circle_circumscribes_within_error(error):
    poly = shapes.circle(0.008, max_error=error)
    np.testing.assert_allclose(_flat_distances(poly), 0.008, rtol=1e-12)
    assert np.linalg.norm(poly, axis=1).max() - 0.008 <= error


@pytest.mark.parametrize('w,h', [(0.03, 0.01), (0.01, 0.03), (0.02, 0.02)])
def test_stadium_has_exact_straight_sides_and_circumscribed_ends(w, h):
    poly = shapes.stadium(w, h)
    assert pg.is_convex(poly) and pg.signed_area(poly) > 0
    long_axis = 0 if w >= h else 1
    half = min(w, h) / 2
    assert np.abs(poly[:, 1 - long_axis]).max() == pytest.approx(half, rel=1e-12)
    assert len(np.unique(np.round(poly, 15), axis=0)) == len(poly)  # no repeated vertex


@pytest.mark.parametrize(
    'fn',
    [
        lambda: shapes.rect(0.01, 0.005),
        lambda: shapes.ngon(6, 0.01),
        lambda: shapes.star(5, 0.01, 0.004),
        lambda: shapes.trapezoid(0.01, 0.008, 0.004),
        lambda: shapes.circle(0.01),
        lambda: shapes.stadium(0.02, 0.01),
    ],
)
def test_outlines_are_ccw_simple(fn):
    poly = fn()
    assert pg.signed_area(poly) > 0 and pg.is_simple(poly)


def test_rotate_is_counterclockwise_degrees():
    poly = shapes.rect(0.02, 0.002, rotate=90)
    assert np.ptp(poly[:, 1]) == pytest.approx(0.02)
    tip = shapes.ngon(3, 0.01, rotate=90)
    assert tip[np.argmax(np.linalg.norm(tip, axis=1)), 0] == pytest.approx(-0.01)  # +y tip to -x


def test_bezier_corner_order_and_chord_error():
    alphas = np.array([0.5, 0, 0.25, 0.4])
    error = 0.0001
    poly = shapes.bezier(CONTROL, alphas, max_error=error)
    np.testing.assert_array_equal(shapes.bezier(CONTROL[::-1], alphas[::-1], max_error=error), poly)
    np.testing.assert_array_equal(shapes.bezier(CONTROL, 0), CONTROL)
    starts, ends = poly, np.roll(poly, -1, axis=0)
    edges = ends - starts
    for i, alpha in enumerate(alphas):
        v = CONTROL[i]
        entry = v + alpha * (CONTROL[i - 1] - v)
        exit_ = v + alpha * (CONTROL[(i + 1) % len(CONTROL)] - v)
        t = np.linspace(0, 1, 501)[:, None]
        curve = (1 - t) ** 2 * entry + 2 * t * (1 - t) * v + t**2 * exit_
        f = np.clip(
            np.einsum('pei,ei->pe', curve[:, None] - starts, edges) / np.sum(edges**2, axis=1), 0, 1
        )
        closest = starts + f[..., None] * edges
        assert np.linalg.norm(curve[:, None] - closest, axis=2).min(axis=1).max() <= error * (
            1 + 1e-10
        )


@pytest.mark.parametrize('rounding,error', [(-0.1, 1e-4), (0.51, 1e-4), ([0.5, 0], 1e-4), (0.5, 0)])
def test_bad_bezier_settings_rejected(rounding, error):
    with pytest.raises(ValueError):
        shapes.bezier(CONTROL, rounding, max_error=error)


def test_specs_match_functions(tmp_path):
    np.testing.assert_array_equal(
        shapes.from_spec(
            {'type': 'star', 'points': 5, 'outer_radius': 0.01, 'inner_radius': 0.004, 'rotate': 15}
        ),
        shapes.star(5, 0.01, 0.004, rotate=15),
    )
    np.savetxt(tmp_path / 'tri.csv', [[0, 0], [0.01, 0], [0, 0.01]], delimiter=',')
    np.testing.assert_array_equal(
        shapes.from_spec({'type': 'csv', 'path': 'tri.csv'}, tmp_path),
        [[0, 0], [0.01, 0], [0, 0.01]],
    )
    with pytest.raises(ValueError, match="unknown key"):
        shapes.from_spec({'type': 'circle', 'radius': 0.01, 'sides': 8})
    with pytest.raises(ValueError, match="unknown shape type"):
        shapes.from_spec({'type': 'hexagon', 'radius': 0.01})


def test_presets():
    for name in shapes.PRESETS:
        poly = shapes.preset(name, 0.01)
        assert pg.signed_area(poly) > 0
    with pytest.raises(ValueError, match='Unknown preset'):
        shapes.preset('heart', 0.01)
