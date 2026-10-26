"""mjfit: exact convex collision geometry for MuJoCo assemblies.

Generators return MJCF strings in which every collision geom is convex, so
MuJoCo collides the parts natively while their clearances stay exact::

    import mjfit
    xml = mjfit.peg(hole=mjfit.shapes.star(5, 0.012, 0.006), clearance=1e-4)
    xml = mjfit.thread(major_diameter=0.010, pitch=0.0015)
    xml = mjfit.gear(teeth=(18, 24))
    xml = mjfit.build('usb_c')         # a bundled config, a YAML path, or a dict
"""
from __future__ import annotations

from . import shapes
from ._config import bundled as _bundled
from ._config import check_keys as _check_keys
from ._config import load
from ._gear import gear
from ._peg import peg
from ._sample import sample_peg, sample_pegs
from ._scene import grid
from ._thread import THREAD_FORMS, thread, thread_profile

__version__ = '0.1.0'
__all__ = ['peg', 'thread', 'gear', 'build', 'load', 'configs', 'sample_peg', 'sample_pegs',
           'grid', 'shapes', 'thread_profile', 'THREAD_FORMS']

GENERATORS = {'peg': peg, 'thread': thread, 'gear': gear}


def build(config, **overrides) -> str:
    """MJCF for a config: a bundled config's name, a YAML path, or a dict.

    A config maps a generator's keyword arguments to values, plus ``kind``
    (``peg``, ``thread`` or ``gear``). Keyword ``overrides`` replace entries.
    """
    cfg = {**load(config), **overrides}
    kind = cfg.pop('kind', None)
    if kind not in GENERATORS:
        raise ValueError(f"A config needs kind: peg, thread or gear (got {kind!r}).")
    fn = GENERATORS[kind]
    _check_keys(fn, cfg, f"{kind} config")
    return fn(**cfg)


def configs() -> dict[str, str]:
    """The bundled configs, as {name: kind}."""
    return {name: load(path).get('kind', '?') for name, path in _bundled().items()}
