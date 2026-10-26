"""Config files: YAML mappings of a generator's keyword arguments plus `kind`."""
from __future__ import annotations

import difflib
import inspect
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / 'configs'
OUTLINE_KEYS = ('hole', 'peg', 'handle', 'plate')


def bundled() -> dict[str, Path]:
    """Every config shipped with the package, by name."""
    return {p.stem: p for p in sorted(CONFIG_DIR.glob('*/*.yaml'))}


def load(source) -> dict:
    """A config as a dict: a mapping, a YAML path, or a bundled config's name.

    Relative CSV paths in outlines are made absolute against the file's folder.
    """
    if isinstance(source, dict):
        return dict(source)
    path = Path(source)
    if not path.exists():
        if path.suffix == '' and str(source) in bundled():
            path = bundled()[str(source)]
        else:
            names = ', '.join(bundled())
            raise FileNotFoundError(f"No config file {str(source)!r}; bundled configs: {names}.")
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: a config must be a mapping of parameter names to values.")
    for key in OUTLINE_KEYS:
        _absolute_csv(cfg.get(key), path.parent)
    for spec in cfg.get('islands') or ():
        _absolute_csv(spec, path.parent)
    return cfg


def _absolute_csv(spec, base: Path) -> None:
    if isinstance(spec, dict) and spec.get('type') == 'csv' and 'path' in spec:
        if not Path(spec['path']).is_absolute():
            spec['path'] = str(base / spec['path'])


def check_keys(fn, params: dict, where: str) -> None:
    """Raise on keys that are not keyword arguments of `fn`, suggesting fixes."""
    valid = [p for p in inspect.signature(fn).parameters]
    bad = [k for k in params if k not in valid]
    if bad:
        hints = []
        for k in bad:
            close = difflib.get_close_matches(k, valid, n=1)
            hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(f"{where}: unknown parameter(s) {', '.join(hints)}.")
