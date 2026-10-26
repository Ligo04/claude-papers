"""Build and load the analytic thread SDF plugin (iso_thread_sdf.cc).

Only the 'sdf_analytic' representation of the thread benchmark needs it. On
first use, library() compiles the plugin with the system C++ compiler against
the libmujoco shared library inside the installed mujoco wheel:

    g++ -std=c++17 -O2 -fPIC -shared -I<mujoco>/include iso_thread_sdf.cc \\
        <mujoco>/libmujoco.so.<version> -Wl,-rpath,<mujoco> \\
        -o benchmark/cache/libiso_thread_sdf_<version>.so

This needs Linux (the wheel's libmujoco.so), g++ with C++17 (GCC 8 or later),
and the headers that ship in the wheel (no separate MuJoCo install). The
library is rebuilt when the source is newer or MuJoCo's version changes; set
CXX to use another compiler. Every other representation runs without it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import mujoco

SRC = Path(__file__).resolve().parent / 'iso_thread_sdf.cc'
NAME = 'iso.sdf.thread'
_loaded = False


def library() -> Path:
    """Compile against the installed mujoco wheel (once per machine and
    version); the library lives in the gitignored cache."""
    from ...representations import CACHE_DIR
    pkg = Path(mujoco.__file__).resolve().parent
    out = CACHE_DIR / f'libiso_thread_sdf_{mujoco.__version__}.so'
    if out.exists() and out.stat().st_mtime >= SRC.stat().st_mtime:
        return out
    libs = sorted(pkg.glob('libmujoco.so*'))
    cxx = os.environ.get('CXX', 'g++')
    if not libs:
        raise RuntimeError(f'The analytic thread SDF plugin links against libmujoco.so, which '
                           f'is not in {pkg}: it builds on Linux only.')
    if shutil.which(cxx) is None:
        raise RuntimeError(f'The analytic thread SDF plugin is compiled on first use, but the '
                           f'C++ compiler {cxx!r} was not found (install g++, or set CXX).')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f'{out.stem}.{os.getpid()}.tmp.so')
    subprocess.run([cxx, '-std=c++17', '-O2', '-fPIC', '-shared', f'-I{pkg / "include"}',
                    str(SRC), str(libs[0]), f'-Wl,-rpath,{pkg}', '-o', str(tmp)], check=True)
    tmp.replace(out)
    return out


def load() -> str:
    """Register the plugin with MuJoCo (idempotent); returns its name."""
    global _loaded
    if not _loaded:
        mujoco.mj_loadPluginLibrary(str(library()))
        _loaded = True
    return NAME
