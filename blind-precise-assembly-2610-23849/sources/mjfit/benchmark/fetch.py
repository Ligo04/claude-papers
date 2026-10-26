"""Download the benchmark's large files from the GitHub release.

    python -m benchmark.fetch cache     # decompositions -> benchmark/cache/
    python -m benchmark.fetch videos    # behavior videos -> benchmark/results/render/
    python -m benchmark.fetch cache --archive benchmark-cache.tar.gz   # a local copy

cache: every decomposition the shipped results were simulated with (CoACD,
V-HACD and mjfit hulls, the exact bolt and nut meshes, and the cached exact
T3/T7 answers), keyed by setting, so reruns load them instead of rebuilding
(CoACD at fine thresholds takes minutes to hours). videos: the rendered
behavior videos. Archives are checked against their SHA-256 before anything
is extracted; files already present are replaced.

Maintainers: `python -m benchmark.fetch pack cache SRC_DIR OUT.tar.gz` builds
an archive (sorted members, no owner names, no gzip timestamp) and prints its
SHA-256 for ASSETS.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
RELEASE = 'https://github.com/younghyopark/mjfit/releases/download/benchmark-v1'
# name -> (asset, directory inside the archive and under benchmark/, SHA-256)
ASSETS = {
    'cache': ('benchmark-cache.tar.gz', 'cache',
              'b7acdd927279a9f2b454f7774e30148c2f342d985d2fed51b111b9f17ae26766'),
    'videos': ('benchmark-videos.tar.gz', 'results/render',
               '8dac21022bf9766415279f8e08d12ab2c388db160aa8f9e23b6468b057570a9b'),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f'downloading {url}', flush=True)
    with urllib.request.urlopen(url) as r, open(dest, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        for block in iter(lambda: r.read(1 << 20), b''):
            f.write(block)
            done += len(block)
            if total:
                print(f'\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB', end='', flush=True)
    print(flush=True)


def _members(tar: tarfile.TarFile, prefix: str) -> list[tarfile.TarInfo]:
    """The regular files, all under prefix/. Links, devices, absolute paths,
    '..' and files elsewhere are refused; directories are made as needed."""
    top = PurePosixPath(prefix).parts
    out = []
    for m in tar.getmembers():
        p = PurePosixPath(m.name)
        if p.is_absolute() or '..' in p.parts or not (m.isfile() or m.isdir()):
            raise ValueError(f'refusing archive member {m.name!r}')
        if m.isdir() and top[:len(p.parts)] == p.parts:
            continue                                  # prefix/ or one of its parents
        if p.parts[:len(top)] != top or not m.isfile():
            raise ValueError(f'archive member {m.name!r} is outside {prefix}/')
        out.append(m)
    return out


def fetch(name: str, archive: Path | None = None, dest: Path = HERE) -> Path:
    asset, prefix, sha = ASSETS[name]
    with tempfile.TemporaryDirectory() as tmp:
        if archive is None:
            archive = Path(tmp) / asset
            _download(f'{RELEASE}/{asset}', archive)
        got = _sha256(archive)
        if sha is not None and got != sha:
            raise SystemExit(f'{archive}: SHA-256 {got}, expected {sha}; not extracting')
        with tarfile.open(archive, 'r:gz') as tar:
            members = _members(tar, prefix)
            kw = {'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}
            tar.extractall(dest, members=members, **kw)
    n = sum(m.isfile() for m in members)
    print(f'{n} files -> {dest / prefix}')
    return dest / prefix


def pack(name: str, src: Path, out: Path) -> str:
    """Archive the files of src as ASSETS[name]'s directory; returns the SHA-256."""
    _, prefix, _ = ASSETS[name]
    files = sorted(p for p in src.iterdir() if p.is_file() and not p.name.startswith('.')
                   and '.tmp' not in p.name and p.suffix != '.so')

    def info(path: Path, arcname: str) -> tarfile.TarInfo:
        ti = tarfile.TarInfo(arcname)
        ti.size, ti.mtime, ti.mode = path.stat().st_size, int(path.stat().st_mtime), 0o644
        return ti

    with open(out, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                                               mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for d in [PurePosixPath(prefix).parents[i] for i in
                  range(len(PurePosixPath(prefix).parts) - 2, -1, -1)] + [PurePosixPath(prefix)]:
            if str(d) != '.':
                ti = tarfile.TarInfo(str(d))
                ti.type, ti.mode, ti.mtime = tarfile.DIRTYPE, 0o755, 0
                tar.addfile(ti)
        for p in files:
            with open(p, 'rb') as f:
                tar.addfile(info(p, f'{prefix}/{p.name}'), f)
    sha = _sha256(out)
    print(f'{out}: {len(files)} files, {out.stat().st_size / 1e6:.1f} MB, sha256 {sha}')
    return sha


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['pack']:
        ap = argparse.ArgumentParser(prog='python -m benchmark.fetch pack')
        ap.add_argument('name', choices=list(ASSETS))
        ap.add_argument('src', type=Path)
        ap.add_argument('out', type=Path)
        a = ap.parse_args(argv[1:])
        pack(a.name, a.src, a.out)
        return
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', nargs='+', choices=list(ASSETS))
    ap.add_argument('--archive', type=Path, help='use a local copy of the archive')
    ap.add_argument('--dest', type=Path, default=HERE,
                    help='directory that receives cache/ or results/render/ (default: benchmark/)')
    a = ap.parse_args(argv)
    a.dest.mkdir(parents=True, exist_ok=True)
    for name in dict.fromkeys(a.name):
        fetch(name, a.archive if len(a.name) == 1 else None, a.dest)


if __name__ == '__main__':
    main()
