"""The shipped raw results, the reports built from them, and the release helpers."""
import gzip
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from benchmark import tables

REPORT = Path(__file__).resolve().parents[1] / 'benchmark' / 'report'


def test_tables_merge_replaces_rows_in_place(tmp_path):
    rows = [{'shape': 'circle', 'label': f'ours(n={i})', 'multiccd': True, 'clearance_um': c,
             'error_um': 0.1 * i + 1 / 3} for i in range(3) for c in (100.0, 20.0)]
    tables.write('peg/probes_cpu/lateral', rows, tmp_path)
    text = gzip.open(tables.path('peg/probes_cpu/lateral', tmp_path)).read()
    tables.write('peg/probes_cpu/lateral', rows[2:3], tmp_path)     # a repeat: no change
    assert gzip.open(tables.path('peg/probes_cpu/lateral', tmp_path)).read() == text
    new = {**rows[2], 'error_um': 9.5}
    extra = {**rows[0], 'label': 'coacd(threshold=0.05)'}
    tables.write('peg/probes_cpu/lateral', [new, extra], tmp_path)
    df = tables.read('peg/probes_cpu/lateral', tmp_path)
    assert len(df) == 7 and df.error_um.iloc[2] == 9.5 and df.label.iloc[-1] == extra['label']
    assert df.error_um.iloc[0] == rows[0]['error_um']                 # exact round trip


def test_peg_report_regenerates_the_tracked_summaries(tmp_path):
    from benchmark import report
    report.main(['--out', str(tmp_path), '--no-figures'])
    for name in ('summary.md', 'summary_circle.csv', 'summary_square.csv',
                 'speed_table.csv', 'speed_table.md'):
        assert (tmp_path / name).read_bytes() == (REPORT / name).read_bytes(), name


def test_thread_report_regenerates_the_tracked_summaries(tmp_path):
    from benchmark.thread import report
    report.main(['--out', str(tmp_path)])
    names = ['thread_summary.md'] + [p.name for p in REPORT.glob('thread_*.csv')]
    assert len(names) == 8
    for name in names:
        assert (tmp_path / name).read_bytes() == (REPORT / name).read_bytes(), name


def test_gpu_world_count_is_an_option():
    from benchmark import report
    p, speed = report.load()
    for n in (256, 512):
        t = report.summarize(p, speed, 'circle', nworld=n)
        assert t[f'GPU us/env-step @{n} (wall)'].notna().sum() > 30
    st = pd.read_csv(REPORT / 'speed_table.csv')
    ours = st[(st.representation == 'ours') & (st.multiccd == 'off')].iloc[0]   # the square
    assert ours['GPU real-time factor @512'] == pytest.approx(970, rel=0.01)


def test_reproduce_narrows_and_skips_commands():
    from benchmark import reproduce
    opt = SimpleNamespace(shapes=['square'], methods=['ours'], clearance=None, workers=None,
                          results=Path('/r'), out=Path('/o'), gpu_nworld=256)
    probes = reproduce.commands('probes', opt)
    assert len(probes) == 1 and probes[0][-6:] == ['--methods', 'ours', '--shapes', 'square',
                                                   '--results', '/r']
    assert reproduce.commands('gpu', opt)[0][3:] == [
        '--parts', 'warp', '--scenarios', 'wall', '--nworlds', '512', '--methods', 'ours',
        '--shapes', 'square', '--results', '/r']
    opt.methods, opt.clearance = ['coacd'], [20.0]
    thread = reproduce.commands('thread', opt)
    assert len(thread) == 2 and all('--clearance' in c and '20' in c for c in thread)


def test_fetch_extracts_only_files_under_its_directory(tmp_path, monkeypatch):
    from benchmark import fetch
    monkeypatch.setitem(fetch.ASSETS, 'cache', ('x.tar.gz', 'cache', None))

    def archive(names):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as t:
            for name in names:
                ti = tarfile.TarInfo(name)
                ti.size = 1
                t.addfile(ti, io.BytesIO(b'x'))
        path = tmp_path / f'{len(names)}.tar.gz'
        path.write_bytes(buf.getvalue())
        return path

    fetch.fetch('cache', archive(['cache/a.npz', 'cache/b.json']), tmp_path / 'dest')
    assert sorted(p.name for p in (tmp_path / 'dest' / 'cache').iterdir()) == ['a.npz', 'b.json']
    for bad in (['cache/../../evil'], ['results/x.npz'], ['/abs/cache/x']):
        with pytest.raises(ValueError):
            fetch.fetch('cache', archive(bad), tmp_path / 'dest')
