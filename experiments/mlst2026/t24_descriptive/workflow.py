"""T24-D run / verify / table. One attempt, write-once outputs outside the sources."""
from __future__ import annotations
import hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
from . import science as sc

PACKAGE = Path(__file__).resolve().parent
EXP = PACKAGE.parents[2]
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
FILES = ('X_test.npy', 'features_test.csv')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write_new(p, v):
    p = Path(p)
    with p.open('x') as f:
        json.dump(v, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def source_identity():
    import noise_geometry
    ng = Path(noise_geometry.__file__).resolve().parent
    from experiments.mlst2026 import t24_cumulant
    files = sorted(PACKAGE.glob('*.py')) + [Path(t24_cumulant.__file__).resolve().parent / 'core.py'] \
        + sorted((ng / 'noise').glob('*.py')) + sorted((ng / 'subspace').glob('*.py')) \
        + [ng / 'cresst' / 'splits.py', ng / 'cresst' / 'loader.py']
    return {str(p.relative_to(EXP)) if p.is_relative_to(EXP) else str(p): sha(p) for p in files}


def environment():
    import scipy, pandas
    return {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pandas.__version__}


def gates(out):
    for k in THREADS:
        if os.environ.get(k) != '1':
            raise ValueError('one numerical thread required: ' + k)
    out = Path(out).resolve()
    roots = [EXP, (EXP.parent / 'arXiv-v1-polished-extended-preprint').resolve()] + ([Path(os.environ['MLST_MS_ROOT']).resolve()] if os.environ.get('MLST_MS_ROOT') else [])
    if any(out.is_relative_to(r) for r in roots):
        raise ValueError('output must be outside the source trees')
    return out


def load(data_root, c):
    from noise_geometry.cresst.loader import load_cresst_release
    x, f, _ = load_cresst_release(Path(data_root), split=c['split'], mmap=True)
    need = {'run', 'channel', 'hours', 'noise', 'clean', 'release_row_index'}
    if not need <= set(f.columns):
        raise ValueError('missing metadata columns: ' + str(need - set(f.columns)))
    if x.shape != (len(f), c['width']):
        raise ValueError('waveform shape/metadata mismatch')
    for k in ('noise', 'clean'):
        v = f[k].astype(str).str.strip().str.lower()
        if not v.isin(['true', 'false', '1', '0', '1.0', '0.0']).all():
            raise ValueError('invalid boolean flag')
        f[k] = v.isin(['true', '1', '1.0'])
    if not np.isfinite(f.hours.to_numpy(float)).all():
        raise ValueError('nonfinite hours')
    keys = sorted({f'{int(r)}/{int(ch)}' for r, ch in zip(f.run, f.channel)})
    if keys != sorted(c['units']):
        raise ValueError(f'unit set differs from the registered seven test channels: {keys}')
    if not np.array_equal(f.release_row_index.to_numpy(np.int64), np.arange(len(f))):
        raise ValueError('release_row_index is not the file row order')
    return x, f


def counts(x, f, unit):
    run, ch = (float(v) for v in unit.split('/'))
    g = f[(f.run.astype(float) == run) & (f.channel.astype(float) == ch)]
    rows = g.release_row_index.to_numpy(np.int64)
    hashes = [hashlib.sha256(np.ascontiguousarray(x[i]).tobytes()).hexdigest() for i in rows]
    finite = bool(np.all([np.isfinite(x[i]).all() for i in rows]))
    if not finite:
        raise ValueError('nonfinite waveform in ' + unit)
    return {'rows': int(len(g)), 'noise_flagged': int(g.noise.sum()), 'noise_and_clean': int((g.noise & g.clean).sum()),
            'clean_pulses': int((g.clean & ~g.noise).sum()), 'duplicate_waveforms': int(len(hashes) - len(set(hashes))),
            'duplicate_hours': int(g.hours.duplicated().sum()),
            'file_order_nonincreasing': int(np.sum(np.diff(g.hours.to_numpy(float)) <= 0))}


def run(data_root, out, c=sc.CONFIG):
    out = gates(out); data_root = Path(data_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    before = {n: {'sha256': sha(data_root / n), 'bytes': (data_root / n).stat().st_size} for n in FILES}
    write_new(out / 'START.json', {'version': c['version'], 'config': c, 'data_root': str(data_root), 'data_files': before,
                                   'sources': source_identity(), 'environment': environment(), 'retries': 0})
    try:
        x, f = load(data_root, c)
        units = {}
        for unit in c['units']:
            n = counts(x, f, unit)
            arrays = sc.unit_arrays(x, f, unit, c)
            tag = unit.replace('/', '_')
            with (out / f'unit_{tag}.npz').open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            units[unit] = sc.unit_summary(arrays, unit, c, n)
        after = {n: {'sha256': sha(data_root / n), 'bytes': (data_root / n).stat().st_size} for n in FILES}
        if after != before:
            raise ValueError('data changed during the run')
        write_new(out / 'summary.json', {'version': c['version'], 'units': units, 'scope': sc.__doc__})
        write_table(out, units)
        write_new(out / 'COMPLETE.json', {'elapsed_seconds': time.time() - started, 'files': inventory(out)})
    except BaseException as e:
        write_new(out / 'FAILED.json', {'error': f'{type(e).__name__}: {e}', 'retry_allowed': False})
        raise


def inventory(root):
    root = Path(root)
    return {p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name not in ('COMPLETE.json', 'VERIFIED.json')}


def fmt(v):
    return 'unavailable' if v is None else f'{v:.4f}'


def write_table(out, units):
    head = 'unit,block,N,p,b2,b2_iid_mean,b2_iid_q05,b2_iid_q95,b2_ratio,b1,b1_iid_mean,b2_early,b2_late'
    lines = [head]
    for unit, u in units.items():
        for block in ('retained', 'residual'):
            d = u[block]; r = d['reference']; h = {x['half']: x.get('b2') for x in d['chronological_halves']}
            lines.append(','.join(str(v) for v in [unit, block, d['N'], d['p'], f"{d['b2']:.4f}", f"{r['b2_mean']:.4f}",
                         f"{r['b2_quantiles'][0]:.4f}", f"{r['b2_quantiles'][1]:.4f}", f"{d['b2_ratio_to_reference_mean']:.4f}",
                         f"{d['b1']:.4f}", f"{r['b1_mean']:.4f}", fmt(h.get('early')), fmt(h.get('late'))]))
    with (Path(out) / 'table.csv').open('x') as f:
        f.write('\n'.join(lines) + '\n')


def verify(root, data_root=None, c=sc.CONFIG):
    """Portable: hashes, statistics and references recomputed from saved blocks,
    basis geometry. With data_root: selection, metric, bases and projections are
    rebuilt from the raw release and compared."""
    root = Path(root)
    if (root / 'FAILED.json').exists() or not (root / 'COMPLETE.json').is_file():
        raise ValueError('failed or incomplete attempt')
    done = read(root / 'COMPLETE.json'); start = read(root / 'START.json')
    if done['files'] != inventory(root):
        raise ValueError('output inventory/hash changed')
    if start['config'] != c or start['version'] != c['version']:
        raise ValueError('config differs from the registered design')
    expected = {'START.json', 'summary.json', 'table.csv'} | {f"unit_{u.replace('/', '_')}.npz" for u in c['units']}
    if set(done['files']) != expected:
        raise ValueError('unexpected or missing outputs')
    summary = read(root / 'summary.json')
    raw = None
    if data_root is not None:
        data_root = Path(data_root).resolve()
        if {n: {'sha256': sha(data_root / n), 'bytes': (data_root / n).stat().st_size} for n in FILES} != start['data_files']:
            raise ValueError('raw data differ from the run')
        raw = load(data_root, c)
    for unit in c['units']:
        a = dict(np.load(root / f"unit_{unit.replace('/', '_')}.npz", allow_pickle=False))
        R, S = a['retained_basis'], a['residual_basis']
        if not np.allclose(R @ R.T, np.eye(len(R)), atol=1e-10) or not np.allclose(S.T @ S, np.eye(S.shape[1]), atol=1e-10) or np.abs(R @ S).max() > 1e-10:
            raise ValueError('basis geometry invalid: ' + unit)
        if raw is not None:
            x, f = raw
            rebuilt = sc.unit_arrays(x, f, unit, c)
            for k, v in rebuilt.items():
                if not np.allclose(a[k], v, rtol=1e-9, atol=1e-10):
                    raise ValueError(f'rebuild mismatch {unit}/{k}')
            if counts(x, f, unit) != summary['units'][unit]['counts']:
                raise ValueError('row counts differ: ' + unit)
        want = sc.unit_summary(a, unit, c, summary['units'][unit]['counts'])
        if json.loads(json.dumps(want)) != summary['units'][unit]:
            raise ValueError('summary not reproduced from saved arrays: ' + unit)
    return {'verified': True, 'mode': 'raw_rebuild' if raw is not None else 'saved_arrays', 'units': len(c['units'])}
