"""DESIGN 1.2–1.3: metadata-only guarded blocks, fixed grid, cyclic block draws."""
import math
import numpy as np
from .protocol import load

ROLES = ('ref', 'train', 'cal', 'eval')


def plan(n_times, sfreq, T=None, *, decimated=False):
    cfg = load(); w = cfg['windows']; pre = cfg['preprocessing']
    T = w['T'] if T is None else int(T)
    if T < 1 or n_times < 1 or sfreq <= 0:
        raise ValueError('positive sample count, rate and T required')
    fs = float(sfreq) if decimated else float(sfreq) / pre['decimation_factor']
    n = int(n_times) if decimated else math.ceil(n_times / pre['decimation_factor'])
    width = math.floor(w['block_seconds'] * fs)
    guard = math.ceil(w['guard_seconds'] * fs)
    edge = math.ceil(pre['edge_seconds'] * fs)
    if width <= guard or width < T:
        raise ValueError('block has no active support')
    records = {r: [] for r in ROLES}; blocks = []
    for b, start in enumerate(range(edge, n-edge, width)):
        end = min(start+width, n-edge)
        if end-start != width:
            continue
        role = w['cycle'][b % len(w['cycle'])]
        active = start+guard
        blocks.append({'block': b, 'start': start, 'active_start': active, 'end': end, 'role': role,
                       'third': min(2, 3*b // max(1, (n-2*edge)//width))})
        for position, off in enumerate(range(active, end-T+1, 1 if role == 'ref' else T)):
            records[role].append({'start': off, 'block': b, 'position': position, 'third': blocks[-1]['third']})
    return {'n_decimated': n, 'sfreq_hz': fs, 'T': T, 'block_samples': width, 'guard_samples': guard,
            'edge_samples': edge, 'blocks': blocks, 'records': records}


def cut(data, window_plan):
    """Input (channels,samples); output (windows,d) with c+C*t indexing."""
    x = np.asarray(data, float)
    if x.ndim != 2 or x.shape[1] != window_plan['n_decimated']:
        raise ValueError('data shape differs from plan')
    T = window_plan['T']; out = {}
    for role in ROLES:
        rows = window_plan['records'][role]
        X = np.stack([x[:, r['start']:r['start']+T].T.reshape(-1) for r in rows]) if rows else np.empty((0,x.shape[0]*T))
        meta = {k: np.array([r[k] for r in rows], dtype=int) for k in ('start','block','position','third')}
        out[role] = (X, meta)
    return out


def sample_blocks(X, meta, count, gen):
    """Uniform full-block order and independent cyclic rotations; no new windows."""
    X = np.asarray(X); count = int(count)
    if not 0 < count <= len(X):
        raise ValueError('block draw count outside retained population')
    blocks = np.unique(meta['block'])
    ids = [np.flatnonzero(meta['block'] == b) for b in blocks]
    if len(set(map(len, ids))) != 1:
        raise ValueError('unequal/partial blocks forbidden')
    drawn = []
    for j in gen.permutation(len(ids)):
        idx = ids[j]
        drawn.extend(np.roll(idx, -int(gen.integers(len(idx)))).tolist())
        if len(drawn) >= count:
            break
    idx = np.asarray(drawn[:count], int)
    return X[idx], idx


def census(units):
    result = {}
    for u in units:
        by_T = {}
        for T in load()['windows']['T_candidates']:
            p = plan(u['n_times'], u['sfreq_hz'], T)
            by_T[str(T)] = {r: len(p['records'][r]) for r in ROLES}
        result[u['measurement_date']] = by_T
    return result


def grids(P):
    if P < 1:
        raise ValueError('empty training pool')
    cfg = load()['budgets']; A = P//5; B = max(5*cfg['N_min_grid'], A/2)
    # Exact 10% margin avoids floor(1111 / 1.1) floating-point error.
    maximum = 10*P//11
    grid = lambda x: sorted({math.floor(x*f+.5) for f in cfg['N_factors'] if 1 <= math.floor(x*f+.5) <= maximum})
    return {'P': P, 'N_max': maximum, 'A': A, 'B': B, 'A_grid': grid(A), 'B_grid': grid(B)}
