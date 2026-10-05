"""X3 science: Sigma^-1 versus isotropic loss for a tied linear, a deep linear and a deep
SiLU autoencoder on the accepted T25 generator (oracle Sigma, theta = 0.25).

The generator, covariance, moments, closed-form unrestricted fit and exact affine risk are
imported read-only from the accepted T25 production code after guard.py has checked their
bytes against the accepted closure (identity 01f2f82f...).
"""
from __future__ import annotations

from . import guard

guard.assert_direct()

import hashlib                                                       # noqa: E402
import numpy as np                                                   # noqa: E402
from scipy.stats import t as student_t                               # noqa: E402
from experiments.mlst2026.t25_production_20260930 import science as t25  # noqa: E402
from experiments.mlst2026.t25_production_20260930.science import (   # noqa: E402,F401
    rng, digest, powers, geometry, map_for, unrestricted, risks, moments)
from .protocol import DEEP                                           # noqa: E402

ACTIVATION = {'deep_linear': None, 'deep_silu': 'silu'}


def key_int(*parts) -> int:
    return int.from_bytes(hashlib.sha256('/'.join(map(str, parts)).encode()).digest()[:8], 'little') & (2 ** 63 - 1)


def import_torch(c):
    """Pinned torch, one thread, deterministic algorithms, float64.  Fails closed."""
    import torch
    if torch.__version__.split('+')[0] != c['torch_version']:
        raise ValueError(f'torch {torch.__version__} != registered {c["torch_version"]}')
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)
    return torch


def make_data(c, seed):
    """T25 generator, oracle Sigma.  Training pool (nested prefixes) and one evaluation set
    per seed, each with its own stream; physical clean signal L s is kept for the truth risk."""
    g = geometry(c)
    sigma = t25.covariance(c)
    L, Li = powers(sigma)
    th, key, d = c['theta'], (c['namespace'], seed), g.d
    root = (g.V(th) * np.sqrt(g.mu)).T
    # evaluated left to right exactly as T25's make_data, so the training draws are T25's
    clean = rng(*key, th, 'signal').normal(size=(max(c['N']), d)) @ root @ L
    train = clean + rng(*key, th, 'noise').normal(size=(max(c['N']), d)) @ L
    M = c['evaluation']['records']
    eclean = rng(*key, th, 'evaluation-signal').normal(size=(M, d)) @ root @ L
    enoisy = eclean + rng(*key, th, 'evaluation-noise').normal(size=(M, d)) @ L
    return g, sigma, L, Li, train, eclean, enoisy


def split(N, fraction):
    n_val = int(np.floor(fraction * N))
    return N - n_val, n_val


def closed_form(c, S, mu, W, k):
    """Tied linear AE optimum under metric W: the unrestricted T25 fit (PCA for I,
    CW-PCA for Sigma^-1), affine map A x + b."""
    B = unrestricted(W, S, k)
    A = map_for(B, W)
    return A, (np.eye(len(A)) - A) @ mu


def metric_loss(model, X, Wt):
    """Mean training-metric reconstruction loss: ||x - f(x)||^2 (isotropic, Wt None) or
    ||Sigma^{-1/2}(x - f(x))||^2 (Wt = Sigma^{-1/2}).  Used by training AND verification."""
    r = X - model(X)
    if Wt is not None:
        r = r @ Wt
    return (r ** 2).sum(1).mean()


def population_references(c):
    g = geometry(c)
    sigma = t25.covariance(c)
    err = float(np.max(np.abs(np.linalg.eigvalsh(g.K(c['theta'])) - np.sort(g.mu))))
    if err > 1e-10:
        raise ValueError('registered spectrum invariant failed')
    return {'D': g.d, 'k': c['k'], 'sigma_trace': float(np.trace(sigma)),
            'sigma_lambda_min': float(np.linalg.eigvalsh(sigma)[0]), 'theta': c['theta'], 'whitened_spectrum_max_error': err}


def train_deep(c, torch, x_train, x_val, whiten, activation, init_key):
    from .models import FlatAE
    rec = c['recipe']
    torch.manual_seed(init_key)
    model = FlatAE(c['C'] * c['T'], c['k'], c['hidden'], activation)
    opt = torch.optim.Adam(model.parameters(), lr=rec['lr'], betas=tuple(rec['betas']), eps=rec['eps'],
                           weight_decay=rec['weight_decay'])
    gen = torch.Generator().manual_seed(init_key ^ 0x5BD1E995)
    Xtr, Xva = torch.from_numpy(x_train), torch.from_numpy(x_val)
    Wt = None if whiten is None else torch.from_numpy(whiten)

    def loss(X):
        return metric_loss(model, X, Wt)
    best, best_state, best_epoch, bad, history = float('inf'), None, -1, 0, []
    n = len(Xtr)
    for epoch in range(rec['max_epochs']):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, rec['batch']):
            opt.zero_grad()
            loss(Xtr[perm[i:i + rec['batch']]]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss(Xva))
        history.append(v)
        if v < best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= rec['patience']:
                break
    if best_state is None or not np.isfinite(best):
        raise ValueError('training produced no finite validation loss')
    model.load_state_dict(best_state)
    with torch.no_grad():
        train_loss = float(loss(Xtr))
    return model, {'best_validation_loss': best, 'best_epoch': best_epoch, 'epochs_run': len(history),
                   'stopped': 'patience' if bad >= rec['patience'] else 'max_epochs',
                   'final_train_loss_at_best': train_loss, 'validation_history': history}


def build_model(c, torch, cls, state):
    from .models import FlatAE
    model = FlatAE(c['C'] * c['T'], c['k'], c['hidden'], ACTIVATION[cls])
    model.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state.items()})
    model.eval()
    return model


def evaluate(c, torch, seed, fn, eclean, enoisy, Li):
    """Truth-referenced clean risks (Sigma^-1 and Euclidean), held-out whitened residual and
    SURE on the shared evaluation set; one Hutchinson probe stream per seed for every model."""
    from .models import sure_terms
    X, Cl, Wt = torch.from_numpy(enoisy), torch.from_numpy(eclean), torch.from_numpy(Li)
    with torch.no_grad():
        out = fn(X)
        err = out - Cl
        rw = float(((err @ Wt) ** 2).sum(1).mean())
        re = float((err ** 2).sum(1).mean())
    gen = torch.Generator().manual_seed(key_int(c['namespace'], seed, 'hutchinson'))
    st = sure_terms(fn, X, Wt, c['evaluation']['hutchinson_probes'], gen)
    return {'clean_whitened': rw, 'clean_euclidean': re, 'observed_whitened': float(st['residual'].mean()),
            'sure': float(st['sure'].mean()), 'divergence': float(st['divergence'].mean())}


def fit_cell(c, seed, N, torch=None):
    torch = torch or import_torch(c)
    g, sigma, L, Li, train, eclean, enoisy = make_data(c, seed)
    x, mu, S = moments(train, N)
    n_tr, n_val = split(N, c['recipe']['validation_fraction'])
    metrics = {'isotropic': None, 'sigma_inverse': Li}
    Wmat = {'isotropic': np.eye(g.d), 'sigma_inverse': Li @ Li}
    Cs = L @ g.K(c['theta']) @ L
    record = {'seed': seed, 'N': N, 'n_train': n_tr, 'n_validation': n_val, 'training_sha': digest(x),
              'evaluation_sha': digest(enoisy), 'models': {}, 'training': {}}
    arrays = {}
    for loss in c['losses']:
        A, b = closed_form(c, S, mu, Wmat[loss], c['k'])
        arrays[f'tied_linear__{loss}__A'], arrays[f'tied_linear__{loss}__b'] = A, b
        At, bt = torch.from_numpy(A), torch.from_numpy(b)
        ev = evaluate(c, torch, seed, lambda X: X @ At.T + bt, eclean, enoisy, Li)
        exact = risks(A, b, Cs, sigma, Li @ Li)
        ev.update(exact_clean_whitened=exact['clean'], exact_clean_euclidean=exact['euclidean_clean'],
                  exact_divergence=float(np.trace(A)))
        record['models'][f'tied_linear__{loss}'] = ev
    for cls in DEEP:
        for loss in c['losses']:
            runs, states = [], []
            for r in range(c['recipe']['restarts']):
                init = key_int(c['namespace'], seed, N, 'init', r)     # shared by both classes and both losses
                model, diag = train_deep(c, torch, x[:n_tr], x[n_tr:N], metrics[loss], ACTIVATION[cls], init)
                runs.append(diag)
                states.append({k: t.detach().numpy().copy() for k, t in model.state_dict().items()})
            sel = int(np.argmin([d['best_validation_loss'] for d in runs]))
            tag = f'{cls}__{loss}'
            for k, v in states[sel].items():
                arrays[f'{tag}__{k}'] = v
            model = build_model(c, torch, cls, states[sel])
            record['models'][tag] = evaluate(c, torch, seed, model, eclean, enoisy, Li)
            record['training'][tag] = {'selected': sel, 'restarts': runs}
    return record, arrays


def state_of(arrays, tag):
    pre = tag + '__'
    return {k[len(pre):]: v for k, v in arrays.items() if k.startswith(pre) and not k.endswith(('__A', '__b'))}


# --------------------------------------------------------------------------------------
# contrasts, inference, registered predictions
# --------------------------------------------------------------------------------------
def contrast_values(models, classes):
    w = lambda tag: models[tag]['clean_whitened']                    # noqa: E731
    e = lambda tag: models[tag]['clean_euclidean']                   # noqa: E731
    out = {}
    for cls in classes:
        out[f'metric/whitened/{cls}'] = w(f'{cls}__sigma_inverse') - w(f'{cls}__isotropic')
        out[f'metric/euclidean/{cls}'] = e(f'{cls}__sigma_inverse') - e(f'{cls}__isotropic')
        for loss in ('isotropic', 'sigma_inverse'):
            out[f'sure_gap/{cls}/{loss}'] = models[f'{cls}__{loss}']['sure'] - w(f'{cls}__{loss}')
    for loss in ('isotropic', 'sigma_inverse'):
        out[f'class/whitened/{loss}/deep_silu-tied_linear'] = w(f'deep_silu__{loss}') - w(f'tied_linear__{loss}')
        out[f'class/whitened/{loss}/deep_silu-deep_linear'] = w(f'deep_silu__{loss}') - w(f'deep_linear__{loss}')
        out[f'class/whitened/{loss}/deep_linear-tied_linear'] = w(f'deep_linear__{loss}') - w(f'tied_linear__{loss}')
    return out


def sign(row):
    return '-' if row['ci'][1] < 0 else '+' if row['ci'][0] > 0 else '?'


def reduce_records(c, records):
    expected = {(s, n) for s in c['seeds'] for n in c['N']}
    got = [(r['seed'], r['N']) for r in records]
    if set(got) != expected or len(got) != len(set(got)):
        raise ValueError('missing/duplicate cells')
    groups = {}
    for r in sorted(records, key=lambda r: (r['seed'], r['N'])):
        for name, val in contrast_values(r['models'], c['classes']).items():
            groups.setdefault((r['N'], name), []).append(val)
    family = len(groups)
    rows = []
    for (N, name), values in sorted(groups.items()):
        v = np.asarray(values, dtype=float)
        sd = float(v.std(ddof=1))
        half = float(student_t.ppf(1 - .05 / (2 * family), len(v) - 1) * sd / np.sqrt(len(v)))
        mean = float(v.mean())
        rows.append({'N': N, 'contrast': name, 'mean': mean, 'ci': [mean - half, mean + half], 'seeds': len(v),
                     'zero_standard_error': sd == 0})
    get = {(r['N'], r['contrast']): r for r in rows}
    p1 = [{'N': n, 'class': cls, 'sign': sign(get[(n, f'metric/whitened/{cls}')])} for cls in c['classes'] for n in c['N']]
    p2 = [{'N': n, 'loss': loss, 'sign': sign(get[(n, f'class/whitened/{loss}/deep_silu-tied_linear')])}
          for loss in c['losses'] for n in c['N']]
    stops = [d['stopped'] for r in records for t in r['training'].values() for d in t['restarts']]
    return {
        'family_size': family, 'interval': 'approximate Bonferroni Student-t over independent seeds (23 df at 24 seeds)',
        'zero_SE_rule': 'degenerate observed-seed interval; zero is undecided',
        'contrasts': rows,
        'training_stops': {k: stops.count(k) for k in ('patience', 'max_epochs')},
        'predictions': {
            'P1': {'rule': 'HOLDS if Sigma^-1-minus-isotropic whitened clean risk is decided negative for every '
                           'class at every N; FAILS if any is decided positive; UNDECIDED otherwise',
                   'verdict': 'HOLDS' if all(x['sign'] == '-' for x in p1) else
                              'FAILS' if any(x['sign'] == '+' for x in p1) else 'UNDECIDED',
                   'locations': p1},
            'P2': {'rule': 'HOLDS if the SiLU-minus-tied-linear whitened clean risk is not decided negative at any '
                           '(loss, N); FAILS if any is decided negative (SiLU better beyond seed variation)',
                   'verdict': 'FAILS' if any(x['sign'] == '-' for x in p2) else 'HOLDS',
                   'counts': {s: sum(x['sign'] == s for x in p2) for s in '-+?'}, 'locations': p2}}}
