"""Route-A bottleneck autoencoder for the TR07 pilot (torch; development only).

``FlatAE`` is copied from the frozen X3 package's ``models.py`` (itself ported from VERITAS
``src/whitebox_subjects/ladder.py``, MIT, same author) so that this development package never imports
or writes into a frozen package: encoder D -> hidden... -> k (linear code), decoder k -> ...hidden -> D
(linear output), biases on, untied, activation between hidden layers.

Nesting (MS plan/onthefly/P1NL_THEORY_NOTE.md, TR08): with ReLU or SiLU hidden layers of width >= 2k,
every rank-k affine map is representable EXACTLY (u = act(u) - act(-u) for both activations), so the
architecture contains the rank-k projector and the rank-k shrinkage map; training need not find them.

Training = the frozen X3 recipe (Adam, early stopping on the validation Sigma^-1 reconstruction loss of the
NOISY input, restarts selected by that validation loss only).  No truth quantity enters training,
stopping or restart selection.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


def key_int(*parts) -> int:
    return int.from_bytes(hashlib.sha256('/'.join(map(str, parts)).encode()).digest()[:8], 'little') & (2 ** 63 - 1)


def import_torch():
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)
    return torch


def build(torch, D: int, k: int, hidden: Sequence[int], activation: str | None):
    from torch import nn
    act = {'silu': nn.SiLU, 'relu': nn.ReLU}

    def mlp(dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation is not None and i < len(dims) - 2:
                layers.append(act[activation]())
        return nn.Sequential(*layers)

    class FlatAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = mlp([D, *hidden, k])
            self.decoder = mlp([k, *reversed(hidden), D])

        def forward(self, x):
            return self.decoder(self.encoder(x))
    return FlatAE()


def tensor(torch, a):
    return torch.tensor(np.ascontiguousarray(a, dtype=np.float64), dtype=torch.float64)


def train(torch, cfg, X_train, X_val, Li, init_key):
    """One restart.  X_* are PHYSICAL noisy inputs (rows); loss = mean ||(x - f(x)) Sigma^{-1/2}||^2."""
    rec = cfg['recipe']
    torch.manual_seed(init_key)
    D = X_train.shape[1]
    model = build(torch, D, cfg['k'], cfg['hidden'], cfg['activation'])
    opt = torch.optim.Adam(model.parameters(), lr=rec['lr'], betas=tuple(rec['betas']), eps=rec['eps'],
                           weight_decay=rec['weight_decay'])
    gen = torch.Generator().manual_seed(init_key ^ 0x5BD1E995)
    Xtr, Xva, Wt = tensor(torch, X_train), tensor(torch, X_val), tensor(torch, Li)

    def loss(X):
        r = (X - model(X)) @ Wt
        return (r ** 2).sum(1).mean()
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
        if not np.isfinite(v):
            return None, {'failed': 'non-finite validation loss', 'epochs_run': epoch + 1, 'validation_history': history}
        history.append(v)
        if v < best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= rec['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {'best_validation_loss': best, 'best_epoch': best_epoch, 'epochs_run': len(history),
                   'stopped': 'patience' if bad >= rec['patience'] else 'max_epochs', 'validation_history': history}


def fit(torch, cfg, X, Li, seed, N, p):
    """All restarts on the first N - floor(0.2 N) records, validation on the rest; select the restart with
    the smallest validation loss (first index on ties).  Returns (model or None, record)."""
    n_val = int(np.floor(cfg['recipe']['validation_fraction'] * N))
    n_tr = N - n_val
    runs, models = [], []
    for r in range(cfg['recipe']['restarts']):
        m, info = train(torch, cfg, X[:n_tr], X[n_tr:N], Li, key_int(cfg['namespace'], seed, N, p, 'init', r))
        runs.append(info)
        models.append(m)
    ok = [i for i, m in enumerate(models) if m is not None]
    if not ok:
        return None, {'selected': None, 'restarts': runs, 'n_train': n_tr, 'n_validation': n_val}
    sel = min(ok, key=lambda i: runs[i]['best_validation_loss'])
    return models[sel], {'selected': sel, 'restarts': runs, 'n_train': n_tr, 'n_validation': n_val,
                         'optimisation_limited': runs[sel]['stopped'] == 'max_epochs'}


def predict(torch, model, X, chunk=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            out.append(model(tensor(torch, X[i:i + chunk])).numpy())
    return np.concatenate(out)


def code(torch, model, X, chunk=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            out.append(model.encoder(tensor(torch, X[i:i + chunk])).numpy())
    return np.concatenate(out)
