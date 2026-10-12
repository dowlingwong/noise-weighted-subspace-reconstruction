"""D8 secondary, truth-free analysis on the real evoked trials (descriptive only; never feeds a PZ reading).

Per subject: fit both classes at rank (k_c, k_t) = (2, 3) under Sigma_ref on the stimulus-locked trials of the
training runs (default 3-5, face and scrambled pooled), then on the held-out run (6):
 (i)  held-out weighted residual  mean (x - f(x))^T Sigma_ref^{-1} (x - f(x))  for U and F.  By Prop. 2.1 its
      ordering matches the whitened clean-risk ordering only if Sigma_ref is the noise covariance and the noise
      is uncorrelated with the signal: a stated condition, not a check;
 (ii) face-vs-scrambled decoding: a shrinkage LDA on the k = 6 reconstruction coordinates (GLS coefficients of
      f(x) on the fitted basis), trained on the training-run trials and tested on the held-out run; the same
      classifier on the raw trials projected on the top-6 Sigma_ref-whitened PCA axes of the training trials is
      reported as a reference.

Command:  python -m experiments.mlst2026.meg_z_20261013.secondary_d8 --derived DIR --subjects ... --out D8.json
Confirmation subjects are refused unless the protocol is frozen (F7).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import common, fit, population
from .common import vec, symmetric


def lda_fit(Z: np.ndarray, y: np.ndarray, shrink: float = 0.1):
    m1, m0 = Z[y].mean(0), Z[~y].mean(0)
    Zc = np.concatenate([Z[y] - m1, Z[~y] - m0])
    S = symmetric(Zc.T @ Zc / max(len(Zc) - 2, 1))
    S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S))
    w = np.linalg.solve(S, m1 - m0)
    return w, -0.5 * w @ (m1 + m0) + np.log(y.mean() / (1 - y.mean()))


def lda_accuracy(Ztr, ytr, Zte, yte) -> float:
    w, c = lda_fit(Ztr, ytr)
    return float(np.mean(((Zte @ w + c) > 0) == yte))


def subject_d8(sd, proto: dict) -> dict:
    cfg = proto['secondary_d8']
    sc = proto['signal']
    kc, kt = int(cfg['rank'][0]), int(cfg['rank'][1])
    ref = population.reference_for(sd, proto)
    W = fit.metric(ref['cov'])
    tr = [sd.trials[r] for r in cfg['train_runs'] if r in sd.trials]
    te = sd.trials.get(int(cfg['test_run']))
    if not tr or te is None:
        return {'status': 'unavailable', 'reason': 'missing trials'}
    Xtr = np.concatenate([vec(t['X']) for t in tr])
    ytr = np.concatenate([t['face'] for t in tr]).astype(bool)
    Xte, yte = vec(te['X']), te['face'].astype(bool)
    f = fit.fit_pair(Xtr, len(Xtr), W, sd.C, sd.T, kc, kt, proto['learners'], (proto['namespace'], sd.subject, 'd8'))
    out = {'n_train': int(len(Xtr)), 'n_test': int(len(Xte)), 'k': kc * kt, 'condition': 'Prop. 2.1 ordering requires '
           'Sigma_ref = noise covariance and noise uncorrelated with signal (not checked)'}
    for cls in ('U', 'F'):
        A, b = f['maps'][cls]
        R = Xte - (Xte @ A.T + b)
        out[f'heldout_weighted_residual_{cls}'] = float(np.mean(np.einsum('ij,ij->i', R @ W, R)))
        # k-dimensional GLS coordinates of the reconstruction on the fitted basis (range of A)
        q, _ = np.linalg.qr(A @ np.linalg.qr(np.random.default_rng(0).normal(size=(sd.d, kc * kt)))[0])
        G = q.T @ W @ q
        coef = lambda X: np.linalg.solve(G, q.T @ W @ (X @ A.T + b).T).T          # noqa: E731
        out[f'decoding_accuracy_{cls}'] = lda_accuracy(coef(Xtr), ytr, coef(Xte), yte)
    _, li = population.spd_powers(ref['cov'])
    Ztr, Zte = (Xtr - Xtr.mean(0)) @ li, (Xte - Xtr.mean(0)) @ li
    v = np.linalg.eigh(symmetric(Ztr.T @ Ztr))[1][:, -kc * kt:]
    out['decoding_accuracy_raw_whitened_pca6'] = lda_accuracy(Ztr @ v, ytr, Zte @ v, yte)
    out['heldout_residual_difference_F_minus_U'] = out['heldout_weighted_residual_F'] - out['heldout_weighted_residual_U']
    out['chance'] = float(max(yte.mean(), 1 - yte.mean()))
    return out


def main(argv=None):
    from .preprocess import load_subject
    from .protocol import load_protocol
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', required=True)
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--protocol')
    a = ap.parse_args(argv)
    proto = load_protocol(a.protocol)
    from .preprocess import refuse_unfrozen_confirmation
    refuse_unfrozen_confirmation(a.subjects.split(','), proto, a.protocol)          # F7: no confirmation subject before the freeze
    res = {'descriptive_only': True, 'protocol_sha256': common.sha_json(proto), 'subjects': {}}
    for s in a.subjects.split(','):
        res['subjects'][s] = subject_d8(load_subject(Path(a.derived) / f'{s}.h5'), proto)
        print(s, json.dumps(res['subjects'][s]))
    common.atomic_json(a.out, res)


if __name__ == '__main__':
    main()
