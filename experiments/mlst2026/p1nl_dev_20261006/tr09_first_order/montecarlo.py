"""Small development Monte Carlo checks of the TR09 first-order note (namespace paper1-tr09-dev).

Not the registered TR09 experiment, not confirmation evidence.  Linear classes only, oracle metric,
true-whitened coordinates.  Per replicate (theta, p, N, r), with Shat the centred sample covariance
(divisor N, the T25 convention) and ybar the sample mean:

* unrestricted: Phat_U = top-k eigenprojector of Shat; Delta_U = tr(P_U M) - tr(Phat_U M);
* product, LOCAL branch: Newton refinement of tr(P Shat) started at the population maximiser P_F*;
  Delta_F = tr(P_F* M) - tr(Phat_F M); q_F = 1/2 g^T H^-1 g with g the gradient of tr(P (Shat - M)) at P_F*
  (the first-order prediction evaluated on the same draw);
* product, GLOBAL selection (T25-like): HOSVD + 3 random starts, refined, maximum training objective
  selected; reported as the fraction of replicates whose selection differs from the local branch;
* risks R = tr K - tr(P K) + k + ||(I - P) ybar||^2 for both classes and the contrast D = R_F - R_U.
Draws are paired across p (sample_keyed) and across classes (same data).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..generator import SparseLatentModel, rng
from . import core

NAMESPACE = 'paper1-tr09-dev'


def replicate(model, pt_star, Hinv, N, key, global_check=False):
    y, _, _ = model.sample_keyed(N, NAMESPACE, *key)
    ybar = y.mean(0)
    yc = y - ybar
    S = yc.T @ yc / N
    M, k, d = model.M, model.k, model.d
    trK = float(np.trace(model.K))
    w, V = np.linalg.eigh(S)
    PU = V[:, -k:] @ V[:, -k:].T
    PUs = model.V[:, :k] @ model.V[:, :k].T
    out = {}
    out['Delta_U'] = float(np.trace(PUs @ M) - np.trace(PU @ M))
    out['offset_U'] = float(ybar @ (np.eye(d) - PU) @ ybar)
    E4 = core.reshape4(S - M, model.C, model.T)
    g = pt_star.gradient(E4)
    out['q_F'] = 0.5 * float(g @ Hinv @ g)
    loc, info = core.refine(S, model.C, model.T, pt_star)
    PF = loc.P()
    PFs = pt_star.P()
    out['Delta_F'] = float(np.trace(PFs @ M) - np.trace(PF @ M))
    out['offset_F'] = float(ybar @ (np.eye(d) - PF) @ ybar)
    out['local_status'] = info['status']
    out['R_U'] = trK - float(np.trace(PU @ model.K)) + k + out['offset_U']
    out['R_F'] = trK - float(np.trace(PF @ model.K)) + k + out['offset_F']
    out['D'] = out['R_F'] - out['R_U']
    if global_check:
        best, bval = None, -np.inf
        for name, Uc, Ut in core.product_starts(S, model.C, model.T, model.kc, model.kt,
                                                rng(NAMESPACE, *key, 'starts'), n_random=3):
            p2, i2 = core.refine(S, model.C, model.T, core.ProductPoint(Uc, Ut))
            if i2['value'] > bval:
                best, bval = p2, i2['value']
        out['global_differs'] = bool(np.linalg.norm(best.P() - PF, 'fro') > 1e-6)
        out['global_minus_local_objective'] = float(bval - info['value'])
        PG = best.P()
        out['R_F_global'] = trK - float(np.trace(PG @ model.K)) + k + float(ybar @ (np.eye(d) - PG) @ ybar)
    return out


def run_cell(theta, p, N, reps, rep0=0, global_every=1):
    model = SparseLatentModel(theta=theta, p=p)
    base = SparseLatentModel(theta=theta, p=1.0)
    pt, dist = core.population_product_optimum(base, rng(NAMESPACE, 'population', theta), n_random=6)
    fo = core.first_order(base, pt)
    rows = []
    t0 = time.time()
    for r in range(rep0, rep0 + reps):
        row = replicate(model, pt, fo['Hinv'], N, (theta, N, r), global_check=(r % global_every == 0))
        row.update(theta=theta, p=p, N=N, rep=r)
        rows.append(row)
    return {'theta': theta, 'p': p, 'N': N, 'reps': reps, 'seconds': time.time() - t0,
            'prediction': core.summary(model, pt, fo), 'rows': rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--theta', type=float, nargs='+', default=[0.25, 0.40])
    ap.add_argument('--p', type=float, nargs='+', default=[1.0, 0.3, 0.1])
    ap.add_argument('--N', type=int, nargs='+', default=[400, 1000, 3200, 12800])
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--rep0', type=int, default=0)
    ap.add_argument('--global-every', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for th in a.theta:
        for N in a.N:
            for p in a.p:
                f = out / f'cell_theta{th}_N{N}_p{p}_r{a.rep0}-{a.rep0 + a.reps - 1}.json'
                if f.exists():
                    continue
                res = run_cell(th, p, N, a.reps, a.rep0, a.global_every)
                f.write_text(json.dumps(res))
                print(f.name, round(res['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
