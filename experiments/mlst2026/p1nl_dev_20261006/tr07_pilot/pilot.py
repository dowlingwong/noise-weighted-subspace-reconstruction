"""TR07 pilot runner (development namespace paper1-g1-dev).  Nothing runs on import.

    python -m experiments.mlst2026.p1nl_dev_20261006.tr07_pilot.pilot run --mode fixture --out DIR
    python -m experiments.mlst2026.p1nl_dev_20261006.tr07_pilot.pilot reduce --run DIR
    python -m experiments.mlst2026.p1nl_dev_20261006.tr07_pilot.pilot resources --out DIR   # bounded timing
    # the pilot itself (NOT run in the preparation pass; requires the recorded author decisions, an accepted
    # independent review of the amended implementation, and P1NL_TR07_PILOT_AUTHORISED=yes; N = 12,800 only):
    python -m experiments.mlst2026.p1nl_dev_20261006.tr07_pilot.pilot run --mode pilot --out DIR

Per (seed, p, N) cell: one keyed training pool per (seed, p) (nested prefixes over N; paired across p by
common random numbers), one keyed evaluation set per seed (paired across p); the autoencoder and both
linear baselines see the same first N - floor(0.2 N) training records; the last floor(0.2 N) records are
the autoencoder's validation split only.  Evaluation draws never enter fitting, stopping or selection.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .. import bayes
from ..generator import SparseLatentModel
from . import baselines, diagnostics, gate, models
from .config import FIXTURE, PILOT


def write_json(path, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with Path(path).open('x') as fh:
        json.dump(obj, fh, indent=1, default=conv, allow_nan=False)


def run_cell(torch, cfg, seed, p, N, Yt, Yev, Sev, Zev, refs, model_dir=None):
    m = SparseLatentModel(cfg['geometry'], cfg['theta'], p, cfg['n_sparse'])
    k = cfg['k']
    X = m.physical(Yt[:N])
    t0 = time.time()
    ae, fitrec = models.fit(torch, cfg, X, m.Li, seed, N, p)
    t_fit = time.time() - t0
    n_tr = fitrec['n_train']
    Ap, bp, ip = baselines.fitted_projector(Yt[:n_tr], k)
    As, bs, is_ = baselines.fitted_shrinkage(Yt[:n_tr], k)
    rec = {'seed': seed, 'p': p, 'N': N, 'k': k, 'fit': fitrec, 'fit_seconds': t_fit,
           'R_proj': baselines.affine_risk(Ap, bp, m.K), 'R_shrink': baselines.affine_risk(As, bs, m.K),
           'proj_info': ip, 'shrink_info': is_,
           'proj_per_coordinate': diagnostics.affine_per_coordinate(Ap, bp, m),
           'shrink_per_coordinate': diagnostics.affine_per_coordinate(As, bs, m), 'ok': ae is not None}
    if ae is None:
        return rec
    f_white = lambda Y: models.predict(torch, ae, m.physical(Y)) @ m.Li     # noqa: E731
    Shat = f_white(Yev)
    ev = diagnostics.mc_risk(Shat, Sev, m)
    n = m.n_sparse
    sh_sparse = np.asarray(refs['shrinkage_per_coordinate'][:n])
    num_sample = sh_sparse.sum() - ev['per_sample_coord'][:, :n].sum(1)
    codes = models.code(torch, ae, m.physical(Yev))
    rec.update({
        'R_AE': ev['risk'], 'mcse': ev['mcse'], 'AE_per_coordinate': ev['per_coordinate'],
        'AE_per_coordinate_se': ev['per_coordinate_se'],
        'sparse_num': float(num_sample.mean()), 'sparse_num_mcse': float(num_sample.std(ddof=1) / np.sqrt(len(num_sample))),
        'groups': {'AE': diagnostics.group_sums(ev['per_coordinate'], m, k),
                   'projector': diagnostics.group_sums(rec['proj_per_coordinate'], m, k),
                   'shrinkage': diagnostics.group_sums(rec['shrink_per_coordinate'], m, k)},
        'leakage_k': diagnostics.leakage(Shat, k), 'leakage_6': diagnostics.leakage(Shat, n),
        'recovery': diagnostics.recovery(Shat, codes, Zev[:, :n], m, cfg['diagnostics']['probe_fit_fraction']),
        'interference': diagnostics.interference(f_white, m, cfg['diagnostics']['interference_amplitudes'])})
    if model_dir is not None:
        np.savez_compressed(Path(model_dir) / f'ae_seed{seed}_p{p}_N{N}.npz',
                            **{k_: v.detach().numpy() for k_, v in ae.state_dict().items()})
    return rec


def run(mode, out, diagnostic_N=False, seeds=None):
    cfg = FIXTURE if mode == 'fixture' else PILOT
    if mode == 'pilot':
        if (not cfg['author_decisions_recorded'] or not cfg.get('independent_review_accepted')
                or os.environ.get('P1NL_TR07_PILOT_AUTHORISED') != 'yes'):
            raise SystemExit('pilot refused: requires the recorded author decisions, an independent review that '
                             'accepts the amended implementation, and P1NL_TR07_PILOT_AUTHORISED=yes')
        if diagnostic_N:
            raise SystemExit('pilot refused: N = 51,200 is not part of the planned pilot (author, 6 Oct); a larger '
                             'grid needs a separately reviewed redesign and may not rescue a failed pilot')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'models').mkdir()
    torch = models.import_torch()
    Ns = list(cfg['N']) + (list(cfg['N_diagnostic']) if diagnostic_N else [])
    write_json(out / 'CONFIG.json', {'mode': mode, 'cfg': cfg, 'Ns': Ns, 'torch': torch.__version__})
    seeds = cfg['seeds'] if seeds is None else seeds
    for seed in seeds:
        base = SparseLatentModel(cfg['geometry'], cfg['theta'], 1.0, cfg['n_sparse'])
        for p in cfg['p']:
            m = SparseLatentModel(cfg['geometry'], cfg['theta'], p, cfg['n_sparse'])
            refs = bayes.references(m, cfg['k'])
            Yt, _, _ = m.sample_keyed(max(Ns), cfg['namespace'], seed, 'train')
            Yev, Sev, Zev = m.sample_keyed(cfg['diagnostics']['eval_records'], cfg['namespace'], seed, 'evaluation')
            for N in Ns:
                rec = run_cell(torch, cfg, seed, p, N, Yt, Yev, Sev, Zev, refs, out / 'models')
                rec['refs'] = {k_: v for k_, v in refs.items()}
                write_json(out / f'cell_seed{seed}_p{p}_N{N}.json', rec)
        del base
    return out


def reduce(run_dir):
    run_dir = Path(run_dir)
    conf = json.loads((run_dir / 'CONFIG.json').read_text())
    cfg = conf['cfg']
    cells = [json.loads(f.read_text()) for f in sorted(run_dir.glob('cell_*.json'))]
    G = cfg['gate']
    m = SparseLatentModel(cfg['geometry'], cfg['theta'], G['p'], cfg['n_sparse'])
    refs = bayes.references(m, cfg['k'])
    at_gate = [c for c in cells if c['p'] == G['p'] and c['N'] == G['N']]
    verdict = gate.evaluate(at_gate, refs, m, cfg)
    diag = [{k_: c.get(k_) for k_ in ('seed', 'p', 'N', 'ok', 'R_AE', 'mcse', 'R_proj', 'R_shrink', 'leakage_k',
                                       'leakage_6', 'groups', 'recovery')}
            | {'optimisation_limited': c['fit'].get('optimisation_limited')} for c in cells
            if not (c['p'] == G['p'] and c['N'] == G['N'])]
    summary = {'verdict': verdict, 'diagnostic_cells': diag,
               'population_references': {k_: refs[k_] for k_ in ('projector', 'shrinkage', 'full_wiener', 'bayes',
                                                                 'bayes_error_estimate', 'gap_shrinkage_to_bayes')},
               'note': 'p = 1 cells are diagnostics and never enter the verdict; there is no N = 51,200 cell '
                       '(Amendment 1)'}
    write_json(run_dir / 'GATE.json', summary)
    return summary


def resources(out, epochs=2, N=1600, p=0.1):
    """Bounded timing at production shape (D = 256, k = 3, hidden [128, 32]) in a resource-check
    namespace: a few epochs at a small N, never the gate cell."""
    cfg = {**PILOT, 'namespace': PILOT['namespace'] + '/resource-check',
           'recipe': {**PILOT['recipe'], 'max_epochs': epochs, 'restarts': 1}}
    torch = models.import_torch()
    m = SparseLatentModel(cfg['geometry'], cfg['theta'], p, cfg['n_sparse'])
    Y, _, _ = m.sample_keyed(N, cfg['namespace'], 0, 'train')
    t0 = time.time()
    _, rec = models.fit(torch, cfg, m.physical(Y), m.Li, 0, N, p)
    sec = time.time() - t0
    steps = epochs * int(np.ceil(rec['n_train'] / cfg['recipe']['batch']))
    per_step = sec / steps
    full = {Nn: per_step * int(np.ceil((Nn - int(0.2 * Nn)) / cfg['recipe']['batch'])) * PILOT['recipe']['max_epochs']
            for Nn in (12800, 51200)}         # upper bound: every restart runs the full max_epochs
    res = {'N': N, 'epochs': epochs, 'seconds': sec, 'steps': steps, 'seconds_per_step': per_step,
           'upper_bound_seconds_per_restart_at_max_epochs': full,
           'pilot_upper_bound_hours': {'gate_grid': full[12800] * 4 * 2 * 3 / 3600,
                                       'with_51200_diagnostic': (full[12800] + full[51200]) * 4 * 2 * 3 / 3600},
           'torch': torch.__version__}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f'resources_{int(time.time())}.json', res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--mode', choices=('fixture', 'pilot'), required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--diagnostic-N', action='store_true')
    d = sub.add_parser('reduce')
    d.add_argument('--run', required=True)
    s = sub.add_parser('resources')
    s.add_argument('--out', required=True)
    s.add_argument('--epochs', type=int, default=2)
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        print(run(a.mode, a.out, a.diagnostic_N))
    elif a.cmd == 'reduce':
        print(json.dumps(reduce(a.run)['verdict'], indent=1, default=str))
    else:
        print(json.dumps(resources(a.out, a.epochs), indent=1))


if __name__ == '__main__':
    main()
