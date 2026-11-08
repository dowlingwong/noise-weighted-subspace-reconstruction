"""Resume-4 Step B: second-host determinism replay of 10 of the 60 seeded Stage 0c rows.

Uses the frozen learner `learning.fit_cell` unchanged, under the fixed
OPENBLAS_CORETYPE used by the verified replay.  Pass criteria (pre-registered in the
resume-4 brief): training objective within 1e-9 relative AND |Delta D| <= the row's
stored `restart_spread`, in 10/10 rows.

    python -m experiments.mlst2026.hcp_xopt_20261106.resume4_replay <n_rows> <out.json>
"""
import json, os, socket, sys, time
import numpy as np

from .config import ROOT
W = ROOT
IN = W + '/inputs/hcp_confirmation_20261020'
sys.path.insert(0, W + '/EXP')
sys.path.insert(1, W + '/EXP/src')

from experiments.mlst2026.hcp_rnoise_20261019.learning import fit_cell
from experiments.mlst2026.hcp_rnoise_20261019.signal import build

N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = sys.argv[2] if len(sys.argv) > 2 else (W + '/out/resume4/replay_host2.json')
SEED = 20261106
OBJ_RTOL = 1e-9


def scan(path):
    recs, off = [], 0
    with open(path, 'rb') as f:
        for line in f:
            r = json.loads(line)
            k = r['kernel']
            recs.append(dict(date=r['date'], geometry=r['geometry'], N=r['N'], arm=r['arm'], m=r['m'],
                             replicate=r['replicate'], lim=bool(k['optimization_limited']),
                             lim_t25=bool(k['optimization_limited_t25_rule']), theta=k['theta'],
                             selected=k['selected'], obj=k['restarts'][k['selected']]['objective'],
                             spread=float(k['restart_spread']), D=r['D'], offset=off))
            off += len(line)
    return recs


def pick_first_n(recs, n):
    """First n rows of the 60-row Stage 0c seeded selection (same construction as replay.py)."""
    bydate = {}
    for rec in recs:
        bydate.setdefault(rec['date'], []).append(rec)
    rng = np.random.default_rng(SEED)
    chosen = []
    for d in sorted(bydate):
        block = bydate[d]
        picks = []
        for pool in ([r for r in block if r['lim']], [r for r in block if not r['lim']]):
            if pool:
                picks.append(pool[int(rng.integers(len(pool)))])
        while len(picks) < 2:
            rest = [r for r in block if r not in picks]
            picks.append(rest[int(rng.integers(len(rest)))])
        chosen.extend(picks[:2])
        if len(chosen) >= n:
            break
    return chosen[:n]


def main():
    t0 = time.time()
    recs = scan(IN + '/FITS.jsonl')
    chosen = pick_first_n(recs, N_ROWS)
    print(f"HOST {socket.gethostname()} OPENBLAS_CORETYPE={os.environ.get('OPENBLAS_CORETYPE')}",
          flush=True)
    print(f"scanned {len(recs)} rows, selected {len(chosen)} rows, seed={SEED}", flush=True)
    cache = {}
    rows = []
    with open(IN + '/FITS.jsonl', 'rb') as f:
        for i, rec in enumerate(chosen):
            f.seek(rec['offset'])
            row = json.loads(f.readline())
            k = row['kernel']
            date, geo, N, arm, m, rep = (row['date'], row['geometry'], row['N'],
                                         row['arm'], row['m'], row['replicate'])
            if date not in cache:
                roles = {}
                for role in ('ref', 'train', 'cal', 'eval'):
                    with np.load(f'{IN}/arrays/{date}/{role}.npz', allow_pickle=False) as a:
                        roles[role] = (a['X'], {kk: a[kk] for kk in a.files if kk != 'X'})
                S = np.asarray(json.loads(open(f'{IN}/arrays/{date}/Sref.json', 'rb').read()), float)
                cache[date] = (roles, S)
            roles, S = cache[date]
            model = build(S, float(k['theta']))
            ta = time.time()
            got = fit_cell(roles, model, S, N, m, arm, int(k['date_index']), rep)
            dt = time.time() - ta
            obj_s = float(k['restarts'][k['selected']]['objective'])
            obj_g = float(got['restarts'][got['selected']]['objective'])
            d_s, d_g = float(k['D']), float(got['D'])
            rel_obj = abs(obj_g - obj_s) / abs(obj_s) if obj_s else abs(obj_g - obj_s)
            abs_dD = abs(d_g - d_s)
            spread = float(k['restart_spread'])
            rows.append(dict(date=date, geometry=geo, N=N, arm=arm, m=m, replicate=rep,
                             date_index=int(k['date_index']), theta=float(k['theta']),
                             obj_stored=obj_s, obj_replay=obj_g, rel_obj=rel_obj,
                             D_stored=d_s, D_replay=d_g, abs_dD=abs_dD,
                             restart_spread=spread,
                             dD_over_spread=(abs_dD / spread if spread > 0 else None),
                             selected_stored=int(k['selected']),
                             selected_replay=int(got['selected']),
                             training_sha_match=(k['training_sha'] == got['training_sha']),
                             seconds_replay=dt))
            print(f"[{i+1}/{len(chosen)}] {date} {geo} N={N} {arm} m={m} rep={rep} "
                  f"rel_obj={rel_obj:.3e} abs_dD={abs_dD:.3e} spread={spread:.3e} "
                  f"Dcrit={'PASS' if abs_dD <= spread else 'FAIL'} {dt:.1f}s", flush=True)
    n = len(rows)
    obj_pass = sum(1 for r in rows if r['rel_obj'] <= OBJ_RTOL)
    d_pass = sum(1 for r in rows if r['abs_dD'] <= r['restart_spread'])
    summary = dict(
        stage='resume4_stepB_second_host_replay',
        host=socket.gethostname(), openblas_coretype=os.environ.get('OPENBLAS_CORETYPE'),
        model='deepseek-flash',
        seed=SEED, requested=n, completed=n,
        criteria=dict(objective_relative_tolerance=OBJ_RTOL,
                      D_criterion='abs_dD <= stored restart_spread'),
        objective_pass=obj_pass, D_pass=d_pass,
        pass_all=(n == N_ROWS and obj_pass == n and d_pass == n),
        max_rel_obj=max(r['rel_obj'] for r in rows),
        max_abs_dD=max(r['abs_dD'] for r in rows),
        max_dD_over_spread=max((r['dD_over_spread'] or 0.0) for r in rows),
        training_sha_match_all=all(r['training_sha_match'] for r in rows),
        wall_seconds=time.time() - t0, rows=rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(json.dumps({k: summary[k] for k in ('host', 'requested', 'completed', 'objective_pass',
                                              'D_pass', 'pass_all', 'max_rel_obj', 'max_abs_dD',
                                              'max_dD_over_spread', 'training_sha_match_all',
                                              'wall_seconds')}, indent=2))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
