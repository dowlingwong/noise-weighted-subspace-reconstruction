"""Stage 0c replay driver (carried in the frozen package): reproduce the stored training
objective and D for 60 rows, to certify that the frozen learner can be rerun.

Run from EXP with PYTHONPATH=$EXP:$EXP/src and OPENBLAS_CORETYPE=Haswell:
  python -m experiments.mlst2026.hcp_xopt_20261106.replay 60 [out.json]
"""
import json, sys, time
import numpy as np

from .config import ROOT
W = ROOT
IN = W + '/inputs/hcp_confirmation_20261020'
sys.path.insert(0, W + '/EXP')
sys.path.insert(1, W + '/EXP/src')

from experiments.mlst2026.hcp_rnoise_20261019.learning import fit_cell
from experiments.mlst2026.hcp_rnoise_20261019.signal import build

SEED = 20261106
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
ONLY_DATE = sys.argv[2] if len(sys.argv) > 2 else None
OUT = sys.argv[3] if len(sys.argv) > 3 else (W + '/out/0c_replay.json')


def scan(path):
    recs = []
    off = 0
    with open(path, 'rb') as f:
        for line in f:
            r = json.loads(line)
            k = r['kernel']
            recs.append(dict(date=r['date'], geometry=r['geometry'], N=r['N'], arm=r['arm'], m=r['m'],
                             replicate=r['replicate'], lim=bool(k['optimization_limited']),
                             lim_t25=bool(k['optimization_limited_t25_rule']), theta=k['theta'],
                             selected=k['selected'], obj=k['restarts'][k['selected']]['objective'],
                             D=r['D'], offset=off, nbytes=len(line)))
            off += len(line)
    return recs


def main():
    t_start = time.time()
    recs = scan(IN + '/FITS.jsonl')
    bydate = {}
    for rec in recs:
        bydate.setdefault(rec['date'], []).append(rec)
    dates = sorted(bydate)
    rng = np.random.default_rng(SEED)
    chosen = []
    for d in dates:
        if ONLY_DATE and d != ONLY_DATE:
            continue
        block = bydate[d]
        picks = []
        for pool in ([r for r in block if r['lim']], [r for r in block if not r['lim']]):
            if pool:
                picks.append(pool[int(rng.integers(len(pool)))])
        while len(picks) < 2:
            rest = [r for r in block if r not in picks]
            picks.append(rest[int(rng.integers(len(rest)))])
        chosen.extend(picks[:2])
    chosen = chosen[:N_ROWS]
    print(f"scanned {len(recs)} rows, {len(dates)} dates, selected {len(chosen)} rows, seed={SEED}", flush=True)

    arrays_cache, model_cache = {}, {}
    results = []
    with open(IN + '/FITS.jsonl', 'rb') as f:
        for i, rec in enumerate(chosen):
            f.seek(rec['offset'])
            row = json.loads(f.readline())
            k = row['kernel']
            date, geo, N, arm, m, rep = row['date'], row['geometry'], row['N'], row['arm'], row['m'], row['replicate']
            if date not in arrays_cache:
                roles = {}
                for role in ('ref', 'train', 'cal', 'eval'):
                    with np.load(f'{IN}/arrays/{date}/{role}.npz', allow_pickle=False) as a:
                        roles[role] = (a['X'], {kk: a[kk] for kk in a.files if kk != 'X'})
                S = np.asarray(json.loads(open(f'{IN}/arrays/{date}/Sref.json', 'rb').read()), float)
                arrays_cache[date] = (roles, S)
            roles, S = arrays_cache[date]
            key = (date, float(k['theta']))
            if key not in model_cache:
                model_cache[key] = build(S, float(k['theta']))
            model = model_cache[key]
            t0 = time.time()
            got = fit_cell(roles, model, S, N, m, arm, int(k['date_index']), rep)
            dt = time.time() - t0
            obj_s = float(k['restarts'][k['selected']]['objective'])
            obj_g = float(got['restarts'][got['selected']]['objective'])
            d_s, d_g = float(k['D']), float(got['D'])
            rel_obj = abs(obj_g - obj_s) / abs(obj_s) if obj_s else abs(obj_g - obj_s)
            rel_D = abs(d_g - d_s) / abs(d_s) if d_s else abs(d_g - d_s)
            results.append(dict(
                date=date, geometry=geo, N=N, arm=arm, m=m, replicate=rep, date_index=int(k['date_index']),
                lim_orig=rec['lim'], lim_t25_orig=rec['lim_t25'],
                obj_stored=obj_s, obj_replay=obj_g, rel_obj=rel_obj,
                D_stored=d_s, D_replay=d_g, rel_D=rel_D,
                selected_stored=int(k['selected']), selected_replay=int(got['selected']),
                status_stored=k['selected_status'], status_replay=got['selected_status'],
                training_sha_match=(k['training_sha'] == got['training_sha']),
                seconds_replay=dt))
            print(f"[{i+1}/{len(chosen)}] {date} {geo} N={N} {arm} m={m} rep={rep} "
                  f"rel_obj={rel_obj:.2e} rel_D={rel_D:.2e} sel={got['selected']} status={got['selected_status']} "
                  f"sha={k['training_sha']==got['training_sha']} {dt:.2f}s", flush=True)

    ok_obj = [r for r in results if r['rel_obj'] <= 1e-9]
    ok_D = [r for r in results if r['rel_D'] <= 1e-9]
    ok_sha = [r for r in results if r['training_sha_match']]
    summary = dict(stage='0c_replay', seed=SEED, selected=len(results), scan_rows=len(recs),
                   dates=len(dates), ok_objective=len(ok_obj), ok_D=len(ok_D), ok_training_sha=len(ok_sha),
                   max_rel_obj=max(r['rel_obj'] for r in results), max_rel_D=max(r['rel_D'] for r in results),
                   selected_index_all_match=all(r['selected_stored'] == r['selected_replay'] for r in results),
                   status_all_match=all(r['status_stored'] == r['status_replay'] for r in results),
                   wall_seconds=time.time() - t_start, rows=results)
    with open(OUT, 'w') as h:
        json.dump(summary, h, indent=2, sort_keys=True)
        h.write('\n')
    print(json.dumps({kk: summary[kk] for kk in ('stage','selected','ok_objective','ok_D','ok_training_sha',
          'max_rel_obj','max_rel_D','selected_index_all_match','status_all_match','wall_seconds')}, indent=2))
    print("wrote", OUT)


if __name__ == '__main__':
    main()
