"""Summarise the TR09 development Monte Carlo cells (python -m ...tr09_first_order.analyse RUN_DIR OUT.json)."""
import json, glob, sys
import numpy as np
cells = {}
for f in glob.glob(sys.argv[1] + '/cell_*.json'):
    r = json.load(open(f)); cells[(r['theta'], r['N'], r['p'])] = r
out = []
def ms(x): x = np.asarray(x, float); return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
for th in sorted({k[0] for k in cells}):
    for N in sorted({k[1] for k in cells}):
        base = {row['rep']: row for row in cells[(th, N, 1.0)]['rows']}
        for p in sorted({k[2] for k in cells}, reverse=True):
            c = cells[(th, N, p)]; rows = c['rows']; pr = c['prediction']
            dU = [r['Delta_U'] for r in rows]; dF = [r['Delta_F'] for r in rows]; q = [r['q_F'] for r in rows]; D = [r['D'] for r in rows]
            g = [r for r in rows if 'global_differs' in r]
            rec = {'theta': th, 'N': N, 'p': p, 'reps': len(rows),
                   'N_mean_Delta_U': [N * v for v in ms(dU)], 'pred_cU': pr['cU'],
                   'N_mean_Delta_F': [N * v for v in ms(dF)], 'N_mean_qF': [N * v for v in ms(q)], 'pred_cF_plus_K': pr['cF_gaussian'] + pr['K'],
                   'mean_D': ms(D), 'pred_D_leading': pr['a'] + pr['B'] / N, 'a': pr['a'],
                   'global_differs_fraction': float(np.mean([r['global_differs'] for r in g])) if g else None,
                   'global_minus_local_RF': ms([r['R_F_global'] - r['R_F'] for r in g]) if g else None,
                   'local_nonstationary': int(sum(r['local_status'] != 'stationary' for r in rows))}
            if p != 1.0:
                pd = lambda key: ms([r[key] - base[r['rep']][key] for r in rows])
                rec['paired_minus_p1'] = {'Delta_U': pd('Delta_U'), 'Delta_F': pd('Delta_F'), 'D': pd('D'),
                                          'pred_Delta_F_shift': pr['K'] / N}
            out.append(rec)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for r in out:
    s = f"th={r['theta']:.2f} N={r['N']:5d} p={r['p']:.1f} | N*dU={r['N_mean_Delta_U'][0]:7.1f}±{r['N_mean_Delta_U'][1]:5.1f} (cU {r['pred_cU']:.0f}) | N*dF={r['N_mean_Delta_F'][0]:6.1f}±{r['N_mean_Delta_F'][1]:4.1f} N*q={r['N_mean_qF'][0]:6.1f}±{r['N_mean_qF'][1]:4.1f} (pred {r['pred_cF_plus_K']:.1f}) | D={r['mean_D'][0]:6.3f}±{r['mean_D'][1]:.3f} pred {r['pred_D_leading']:6.3f} | glob {r['global_differs_fraction']}"
    if 'paired_minus_p1' in r:
        pp = r['paired_minus_p1']; s += f" | dU-p1={pp['Delta_U'][0]:+.3f}±{pp['Delta_U'][1]:.3f} dF-p1={pp['Delta_F'][0]:+.4f}±{pp['Delta_F'][1]:.4f} (pred {pp['pred_Delta_F_shift']:+.4f}) dD-p1={pp['D'][0]:+.3f}±{pp['D'][1]:.3f}"
    print(s)
