"""Resume-4 Step C: build the three scope row lists, hash them, and compute the ladder.

Estimator (fixed before the strengthened result exists):
  point_seconds(scope) = sum_arm n_arm * mean_arm / (P_dt + P_dt2/rho) + 3600 s
  mean_arm  = resume-3 per-arm single-process row-mean seconds (deepthought)
  rho       = loaded/unloaded per-row time on deepthought2, measured in the Step C
              concurrency test of the same rows (same host, so host arithmetic cancels)
  P_dt = 16, P_dt2 = 64, P_total = 80.
The dt2 share therefore uses the measured loaded per-row time; assuming equal per-core
cost to deepthought is the conservative branch, since deepthought2 was measured faster.

  python -m experiments.mlst2026.hcp_xopt_20261106.resume4_scope <outdir>
"""
import hashlib, json, os, socket, sys
import numpy as np

from . import scopes, select

W = select.ROOT
FITS = select.IN_DEFAULT + '/FITS.jsonl'
TIMING = W + '/out/resume3/timing_full.json'
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else W + '/out/resume4'

# Rows per arm at 30 dates x 15 cells x 50 replicates: reference and isotropic have one
# m setting (22,500 rows each); lw and kron each have m=256 and m=512 (45,000 rows each).
COUNTS = {
    'S_full':    {'reference': 22500, 'isotropic': 22500, 'lw': 45000, 'kron': 45000},
    'S_primary': {'reference': 22500, 'isotropic': 22500, 'lw': 4500,  'kron': 4500},
    'S_core':    {'reference': 22500, 'isotropic': 4500,  'lw': 2250,  'kron': 2250},
}
P_DT = 16
P_DT2 = 80
OVERHEAD = 3600.0
FIELDS = ('offset', 'date', 'geometry', 'N', 'arm', 'm', 'replicate', 'date_index', 'theta')


def all_rows():
    out = []
    for row in select.scan(FITS):
        out.append({k: row[k] for k in FIELDS})
    return out


def build(scope):
    rows = all_rows() if scope == 'S_full' else scopes.rows_for_scope(scope, FITS)
    rows.sort(key=lambda r: r['offset'])
    blob = ''.join(json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n' for r in rows).encode()
    path = os.path.join(OUTDIR, 'scopes', f'{scope}_rows.jsonl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(blob)
    ids = {(r['date'], r['replicate'], r['geometry'], r['N'], r['arm'], r['m']) for r in rows}
    arm_counts = {a: sum(1 for r in rows if r['arm'] == a) for a in COUNTS[scope]}
    return dict(scope=scope, rows=len(rows), unique=len(ids), path=path,
                sha256=hashlib.sha256(blob).hexdigest(), arm_counts=arm_counts,
                rows_ok=(len(rows) == sum(COUNTS[scope].values())) and len(ids) == len(rows))


def arm_means(timing):
    """Per-arm arithmetic mean row seconds from the 20 timing rows, pooling m within arm."""
    rows = timing['rows']
    out = {}
    for a in ('reference', 'isotropic', 'lw', 'kron'):
        v = [float(r['t_row']) for r in rows if r['arm'] == a]
        if not v:
            raise ValueError(f'no timing rows for arm {a}')
        out[a] = dict(n=len(v), mean_seconds=float(np.mean(v)), max_seconds=float(max(v)))
    return out


def ladder(timing, rho):
    means = arm_means(timing)
    res = {}
    for scope, counts in COUNTS.items():
        num = sum(counts[a] * means[a]['mean_seconds'] for a in counts)
        point = num / (P_DT + P_DT2 / rho) + OVERHEAD
        res[scope] = dict(rows=sum(counts.values()), arm_counts=counts,
                          numerator_seconds=num, point_hours=point / 3600.0,
                          passes_48h=point / 3600.0 <= 48.0)
    return res


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    timing = json.load(open(TIMING))
    quant = json.load(open(W + '/out/resume4/concurrency_loaded_p80.json'))
    base = json.load(open(W + '/out/resume4/concurrency_unloaded.json'))
    rho = quant['t_row_mean'] / base['t_row_mean']
    rows_info = {s: build(s) for s in COUNTS}
    est = ladder(timing, rho)
    chosen = None
    for s in ('S_full', 'S_primary', 'S_core'):
        if est[s]['passes_48h']:
            chosen = s
            break
    out = dict(stage='resume4_stepC_scope', model='deepseek-flash',
               host=socket.gethostname(), scope_seed=scopes.SEED,
               process_counts=dict(P_dt=P_DT, P_dt2=P_DT2, P_total=P_DT + P_DT2),
               memory=dict(peak_rss_kb_per_worker=quant['peak_rss_kb_max'],
                           n_workers=P_DT2,
                           required_gib=1.25 * P_DT2 * quant['peak_rss_kb_max'] / 1024 ** 2,
                           available_gib=float(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 1024 ** 2),
               concurrency=dict(workers=quant['workers'], wall_seconds=quant['wall_seconds'],
                                unloaded_t_row_mean=base['t_row_mean'],
                                loaded_t_row_mean=quant['t_row_mean'],
                                loaded_t_row_max=quant['t_row_max'],
                                inflation_rho=rho,
                                peak_rss_kb_max=quant['peak_rss_kb_max']),
               arm_means_seconds=arm_means(timing),
               overhead_allowance_seconds=OVERHEAD,
               estimator='sum_arm n_arm*mean_arm/(P_dt + P_dt2/rho) + 3600',
               scopes=rows_info, estimates=est, chosen_rung=chosen,
               threshold_hours=48.0,
               graceful_stop_hours=(1.5 * est[chosen]['point_hours'] if chosen else None))
    path = os.path.join(OUTDIR, 'scope_estimates.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
