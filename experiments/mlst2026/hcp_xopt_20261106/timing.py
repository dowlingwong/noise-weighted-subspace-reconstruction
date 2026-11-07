"""Stage 0d timing run: optimizer cost only.  Computes no clean risk at all (no Evaluation object)."""
from __future__ import annotations

import json
import time

import numpy as np

from . import kernel, select
from experiments.mlst2026.hcp_rnoise_20261019.signal import build
from experiments.mlst2026.t25_production_20260930 import science as t25


def main(n_rows=20, seed=20261106):
    fits = select.IN_DEFAULT + '/FITS.jsonl'
    metas = select.pick_rows(fits, seed=seed, per_date=2, max_rows=n_rows)
    rows = []
    t_all = time.time()
    for i, meta in enumerate(metas):
        roles, Sref = select.load_inputs(meta['date'])
        model = build(Sref, float(meta['theta']))
        t0 = time.time()
        asm = kernel.assemble(roles, model, Sref, meta['N'], meta['m'], meta['arm'],
                              int(meta['date_index']), meta['replicate'])
        t_asm = time.time() - t0
        t0 = time.time()
        B0, diag0, bases0 = t25.structured(asm['W'], asm['S'], asm['initial'],
                                           kernel.S0_MAX_ITER, kernel.S0_TOL)
        t_s0 = time.time() - t0
        cap_s = dict(W=asm['W'], S=asm['S'], initial=asm['initial'], B=B0, diag=diag0, bases=bases0)
        t0 = time.time()
        out, _, _, _ = kernel.strengthened(cap_s, None, model.C, model.T, asm['key'])
        out['route_seconds']['S0'] = t_s0
        t_routes = time.time() - t0
        total = t_asm + t_s0 + t_routes
        rows.append(dict(i=i, date=meta['date'], geometry=meta['geometry'], N=meta['N'],
                         arm=meta['arm'], m=meta['m'], replicate=meta['replicate'],
                         date_index=meta['date_index'], theta=meta['theta'],
                         t_assemble=t_asm, t_S0=t_s0, t_routes=t_routes, t_row=total,
                         route_seconds=out['route_seconds'],
                         S0_status=out['S0']['status'], S0_iters=out['S0']['iterations'],
                         S1_status=out['S1']['status'], S1_iters=out['S1']['iterations'],
                         S2_statuses=sorted({c['status'] for c in out['candidates'] if c['route'].startswith('S2')}),
                         S2_iters=[c['iterations'] for c in out['candidates'] if c['route'].startswith('S2')],
                         S3_status=out['S3']['status'], S3_iters=out['S3']['iterations'],
                         selected_route=out['selected_route'], n_tied=out['n_tied'],
                         lim_t25=meta['lim_t25'], lim=meta['lim']))
        print(f"[{i+1}/{len(metas)}] {meta['date']} {meta['geometry']:7s} N={meta['N']:3d} {meta['arm']:9s} "
              f"m={str(meta['m']):4s} rep={meta['replicate']:2d} asm={t_asm:.2f} S0={t_s0:.2f} "
              f"routes={t_routes:.2f} total={total:.2f}s win={out['selected_route']}", flush=True)
    times = np.array([r['t_row'] for r in rows])
    summary = dict(stage='0d_timing', n_rows=len(rows), seed=seed,
                   mean_row_seconds=float(times.mean()), median_row_seconds=float(np.median(times)),
                   max_row_seconds=float(times.max()), total_seconds=time.time() - t_all,
                   route_seconds_sum={k: float(sum(r['route_seconds'].get(k, 0.) for r in rows))
                                      for k in ('S0', 'S1', 'S2', 'S3')},
                   rows=rows)
    print(json.dumps({k: summary[k] for k in ('stage', 'n_rows', 'mean_row_seconds',
                                              'median_row_seconds', 'max_row_seconds', 'route_seconds_sum')},
                     indent=2))
    with open(select.ROOT + "/out/0d_timing.json", 'w') as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print('wrote /ceph/dwong/hcp_xopt_20261106/out/0d_timing.json')


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
