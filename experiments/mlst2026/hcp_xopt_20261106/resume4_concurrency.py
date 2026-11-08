"""Resume-4 Step C concurrency/resource test on deepthought2 (no clean risk; output discarded).

Each child process runs the 2 timing rows through the *timing* code path only
(assemble + S0 + strengthened routes; no Evaluation object, no clean risk), records
per-row seconds and the child's peak RSS, then exits.  The parent starts P children
simultaneously, exactly the concurrency the production share will see.

  python -m experiments.mlst2026.hcp_xopt_20261106.resume4_concurrency \
      --workers 64 --out <summary.json>
  python -m ...resume4_concurrency --child <i> --outdir <dir>          (internal)
"""
from __future__ import annotations
import argparse, json, os, resource, socket, subprocess, sys, time
import numpy as np

from . import kernel, select
from experiments.mlst2026.hcp_rnoise_20261019.signal import build
from experiments.mlst2026.t25_production_20260930 import science as t25


def timing_rows(n=2):
    return select.pick_rows(select.IN_DEFAULT + '/FITS.jsonl', seed=20261106,
                            per_date=2, max_rows=n)


def run_rows(metas):
    out = []
    cache = {}
    for meta in metas:
        if meta['date'] not in cache:
            cache[meta['date']] = select.load_inputs(meta['date'])
        roles, Sref = cache[meta['date']]
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
        res, _, _, _ = kernel.strengthened(cap_s, None, model.C, model.T, asm['key'])
        t_routes = time.time() - t0
        t0 = time.time()
        asm2 = kernel.assemble(roles, model, Sref, meta['N'], meta['m'], meta['arm'],
                               int(meta['date_index']), meta['replicate'])
        t_asm2 = time.time() - t0
        out.append(dict(date=meta['date'], geometry=meta['geometry'], N=meta['N'], arm=meta['arm'],
                        m=meta['m'], replicate=meta['replicate'], date_index=meta['date_index'],
                        theta=meta['theta'], t_assemble=t_asm, t_S0=t_s0, t_routes=t_routes,
                        t_assemble_check=t_asm2,
                        t_row=t_asm + t_s0 + t_routes + t_asm2,
                        selected_route=res['selected_route'], n_tied=res['n_tied'],
                        route_seconds=res['route_seconds']))
    return out


def child(i, outdir):
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rows = run_rows(timing_rows(2))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec = dict(child=i, host=socket.gethostname(), pid=os.getpid(), peak_rss_kb=int(rss),
               rss_before_kb=int(rss0), rows=rows,
               openblas_coretype=os.environ.get('OPENBLAS_CORETYPE'),
               start=time.time())
    with open(os.path.join(outdir, f'child_{i:03d}.json'), 'w') as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
        fh.write('\n')


def parent(args):
    os.makedirs(args.outdir, exist_ok=True)
    for f in os.listdir(args.outdir):
        if f.startswith('child_'):
            os.unlink(os.path.join(args.outdir, f))
    env = dict(os.environ, OPENBLAS_CORETYPE=args.kernel,
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    cmd = [sys.executable, '-m', 'experiments.mlst2026.hcp_xopt_20261106.resume4_concurrency',
           '--child', '{i}', '--outdir', args.outdir, '--kernel', args.kernel]
    t0 = time.time()
    procs = []
    for i in range(args.workers):
        procs.append(subprocess.Popen([c.format(i=i) for c in cmd], env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                      preexec_fn=lambda: os.nice(10)))
    rcs = [p.wait() for p in procs]
    wall = time.time() - t0
    recs = []
    for i in range(args.workers):
        p = os.path.join(args.outdir, f'child_{i:03d}.json')
        if os.path.exists(p):
            recs.append(json.load(open(p)))
    rows = [r for rec in recs for r in rec['rows']]
    per_row = {}
    for r in rows:
        per_row.setdefault((r['date'], r['N'], r['arm'], r['m'], r['replicate']), []).append(r['t_row'])
    summary = dict(stage='resume4_stepC_concurrency', host=socket.gethostname(),
                   model='deepseek-flash', workers=args.workers,
                   kernel=args.kernel, children_ok=int(sum(1 for rc in rcs if rc == 0)),
                   children_rc=rcs, wall_seconds=wall,
                   peak_rss_kb_max=int(max(r['peak_rss_kb'] for r in recs)),
                   peak_rss_kb_median=float(np.median([r['peak_rss_kb'] for r in recs])),
                   n_rows=int(len(rows)),
                   rows_per_child=len(timing_rows(2)),
                   per_identity_times={f'{k[0]}|N{k[1]}|{k[2]}|m{k[3]}|r{k[4]}':
                                       dict(n=len(v), mean=v_mean, min=min(v), max=max(v))
                                       for k, v in per_row.items() for v_mean in [float(np.mean(v))]},
                   t_row_mean=float(np.mean([r['t_row'] for r in rows])),
                   t_row_median=float(np.median([r['t_row'] for r in rows])),
                   t_row_max=float(max(r['t_row'] for r in rows)),
                   slots=args.workers * len(timing_rows(2)),
                   seconds_per_slot=wall / (args.workers * len(timing_rows(2))))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(json.dumps({k: summary[k] for k in ('host', 'workers', 'children_ok', 'wall_seconds',
                                              'peak_rss_kb_max', 'peak_rss_kb_median', 'n_rows',
                                              't_row_mean', 't_row_median', 't_row_max',
                                              'slots', 'seconds_per_slot')}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--out', default=None)
    ap.add_argument('--outdir', default=select.ROOT + '/out/resume4/concurrency_children')
    ap.add_argument('--kernel', default='Haswell')
    ap.add_argument('--child', type=int, default=None)
    args = ap.parse_args()
    if args.child is not None:
        child(args.child, args.outdir)
    else:
        parent(args)


if __name__ == '__main__':
    main()
