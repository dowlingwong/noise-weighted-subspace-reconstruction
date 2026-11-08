"""Prespecified resume-3 per-arm compute budget; no clean-risk computation."""
import json
from collections import defaultdict
from pathlib import Path
from .config import ROOT

def main():
    w=Path(ROOT)
    timing=json.loads((w/'out/resume3/timing_full.json').read_text())
    assert timing['n_rows']==20 and timing['seed']==20261106
    arms=defaultdict(list); settings=defaultdict(list)
    for row in timing['rows']:
        arms[row['arm']].append(row['t_row'])
        settings[f"{row['arm']}/m{row['m']}"].append(row['t_row'])
    assert set(arms)=={'reference','isotropic','lw','kron'}
    def stats(v):return dict(n=len(v),mean_seconds=sum(v)/len(v),max_seconds=max(v))
    arm_stats={k:stats(v) for k,v in sorted(arms.items())}
    scopes={'S_primary':dict(reference=22500,isotropic=22500,lw=4500,kron=4500),
            'S_core':dict(reference=22500,isotropic=4500,lw=2250,kron=2250)}
    results={}
    for name,counts in scopes.items():
        serial=sum(arm_stats[a]['mean_seconds']*n for a,n in counts.items())
        max_serial=sum(arm_stats[a]['max_seconds']*n for a,n in counts.items())
        results[name]=dict(rows=sum(counts.values()),arm_counts=counts,processes={str(p):dict(
            point_seconds=serial/p+3600,point_hours=(serial/p+3600)/3600,
            per_arm_max_row_hours=(max_serial/p+3600)/3600,
            global_max_row_hours=(timing['max_row_seconds']*sum(counts.values())/p+3600)/3600,
            point_pass_40h=(serial/p+3600)<=40*3600) for p in (8,16)})
    out=dict(model='gpt-6.1-sol',timing_rows=20,seed=20261106,arm_statistics=arm_stats,
             arm_m_statistics={k:stats(v) for k,v in sorted(settings.items())},
             overhead_allowance_seconds=3600,overhead_is_measured=False,
             overhead_authorization=str(w/'records/resume3/OVERHEAD_AUTHORIZATION.md'),
             estimator='per-arm arithmetic mean * scoped row count / P + fixed 1h allowance',
             caveat='max-row extrapolations are finite-sample estimates, not guarantees or confidence bounds',
             scopes=results)
    path=w/'out/resume3/scope_estimates.json'
    path.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,sort_keys=True,indent=2),flush=True)

if __name__=='__main__':main()
