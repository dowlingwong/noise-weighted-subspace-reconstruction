"""Prespecified SD-trigger and planning-power decision, never using benchmark means."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import t,nct,chi2

def power(mean,sd,n):
    cutoff=t.ppf(1-.05/8,n-1)
    noncentrality=abs(mean)*np.sqrt(n)/sd
    return float(nct.sf(cutoff,n-1,noncentrality)+nct.cdf(-cutoff,n-1,noncentrality))
def decide(summary,dev2):
    rows=[];trigger=False;needed=120
    for row in summary['families']['primary']:
        key=f"{row['theta']}|{row['N']}|0.1|D";ref=dev2[key]
        sd=row['sd'];n=row['seeds'];target=power(ref['mean'],ref['sd'],120)
        over=sd>1.25*ref['sd'];trigger|=over
        df=n-1;interval=[float(sd*np.sqrt(df/chi2.ppf(.975,df))),float(sd*np.sqrt(df/chi2.ppf(.025,df)))]
        candidate=next((m for m in range(120,1001) if power(ref['mean'],sd,m)>=target),None)
        rows.append(dict(theta=row['theta'],N=row['N'],benchmark_sd=sd,sd_planning_interval=interval,
                         dev2_mean=ref['mean'],dev2_sd=ref['sd'],trigger_threshold=1.25*ref['sd'],trigger=over,
                         original_n120_power=target,new_n120_power=power(ref['mean'],sd,120),required_n=candidate))
    if trigger:
        if any(r['required_n'] is None for r in rows):needed=None
        else:needed=max(r['required_n'] for r in rows)
    return dict(trigger=bool(trigger),confirmation_seed_count=needed,launch_blocked=needed is None,
                primary=rows,rule='point SD >1.25*exact dev2 SD; if triggered retain each original n120 Bonferroni-4 power using dev2 mean and T25 SD; cap1000',
                uncertainty='SD chi-square interval is planning-only under Gaussian assumption; non-Gaussian differences may invalidate it',
                confirmation_draws=0,development_evidence_only=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',required=True);ap.add_argument('--dev2',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=decide(json.loads(Path(a.summary).read_text()),json.loads(Path(a.dev2).read_text()))
    Path(a.out).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
