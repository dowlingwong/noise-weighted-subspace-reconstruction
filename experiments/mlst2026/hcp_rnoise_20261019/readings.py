"""PZ criteria and between-date intervals. No window/block pseudoreplication."""
import numpy as np
from experiments.mlst2026.meg_z_20261013.runner import interval
from .protocol import load


def pz1(rows,tol=1e-9):
    rows=sorted(rows,key=lambda r:r['theta']);values=[r.get('B_cert') for r in rows]
    available=bool(rows) and all(v is not None for v in values)
    zero=available and rows[0]['theta']==0 and abs(values[0])<=tol
    monotone=available and bool(np.all(np.diff(values)>=-tol))
    return {'quantity':'B_dagger','values':values,'zero_at_theta0':bool(zero),'nondecreasing':bool(monotone),
            'met':bool(zero and monotone),'tolerance':tol}


def date_interval(records, family):
    if family<1 or not records:
        raise ValueError('registered positive family and records required')
    grouped={}
    for row in records:
        grouped.setdefault(row['date'],[]).append(float(row['D']))
    means={d:float(np.mean(v)) for d,v in sorted(grouped.items())}
    return {**interval(list(means.values()),family,load()['inference']['family_alpha']),
            'date_means':means,'unit':'recording date','resamples_per_date':{d:len(v) for d,v in grouped.items()}}


def pz2(below,above,in_range):
    return {'applicable':bool(in_range),'met':bool(in_range and below['decision']=='negative' and above['decision']=='positive')}


def pz3(reference_theta0):
    return {'met':bool(reference_theta0) and all(r['decision']!='positive' for r in reference_theta0)}


def pz4(low,high):
    # Inputs are paired estimated-minus-reference interaction intervals.
    return {'met':len(low)==len(high)==2 and all(lo['decision']=='negative' and abs(hi['mean'])<abs(lo['mean']) for lo,hi in zip(low,high))}


def pz5(isotropic):
    return {'met':bool(isotropic) and all(r['decision']!='negative' for r in isotropic)}
