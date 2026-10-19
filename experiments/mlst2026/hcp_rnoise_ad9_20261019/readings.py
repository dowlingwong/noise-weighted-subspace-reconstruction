"""Finite, complete date-level evidence is required before any affirmative reading."""
import numbers
import numpy as np
from experiments.mlst2026.meg_z_20261013.runner import interval
from .protocol import load


def _integer(value, low, high=None):
    return (isinstance(value,numbers.Integral) and not isinstance(value,(bool,np.bool_))
            and value>=low and (high is None or value<=high))


def pz1(rows,tol=1e-9):
    required=load()['signal']['theta_grid']
    if len(rows)!=len(required):
        return {'available':False,'met':False,'reason':'complete four-theta family required'}
    try:
        rows=sorted(rows,key=lambda r:r['theta'])
        theta=np.array([r['theta'] for r in rows],float)
        values=np.array([r['B_cert'] for r in rows],float)
    except (KeyError,ValueError,TypeError):
        return {'available':False,'met':False,'reason':'invalid certificates'}
    valid=(np.isfinite(theta).all() and np.isfinite(values).all()
           and len(set(theta))==4 and np.allclose(theta,required,atol=1e-12,rtol=0))
    if not valid:
        return {'available':False,'met':False,'reason':'missing, duplicate or nonfinite fixed-theta evidence'}
    zero=abs(values[0])<=tol;monotone=bool(np.all(np.diff(values)>=-tol))
    return {'available':True,'quantity':'B_dagger','values':values.tolist(),'zero_at_theta0':bool(zero),
            'nondecreasing':monotone,'met':bool(zero and monotone),'tolerance':tol}


def date_interval(records, family, *, expected_dates, qualification, resamples=50):
    """One frozen cell, exact expected date x replicate ledger; explicit R remedy."""
    from .protocol import cohort
    selection = cohort(qualification)
    dates=list(expected_dates)
    if selection['status'] != 'PASS' or dates != selection['included']:
        raise ValueError('STOP inference requires every and only gate-passing confirmation date')
    if (family != 34 or not _integer(resamples,50,50)
            or not 18<=len(dates)<=60 or len(set(dates))!=len(dates) or not records):
        raise ValueError('family34,18..60 unique dates and R50 required')
    expected={(d,r) for d in dates for r in range(resamples)};seen=set();grouped={d:[] for d in dates}
    cells=set()
    for row in records:
        try:
            d=row['date'];r=row['replicate'];v=float(row['D'])
        except (KeyError,ValueError,TypeError) as exc:
            raise ValueError('finite contrast and date/replicate identity required') from exc
        if not _integer(r,0,resamples-1) or not np.isfinite(v) or (d,r) not in expected or (d,r) in seen:
            raise ValueError('nonfinite, duplicate or unexpected date/replicate')
        seen.add((d,r));grouped[d].append(v)
        cells.add(tuple(row.get(k) for k in ('theta','N','m','arm')))
    if seen!=expected or len(cells)!=1:
        raise ValueError('incomplete date/resample ledger or mixed cells')
    means={d:float(np.mean(grouped[d])) for d in dates}
    result=interval(list(means.values()),family,load()['inference']['family_alpha'])
    if not np.isfinite([result['mean'],result['sd'],*result['ci']]).all():
        raise ValueError('nonfinite interval')
    return {**result,'available':True,'date_means':means,'expected_dates':dates,'unit':'recording date',
            'resamples':resamples,'resamples_per_date':{d:len(v) for d,v in grouped.items()}}


def _valid_interval(row):
    """Do not confuse a failed computation with a valid undecided interval."""
    try:
        if row.get('available') is not True or not _integer(row['n'],18,60):
            return False
        numeric=np.array([row['mean'],row['sd'],*row['ci']],float)
        if numeric.shape!=(4,) or not np.isfinite(numeric).all() or row['sd']<0 or row['ci'][0]>row['ci'][1]:
            return False
        R=row['resamples'];dates=row['expected_dates']
        if (not _integer(R,50,50) or len(set(dates))!=row['n'] or set(row['date_means'])!=set(dates)
                or not np.isfinite(list(row['date_means'].values())).all()
                or any(row['resamples_per_date'].get(d)!=R for d in dates)):
            return False
        expected=('undecided' if row.get('zero_SE') else 'negative' if row['ci'][1]<0
                  else 'positive' if row['ci'][0]>0 else 'undecided')
        return row['decision']==expected
    except (KeyError,TypeError,ValueError):
        return False


def _family(rows, expected_cells):
    if expected_cells is None:
        return False
    expected=list(expected_cells)
    if not expected or len(set(expected))!=len(expected) or len(rows)!=len(expected):
        return False
    cells=[r.get('cell') for r in rows]
    return len(set(cells))==len(cells) and set(cells)==set(expected) and all(_valid_interval(r) for r in rows)


def pz2(below,above,in_range):
    valid=bool(in_range and _valid_interval(below) and _valid_interval(above))
    return {'applicable':bool(in_range),'available':valid,
            'met':bool(valid and below['decision']=='negative' and above['decision']=='positive')}


def pz3(reference_theta0, expected_cells=None):
    valid=_family(reference_theta0,expected_cells)
    return {'available':valid,'met':bool(valid and all(r['decision']!='positive' for r in reference_theta0))}


def pz4(low,high):
    valid=(isinstance(low,dict) and isinstance(high,dict) and set(low)==set(high)=={'lw','kron'}
           and all(_valid_interval(v) for v in list(low.values())+list(high.values())))
    return {'available':valid,'met':bool(valid and all(low[e]['decision']=='negative'
            and abs(high[e]['mean'])<abs(low[e]['mean']) for e in ('lw','kron')))}


def pz5(isotropic, expected_cells=None):
    valid=_family(isotropic,expected_cells)
    return {'available':valid,'met':bool(valid and all(r['decision']!='negative' for r in isotropic))}
