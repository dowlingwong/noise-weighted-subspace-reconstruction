"""Finite, complete date-level evidence is required before any affirmative reading."""
import numbers
import hashlib
import json
import numpy as np
from experiments.mlst2026.meg_z_20261013.runner import interval
from .protocol import load
from .prediction import GRIDS

PZ3_CELLS = [f'control/{i}/reference' for i in range(5)]
REFERENCE_CELLS = [f'{g}/{i}/reference' for g in GRIDS for i in range(5)]
PZ5_CELLS = [f'{g}/{i}/isotropic' for g in GRIDS for i in range(5)]
PAIRED_CELLS = [f'{g}/{i}' for g in ('A', 'B') for i in range(5)]


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
    g,N,m,arm = next(iter(cells))
    if (g not in GRIDS or N not in GRIDS[g] or
            (arm,m) not in (('reference',None),('isotropic',None),('lw',256),('lw',512),('kron',256),('kron',512))):
        raise ValueError('fixed geometry/grid/six-arm identity required')
    key = arm if m is None else f'{arm}_m{m}'
    return {**result,'available':True,'date_means':means,'expected_dates':dates,'unit':'recording date',
            'cell':f'{g}/{GRIDS[g].index(N)}/{key}', 'family_size':34, 'alpha':.05,
            'qualification':json.loads(json.dumps(qualification)),
            'resamples':resamples,'resamples_per_date':{d:len(v) for d,v in grouped.items()}}


def _valid_interval(row):
    """Do not confuse a failed computation with a valid undecided interval."""
    try:
        if row.get('available') is not True or not _integer(row['n'],18,60) or row.get('family_size') != 34 or row.get('alpha') != .05:
            return False
        numeric=np.array([row['mean'],row['sd'],*row['ci']],float)
        if numeric.shape!=(4,) or not np.isfinite(numeric).all() or row['sd']<0 or row['ci'][0]>row['ci'][1]:
            return False
        R=row['resamples'];dates=row['expected_dates']
        from .protocol import cohort
        selection=cohort(row['qualification'])
        if selection['status'] != 'PASS' or selection['included'] != dates:
            return False
        if (not _integer(R,50,50) or len(set(dates))!=row['n'] or set(row['date_means'])!=set(dates)
                or not np.isfinite(list(row['date_means'].values())).all()
                or any(row['resamples_per_date'].get(d)!=R for d in dates)):
            return False
        expected=('undecided' if row.get('zero_SE') else 'negative' if row['ci'][1]<0
                  else 'positive' if row['ci'][0]>0 else 'undecided')
        recomputed = interval([row['date_means'][d] for d in dates],34,.05)
        if any(not np.allclose(row[k],recomputed[k],atol=1e-12,rtol=1e-12) for k in ('mean','sd','ci','half_width')):
            return False
        return row['decision']==expected and row['zero_SE']==recomputed['zero_SE']
    except (KeyError,TypeError,ValueError):
        return False


def _same_cohort(rows):
    return bool(rows and all(r['expected_dates']==rows[0]['expected_dates'] for r in rows))


def _family(rows, expected_cells):
    if expected_cells is None:
        return False
    expected=list(expected_cells)
    if not expected or len(set(expected))!=len(expected) or len(rows)!=len(expected):
        return False
    cells=[r.get('cell') for r in rows]
    return (len(set(cells))==len(cells) and set(cells)==set(expected)
            and all(_valid_interval(r) for r in rows) and _same_cohort(rows))


def pz2(reference_rows, *, predictions, qualification):
    """Derive every A/B sign claim from all included per-date certificates.

    There is no caller-supplied in_range flag or favorable-pair selection. Even
    certificate-boundary/interior cells are retained in the complete family,
    while carrying no sign requirement. Theta0 is never a PZ2 geometry.
    """
    from .protocol import cohort
    from .prediction import validate_record
    unavailable={'applicable':False,'available':False,'met':False,
                 'reading':'UNAVAILABLE','geometries':{}}
    try:
        selection=cohort(qualification)
        dates=selection['included']
        if (selection['status'] != 'PASS' or not _family(reference_rows,REFERENCE_CELLS)
                or reference_rows[0]['expected_dates'] != dates or set(predictions) != set(dates)):
            return {**unavailable,'reason':'complete common-cohort reference family and prediction records required'}
        for d in dates:
            validate_record(predictions[d],d)
        rows={r['cell']:r for r in reference_rows}
        result={}
        for g in ('A','B'):
            geos=[predictions[d]['geometries'][g] for d in dates]
            valid=all(geo['available'] and geo['row']['PZ2']['available'] for geo in geos)
            in_range=valid and all(geo['row']['PZ2']['in_range'] for geo in geos)
            below=[];above=[]
            if valid:
                intervals=[geo['row']['N_star_interval'] for geo in geos]
                below=[f'{g}/{i}/reference' for i,N in enumerate(GRIDS[g])
                       if all(N<bounds[0] for bounds in intervals)]
                above=[f'{g}/{i}/reference' for i,N in enumerate(GRIDS[g])
                       if all(N>bounds[1] for bounds in intervals)]
            applicable=bool(in_range and below and above)
            met=bool(applicable and all(rows[c]['decision']=='negative' for c in below)
                     and all(rows[c]['decision']=='positive' for c in above))
            result[g]={'applicable':applicable,'available':bool(valid),'below':below,'above':above,
                'met':met,'reason':None if applicable else 'unavailable or out-of-grid crossing/certificate in an included date; no PZ2 claim'}
        applicable=[r for r in result.values() if r['applicable']]
        met=bool(applicable and all(r['met'] for r in result.values()))
        return {'available':True,'applicable':bool(applicable),'met':met,'geometries':result,
                'reading':'NOT APPLICABLE' if not applicable else 'MET' if met else 'NOT MET',
                'reason':None,'expected_dates':dates}
    except (PermissionError,KeyError,TypeError,ValueError):
        return {**unavailable,'reason':'invalid qualification, interval or per-date prediction evidence'}


def pz3(reference_theta0, expected_cells=None):
    valid=(expected_cells is None or set(expected_cells)==set(PZ3_CELLS)) and _family(reference_theta0,PZ3_CELLS)
    return {'available':valid,'met':bool(valid and all(r['decision']!='positive' for r in reference_theta0))}


def paired_interactions(records, *, qualification):
    """Construct all four PZ4 estimands from the full paired 10-cell x5-arm ledger."""
    from .protocol import cohort
    selection=cohort(qualification)
    if selection['status'] != 'PASS': raise ValueError('STOP below n_min; no PZ4 claim')
    dates=selection['included']
    arms=(('reference',None),('lw',256),('lw',512),('kron',256),('kron',512))
    expected={(d,r,g,N,a,m) for d in dates for r in range(50)
              for g in ('A','B') for N in GRIDS[g] for a,m in arms}
    seen={}
    for row in records:
        try:
            key=(row['date'],row['replicate'],row['theta'],row['N'],row['arm'],row.get('m'))
            value=float(row['D'])
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError('STOP complete PZ4 paired identity required') from exc
        if not _integer(key[1],0,49) or key not in expected or key in seen or not np.isfinite(value):
            raise ValueError('STOP duplicate, nonfinite or unexpected paired record')
        seen[key]=value
    if set(seen)!=expected: raise ValueError('STOP incomplete paired ten-cell PZ4 ledger')
    # Same inherited estimand: mean over10 paired per-date cell-mean differences.
    digest=hashlib.sha256(json.dumps(sorted((list(k),v) for k,v in seen.items()),
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    result={}
    for arm,m in arms[1:]:
        means={d:float(np.mean([float(np.mean([seen[(d,r,g,N,arm,m)] for r in range(50)]))
                               -float(np.mean([seen[(d,r,g,N,'reference',None)] for r in range(50)]))
                               for g in ('A','B') for N in GRIDS[g]])) for d in dates}
        key=f'{arm}_m{m}'
        result[key]={**interval(list(means.values()),34,.05),'available':True,
            'cell':key,'family_size':34,'alpha':.05,'date_means':means,'expected_dates':dates,
            'qualification':json.loads(json.dumps(qualification)),
            'resamples':50,'resamples_per_date':dict.fromkeys(dates,50),
            'paired_cells':PAIRED_CELLS,'paired_record_sha256':digest,
            'construction':'mean of10 estimated-minus-reference date-cell means; exact paired R50 ledger'}
    return result


def pz4(low,high):
    valid=(isinstance(low,dict) and isinstance(high,dict) and set(low)==set(high)=={'lw','kron'}
           and all(_valid_interval(v) for v in list(low.values())+list(high.values())))
    if valid:
        rows=list(low.values())+list(high.values())
        valid=(_same_cohort(rows) and all(v.get('paired_cells')==PAIRED_CELLS for v in rows)
            and len({v.get('paired_record_sha256') for v in rows})==1
            and all(isinstance(v.get('paired_record_sha256'),str) and len(v['paired_record_sha256'])==64 for v in rows)
            and all(low[e]['cell']==f'{e}_m256' and high[e]['cell']==f'{e}_m512' for e in ('lw','kron')))
    return {'available':valid,'met':bool(valid and all(low[e]['decision']=='negative'
            and abs(high[e]['mean'])<abs(low[e]['mean']) for e in ('lw','kron')))}


def pz5(isotropic, expected_cells=None):
    valid=(expected_cells is None or set(expected_cells)==set(PZ5_CELLS)) and _family(isotropic,PZ5_CELLS)
    return {'available':valid,'met':bool(valid and all(r['decision']!='negative' for r in isotropic))}
