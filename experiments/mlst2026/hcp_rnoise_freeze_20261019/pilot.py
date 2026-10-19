"""F3 dev-only pilot on unchanged sealed sources; F4 R50 reference-only remedy."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.mlst2026.hcp_rnoise_20261019 import learning, windows, gates
from experiments.mlst2026.hcp_rnoise_dev_20261019 import access
from experiments.mlst2026.hcp_rnoise_dev_20261019.pipeline import new_file
from .prediction import DEV, OUT, PUB, MS, evidence, committed

ARMS=(('reference',None),('lw',256),('lw',512),('kron',256),('kron',512),('isotropic',None))
PRED_COMMIT='f382102'


def arm_key(arm,m):return arm if m is None else f'{arm}_m{m}'


def source(date,q,s):
    unit=access.date_guard(date)
    if date not in q['E1']['included']: raise PermissionError('STOP excluded dev unit')
    path=DEV/(date+'_decimated.npy')
    expected=s['moments'][date]['source_sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('STOP source changed')
    X=np.load(path,allow_pickle=False)
    roles=windows.cut(X,windows.plan(unit['n_times'],unit['sfreq_hz']))
    S=np.asarray(s['moments'][date]['Sref'])
    if not np.array_equal(gates.reference(roles['ref'][0]),S):
        raise RuntimeError('STOP Sref differs from seal')
    return roles,S


def one_group(task):
    """One date/geometry/replicate; exact inherited fit_cell for all required cells."""
    date,geometry,replicate,remedy=task
    q,s=evidence();committed(PUB/'DEV_PREDICTIONS.json',MS,PRED_COMMIT,remote=True)
    pred=json.loads((PUB/'DEV_PREDICTIONS.json').read_text())['units'][date]
    geo=pred['geometries'][geometry]
    if geometry!='control' and not geo['available']:
        raise RuntimeError('STOP unavailable adaptive geometry; no silent family deletion')
    roles,S=source(date,q,s)
    from experiments.mlst2026.hcp_rnoise_20261019.signal import build
    model=build(S,geo['theta']);index=access.config()['access']['dev_dates'].index(date)
    rows=[];start=time.monotonic()
    for N in geo['grid']:
        for arm,m in (ARMS[:1] if remedy else ARMS):
            result=learning.fit_cell(roles,model,S,N,m,arm,index,replicate)
            if not np.isfinite(result['D']):raise RuntimeError('STOP nonfinite learning result')
            rows.append({'date':date,'geometry':geometry,'arm_id':arm_key(arm,m),**result})
    record={'date':date,'geometry':geometry,'replicate':replicate,'label':'DEV PILOT; not confirmatory',
            'prediction_commit':PRED_COMMIT,'seed_namespace':'inherited learning law8/cells100 training,101 evaluation,102 calibration; original sorted12 dev index',
            'records':rows,'seconds':time.monotonic()-start,'remedy':remedy}
    path=OUT/('remedy_records' if remedy else 'pilot_records')/date/geometry/f'r{replicate:02d}.json'
    new_file(path,record)
    print('PILOT',date,geometry,replicate,len(rows),'seconds',round(record['seconds'],2),flush=True)
    return str(path)


def run(remedy=False):
    q,s=evidence();expected=(OUT/('remedy_records' if remedy else 'pilot_records'))
    expected.mkdir(exist_ok=False)
    tasks=[(d,g,r,remedy) for d in q['E1']['included'] for g in ('control','A','B')
           for r in (range(20,50) if remedy else range(20))]
    new_file(OUT/('REMEDY_STARTED.json' if remedy else 'PILOT_STARTED.json'),
             {'tasks':len(tasks),'workers':16,'resamples':50 if remedy else 20,
              'arms':'reference only; extends existing R20 without recomputing' if remedy else list(map(lambda a:arm_key(*a),ARMS)),
              'prediction_commit':PRED_COMMIT,'confirmation_access':False})
    with ProcessPoolExecutor(max_workers=16) as pool:list(pool.map(one_group,tasks))
    new_file(OUT/('REMEDY_COMPLETED.json' if remedy else 'PILOT_COMPLETED.json'),{'status':'COMPLETE','tasks':len(tasks)})


if __name__=='__main__':
    import sys
    run(remedy='--remedy' in sys.argv)
