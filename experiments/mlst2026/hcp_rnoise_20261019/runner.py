"""Identity/split/census/synthetic timing CLI. Real stages fail closed."""
import argparse
import json
import os
import platform
import signal as process_signal
import time
from pathlib import Path
import numpy as np
from experiments.mlst2026.meg_z_20261013.runner import kit_hashes
from experiments.mlst2026.meg_z_r1_20261014.runner import verify_kit as r1_verify
from experiments.mlst2026.meg_z_20261013.common import atomic_json
from .protocol import PACKAGE, load, split, split_check, geometry, sha, guard
from . import controls, windows


def verify_kit():
    cfg=load(); expected=json.loads((PACKAGE/'KIT_SHA256.json').read_text()); got=kit_hashes(PACKAGE)
    if got!=expected:
        raise RuntimeError('STOP HCP kit identity mismatch')
    base=r1_verify()
    if base['kit_identity']!=cfg['base']['r1_identity'] or base['old_identity']!=cfg['base']['old_identity']:
        raise RuntimeError('STOP frozen inherited kits differ')
    split();geometry()
    return {'kit_identity':got['kit_identity'],'files':len(got['files']),
            'r1_identity':base['kit_identity'],'old_identity':base['old_identity'],'CONFIG_sha256':sha(PACKAGE/'protocol.json')}


def census_check():
    dev=[u for u in split()['units'] if u['split']=='dev']; got=windows.census(dev)
    expected=json.loads((PACKAGE/'fixtures/BUDGETS.json').read_text())
    for row in expected['rows']:
        if got[row['date']][str(row['T'])]!=row['counts']:
            raise RuntimeError('STOP metadata census differs')
    p=min(got[u['measurement_date']]['6']['train'] for u in dev)
    grids=windows.grids(p)
    if grids['A_grid']!=expected['A_grid'] or grids['B_grid']!=expected['B_grid'] or grids['N_max']!=expected['N_max']:
        raise RuntimeError('STOP derived grids differ')
    from .gates import choose_m_hi
    high=choose_m_hi({u['measurement_date']:got[u['measurement_date']]['6']['cal'] for u in dev})
    return {'status':'PASS','metadata_only':True,'T':6,'C':12,'d':72,'grids':grids,'m_high':high,'units':len(dev)}


def environment():
    import scipy,mne
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE'):
        if os.environ.get(key)!='1': raise RuntimeError(f'STOP {key} must be1')
    if 'PYTHONPYCACHEPREFIX' in os.environ: raise RuntimeError('STOP unset PYTHONPYCACHEPREFIX')
    if (np.__version__,scipy.__version__,mne.__version__)!=('2.2.6','1.15.3','1.8.0'):
        raise RuntimeError('STOP reference numerical environment differs')
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'mne':mne.__version__,
            'threads':1,'bytecode_disabled':True,'clock':'perf_counter wall seconds, single process'}


def preflight(out):
    """Five timing probes, NOT controls campaign; all inputs fabricated."""
    out=Path(out)
    if out.exists(): raise ValueError('STOP output exists; choose a new evidence path')
    env=environment(); identity=verify_kit(); cfg=load(); census=census_check()
    u=next(u for u in split()['units'] if u['split']=='dev')
    shape=controls.synthetic_shape(); rows=[]
    for law,cell in ((1,0),(2,0),(4,2),(6,5),(7,12)):
        t0=time.perf_counter(); row=controls.replicate(shape,u,law,cell,0,999,details=True,timing=True)
        rows.append({'law':law,'cell':cell,'seconds':time.perf_counter()-t0,'seed_entropy':row['seed_entropy']})
    from .population import general_noise
    from .signal import build
    from .gates import reference
    timed=[]
    class Limit(Exception): pass
    def alarm(signum,frame): raise Limit()
    previous=process_signal.signal(process_signal.SIGALRM,alarm)
    try:
        for theta in (.12,.4):
            t0=time.perf_counter(); complete=False
            process_signal.setitimer(process_signal.ITIMER_REAL,128.)
            try:
                gen,_=controls.rng(2,99,0,int(theta*100))
                S=reference(controls.generate(shape,u,2,gen)['ref'][0]); model=build(S,theta)
                result=general_noise(model,S,shape['population'],gen,load()['signal']['population_random_starts'])
                complete=True
            except Limit: result={'censored_lower_bound':True}
            finally: process_signal.setitimer(process_signal.ITIMER_REAL,0)
            timed.append({'theta':theta,'seconds':time.perf_counter()-t0,'complete':complete,'result':result})
            if not complete: break
    finally:
        process_signal.setitimer(process_signal.ITIMER_REAL,0);process_signal.signal(process_signal.SIGALRM,previous)
    ctrl_per=max(v['seconds'] for v in rows); d3_per=max(v['seconds'] for v in timed)
    controls_n=cfg['controls']['shapes']*len(cfg['controls']['laws'])*len(cfg['controls']['cells'])*cfg['controls']['replicates_per_shape']
    d3_rows=12*cfg['reference']['D3S_estimates']*cfg['reference']['D3S_rows_per_estimate']
    projections={str(w):{'controls_hours_ideal':ctrl_per*controls_n/w/3600,'D3S_hours_ideal':d3_per*d3_rows/w/3600,
                         'controls_hours_25pct_overhead':1.25*ctrl_per*controls_n/w/3600,'D3S_hours_25pct_overhead':1.25*d3_per*d3_rows/w/3600} for w in (4,16)}
    report={'synthetic_only':True,'no_real_moments':True,'controls_campaign_run':False,'timing_probes_only':True,
            'environment':env,**identity,'census':census,'d':72,'control_timings':rows,'D3S_timings':timed,
            'controls_total_replicates':controls_n,'D3S_total_rows_bound':d3_rows,'seconds_per_control_worst':ctrl_per,
            'seconds_per_D3S_worst':d3_per,'projections':projections,
            'projection_limits':'five synthetic controls and two theta rows only; no real dev shapes, no worker throughput measured;25% overhead is illustrative, not a confidence interval',
            'status':'PASS' if all(v['complete'] for v in timed) and projections['4']['controls_hours_25pct_overhead']<=6 and projections['16']['D3S_hours_25pct_overhead']<=8 else 'STOP'}
    atomic_json(out,report); return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='command',required=True)
    for name in ('verify-kit','census-check'): sp.add_parser(name)
    p=sp.add_parser('split-check');p.add_argument('--inventory')
    p=sp.add_parser('preflight');p.add_argument('--out',required=True)
    for name in ('decode-dev','gates','predict','controls','fit','confirmation'):
        p=sp.add_parser(name,help='reserved future stage; candidate always refuses real execution')
        p.add_argument('--date',required=True);p.add_argument('--subject')
    a=ap.parse_args(argv)
    if a.command=='verify-kit': result=verify_kit()
    elif a.command=='split-check': result=split_check(a.inventory)
    elif a.command=='census-check': result=census_check()
    elif a.command=='preflight': result=preflight(a.out)
    else:
        guard(a.date,a.subject,header_only=False)
        raise PermissionError('future execution requires author adoption and independent launch review')
    print(json.dumps(result,indent=2))
    if result.get('status')=='STOP': raise SystemExit(2)

if __name__=='__main__': main()
