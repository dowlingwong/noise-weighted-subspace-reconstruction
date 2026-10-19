"""Prospective per-unit procedure. No recording reader is imported or called."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from experiments.mlst2026.hcp_rnoise_20261019 import population, protocol, windows
from experiments.mlst2026.hcp_rnoise_dev_20261019 import access
from experiments.mlst2026.hcp_rnoise_dev_20261019.pipeline import clean, new_file

SEAL_SHA = 'a1d35af344f5a7a619198d7f92ed77a051f52eee03c41b20ae3f6c4632984ff8'
QUAL_SHA = '10c20076a2bec80f2b77e0885227f091d8225329ef694d2fa5c1e829a71db470'
MS = Path('/ceph/dwong/arXiv-v1-polished-extended-preprint')
DEV = Path('/ceph/dwong/hcp_runs/dev_20261019/qualification')
OUT = Path('/ceph/dwong/hcp_runs/freeze_20261019')
PUB = MS / 'reviews/hcp_rnoise_freeze_20261019'


def committed(path, ms, commit, *, remote=False):
    """Require exact bytes at a commit; origin/main must contain it when remote=True."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    cmd = ['git', '-C', str(ms)]
    if remote:
        subprocess.run(cmd+['merge-base','--is-ancestor',commit,'origin/main'],env=env,check=True)
    rel = str(Path(path).resolve().relative_to(Path(ms).resolve()))
    blob = subprocess.check_output(cmd+['show',f'{commit}:{rel}'],env=env)
    if blob != Path(path).read_bytes():
        raise PermissionError('STOP predictions differ from committed record')
    return hashlib.sha256(blob).hexdigest()


def evidence():
    access.verify()
    seal = DEV/'SEALED_MOMENTS.json'; qual = DEV/'QUALIFICATION.json'
    if protocol.sha(seal)!=SEAL_SHA or protocol.sha(qual)!=QUAL_SHA:
        raise RuntimeError('STOP dev evidence identity mismatch')
    committed(MS/'reviews/hcp_rnoise_dev_20261019/SEALED_MOMENTS.json',MS,'76fa050')
    q=json.loads(qual.read_text()); s=json.loads(seal.read_text())
    if q['status']!='PASS' or q['E1']['included'] != [d for d,r in q['units'].items()
            if all(r[k]['pass'] for k in ('artifact','G1','G2','budget'))]:
        raise RuntimeError('STOP inconsistent E1')
    return q,s


def annotate(row, grid):
    """Keep the numerical prediction untouched; classify unavailable/out-of-grid explicitly."""
    r=dict(row); n=r.get('N_star_pred'); interval=r.get('N_star_interval')
    valid = n is not None and np.isfinite(n) and interval is not None and len(interval)==2 and np.isfinite(interval).all()
    valid = bool(valid and interval[0]<=interval[1] and all(r['checks'].values()) and r['restart_agreement'])
    below=[v for v in grid if valid and v<interval[0]]
    above=[v for v in grid if valid and v>interval[1]]
    r['PZ2']={'available':valid,'in_range':bool(valid and below and above),
              'below_certificate':below,'above_certificate':above,
              'reason':None if valid and below and above else 'unavailable crossing/certificate/diagnostics' if not valid else 'out-of-grid crossing; no PZ2 claim'}
    return r


def predict_unit(Sref, P, date_index):
    """Unchanged frozen Sref -> K -> Bdagger and 64-start primary population kernels."""
    Sref=np.asarray(Sref,float)
    if Sref.shape!=(72,72) or not np.isfinite(Sref).all():
        raise ValueError('finite d72 Sref required')
    np.linalg.cholesky(Sref)
    result=population.predictions(Sref,P,date_index)
    gg=result['grids']; union=sorted(set(gg['A_grid']+gg['B_grid']))
    result['fixed']=[annotate(r,union) for r in result['fixed']]
    for lab,res in result['adaptive'].items():
        if res['root']:
            res['row']=annotate(res['row'],gg[lab+'_grid'])
            res['available']=bool(res['row']['PZ2']['available'])
        else: res['available']=False
    # Theta0 control uses the A absolute grid; adaptive A/B keep their metadata-fixed grids.
    result['geometries']={'control':{'theta':0.,'grid':gg['A_grid'],'row':result['fixed'][0]}}
    for lab in ('A','B'):
        res=result['adaptive'][lab]
        result['geometries'][lab]={'available':res['available'],'grid':gg[lab+'_grid'],
                                  'theta':res.get('theta'),'row':res.get('row'),'reason':res.get('reason')}
    result['prediction_order']='Per confirmation date: qualify and construct unshrunk Sref; compute all predictions; commit and push exact per-date prediction bytes to MS origin/main; only then any learning fit for that date.'
    result['seed_namespace']='inherited prediction: law=0, cell=uint32 little-endian SHA256(stream)[:4], original sorted dev date index, replicate=0; PCG64DXSM SeedSequence([20261019,law,cell,index,replicate])'
    return result


def dev_unit(date):
    access.date_guard(date); q,s=evidence()
    if date not in q['E1']['included']: raise PermissionError('STOP pilot requires E1-passing dev date')
    index=access.config()['access']['dev_dates'].index(date)
    result={'date':date,'date_index':index,'label':'DEV PREDICTION; not confirmation',
            'sealed_moments_sha256':SEAL_SHA,'Sref_sha256_f64le':hashlib.sha256(np.asarray(s['moments'][date]['Sref'],dtype='<f8').tobytes()).hexdigest(),
            **predict_unit(s['moments'][date]['Sref'],1111,index)}
    new_file(OUT/'predictions'/f'{date}.json',result)
    print('PREDICTED',date,flush=True)
    return date


def collect():
    q,s=evidence(); dates=q['E1']['included']
    rows={d:json.loads((OUT/'predictions'/f'{d}.json').read_text()) for d in dates}
    result={'label':'DEV PREDICTIONS; not confirmatory','dates':dates,'units':rows,
            'sealed_moments_sha256':SEAL_SHA,'confirmation_access':False}
    new_file(PUB/'DEV_PREDICTIONS.json',result)
    return result
