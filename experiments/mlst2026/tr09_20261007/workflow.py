"""Exclusive, source-bound execution and saved-map risk audit. No automatic retries."""
import argparse, hashlib, json, os, platform, sys, time, tarfile
from pathlib import Path
from . import guard
PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sources():
    own=[str(p.relative_to(ROOT)) for p in PACKAGE.rglob('*') if p.suffix in ('.py','.json')]
    return {r:sha(ROOT/r) for r in sorted(set(own+guard.T25_REUSED))}
def environment():
    import numpy,scipy
    return dict(python=sys.version,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,threads={k:os.getenv(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
def config(mode):
    return json.loads((PACKAGE/(mode+'.json')).read_text())
def validate(c,mode):
    if mode not in ('fixture','benchmark','confirmation'):raise ValueError('unknown mode')
    if c!=config(mode):raise ValueError('configuration differs from shipped design')
    if mode!='confirmation' and any(202610060300<=s<=202610060999 for s in c['seeds']):raise ValueError('confirmation seeds forbidden')
    if mode=='fixture' and (c['C']*c['T']>=256 or max(c['N'])>=400):raise ValueError('fixture must be small')
    if mode=='benchmark' and c['namespace']!='paper1-tr09-t25-benchmark-20261007/development':raise ValueError('development namespace required')
    if mode=='confirmation' and os.getenv('TR09_CONFIRMATION_AUTHORISED')!='yes':raise ValueError('authorisation required')
def freeze(mode,out):
    c=config(mode);validate(c,mode);guard.assert_direct()
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    body=dict(mode=mode,config=c,sources=sources(),environment=environment(),independent_review_accepted=False,registered=False)
    write(out/'freeze.json',body)
    return body

def run(mode,out,freeze_path=None,acceptance=None):
    c=config(mode);validate(c,mode);guard.assert_direct()
    from . import science
    import numpy as np
    guard.assert_loaded_modules()
    before=sources();env=environment()
    if mode=='confirmation':
        if not freeze_path or not acceptance:raise ValueError('frozen identity and independent acceptance required')
        f=json.loads(Path(freeze_path).read_text());a=json.loads(Path(acceptance).read_text())
        if f['sources']!=before or f['config']!=c or f['environment']!=env:raise ValueError('freeze identity mismatch')
        if a.get('verdict')!='ACCEPTED' or a.get('freeze_sha256')!=sha(freeze_path):raise ValueError('independent acceptance does not bind freeze')
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    write(out/'START.json',dict(mode=mode,time=time.time(),config=c,sources=before,environment=env))
    snap=out/'source_snapshot'
    for rel in before:
        dest=snap/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    records=[];start=time.time()
    try:
      for seed in c['seeds']:
       for theta in c['theta']:
        for N in c['N']:
         for p in c['p']:
          then=time.time();row,arrays=science.fit(c,seed,theta,N,p);row['seconds']=time.time()-then
          name=f'cell_{seed}_{theta}_{N}_{p}';row['arrays']=name+'.npz'
          np.savez_compressed(out/row['arrays'],**arrays);write(out/(name+'.json'),row);records.append(row)
          print(name,round(row['seconds'],3),flush=True)
      if before!=sources():raise ValueError('source changed during campaign')
      write(out/'SUMMARY.json',science.reduce(c,records))
      audit(out)
      write(out/'EXIT.json',dict(exit_code=0,seconds=time.time()-start,source_unchanged=True))
      export(out)
    except BaseException as e:
      write(out/'EXIT.json',dict(exit_code=1,seconds=time.time()-start,error=repr(e),source_unchanged=before==sources()))
      raise

def audit(out):
    from . import science
    from experiments.mlst2026.t25_production_20260930 import science as t25
    import numpy as np
    out=Path(out);start=json.loads((out/'START.json').read_text());c=start['config'];rows=[]
    if start['sources']!=sources():raise ValueError('audit source identity mismatch')
    for path in sorted(out.glob('cell_*.json')):
        row=json.loads(path.read_text());a=np.load(out/row['arrays'],allow_pickle=False)
        g=t25.geometry(c);sig=t25.covariance(c);L,Li=t25.powers(sig);Cs=L@g.K(row['theta'])@L;W=Li@Li
        for key,value in [('Sigma',sig),('Cs',Cs),('W',W)]:
            if not np.allclose(a[key],value,rtol=1e-10,atol=1e-10):raise ValueError('truth matrices differ')
        recomputed={}
        for label in ['U','F']:
            risk=t25.risks(a['A_'+label],a['b_'+label],Cs,sig,W)
            for endpoint in risk:
                if not np.isclose(risk[endpoint],row['risks'][label][endpoint],rtol=1e-9,atol=1e-9):raise ValueError('saved-map risk mismatch')
            recomputed['R_'+label]=risk['clean']
        recomputed['D']=recomputed['R_F']-recomputed['R_U']
        for key,val in recomputed.items():
            if not np.isclose(val,row[key],rtol=1e-9,atol=1e-9):raise ValueError('endpoint mismatch')
        rr=[t25.risks(a[f'A_restart_{i}'],a[f'b_restart_{i}'],Cs,sig,W)['clean'] for i in range(4)]
        if not np.allclose(rr,row['restart_clean_risks'],rtol=1e-9,atol=1e-9):raise ValueError('restart risks mismatch')
        selected=row['optimization']['selected'];objs=[r['objective'] for r in row['optimization']['restarts']]
        if selected!=int(np.argmax(objs)) or not np.allclose(a['A_F'],a[f'A_restart_{selected}'],rtol=1e-9,atol=1e-9):raise ValueError('selection mismatch')
        row.update(recomputed);rows.append(row)
    reduced=science.reduce(c,rows);summary=json.loads((out/'SUMMARY.json').read_text())
    def close(a,b):
        if isinstance(a,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
        if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
        if isinstance(a,float):return bool(np.isclose(a,b,rtol=1e-9,atol=1e-9))
        return a==b
    if not close(reduced,summary):raise ValueError('summary differs from raw maps')
    write(out/'AUDIT.json',dict(accepted=True,cells=len(rows),tolerance='rtol=atol=1e-9; truth 1e-10',source_bound=True))
    return reduced

def export(out):
    out=Path(out);files=sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS','return.tgz'])
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    with tarfile.open(out/'return.tgz','w:gz') as tar:
        for p in files+[out/'SHA256SUMS']:tar.add(p,arcname=str(p.relative_to(out)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','freeze','audit']);ap.add_argument('--mode',choices=['fixture','benchmark','confirmation']);ap.add_argument('--out',required=True);ap.add_argument('--freeze');ap.add_argument('--acceptance');a=ap.parse_args()
    if a.action=='run':run(a.mode,a.out,a.freeze,a.acceptance)
    elif a.action=='freeze':freeze(a.mode,a.out)
    else:audit(a.out)
if __name__=='__main__':main()
