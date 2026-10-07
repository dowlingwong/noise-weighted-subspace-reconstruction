"""Exclusive, source-bound execution and saved-map risk audit. No automatic retries."""
import argparse, hashlib, json, os, platform, sys, time, tarfile, signal
from pathlib import Path
from . import guard
PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sources():
    own=[str(p.relative_to(ROOT)) for p in PACKAGE.rglob('*') if p.suffix in ('.py','.json','.sh')]
    return {r:sha(ROOT/r) for r in sorted(set(own+guard.T25_REUSED+guard.loaded_repo_modules()))}
def environment():
    import numpy,scipy
    return dict(python=sys.version,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,threads={k:os.getenv(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
def config(mode):
    return json.loads((PACKAGE/(mode+'.json')).read_text())
def validate(c,mode):
    if mode not in ('fixture','benchmark','confirmation'):raise ValueError('unknown mode')
    if c!=config(mode):raise ValueError('configuration differs from shipped design')
    cells=len(c['seeds'])*len(c['theta'])*len(c['N'])*len(c['p'])
    if cells>c['budget']['max_cells'] or cells*4>c['budget']['max_product_restarts'] or c['budget']['retries']!=0:raise ValueError('allocation exceeds frozen caps')
    if mode!='confirmation' and any(202610060300<=s<=202610060999 for s in c['seeds']):raise ValueError('confirmation seeds forbidden')
    if mode=='fixture' and (c['C']*c['T']>=256 or max(c['N'])>=400):raise ValueError('fixture must be small')
    if mode in ('benchmark','confirmation') and any(os.getenv(k)!='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('single-thread environment required')
    if mode=='benchmark' and c['namespace']!='paper1-tr09-t25-benchmark-20261007/development':raise ValueError('development namespace required')
    if mode=='confirmation' and os.getenv('TR09_CONFIRMATION_AUTHORISED')!='yes':raise ValueError('authorisation required')
def freeze(mode,out):
    c=config(mode);validate(c,mode);guard.assert_direct()
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    body=dict(mode=mode,config=c,sources=sources(),environment=environment(),independent_review_accepted=False,registered=False,ledger_path='/ceph/dwong/tr09_runs/registry/paper1-tr09-v1-confirm.attempt.json' if mode=='confirmation' else None)
    write(out/'freeze.json',body)
    return body

def run(mode,out,freeze_path=None,acceptance=None,ledger=None):
    c=config(mode);validate(c,mode);guard.assert_direct()
    from . import science
    import numpy as np
    guard.assert_loaded_modules()
    before=sources();env=environment()
    if mode=='confirmation':
        if not freeze_path or not acceptance or not ledger:raise ValueError('frozen identity, independent acceptance and namespace ledger required')
        f=json.loads(Path(freeze_path).read_text());a=json.loads(Path(acceptance).read_text())
        if f['mode']!='confirmation' or f['sources']!=before or f['config']!=c or f['environment']!=env:raise ValueError('freeze identity mismatch')
        if str(Path(ledger).resolve())!=f.get('ledger_path'):raise ValueError('ledger differs from frozen namespace registry')
        if a.get('registered') is not True or a.get('namespace')!=c['namespace']:raise ValueError('registered namespace acceptance required')
        if a.get('verdict')!='ACCEPTED' or a.get('freeze_sha256')!=sha(freeze_path):raise ValueError('independent acceptance does not bind freeze')
    out=Path(out)
    if out.exists():raise ValueError('output already exists; no retry')
    if mode=='confirmation':
        if Path(ledger).exists():raise FileExistsError('namespace attempt already claimed')
        if os.name!='posix' or not Path('/proc').exists():raise ValueError('registered Linux supervisor required')
        parent=os.getppid();status=Path(f'/proc/{parent}/status').read_text();grandparent=int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
        parent_cmd=Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
        grand_cmd=Path(f'/proc/{grandparent}/cmdline').read_bytes().split(b'\0')
        if not any(Path(os.fsdecode(x)).name=='timeout' for x in parent_cmd if x) or os.fsencode(str(PACKAGE/'supervisor.sh')) not in grand_cmd:raise ValueError('registered supervisor ancestry required')
        fd=os.open(ledger,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:json.dump(dict(namespace=c['namespace'],freeze_sha256=sha(freeze_path),out=str(out),time=time.time(),supervisor=dict(parent_cmd=[os.fsdecode(x) for x in parent_cmd if x],grandparent_cmd=[os.fsdecode(x) for x in grand_cmd if x],timeout_seconds=28800,parent_pid=parent,grandparent_pid=grandparent)),stream)
    out.mkdir(exist_ok=False,parents=True)
    if mode=='confirmation':
        (out/'FREEZE.json').write_bytes(Path(freeze_path).read_bytes());(out/'ACCEPTANCE.json').write_bytes(Path(acceptance).read_bytes());(out/'ATTEMPT.json').write_bytes(Path(ledger).read_bytes())
    write(out/'START.json',dict(mode=mode,time=time.time(),config=c,sources=before,environment=env,loaded_modules_before=guard.loaded_repo_modules()))
    snap=out/'source_snapshot'
    for rel in before:
        dest=snap/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    records=[];start=time.time()
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
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
      audit(out,report=out/'AUDIT.json',in_campaign=True)
      write(out/'EXIT.json',dict(exit_code=0,seconds=time.time()-start,source_unchanged=True,loaded_modules_after=guard.loaded_repo_modules()))
      if mode!='confirmation':export(out)
    except BaseException as e:
      write(out/'EXIT.json',dict(exit_code=1,seconds=time.time()-start,error=repr(e),source_unchanged=before==sources()))
      if mode!='confirmation':export(out)
      raise

def audit(out,report=None,in_campaign=False):
    from . import science
    from experiments.mlst2026.t25_production_20260930 import science as t25
    import numpy as np
    out=Path(out);start=json.loads((out/'START.json').read_text());c=start['config'];rows=[]
    if not in_campaign and json.loads((out/'EXIT.json').read_text()).get('exit_code')!=0:raise ValueError('incomplete/failed campaign cannot pass audit')
    if start['mode']=='confirmation':
        frozen=json.loads((out/'FREEZE.json').read_text());acceptance=json.loads((out/'ACCEPTANCE.json').read_text());attempt=json.loads((out/'ATTEMPT.json').read_text());freeze_sha=sha(out/'FREEZE.json')
        if not in_campaign:
            supervision=json.loads((out/'SUPERVISOR_EXIT.json').read_text())
            if supervision.get('exit_code')!=0 or supervision.get('attempt_sha256')!=sha(out/'ATTEMPT.json'):raise ValueError('supervisor exit binding mismatch')
        if attempt.get('supervisor',{}).get('timeout_seconds')!=28800:raise ValueError('supervisor timeout missing')
        if frozen['sources']!=start['sources'] or frozen['config']!=c or frozen['environment']!=start['environment'] or acceptance.get('registered') is not True or acceptance.get('verdict')!='ACCEPTED' or acceptance.get('freeze_sha256')!=freeze_sha or acceptance.get('namespace')!=c['namespace'] or attempt.get('freeze_sha256')!=freeze_sha or attempt.get('namespace')!=c['namespace']:raise ValueError('supervised attempt identity mismatch')
    if c!=config(start['mode']):raise ValueError('returned config differs from shipped design')
    for rel,want in start['sources'].items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts or sha(out/'source_snapshot'/p)!=want:raise ValueError('source snapshot mismatch')
    if start['sources']!=sources():raise ValueError('audit source identity mismatch')
    for path in sorted(out.glob('cell_*.json')):
        row=json.loads(path.read_text());array_path=out/row['arrays']
        if Path(row['arrays']).name!=row['arrays'] or not row['arrays'].startswith('cell_'):raise ValueError('unsafe array path')
        a=np.load(array_path,allow_pickle=False)
        g=t25.geometry(c);sig=t25.covariance(c);L,Li=t25.powers(sig);Cs=L@g.K(row['theta'])@L;W=Li@Li
        for key,value in [('Sigma',sig),('Cs',Cs),('W',W)]:
            if not np.allclose(a[key],value,rtol=1e-10,atol=1e-10):raise ValueError('truth matrices differ')
        _,_,_,_,pool=science.data(c,row['seed'],row['theta'],row['p'])
        x,mean,S=t25.moments(pool,row['N'])
        if t25.digest(x)!=row['training_sha'] or not np.allclose(mean,a['mean'],rtol=1e-10,atol=1e-10) or not np.allclose(S,a['S'],rtol=1e-10,atol=1e-10):raise ValueError('training provenance mismatch')
        recomputed={}
        for label in ['U','F']:
            B=a['B_'+label]
            if not np.allclose(t25.map_for(B,W),a['A_'+label],rtol=1e-9,atol=1e-9) or not np.allclose((np.eye(g.d)-a['A_'+label])@mean,a['b_'+label],rtol=1e-9,atol=1e-9):raise ValueError('basis map or intercept mismatch')
            if B.shape!=(g.d,g.k) or np.linalg.matrix_rank(B)!=g.k:raise ValueError('basis rank mismatch')
            risk=t25.risks(a['A_'+label],a['b_'+label],Cs,sig,W)
            for endpoint in risk:
                if not np.isclose(risk[endpoint],row['risks'][label][endpoint],rtol=1e-9,atol=1e-9):raise ValueError('saved-map risk mismatch')
            recomputed['R_'+label]=risk['clean']
        recomputed['D']=recomputed['R_F']-recomputed['R_U']
        for key,val in recomputed.items():
            if not np.isclose(val,row[key],rtol=1e-9,atol=1e-9):raise ValueError('endpoint mismatch')
        rr=[t25.risks(a[f'A_restart_{i}'],a[f'b_restart_{i}'],Cs,sig,W)['clean'] for i in range(4)]
        if not np.allclose(rr,row['restart_clean_risks'],rtol=1e-9,atol=1e-9):raise ValueError('restart risks mismatch')
        for i in range(4):
            uc,ut=a[f'uc_{i}'],a[f'ut_{i}']
            if uc.shape!=(g.C,g.k_c) or ut.shape!=(g.T,g.k_t) or not np.allclose(uc.T@uc,np.eye(g.k_c),atol=1e-9) or not np.allclose(ut.T@ut,np.eye(g.k_t),atol=1e-9):raise ValueError('product factors mismatch')
            B=np.kron(ut,uc)
            if not np.allclose((np.eye(g.d)-a[f'A_restart_{i}'])@mean,a[f'b_restart_{i}'],rtol=1e-9,atol=1e-9):raise ValueError('restart intercept mismatch')
            objective,gc,gt=t25.objective_gradient(uc,ut,W,S)
            scale=float(np.trace(W@S));grad=float(np.sqrt(np.sum(((gc-uc@(uc.T@gc))/scale)**2)+np.sum(((gt-ut@(ut.T@gt))/scale)**2)))
            diagnostic=row['optimization']['restarts'][i]
            if not np.isclose(grad,diagnostic['gradient_norm'],rtol=1e-8,atol=1e-10) or (diagnostic['status']=='stationary')!=(grad<=1e-8):raise ValueError('stationarity diagnostic mismatch')
            history=diagnostic['trajectory']
            if len(history)>151 or not np.all(np.diff(history)>=-1e-8) or not np.isclose(history[-1],objective,rtol=1e-9,atol=1e-9):raise ValueError('optimizer trajectory mismatch')
            if not np.isclose(objective,row['optimization']['restarts'][i]['objective'],rtol=1e-9,atol=1e-9) or not np.allclose(t25.map_for(B,W),a[f'A_restart_{i}'],rtol=1e-9,atol=1e-9):raise ValueError('restart objective/map mismatch')
        if not np.allclose(t25.map_for(t25.unrestricted(W,S,g.k),W),a['A_U'],rtol=1e-8,atol=1e-8):raise ValueError('unrestricted fit mismatch')
        selected=row['optimization']['selected'];objs=[r['objective'] for r in row['optimization']['restarts']]
        if selected!=int(np.argmax(objs)) or not np.allclose(a['A_F'],a[f'A_restart_{selected}'],rtol=1e-9,atol=1e-9):raise ValueError('selection mismatch')
        spread=float(max(rr)-min(rr));limited=row['optimization']['restarts'][selected]['status']!='stationary' or spread>=abs(recomputed['D']);agreement=bool(np.linalg.norm(a['A_F']-a['A_restart_0'],'fro')<=1e-6)
        if not np.isclose(spread,row['restart_spread'],rtol=1e-9,atol=1e-9) or limited!=row['optimization_limited'] or agreement!=row['selected_equals_hosvd']:raise ValueError('optimization flag or branch agreement mismatch')
        row.update(recomputed);rows.append(row)
    reduced=science.reduce(c,rows);summary=json.loads((out/'SUMMARY.json').read_text())
    def close(a,b):
        if isinstance(a,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
        if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
        if isinstance(a,float):return bool(np.isclose(a,b,rtol=1e-9,atol=1e-9))
        return a==b
    if not close(reduced,summary):raise ValueError('summary differs from raw maps')
    guard.assert_loaded_modules()
    if report is not None:write(report,dict(accepted=True,cells=len(rows),tolerance='rtol=atol=1e-9; truth 1e-10',source_bound=True))
    return reduced

def export(out):
    out=Path(out);files=sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS','return.tgz'])
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    with tarfile.open(out/'return.tgz','w:gz') as tar:
        for p in files+[out/'SHA256SUMS']:tar.add(p,arcname=str(p.relative_to(out)))
def verify_receipt(out):
    out=Path(out)
    if out.is_symlink() or any(p.is_symlink() for p in out.rglob('*')):raise ValueError('symlink in receipt')
    manifest=out/'SHA256SUMS'; expected={}
    for line in manifest.read_text().splitlines():
        digest,rel=line.split('  ',1);p=Path(rel)
        if p.is_absolute() or '..' in p.parts or rel in expected:raise ValueError('unsafe/duplicate manifest path')
        if sha(out/p)!=digest:raise ValueError('checksum mismatch')
        expected[rel]=digest
    if any(p.is_symlink() for p in out.rglob('*')):raise ValueError('symlink in receipt')
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS','return.tgz']}
    if actual!=set(expected):raise ValueError('manifest census mismatch')
    if (out/'return.tgz').exists():
        with tarfile.open(out/'return.tgz') as archive:
            members=archive.getmembers()
            if len(members)!=len(expected)+1 or {m.name for m in members}!=set(expected)|{'SHA256SUMS'}:raise ValueError('archive census mismatch')
            for m in members:
                if not m.isfile():raise ValueError('non-file archive member')
                digest=hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                if digest!=(sha(manifest) if m.name=='SHA256SUMS' else expected[m.name]):raise ValueError('archive checksum mismatch')
    return expected

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','freeze','audit','seal']);ap.add_argument('--mode',choices=['fixture','benchmark','confirmation']);ap.add_argument('--out',required=True);ap.add_argument('--freeze');ap.add_argument('--acceptance');ap.add_argument('--ledger');ap.add_argument('--exit-code',type=int);a=ap.parse_args()
    if a.action=='run':run(a.mode,a.out,a.freeze,a.acceptance,a.ledger)
    elif a.action=='freeze':freeze(a.mode,a.out)
    elif a.action=='audit':verify_receipt(a.out);audit(a.out)
    else:
        out=Path(a.out)
        if a.exit_code is None:raise ValueError('supervisor exit required')
        write(out/'SUPERVISOR_EXIT.json',dict(exit_code=a.exit_code,attempt_sha256=sha(out/'ATTEMPT.json'),time=time.time()))
        export(out)
if __name__=='__main__':main()
