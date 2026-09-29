"""Local candidate workflow. Production freeze is intentionally closed pending review.

Raw computational kernels are testable, but no production run can be created by
this CLI until a separately reviewed deployment version supplies qualification.
"""
from pathlib import Path
import hashlib,json,os,platform,subprocess,sys,time
import numpy as np
from .protocol import config,enumeration
from .science import fit_cell,make_data,estimate,risks,map_for,qr,top,powers,covariance,geometry,digest,contrast_values,reduce_records,representation,objective_gradient
ROOT=Path(__file__).resolve().parents[3]
PACKAGE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write_new(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def sources():
    # Broad exact-byte closure; additions/removals are also detected. Never scans outcomes.
    files=sorted(set(list((ROOT/'src').rglob('*.py'))+list((ROOT/'experiments').rglob('*.py'))+list(PACKAGE.glob('*.json'))+[ROOT/'pyproject.toml']))
    ms=Path(os.environ.get('MLST_MS_ROOT',str(ROOT.parent/'arXiv-v1-polished-extended-preprint'))).resolve()
    msfiles=[ms/'plan/T25_INTERACTION_PROTOCOL_2026-09-29.md',ms/'plan/T24_CUMULANT_PROTOCOL_v2_2026-09-29.md',ms/'03_theory.tex',ms/'app_main_proofs.tex']
    return {'dirty_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),'MS_files':{str(p):sha(p) for p in msfiles},'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'psutil':__import__('psutil').__version__}}

def verify_source(f):
    if sources()!=f['sources']:raise ValueError('source/environment mutated')

def freeze(c,out):
    if not c['development_only']:raise ValueError('PRODUCTION BLOCKED: candidate requires independent implementation acceptance and actual Deepthought hardware/allocation qualification; no override')
    from .science import population_references
    write_new(out,{'population_references':population_references(c),'status':'fixture_only','config':c,'enumeration':enumeration(c),'sources':sources(),'config_identity':identity(c)})

def worker(fpath,index,out):
    import threading,signal,psutil
    parent=os.getppid()
    def watch():
        while True:
            if os.getppid()!=parent or parent==1:os.kill(os.getpid(),signal.SIGKILL)
            time.sleep(.25)
    threading.Thread(target=watch,daemon=True).start()
    f=read(fpath);verify_source(f);row=f['enumeration']['cells'][index]
    if not f['config']['development_only']:raise ValueError('production not accepted')
    result,arrays=fit_cell(f['config'],row['seed'],row['theta'],row['N'])
    dest=Path(out);dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'arrays.npz',**arrays)
    write_new(dest/'record.json',result);verify_source(f)
    write_new(dest/'COMPLETE.json',{'arrays_sha':sha(dest/'arrays.npz'),'record_sha':sha(dest/'record.json')})

def run(fpath,root):
    import psutil
    f=read(fpath);verify_source(f)
    if f.get('status')!='fixture_only' or not f['config']['development_only']:raise ValueError('production not accepted')
    root=Path(root).resolve()
    ms=Path(os.environ.get('MLST_MS_ROOT',str(ROOT.parent/'arXiv-v1-polished-extended-preprint'))).resolve()
    if root.is_relative_to(ROOT) or root.is_relative_to(ms):raise ValueError('run directory must be outside both source repositories')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError('one thread required: '+key)
    root.mkdir(parents=True,exist_ok=False);write_new(root/'freeze.json',f)
    start=time.monotonic();budget=f['config']['budget']
    try:
        for i,row in enumerate(f['enumeration']['cells']):
            verify_source(f)
            write_new(root/f'{i:04d}.START.json',{'cell':row,'reserved_charges':53,'timestamp':time.time()})
            with (root/f'{i:04d}.log').open('x') as log:
                p=subprocess.Popen([sys.executable,'-m','experiments.mlst2026.t25_interaction','_worker','--freeze',str(root/'freeze.json'),'--index',str(i),'--out',str(root/f'cell-{i:04d}')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                began=time.monotonic();peak=0
                while p.poll() is None:
                    try:peak=max(peak,psutil.Process(p.pid).memory_info().rss)
                    except psutil.NoSuchProcess:pass
                    if peak>budget['rss_bytes'] or time.monotonic()-began>budget['cell_seconds'] or time.monotonic()-start>budget['campaign_seconds']:
                        import signal
                        os.killpg(p.pid,signal.SIGKILL);p.wait();raise ValueError('resource cap exceeded')
                    time.sleep(.05)
                if p.returncode:raise ValueError(f'cell {i} exited {p.returncode}; no retry')
            verify_source(f);write_new(root/f'{i:04d}.END.json',{'elapsed':time.monotonic()-began,'peak_rss':peak,'status':'COMPLETE'})
        verify(root,require_complete=False)
        write_new(root/'COMPLETE.json',{'development_only':True,'elapsed':time.monotonic()-start,'cells':len(f['enumeration']['cells'])})
    except BaseException as e:
        write_new(root/'FAILED.json',{'error':str(e),'elapsed':time.monotonic()-start,'retry_allowed':False});raise


def assert_close(got,want):
    if isinstance(want,dict):
        if set(got)!=set(want):raise ValueError('record inventory changed')
        for k in want:assert_close(got[k],want[k])
    elif isinstance(want,list):
        if len(got)!=len(want):raise ValueError('record length changed')
        for a,b in zip(got,want):assert_close(a,b)
    elif isinstance(want,(float,np.floating)):
        if not np.isclose(got,want,rtol=1e-8,atol=1e-9):raise ValueError('numerical reconstruction mismatch')
    elif got!=want:raise ValueError('record identity changed')


def verify(root,require_complete=True):
    from .science import response,starts
    root=Path(root)
    if (root/'FAILED.json').exists() or (require_complete and not (root/'COMPLETE.json').is_file()):raise ValueError('failed/incomplete attempt')
    f=read(root/'freeze.json');verify_source(f);c=f['config']
    if c not in (__import__(__package__+'.protocol',fromlist=['BASE']).BASE,__import__(__package__+'.protocol',fromlist=['FIXTURE']).FIXTURE):raise ValueError('unknown config')
    if f['config_identity']!=identity(c) or f['enumeration']!=enumeration(c):raise ValueError('freeze inventory mismatch')
    from .science import population_references,structured
    assert_close(f['population_references'],population_references(c))
    records=[]
    expected_dirs={f'cell-{i:04d}' for i in range(len(f['enumeration']['cells']))}
    if {p.name for p in root.glob('cell-*')}!=expected_dirs:raise ValueError('incomplete cell inventory')
    for i,cell in enumerate(f['enumeration']['cells']):
        if not (root/f'{i:04d}.START.json').is_file() or not (root/f'{i:04d}.END.json').is_file():raise ValueError('interrupted attempt')
        assert_close(read(root/f'{i:04d}.START.json')['cell'],cell)
        dest=root/f'cell-{i:04d}';done=read(dest/'COMPLETE.json')
        if done!={'arrays_sha':sha(dest/'arrays.npz'),'record_sha':sha(dest/'record.json')}:raise ValueError('tampered record/arrays')
        r=read(dest/'record.json');a=dict(np.load(dest/'arrays.npz',allow_pickle=False))
        seed,th,N=cell['seed'],cell['theta'],cell['N'];g,sigma,L,Li,pool,x=make_data(c,seed,th);x=x[:N];mu=x.mean(0);S=(x-mu).T@(x-mu)/N;K=g.K(th);Cs=L@K@L;truth=top(K,g.k);W0=Li@Li
        for name,want in {'training':x,'calibration':pool,'S':S,'mean':mu,'Sigma':sigma,'K':K,'Cs':Cs}.items():
            if not np.allclose(a[name],want,rtol=1e-10,atol=1e-10):raise ValueError('data/moment mismatch '+name)
        if (r['seed'],r['theta'],r['N'],r['training_sha'],r['calibration_sha'])!=(seed,th,N,digest(x),digest(pool)):raise ValueError('seed/data identities')
        covs={'iso':np.eye(g.d),'oracle':sigma};cd={}
        for m in c['m']:
            cc,dd=estimate(pool[:m],g.C,g.T);cd[str(m)]=dd
            covs.update({f'{n}_{m}':z for n,z in cc.items()})
        assert_close(r['covariance'],cd);methods={};expected_arrays=set(('S','mean','Sigma','Cs','K','training','calibration'))
        for name,cov in covs.items():
            expected_arrays.add(name+'_cov')
            if not np.allclose(a[name+'_cov'],cov):raise ValueError('covariance differs')
            W=np.linalg.inv(cov)
            for cls in ('restricted','unrestricted'):
                tag=name+'__'+cls;B=a[tag+'_B'];A=map_for(B,W);b=(np.eye(g.d)-A)@mu
                expected_arrays.update(tag+'_'+s for s in ('B','A','b'))
                if not np.allclose(a[tag+'_A'],A) or not np.allclose(a[tag+'_b'],b):raise ValueError('map mismatch')
                if np.linalg.matrix_rank(B)!=g.k:raise ValueError('rank mismatch')
                row=risks(A,b,Cs,sigma,W0)|representation(B,Li,truth)
                row['training_loss']=float(np.trace(W@S)-np.trace(np.linalg.solve(B.T@W@B,B.T@W@S@W@B)))
                if cls=='restricted':
                    opt=r['methods'][tag]['optimization'];rr=[];objectives=[]
                    initial=starts(S,g.C,g.T,g.k_c,g.k_t,(c['namespace'],seed,th,N))
                    replayB,replayOpt,replayFactors=structured(W,S,initial,c['max_iter'],c['gradient_tolerance'])
                    assert_close(opt,replayOpt)
                    if not np.allclose(B,replayB,atol=1e-9):raise ValueError('optimizer replay mismatch')
                    if len(opt['restarts'])!=4:raise ValueError('restart inventory')
                    for j in range(4):
                        uc,ut=a[tag+f'_uc_{j}'],a[tag+f'_ut_{j}'];expected_arrays.update([tag+f'_uc_{j}',tag+f'_ut_{j}'])
                        if not np.allclose(uc.T@uc,np.eye(g.k_c)) or not np.allclose(ut.T@ut,np.eye(g.k_t)):raise ValueError('factor orthogonality')
                        val,gc,gt=objective_gradient(uc,ut,W,S);objectives.append(val)
                        scale=np.trace(W@S);gn=np.sqrt(np.sum(((gc-uc@(uc.T@gc))/scale)**2)+np.sum(((gt-ut@(ut.T@gt))/scale)**2))
                        assert_close(opt['restarts'][j]['objective'],val);assert_close(opt['restarts'][j]['gradient_norm'],gn)
                        trajectory=opt['restarts'][j]['trajectory']
                        if not np.isfinite(trajectory).all() or np.min(np.diff(trajectory))<-1e-8:raise ValueError('invalid optimization trajectory')
                        if opt['restarts'][j]['status']=='stationary' and gn>c['gradient_tolerance']*1.001:raise ValueError('false stationarity')
                        ar=map_for(np.kron(ut,uc),W);br=(np.eye(g.d)-ar)@mu;rr.append(risks(ar,br,Cs,sigma,W0)['clean'])
                    selected=int(np.argmax(objectives))
                    if opt['selected']!=selected or not np.allclose(B,np.kron(a[tag+f'_ut_{selected}'],a[tag+f'_uc_{selected}'])):raise ValueError('class/selection mismatch')
                    row.update(optimization=opt,restart_clean_risks=rr,restart_risk_spread=float(max(rr)-min(rr)))
                else:
                    from .science import unrestricted
                    oracle=unrestricted(W,S,g.k)
                    if not np.allclose(qr(B)@qr(B).T,oracle@oracle.T,atol=1e-8):raise ValueError('unrestricted fit mismatch')
                methods[tag]=row
            if name not in ('iso','oracle'):methods[name+'__domain']=response(Li@S@Li,Li@cov@Li-np.eye(g.d),g.k)[0]
        if set(a)!=expected_arrays:raise ValueError('unexpected/missing arrays')
        assert_close(r['methods'],methods);records.append(r)
    return records


def reduce(root,out):
    root=Path(root)
    if not (root/'COMPLETE.json').is_file() or (root/'FAILED.json').exists():raise ValueError('incomplete/failed run')
    records=verify(root);c=read(root/'freeze.json')['config'];reduction=reduce_records(c,records)
    write_new(out,{'development_only':c['development_only'],**reduction})


def export(root,out):
    verify(root)
    if read(Path(root)/'freeze.json')['config']['development_only']:raise ValueError('fixture export forbidden')
    raise ValueError('production export blocked pending independent implementation acceptance')
