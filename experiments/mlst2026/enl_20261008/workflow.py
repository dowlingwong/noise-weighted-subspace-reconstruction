"""Source-bound E-NL execution/qualification and read-only saved-artifact auditing."""
import argparse, hashlib, json, os, platform, signal, sys, tarfile, time
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[2]
LEDGER='/ceph/dwong/enl_runs/registry/paper1-enl-v1-confirm.attempt.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)

def config(mode):
    if mode not in ['fixture','resource','confirmation']:raise ValueError('unknown mode')
    return read(PACKAGE/(mode+'.json'))

def sources():
    deps=read(PACKAGE/'dependency_sources.json')
    for rel,want in deps.items():
        if sha(ROOT/rel)!=want:raise ValueError('accepted dependency bytes differ: '+rel)
    own={str(p.relative_to(ROOT)):sha(p) for p in PACKAGE.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.sh']}
    return {**deps,**own,'pyproject.toml':sha(ROOT/'pyproject.toml')}

def no_cache():
    if os.getenv('PYTHONPYCACHEPREFIX'):raise ValueError('external executable cache prefix refused')
    for folder in ['experiments','src']:
        base=ROOT/folder
        if base.is_symlink() or any(p.is_symlink() for p in base.rglob('*')):raise ValueError('repository source/cache symlink refused')
        if any((ROOT/folder).rglob('*.pyc')):raise ValueError('repository executable bytecode cache refused')
    if not sys.dont_write_bytecode:raise ValueError('bytecode writes must be disabled')

def numerical():
    from . import runtime
    if not any(isinstance(x,runtime.Finder) for x in sys.meta_path):runtime.install()
    from . import science
    torch=science.models.import_torch()
    # Force optimizer imports without random initialization or fitting.
    warm=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))],lr=.001)
    del warm
    return science,torch

def environment():
    from experiments.mlst2026.tr09_20261007.workflow import environment as base_environment
    import torch, importlib.metadata
    from . import runtime
    env=base_environment();dist=importlib.metadata.distribution('torch')
    cached=dict(env['binary_metadata_sha256']);cached[env['executable']]=env['executable_sha256']
    executable_files=runtime.executable_files(sha,cached);binaries={}
    for relative in dist.files or []:
        rel=Path(relative)
        if not ('.so' in rel.name or rel.suffix in ['.py','.dylib','.dll'] or rel.name in ['METADATA','RECORD']):continue
        p=Path(dist.locate_file(relative)).resolve();key=str(p)
        if p.is_file():binaries[key]=executable_files[key] if key in executable_files else sha(p)
    env.update(executable_files=executable_files,source_only_imports=True,torch=torch.__version__,torch_config=torch.__config__.show(),torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),torch_dtype=str(torch.get_default_dtype()),torch_deterministic=torch.are_deterministic_algorithms_enabled(),torch_binaries=binaries)
    return env

def loaded_identity(bound):
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if not f or not Path(f).is_absolute() or not Path(f).is_file():continue
        p=Path(f).resolve()
        if name.split('.')[0] in ['experiments','noise_geometry'] and not p.is_relative_to(ROOT):raise ValueError('external repository module origin')
        if p.suffix=='.py' and p.is_relative_to(ROOT):
            rel=str(p.relative_to(ROOT))
            if rel not in bound or sha(p)!=bound[rel]:raise ValueError('unbound loaded source: '+rel)

def validate(c,mode):
    if c!=config(mode):raise ValueError('config differs from source-bound shipped design')
    cells=len(c['seeds'])*sum(len(Ns) for Ns in c['arms'].values())*len(c['p'])
    if len(c['seeds'])!=len(set(c['seeds'])) or cells!=c['budget']['max_cells'] or cells*c['recipe']['restarts']!=c['budget']['max_restarts'] or c['budget']['retries']!=0:raise ValueError('allocation census/budget mismatch')
    if mode!='confirmation' and set(c['seeds'])&set(config('confirmation')['seeds']):raise ValueError('confirmation seeds forbidden')
    if mode=='fixture' and c['geometry']['C']*c['geometry']['T']>=256:raise ValueError('production fixture geometry forbidden')
    if any(os.getenv(k)!='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('single thread environment required')

def ancestry(c):
    if not Path('/proc').is_dir():raise ValueError('Linux source-bound supervisor required')
    parent=os.getppid();status=Path(f'/proc/{parent}/status').read_text();grand=int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
    cmd=lambda pid:[os.fsdecode(x) for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    pc,gc=cmd(parent),cmd(grand)
    if not any(Path(x).name=='timeout' for x in pc) or str(c['budget']['run_seconds']) not in pc or '--kill-after=60' not in pc or str(PACKAGE/'supervisor.sh') not in gc:raise ValueError('source-bound supervisor ancestry refused')
    coordinator=int(os.getenv('ENL_COORDINATOR_PID','0'))
    gstatus=Path(f'/proc/{grand}/status').read_text();actual=int(next(x.split()[1] for x in gstatus.splitlines() if x.startswith('PPid:')))
    if coordinator!=actual or 'experiments.mlst2026.enl_20261008.coordinator' not in cmd(coordinator):raise ValueError('source-bound whole-campaign coordinator required')
    cs=Path(f'/proc/{coordinator}/status').read_text();outer=int(next(x.split()[1] for x in cs.splitlines() if x.startswith('PPid:')));oc=cmd(outer)
    if not any(Path(x).name=='timeout' for x in oc) or '259140' not in oc or '--kill-after=60' not in oc:raise ValueError('whole72h process-tree deadline required')
    return {'coordinator_pid':coordinator,'outer_cmd':oc,'parent_cmd':pc,'grandparent_cmd':gc,'parent_pid':parent,'grandparent_pid':grand,'timeout_seconds':c['budget']['run_seconds']}

def freeze(out,qualification):
    no_cache();c=config('confirmation');validate(c,'confirmation');before=sources();numerical();env=environment();loaded_identity(before)
    q=read(qualification)
    if q.get('qualified') is not True or q.get('source_sha256')!=before or q.get('config')!=c or q.get('runtime')!=env:raise ValueError('qualified exact-source runtime report required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write(out/'freeze.json',{'qualification_sha256':sha(qualification),'qualification_path':str(Path(qualification).resolve()),'mode':'confirmation','config':c,'sources':before,'environment':env,'ledger_path':LEDGER,'registered':False,'independent_review_accepted':False})

def authorize(mode,c,out,freeze_path,acceptance,ledger):
    """All standard-library refusal gates precede numerical imports or pool draws."""
    if mode!='confirmation':return None
    if os.getenv('ENL_CONFIRMATION_AUTHORISED')!='yes':raise ValueError('explicit confirmation authorization required')
    if not freeze_path or not acceptance or ledger!=LEDGER:raise ValueError('exact freeze/acceptance/permanent ledger required')
    if Path(out).exists() or Path(ledger).exists():raise ValueError('permanent attempt/output exists; no retry')
    f,a=read(freeze_path),read(acceptance)
    if f.get('mode')!='confirmation' or f.get('config')!=c or f.get('sources')!=sources() or f.get('ledger_path')!=LEDGER:raise ValueError('freeze source/config/ledger mismatch')
    qp=Path(f.get('qualification_path',''))
    if not qp.is_file() or sha(qp)!=f.get('qualification_sha256') or read(qp).get('qualified') is not True:raise ValueError('frozen resource qualification binding mismatch')
    if a.get('registered') is not True or a.get('verdict')!='ACCEPTED' or a.get('namespace')!=c['namespace'] or a.get('freeze_sha256')!=sha(freeze_path):raise ValueError('registered independent acceptance mismatch')
    rp=Path(a.get('review_path',''))
    if not rp.is_file() or a.get('review_sha256')!=sha(rp):raise ValueError('independent review byte binding mismatch')
    review=read(rp)
    if review.get('verdict')!='ACCEPTED' or review.get('namespace')!=c['namespace'] or review.get('freeze_sha256')!=sha(freeze_path) or review.get('qualification_sha256')!=f['qualification_sha256']:raise ValueError('independent review does not accept exact freeze/qualification')
    import resource
    vmem=resource.getrlimit(resource.RLIMIT_AS)[0];cpu=resource.getrlimit(resource.RLIMIT_CPU)[0]
    if vmem<0 or vmem>c['budget']['virtual_memory_bytes'] or cpu<0 or cpu>c['budget']['campaign_seconds']:raise ValueError('required virtual-memory/CPU limits absent')
    from . import runtime
    runtime.preimport(f['environment'],sha)
    return f, a, ancestry(c)

def usage(out):return sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())

def close(a,b):
    import numpy as np
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(float,int)) and bool(np.isclose(a,b,rtol=1e-9,atol=1e-9))
    return a==b

def run(mode,out,freeze_path=None,acceptance=None,ledger=None,timings=None):
    outer_begin=time.monotonic();pool_seconds=0.
    c=config(mode);validate(c,mode);no_cache();before=sources();binding=authorize(mode,c,out,freeze_path,acceptance,ledger)
    science,torch=numerical();env=environment();loaded_identity(before)
    if binding and binding[0]['environment']!=env:raise ValueError('frozen runtime mismatch')
    out=Path(out)
    if out.exists():raise ValueError('output already exists; no resume/retry')
    if binding:
        Path(ledger).parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(ledger,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump({'namespace':c['namespace'],'freeze_sha256':sha(freeze_path),'acceptance_sha256':sha(acceptance),'review_sha256':binding[1]['review_sha256'],'out':str(out.resolve()),'supervisor':binding[2],'time':time.time()},f)
    out.mkdir(parents=True,exist_ok=False)
    if binding:
        for name,path in [('FREEZE.json',freeze_path),('ACCEPTANCE.json',acceptance),('ATTEMPT.json',ledger),('REVIEW.json',binding[1]['review_path']),('QUALIFICATION.json',binding[0]['qualification_path'])]: (out/name).write_bytes(Path(path).read_bytes())
    for rel in before:
        p=out/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    write(out/'START.json',{'mode':mode,'time':time.time(),'config':c,'sources':before,'environment':env,'cell_order':science.order(c)})
    begin=time.monotonic();setup_seconds=begin-outer_begin;rows=[];pool=None;current_seed=None
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    try:
        for seed,k,N,p in science.order(c):
            if time.monotonic()-begin>c['budget']['run_seconds']:raise TimeoutError('campaign wall budget')
            if current_seed!=seed:
                pool_begin=time.monotonic();pool=science.raw_pools(c,seed);current_seed=seed
                import numpy as np
                np.savez_compressed(out/f'pool_{seed}.npz',**pool)
                write(out/f'pool_{seed}.json',{'seed':seed,'namespace':c['namespace'],'arrays':{key:{'shape':list(value.shape),'sha256':science.digest(value)} for key,value in pool.items()}})
            if current_seed==seed and 'pool_begin' in locals():
                pool_seconds+=time.monotonic()-pool_begin;del pool_begin
            def alarm(signum,frame):raise TimeoutError('cell time budget')
            signal.signal(signal.SIGALRM,alarm);signal.alarm(c['budget']['cell_seconds']);then=time.monotonic()
            row=science.fit(c,seed,k,N,p,pool,out);row['seconds']=time.monotonic()-then;signal.alarm(0)
            write(out/f'cell_{seed}_{k}_{N}_{p}.json',row);rows.append(row)
            if usage(out)>c['budget']['scratch_bytes']//2-16*2**20:raise RuntimeError('scratch reserve for archive exceeded')
            print(seed,k,N,p,round(row['seconds'],3),flush=True)
        if before!=sources():raise ValueError('source changed during campaign')
        write(out/'SUMMARY.json',science.reduce(c,rows))
        internal_audit_timing={};audit(out,in_campaign=True,report=out/'AUDIT.json',timings=internal_audit_timing)
        loaded_identity(before);no_cache()
        write(out/'EXIT.json',{'exit_code':0,'seconds':time.monotonic()-begin,'source_unchanged':before==sources()})
        export_begin=time.monotonic()
        if mode!='confirmation':export(out)
        if timings is not None:timings.update(setup_seconds=setup_seconds,pool_seconds=pool_seconds,internal_audit=internal_audit_timing,export_seconds=time.monotonic()-export_begin,total_seconds=time.monotonic()-outer_begin)
    except BaseException as error:
        signal.alarm(0)
        failure_name='FAILURE.json' if (out/'EXIT.json').exists() else 'EXIT.json'
        write(out/failure_name,{'exit_code':1,'seconds':time.monotonic()-begin,'error':repr(error),'source_unchanged':before==sources()});raise


def audit(out,in_campaign=False,report=None,timings=None):
    no_cache();science,torch=numerical();import numpy as np
    audit_begin=time.monotonic();out=Path(out);start=read(out/'START.json');c=start['config'];mode=start['mode'];validate(c,mode)
    if start['sources']!=sources() or start['environment']!=environment():raise ValueError('source/runtime audit mismatch')
    if not in_campaign and read(out/'EXIT.json').get('exit_code')!=0:raise ValueError('failed/incomplete scientific exit')
    if mode=='confirmation':
        f,a,attempt=read(out/'FREEZE.json'),read(out/'ACCEPTANCE.json'),read(out/'ATTEMPT.json')
        if f['config']!=c or f['sources']!=start['sources'] or f['environment']!=start['environment'] or a.get('registered') is not True or a.get('verdict')!='ACCEPTED' or a.get('freeze_sha256')!=sha(out/'FREEZE.json') or attempt.get('freeze_sha256')!=sha(out/'FREEZE.json') or attempt.get('acceptance_sha256')!=sha(out/'ACCEPTANCE.json') or a.get('namespace')!=c['namespace'] or attempt.get('namespace')!=c['namespace'] or attempt.get('review_sha256')!=a.get('review_sha256') or attempt.get('supervisor',{}).get('timeout_seconds')!=c['budget']['run_seconds']:raise ValueError('attempt/registration/supervision binding mismatch')
        review=read(out/'REVIEW.json');q=read(out/'QUALIFICATION.json')
        if sha(out/'REVIEW.json')!=a.get('review_sha256') or review.get('verdict')!='ACCEPTED' or review.get('freeze_sha256')!=sha(out/'FREEZE.json') or review.get('namespace')!=c['namespace'] or review.get('qualification_sha256')!=f.get('qualification_sha256') or sha(out/'QUALIFICATION.json')!=f.get('qualification_sha256') or q.get('qualified') is not True or q.get('source_sha256')!=start['sources'] or q.get('runtime')!=start['environment'] or q.get('config')!=c:raise ValueError('review/resource receipt binding mismatch')
        if not in_campaign:
            s=read(out/'SUPERVISOR_EXIT.json')
            if s.get('exit_code')!=0 or s.get('attempt_sha256')!=sha(out/'ATTEMPT.json'):raise ValueError('sealed supervisor exit mismatch')
    setup_seconds=time.monotonic()-audit_begin
    rows=[];required={'START.json','SUMMARY.json'}|{'source_snapshot/'+rel for rel in start['sources']}
    if mode=='confirmation':required|={'FREEZE.json','ACCEPTANCE.json','ATTEMPT.json','REVIEW.json','QUALIFICATION.json'}
    if not in_campaign:required|={'EXIT.json','AUDIT.json','SHA256SUMS'}|({'SUPERVISOR_EXIT.json'} if mode=='confirmation' else set())
    for rel,h in start['sources'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or sha(out/'source_snapshot'/rel)!=h:raise ValueError('snapshot bytes mismatch')
    for seed in c['seeds']:
        required|={f'pool_{seed}.npz',f'pool_{seed}.json'}
        with np.load(out/f'pool_{seed}.npz',allow_pickle=False) as a:pool={key:a[key] for key in a.files}
        pd=read(out/f'pool_{seed}.json')
        if set(pd['arrays'])!=set(pool) or pd['seed']!=seed or pd['namespace']!=c['namespace']:raise ValueError('pool array/identity census')
        for role,n in [('train',max(N for Ns in c['arms'].values() for N in Ns)),('evaluation',c['eval_records']),('representation',c['representation_records'])]:
            for suffix,width in [('G',c['geometry']['C']*c['geometry']['T']),('E',c['geometry']['C']*c['geometry']['T']),('U',c['n_sparse'])]:
                if pool[role+'_'+suffix].shape!=(n,width):raise ValueError('declared raw pool shape')
            if np.any(pool[role+'_U']<0) or np.any(pool[role+'_U']>=1):raise ValueError('activity uniform outside support')
        for key,v in pool.items():
            if list(v.shape)!=pd['arrays'][key]['shape'] or science.digest(v)!=pd['arrays'][key]['sha256'] or not np.isfinite(v).all():raise ValueError('raw pool digest/shape/finite')
        for k,N,p in science.locations(c):
            stem=f'cell_{seed}_{k}_{N}_{p}';required|={stem+'.json',stem+'_linear.npz'};row=read(out/(stem+'.json'))
            if (row['seed'],row['k'],row['N'],row['p'])!=(seed,k,N,p):raise ValueError('cell identity mismatch')
            m=science.SparseLatentModel(c['geometry'],c['theta'],p,c['n_sparse']);Y,_,_=science.data(pool,'train',m);ntrain=N-int(.2*N)
            if row['n_train']!=ntrain or row['n_validation']!=N-ntrain or science.digest(Y[:ntrain])!=row['train_digest'] or science.digest(Y[ntrain:N])!=row['validation_digest']:raise ValueError('split/prefix digest mismatch')
            with np.load(out/(stem+'_linear.npz'),allow_pickle=False) as a:
                mean=Y[:ntrain].mean(0);S=(Y[:ntrain]-mean).T@(Y[:ntrain]-mean)/ntrain;V=a['eigenvectors'];ell=a['eigenvalues']
                if not np.allclose(a['cov'],S,atol=1e-9) or not np.allclose(a['mean'],mean,atol=1e-9) or not np.allclose(V.T@V,np.eye(m.d),atol=1e-9) or not np.allclose(S@V,V*ell,atol=1e-8) or np.any(np.diff(ell)>1e-10):raise ValueError('saved eigensystem/covariance mismatch')
                expected_Ap=V[:,:k]@V[:,:k].T;gains=np.where(ell[:k]>1,(ell[:k]-1)/np.where(ell[:k]>1,ell[:k],1),0);expected_As=(V[:,:k]*gains)@V[:,:k].T
                if not close(row['proj_info'],{'sample_eigenvalues':ell[:k+1].tolist()}) or not close(row['shrink_info'],{'sample_eigenvalues':ell[:k].tolist(),'gains':gains.tolist(),'zero_gain_count':int(np.sum(gains==0))}):raise ValueError('baseline diagnostic metadata mismatch')
                if not np.allclose(a['Ap'],expected_Ap,atol=1e-8) or not np.allclose(a['As'],expected_As,atol=1e-8):raise ValueError('rank/gain spectral selection mismatch')
                for label,A,b in [('projector',a['Ap'],a['bp']),('shrinkage',a['As'],a['bs'])]:
                    mean=Y[:ntrain].mean(0);S=(Y[:ntrain]-mean).T@(Y[:ntrain]-mean)/ntrain
                    if not np.allclose((np.eye(m.d)-A)@mean,b,rtol=1e-9,atol=1e-9):raise ValueError('baseline intercept mismatch')
                    # Independent Frobenius signal/noise risk, not production trace implementation.
                    R=float(np.linalg.norm((A-np.eye(m.d))@(m.V*np.sqrt(m.mu)),'fro')**2+np.linalg.norm(A,'fro')**2+b@b)
                    name='R_proj' if label=='projector' else 'R_shrink'
                    if not np.isclose(R,row[name],rtol=1e-9,atol=1e-9):raise ValueError('affine risk mismatch')
                    if not np.allclose(A,A.T,atol=1e-9) or not np.allclose(A@S,S@A,atol=1e-8):raise ValueError('sample eigenmap certificate')
                saved_baselines={key:a[key].copy() for key in a.files}
            groups={name:science.diagnostics.group_sums(science.diagnostics.affine_per_coordinate(saved_baselines[ak],saved_baselines[bk],m),m,k) for name,ak,bk in [('projector','Ap','bp'),('shrinkage','As','bs')]}
            if not close(groups,row['baseline_groups']):raise ValueError('baseline group attribution mismatch')
            ok=[]
            if len(row['restarts'])!=c['recipe']['restarts']:raise ValueError('restart count')
            X=m.physical(Y[:N]);Yev,Sev,_=science.data(pool,'evaluation',m)
            for r,rr in enumerate(row['restarts']):
                if rr['index']!=r or rr['init_key']!=science.models.key_int(c['namespace'],seed,k,N,p,'init',r):raise ValueError('restart key mismatch')
                if rr['checkpoint'] is None:
                    if 'failed' not in rr['fit']:raise ValueError('unexplained failed restart')
                    continue
                name=stem+f'_restart{r}.npz'
                if rr['checkpoint']!=name:raise ValueError('checkpoint path mismatch')
                required|={name,stem+f'_evaluation{r}.npz'};ae=science.load_model(torch,c,k,out/name)
                val,grad=science.validation(torch,ae,X[ntrain:],m.Li,c['recipe']['batch']);fit=rr['fit'];history=fit['validation_history']
                if not history or not np.isfinite(history).all():raise ValueError('nonfinite/empty validation trajectory')
                bad=0;best=float('inf');early=False
                for j,v in enumerate(history):
                    bad=0 if v<best else bad+1;best=min(best,v)
                    if bad>=c['recipe']['patience'] and j<len(history)-1:early=True
                stop='patience' if bad>=c['recipe']['patience'] else 'max_epochs'
                if early or fit['stopped']!=stop or (stop=='max_epochs' and len(history)!=c['recipe']['max_epochs']) or not np.isclose(val,fit['best_validation_loss'],rtol=1e-9,atol=1e-9):raise ValueError('selection loss/stop metadata mismatch')
                if len(history)!=fit['epochs_run'] or fit['epochs_run']>c['recipe']['max_epochs'] or fit['best_epoch']!=int(np.argmin(history)) or not np.isclose(val,min(history),rtol=1e-9,atol=1e-9) or not np.isclose(grad,rr['validation_gradient_norm'],rtol=1e-8,atol=1e-9):raise ValueError('checkpoint/trajectory/gradient mismatch')
                ev,per=science.evaluate(torch,ae,m,k,c,pool)
                if not close(ev,rr['evaluation']):raise ValueError('saved restart evaluation mismatch')
                with np.load(out/(stem+f'_evaluation{r}.npz'),allow_pickle=False) as a:
                    ep=((Yev@saved_baselines['Ap'].T+saved_baselines['bp']-Sev)**2).sum(1);es=((Yev@saved_baselines['As'].T+saved_baselines['bs']-Sev)**2).sum(1)
                    if not np.allclose(per,a['AE'],rtol=1e-9,atol=1e-9) or not np.allclose(ep,a['projector'],rtol=1e-9,atol=1e-9) or not np.allclose(es,a['shrinkage'],rtol=1e-9,atol=1e-9):raise ValueError('paired evaluation mismatch')
                    for label,v in [('projector',ep),('shrinkage',es)]:
                        if not np.isclose((per-v).std(ddof=1)/np.sqrt(len(per)),rr['paired_mcse'][label],rtol=1e-9,atol=1e-9):raise ValueError('paired MCSE mismatch')
                ok.append(r)
            if not ok:raise ValueError('no valid restart')
            sel=min(ok,key=lambda r:row['restarts'][r]['fit']['best_validation_loss'])
            if sel!=row['selected'] or row['optimization_limited']!=(row['restarts'][sel]['fit']['stopped']=='max_epochs') or not close(row['evaluation'],row['restarts'][sel]['evaluation']) or not np.isclose(row['R_AE'],row['evaluation']['R_AE']):raise ValueError('validation-only selection mismatch')
            if not close(row['contrasts'],{'projector':row['R_AE']-row['R_proj'],'shrinkage':row['R_AE']-row['R_shrink']}):raise ValueError('contrast mismatch')
            refs={name:science.interference(f,m,c) for name,f in [('projector',lambda y:y@saved_baselines['Ap'].T+saved_baselines['bp']),('shrinkage',lambda y:y@saved_baselines['As'].T+saved_baselines['bs']),('bayes',lambda y:science.bayes.posterior_mean(m,y))]}
            if not close(refs,row['reference_interference']):raise ValueError('interference reference mismatch')
            rows.append(row)
    if not close(science.reduce(c,rows),read(out/'SUMMARY.json')):raise ValueError('raw model summary mismatch')
    if start['cell_order']!=[list(x) for x in science.order(c)]:raise ValueError('cell order mismatch')
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name!='return.tgz'}
    if actual!=required:raise ValueError('complete receipt census mismatch')
    loaded_identity(start['sources']);no_cache()
    if timings is not None:timings.update(setup_seconds=setup_seconds,total_seconds=time.monotonic()-audit_begin)
    if report is not None:write(report,{'setup_seconds':setup_seconds,'accepted':True,'cells':len(rows),'restarts':sum(len(r['restarts']) for r in rows),'read_only_model_pool_re_evaluation':True,'seconds':time.monotonic()-audit_begin})
    return rows

def export(out):
    out=Path(out)
    if (out/'return.tgz').exists() or (out/'SHA256SUMS').exists():raise ValueError('receipt already sealed; immutable')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    if out.is_symlink() or any(p.is_symlink() for p in out.rglob('*')):raise ValueError('unsafe receipt symlink')
    (out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in files))
    with tarfile.open(out/'return.tgz','w:gz') as a:
        for p in files+[out/'SHA256SUMS']:
            a.add(p,arcname=str(p.relative_to(out)),recursive=False)
            if usage(out)>read(out/'START.json')['config']['budget']['scratch_bytes']:raise RuntimeError('live archive scratch cap exceeded')

def verify_receipt(out):
    out=Path(out)
    if out.is_symlink() or any(p.is_symlink() for p in out.rglob('*')):raise ValueError('unsafe receipt symlink')
    entries={}
    for line in (out/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1);p=Path(rel)
        if p.is_absolute() or '..' in p.parts or rel in entries or sha(out/p)!=h:raise ValueError('unsafe/duplicate/checksum receipt')
        entries[rel]=h
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS','return.tgz']}
    if actual!=set(entries):raise ValueError('manifest census mismatch')
    with tarfile.open(out/'return.tgz') as a:
        members=a.getmembers()
        if len(members)!=len(entries)+1 or {m.name for m in members}!=set(entries)|{'SHA256SUMS'}:raise ValueError('archive census')
        for m in members:
            if not m.isfile():raise ValueError('unsafe archive type')
            stream=a.extractfile(m);h=hashlib.sha256()
            for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
            if h.hexdigest()!=(sha(out/'SHA256SUMS') if m.name=='SHA256SUMS' else entries[m.name]):raise ValueError('archive member checksum')
    return entries

def seal(out,code):
    out=Path(out)
    if (out/'SUPERVISOR_EXIT.json').exists() or (out/'return.tgz').exists():raise ValueError('attempt already sealed')
    write(out/'SUPERVISOR_EXIT.json',{'exit_code':code,'attempt_sha256':sha(out/'ATTEMPT.json'),'time':time.time()});export(out)

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['run','freeze','audit','seal']);a.add_argument('--mode',choices=['fixture','resource','confirmation']);a.add_argument('--out',required=True);a.add_argument('--freeze');a.add_argument('--qualification');a.add_argument('--acceptance');a.add_argument('--ledger');a.add_argument('--exit-code',type=int);x=a.parse_args()
    if x.action=='run':run(x.mode,x.out,x.freeze,x.acceptance,x.ledger)
    elif x.action=='freeze':
        if not x.qualification:raise ValueError('qualification report required')
        freeze(x.out,x.qualification)
    elif x.action=='audit':verify_receipt(x.out);audit(x.out)
    else:
        if x.exit_code is None:raise ValueError('supervisor exit required')
        seal(x.out,x.exit_code)
if __name__=='__main__':main()
