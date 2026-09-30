"""Additive T25 deployment. One bounded attempt, including replay/reduction/export.

Scientific settings/seeds are inherited unchanged. Production requires a byte-bound
independent acceptance and actual-host qualification. No resume or automatic retry.
"""
from pathlib import Path
import hashlib,json,os,platform,shutil,signal,socket,subprocess,sys,time,threading,re,secrets,io,contextlib
import numpy as np
from .protocol import BASE,FIXTURE,config,enumeration
from .science import fit_cell,population_references,reduce_records
ROOT=Path(__file__).resolve().parents[3]
PACKAGE=Path(__file__).resolve().parent
MODULE='experiments.mlst2026.t25_production_20260930'
VERSION='t25-deployment/20260930-v1'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def identity(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write_new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def msroot():return Path(os.environ.get('MLST_MS_ROOT',str(ROOT.parent/'arXiv-v1-polished-extended-preprint'))).resolve()
def external(p):
    p=Path(p).resolve()
    if p.is_relative_to(ROOT) or p.is_relative_to(msroot()):raise ValueError('output inside source tree')
    return p
def environment():
    import scipy,psutil
    try:
        from threadpoolctl import threadpool_info
        # Load order differs between processes; identity must not depend on it.
        pools=sorted(threadpool_info(),key=lambda r:json.dumps(r,sort_keys=True))
    except ImportError:pools=None
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):np.show_config()
    return {'python':sys.version,'platform':platform.platform(),'executable':sys.executable,
            'numpy':np.__version__,'scipy':scipy.__version__,'psutil':psutil.__version__,
            'import_origins':{'numpy':np.__file__,'scipy':scipy.__file__,'psutil':psutil.__file__},
            'blas_build':stream.getvalue(),'threadpools':pools,
            'dont_write_bytecode':bool(sys.flags.dont_write_bytecode),'package_origin':str(PACKAGE)}

def git(root,args):
    return subprocess.check_output(['git','-c','core.fsmonitor=false']+args,cwd=root,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).strip()
def source_files():
    return {k:sources(include_git=False)[k] for k in ('EXP_files','MS_files')}

def sources(include_git=True):
    files=sorted(set(list((ROOT/'src').rglob('*.py'))+list((ROOT/'experiments').rglob('*.py'))+list(PACKAGE.glob('*.json'))+[ROOT/'pyproject.toml']))
    msfiles=['03_theory.tex','app_main_proofs.tex','plan/T25_INTERACTION_PROTOCOL_2026-09-29.md','plan/T25_PRODUCTION_ADDENDUM_2026-09-30.md']
    out={'EXP_files':{str(p.relative_to(ROOT)):sha(p) for p in files},'MS_files':{p:sha(msroot()/p) for p in msfiles}}
    if not include_git:return out
    return {**out,'EXP_head':git(ROOT,['rev-parse','HEAD']),'MS_head':git(msroot(),['rev-parse','HEAD']),'EXP_status':git(ROOT,['status','--short']),'MS_status':git(msroot(),['status','--short'])}
def code_identity(s):return identity({k:s[k] for k in ('EXP_files','MS_files')})
def verify_source(f):
    if sources()!=f['sources'] or environment()!=f['environment']:raise ValueError('source/environment changed since freeze')
def thread_gate():
    for k in THREADS:
        if os.environ.get(k)!='1':raise ValueError('one numerical thread required: '+k)

def validate_freeze(f,portable=False):
    """Execution role: full host/source/environment gate. Portable role (receiver
    that is not the qualified host): frozen identities and current source-file
    bytes only; environment, host and Git state are not compared."""
    if set(f)!={'deployment','status','config','config_identity','enumeration','population_references','sources','environment','acceptance','qualification'}:raise ValueError('freeze schema')
    c=f['config']
    if f['deployment']!=VERSION or c not in (BASE,FIXTURE) or f['config_identity']!=identity(c) or f['enumeration']!=enumeration(c):raise ValueError('freeze identity/config')
    if f['status']!=('fixture_only' if c['development_only'] else 'confirmation'):raise ValueError('freeze status')
    if portable:
        if not c['development_only']:frozen_evidence_identities(f)
        if code_identity(source_files())!=code_identity(f['sources']):raise ValueError('portable audit must run from the frozen source bytes')
        return
    if not c['development_only']:
        q=f['qualification']
        if q.get('host')!=socket.gethostname():raise ValueError('host-scoped operation: run on the qualified host '+str(q.get('host'))+'; use the portable audit elsewhere')
        production_gate(f['acceptance'],f['qualification'],f['sources'],f['environment'])
    verify_source(f)

def frozen_evidence_identities(f):
    """Recorded acceptance/qualification identities, checkable without the host."""
    a,q,cid=f['acceptance'],f['qualification'],code_identity(f['sources'])
    if a.get('verdict')!='ACCEPTED_PRODUCTION_IMPLEMENTATION' or a.get('deployment')!=VERSION or a.get('code_identity')!=cid:raise ValueError('frozen acceptance identity')
    if q.get('code_identity')!=cid or q.get('execution_mode')!='direct' or q.get('environment')!=f['environment'] or not q.get('host'):raise ValueError('frozen qualification identity')
    for record,names in [(a,('independent_review',)),(q,('fixture_report','fixture_log','benchmark_report','author_record','snapshot_record'))]:
        for n in names:
            if not re.fullmatch('[0-9a-f]{64}',str(record.get(n+'_sha256',''))):raise ValueError('frozen evidence hash absent: '+n)

def evidence(record,name,json_body=True):
    path=record.get(name+'_path');digest=record.get(name+'_sha256')
    if not isinstance(path,str) or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('evidence identity absent: '+name)
    p=Path(path)
    if not p.is_file() or sha(p)!=digest:raise ValueError('evidence missing/changed: '+name)
    if json_body:
        if p.stat().st_size>4*1024**2:raise ValueError('oversized evidence')
        return read(p)
    return p

def production_gate(acceptance,qualification,s,env):
    if acceptance.get('verdict')!='ACCEPTED_PRODUCTION_IMPLEMENTATION' or acceptance.get('deployment')!=VERSION or acceptance.get('code_identity')!=code_identity(s):raise ValueError('independent production acceptance missing or stale')
    review=evidence(acceptance,'independent_review',False).read_text(errors='strict')
    if code_identity(s) not in review or 'ACCEPTED_PRODUCTION_IMPLEMENTATION' not in review or 'NOT ACCEPTED' in review:raise ValueError('independent review does not accept this exact code identity')
    if env.get('dont_write_bytecode') is not True:raise ValueError('PYTHONDONTWRITEBYTECODE=1 required for a bound source tree')
    q=qualification
    if q.get('host')!=socket.gethostname() or q.get('execution_mode')!='direct' or q.get('code_identity')!=code_identity(s) or q.get('environment')!=env:raise ValueError('host/source/environment qualification missing or stale')
    fixture=evidence(q,'fixture_report');evidence(q,'fixture_log',False)
    benchmark=evidence(q,'benchmark_report');author=evidence(q,'author_record');snapshot=evidence(q,'snapshot_record')
    if fixture.get('exit')!=0 or fixture.get('tests_passed',0)<42 or fixture.get('code_identity')!=code_identity(s) or fixture.get('environment')!=env or fixture.get('log_sha256')!=q['fixture_log_sha256']:raise ValueError('fixture qualification failed')
    if fixture.get('peak_rss_bytes',float('inf'))>BASE['budget']['rss_bytes']:raise ValueError('fixture RSS cap')
    if author.get('host')!='deepthought' or author.get('execution_mode')!='direct' or author.get('protocol_resource_caps_remain') is not True:raise ValueError('author direct-run record absent')
    if snapshot.get('code_identity')!=code_identity(s) or snapshot.get('EXP_root')!=str(ROOT) or snapshot.get('MS_root')!=str(msroot()) or snapshot.get('writable_source_files')!=[] or snapshot.get('stale_bytecode')!=[]:raise ValueError('isolated snapshot identity absent or not immutable')
    for role,root in [('EXP',ROOT),('MS',msroot())]:
        for relative in s[role+'_files']:
            if (root/relative).stat().st_mode & 0o222:raise ValueError('source file is writable: '+relative)
    if not env['threadpools'] or any(p['num_threads']!=1 for p in env['threadpools']):raise ValueError('actual BLAS thread count unqualified')
    import psutil
    if psutil.virtual_memory().total<8*1024**3:raise ValueError('hardware RAM below requirement')
    scratch=q.get('scratch_root')
    if not scratch or shutil.disk_usage(scratch).free<BASE['budget']['scratch_bytes']:raise ValueError('scratch qualification failed')
    if benchmark.get('code_identity')!=code_identity(s) or benchmark.get('environment')!=env or benchmark.get('scientific_endpoints_evaluated') is not False or not 0<benchmark.get('projected_lifecycle_seconds_with_margin',0)<=BASE['budget']['campaign_seconds'] or benchmark.get('worst_cell_within_half_cap') is not True or not 0<benchmark.get('peak_rss_bytes',0)<=BASE['budget']['rss_bytes']:raise ValueError('lifecycle benchmark absent/stale/too slow')

def freeze(c,out,acceptance=None,qualification=None):
    thread_gate();out=external(out);s=sources();env=environment()
    a=read(acceptance) if acceptance else {};q=read(qualification) if qualification else {}
    if not c['development_only']:production_gate(a,q,s,env)
    f={'deployment':VERSION,'status':'fixture_only' if c['development_only'] else 'confirmation','config':c,'config_identity':identity(c),'enumeration':enumeration(c),'population_references':population_references(c),'sources':s,'environment':env,'acceptance':a,'qualification':q}
    validate_freeze(f);write_new(out,f)

def tree_bytes(root):
    total=0
    for base,dirs,files in os.walk(root,followlinks=False):
        for n in dirs+files:
            p=Path(base)/n
            if p.is_symlink():raise ValueError('symlink in attempt tree')
        for n in files:total+=(Path(base)/n).stat().st_size
    return total

def kill_group(p):
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    p.wait()

def parent_watch(group=False):
    parent=os.getppid()
    def watch():
        while True:
            if os.getppid()!=parent or parent==1:
                if group:os.killpg(os.getpgrp(),signal.SIGKILL)
                else:os.kill(os.getpid(),signal.SIGKILL)
            time.sleep(.2)
    threading.Thread(target=watch,daemon=True).start()

class Interrupted(BaseException):pass

def _raise_interrupt(signum,frame):raise Interrupted('supervisor received signal '+str(signum))

def supervise(argv,log,root,budget,extra_paths=(),extra_env=None):
    """One child group; all descendant RSS, wall time and total scratch monitored.
    SIGTERM/SIGHUP/SIGINT to the supervisor terminate the whole group and fail the
    attempt; surviving group members after leader exit fail the attempt."""
    import psutil
    start=time.monotonic();peak=0;p=None;saved={}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):saved[sig]=signal.signal(sig,_raise_interrupt)
    try:
        with Path(log).open('x') as stream:
            p=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,**(extra_env or {})})
            proc=psutil.Process(p.pid)
            while p.poll() is None:
                try:
                    children=[proc]+proc.children(recursive=True);rss=0
                    for child in children:
                        try:rss+=child.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,rss)
                except psutil.NoSuchProcess:pass
                used=tree_bytes(root)+sum(tree_bytes(p) for p in extra_paths)
                if peak>budget['rss_bytes'] or time.monotonic()-start>budget['seconds'] or used>budget['scratch_bytes']:raise ValueError('resource cap exceeded')
                time.sleep(.1)
            # A leader that abandons live descendants is a failed attempt.
            survivors=[]
            for candidate in psutil.process_iter(['pid','status']):
                try:
                    if candidate.pid!=p.pid and os.getpgid(candidate.pid)==p.pid and candidate.status()!=psutil.STATUS_ZOMBIE:survivors.append(candidate.pid)
                except (psutil.NoSuchProcess,ProcessLookupError,PermissionError):pass
            if survivors:
                kill_group(p);raise ValueError('orphan descendants after group leader exit')
            elapsed=time.monotonic()-start;used=tree_bytes(root)+sum(tree_bytes(p) for p in extra_paths)
            if p.returncode:raise ValueError('worker exit '+str(p.returncode)+'; attempt cannot resume')
            if elapsed>budget['seconds'] or used>budget['scratch_bytes'] or peak>budget['rss_bytes']:raise ValueError('final resource cap exceeded')
            return {'elapsed_seconds':elapsed,'peak_tree_rss_bytes':peak,'scratch_bytes':used,'exit':0}
    finally:
        if p is not None:kill_group(p)
        for sig,handler in saved.items():signal.signal(sig,handler)

def worker_nonce(start_record):
    nonce=os.environ.get('PAPER1_T25_WORKER_NONCE','')
    if not nonce or identity(nonce)!=start_record.get('worker_nonce_identity') or os.getppid()!=start_record.get('supervisor_pid'):raise ValueError('internal worker lacks its supervisor reservation')
    return nonce

def worker_cell(fpath,index,out):
    parent_watch();thread_gate();f=read(fpath);validate_freeze(f);row=f['enumeration']['cells'][index]
    root=Path(fpath).parent;reservation=read(root/f'{index:04d}.START.json')
    nonce=os.environ.get('PAPER1_T25_WORKER_NONCE','')
    if not nonce or identity(nonce)!=read(root/'ATTEMPT.START.json')['worker_nonce_identity'] or os.getppid()!=read(root/'WORKER.START.json')['pid'] or reservation['cell']!=row or Path(out).resolve()!=root.resolve()/f'cell-{index:04d}':raise ValueError('worker lacks matching parent/reservation')
    if (root/f'{index:04d}.END.json').exists() or Path(out).exists():raise ValueError('duplicate worker claim')
    record,arrays=fit_cell(f['config'],row['seed'],row['theta'],row['N'])
    out=Path(out);out.mkdir(exist_ok=False)
    with (out/'arrays.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    write_new(out/'record.json',record);verify_source(f)
    write_new(out/'COMPLETE.json',{'arrays_sha':sha(out/'arrays.npz'),'record_sha':sha(out/'record.json')})

def worker_campaign(root):
    parent_watch(group=True);thread_gate();root=Path(root);f=read(root/'freeze.json');validate_freeze(f);c=f['config'];b=c['budget']
    reservation=read(root/'ATTEMPT.START.json');nonce=os.environ.get('PAPER1_T25_WORKER_NONCE','')
    if not nonce or identity(nonce)!=reservation['worker_nonce_identity'] or os.getppid()!=reservation['supervisor_pid']:raise ValueError('campaign worker lacks parent reservation')
    write_new(root/'WORKER.START.json',{'pid':os.getpid(),'supervisor_pid':os.getppid(),'nonce_identity':identity(nonce)})
    began=time.monotonic()
    for i,row in enumerate(f['enumeration']['cells']):
        verify_source(f);write_new(root/f'{i:04d}.START.json',{'cell':row,'reserved_charges':53,'timestamp':time.time()})
        p=None;start=time.monotonic()
        try:
            with (root/f'{i:04d}.log').open('x') as stream:
                # Same process group as campaign: outer watchdog can kill the full tree.
                p=subprocess.Popen([sys.executable,'-m',MODULE,'_cell','--freeze',str(root/'freeze.json'),'--index',str(i),'--out',str(root/f'cell-{i:04d}')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                while p.poll() is None:
                    if time.monotonic()-start>b['cell_seconds']:p.kill();p.wait();raise ValueError('paired-cell wall cap')
                    time.sleep(.05)
                if p.returncode:raise ValueError('paired-cell failed '+str(i))
        finally:
            if p is not None and p.poll() is None:p.kill();p.wait()
        write_new(root/f'{i:04d}.END.json',{'status':'COMPLETE','elapsed':time.monotonic()-start,'charges':53})
    verify_source(f)
    write_new(root/'FIT_COMPLETE.json',{'cells':len(f['enumeration']['cells']),'elapsed':time.monotonic()-began})
    write_new(root/'VERIFY.START.json',{'reserved_representation_replay_charges':f['enumeration']['representation_charges'],'retries':0})
    from .reconstruction import verify
    records=verify(root,require_complete=False)
    write_new(root/'VERIFY.END.json',{'cells_replayed':len(records),'all_restarts_replayed':True})
    write_new(root/'REDUCE.START.json',{'cell_count':len(records)})
    reduction=reduce_records(c,records)
    domains=[v['domain'] for r in records for k,v in r['methods'].items() if k.endswith('__domain')]
    reduction['domain_census']={k:domains.count(k) for k in ('in_domain','out_of_domain','undefined_gap')}
    if sum(reduction['domain_census'].values())!=len(records)*6:raise ValueError('domain census incomplete')
    write_new(root/'reduction.json',{'development_only':c['development_only'],**reduction})
    write_new(root/'REDUCE.END.json',{'reduction_sha256':sha(root/'reduction.json')})
    verify_source(f)
    # This worker receipt is necessary but insufficient: supervisor adds COMPLETE
    # only after the entire fit+replay+reduction lifetime passes resource checks.
    write_new(root/'WORKER_COMPLETE.json',{'development_only':c['development_only'],'cells':len(records),'elapsed':time.monotonic()-began})

def execute(fpath,root):
    thread_gate();f=read(fpath);validate_freeze(f);root=external(root);root.mkdir(parents=True,exist_ok=False)
    nonce=secrets.token_hex(32)
    write_new(root/'freeze.json',f);write_new(root/'ATTEMPT.START.json',{'deployment':VERSION,'confirmation':not f['config']['development_only'],'reserved_fit_charges':f['enumeration']['total_charges'],'reserved_replay_charges':f['enumeration']['total_charges'],'retries':0,'timestamp':time.time(),'supervisor_pid':os.getpid(),'worker_nonce_identity':identity(nonce)})
    b=f['config']['budget'];budget={'seconds':b['campaign_seconds'],'rss_bytes':b['rss_bytes'],'scratch_bytes':b['scratch_bytes']}
    start=time.monotonic()
    try:
        report=supervise([sys.executable,'-m',MODULE,'_campaign','--root',str(root)],root/'campaign.log',root,budget,extra_env={'PAPER1_T25_WORKER_NONCE':nonce})
        if not (root/'WORKER_COMPLETE.json').is_file():raise ValueError('missing worker completion')
        verify_source(f);write_new(root/'COMPLETE.json',{'development_only':f['config']['development_only'],'cells':f['enumeration']['paired_cells'],'resources':report,'deployment':VERSION})
        write_new(root/'ATTEMPT.END.json',{'elapsed_seconds':time.monotonic()-start,'status':'COMPLETE'})
    except BaseException as e:
        write_new(root/'FAILED.json',{'error':str(e),'elapsed_seconds':time.monotonic()-start,'retry_allowed':False});raise

def inventory(root):
    root=Path(root);files={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('symlink in export')
        if p.is_file():files[str(p.relative_to(root))]=sha(p)
    return files

def finite_between(v,low,high):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and bool(np.isfinite(v)) and low<=v<=high

def check_export(root,f,exporting=False):
    """Bundle roles: EXPORT.START is the original run's single reservation, copied
    into the bundle; EXPORT.CANDIDATE is written by the export worker after the
    copy; EXPORT.COMPLETE is written last by the supervisor. EXPORT.END lives only
    in the original run. Any other combination is an interrupted/failed export."""
    b=f['config']['budget']
    if (root/'EXPORT.FAILED.json').exists():raise ValueError('export failed; retain attempt')
    if not (root/'EXPORT.CANDIDATE.json').exists():
        if (root/'EXPORT.COMPLETE.json').exists():raise ValueError('export completion without candidate')
        if (root/'EXPORT.START.json').exists():
            if (root/'EXPORT.END.json').exists():
                end=read(root/'EXPORT.END.json');start=read(root/'EXPORT.START.json');b=f['config']['budget'];r=end.get('resources',{})
                if set(end)!={'out','resources'} or end['out']!=start.get('out') or set(r)!={'elapsed_seconds','peak_tree_rss_bytes','scratch_bytes','exit'} or r['exit']!=0 or not finite_between(r['elapsed_seconds'],0,start.get('remaining_seconds',0)) or not finite_between(r['peak_tree_rss_bytes'],0,b['rss_bytes']) or not finite_between(r['scratch_bytes'],0,b['scratch_bytes']):raise ValueError('original-run export receipt invalid')
                return False
            if not exporting:raise ValueError('interrupted export: reservation without completion; no retry')
        elif (root/'EXPORT.END.json').exists():raise ValueError('export receipt without reservation')
        return False
    if f['config']['development_only']:raise ValueError('fixture export forbidden')
    if not (root/'EXPORT.COMPLETE.json').is_file():raise ValueError('incomplete export')
    if (root/'EXPORT.END.json').exists():raise ValueError('original-run export receipt inside bundle')
    meta=read(root/'EXPORT.CANDIDATE.json');done=read(root/'EXPORT.COMPLETE.json');start=read(root/'EXPORT.START.json')
    if set(done)!={'status','candidate_sha256','resources'} or done['status']!='COMPLETE_CONFIRMATION' or done['candidate_sha256']!=sha(root/'EXPORT.CANDIDATE.json'):raise ValueError('export completion identity')
    if set(meta)!={'deployment','status','files','verification'} or meta['deployment']!=VERSION or meta['status']!='CANDIDATE_CONFIRMATION' or meta['verification'].get('verified') is not True or meta['verification'].get('development_only') is not False:raise ValueError('export candidate identity')
    observed=inventory(root);observed.pop('EXPORT.CANDIDATE.json');observed.pop('EXPORT.COMPLETE.json')
    if observed!=meta['files']:raise ValueError('export inventory/hash changed')
    remaining=b['campaign_seconds']-read(root/'ATTEMPT.END.json')['elapsed_seconds']
    if set(start)!={'out','remaining_seconds','retries','supervisor_pid','worker_nonce_identity'} or start['retries']!=0 or start['remaining_seconds']!=remaining:raise ValueError('export reservation invalid')
    r=done['resources']
    if set(r)!={'elapsed_seconds','peak_tree_rss_bytes','scratch_bytes','exit'} or r['exit']!=0 or not finite_between(r['elapsed_seconds'],0,remaining) or not finite_between(r['peak_tree_rss_bytes'],0,b['rss_bytes']) or not finite_between(r['scratch_bytes'],0,b['scratch_bytes']):raise ValueError('export resource/cumulative limit')
    return True

def check_ledgers(root,f):
    """Exact schemas, identities, charges and finite nonnegative times for every
    attempt/phase/cell receipt; presence alone is insufficient."""
    b=f['config']['budget'];e=f['enumeration'];dev=f['config']['development_only']
    if (root/'FAILED.json').exists():raise ValueError('failed attempt')
    for n in ('ATTEMPT.START.json','WORKER.START.json','FIT_COMPLETE.json','VERIFY.START.json','VERIFY.END.json','REDUCE.START.json','REDUCE.END.json','WORKER_COMPLETE.json','COMPLETE.json','ATTEMPT.END.json','reduction.json'):
        if not (root/n).is_file():raise ValueError('attempt not complete: '+n)
    start=read(root/'ATTEMPT.START.json');end=read(root/'ATTEMPT.END.json');worker=read(root/'WORKER_COMPLETE.json');fit=read(root/'FIT_COMPLETE.json');ws=read(root/'WORKER.START.json');complete=read(root/'COMPLETE.json')
    expected_start={'deployment':VERSION,'confirmation':not dev,'reserved_fit_charges':e['total_charges'],'reserved_replay_charges':e['total_charges'],'retries':0}
    if set(start)!=set(expected_start)|{'timestamp','supervisor_pid','worker_nonce_identity'} or any(start[k]!=v for k,v in expected_start.items()) or not isinstance(start['supervisor_pid'],int) or not finite_between(start['timestamp'],0,float('inf')) or not re.fullmatch('[0-9a-f]{64}',str(start['worker_nonce_identity'])):raise ValueError('attempt reservation invalid')
    if set(ws)!={'pid','supervisor_pid','nonce_identity'} or ws['supervisor_pid']!=start['supervisor_pid'] or ws['nonce_identity']!=start['worker_nonce_identity'] or not isinstance(ws['pid'],int):raise ValueError('worker reservation invalid')
    if set(end)!={'elapsed_seconds','status'} or end['status']!='COMPLETE' or not finite_between(end['elapsed_seconds'],0,b['campaign_seconds']):raise ValueError('attempt completion invalid')
    if set(worker)!={'development_only','cells','elapsed'} or worker['development_only']!=dev or worker['cells']!=e['paired_cells'] or not finite_between(worker['elapsed'],0,end['elapsed_seconds']):raise ValueError('worker completion invalid')
    if set(fit)!={'cells','elapsed'} or fit['cells']!=e['paired_cells'] or not finite_between(fit['elapsed'],0,worker['elapsed']):raise ValueError('fit completion invalid')
    if read(root/'VERIFY.START.json')!={'reserved_representation_replay_charges':e['representation_charges'],'retries':0} or read(root/'VERIFY.END.json')!={'cells_replayed':e['paired_cells'],'all_restarts_replayed':True} or read(root/'REDUCE.START.json')!={'cell_count':e['paired_cells']}:raise ValueError('phase reservation/completion invalid')
    if read(root/'REDUCE.END.json')!={'reduction_sha256':sha(root/'reduction.json')}:raise ValueError('reduction receipt mismatch')
    if set(complete)!={'development_only','cells','resources','deployment'} or complete['development_only']!=dev or complete['deployment']!=VERSION or complete['cells']!=e['paired_cells']:raise ValueError('completion census')
    r=complete['resources']
    if set(r)!={'elapsed_seconds','peak_tree_rss_bytes','scratch_bytes','exit'} or r['exit']!=0 or not finite_between(r['elapsed_seconds'],worker['elapsed'],end['elapsed_seconds']) or not finite_between(r['peak_tree_rss_bytes'],0,b['rss_bytes']) or not finite_between(r['scratch_bytes'],0,b['scratch_bytes']):raise ValueError('recorded resource failure or inconsistent cumulative elapsed ledger')
    n=e['paired_cells'];allowed={'freeze.json','campaign.log','ATTEMPT.START.json','WORKER.START.json','FIT_COMPLETE.json','VERIFY.START.json','VERIFY.END.json','REDUCE.START.json','REDUCE.END.json','WORKER_COMPLETE.json','COMPLETE.json','ATTEMPT.END.json','reduction.json','EXPORT.START.json','EXPORT.END.json','EXPORT.CANDIDATE.json','EXPORT.COMPLETE.json'}
    allowed|={f'{i:04d}.{s}' for i in range(n) for s in ('START.json','END.json','log')}|{f'cell-{i:04d}' for i in range(n)}
    unknown=sorted({p.name for p in root.iterdir()}-allowed)
    if unknown:raise ValueError('unexpected entries in run tree: '+', '.join(unknown[:5]))
    for i in range(n):
        d=root/f'cell-{i:04d}'
        if not d.is_dir() or {p.name for p in d.iterdir()}!={'arrays.npz','record.json','COMPLETE.json'}:raise ValueError('cell directory inventory')
    cell_starts={p.name for p in root.glob('[0-9][0-9][0-9][0-9].START.json')};cell_ends={p.name for p in root.glob('[0-9][0-9][0-9][0-9].END.json')}
    if cell_starts!={f'{i:04d}.START.json' for i in range(e['paired_cells'])} or cell_ends!={f'{i:04d}.END.json' for i in range(e['paired_cells'])}:raise ValueError('cell ledger inventory')
    for i,cell in enumerate(e['cells']):
        begin=read(root/f'{i:04d}.START.json');stop=read(root/f'{i:04d}.END.json')
        if set(begin)!={'cell','reserved_charges','timestamp'} or begin['cell']!=cell or begin['reserved_charges']!=53 or not finite_between(begin['timestamp'],start['timestamp'],float('inf')):raise ValueError('cell reservation mismatch')
        if set(stop)!={'status','elapsed','charges'} or stop['charges']!=53 or stop['status']!='COMPLETE' or not finite_between(stop['elapsed'],0,b['cell_seconds']):raise ValueError('cell ledger/resource mismatch')
    if (root/'EXPORT.COMPLETE.json').exists():
        exported=read(root/'EXPORT.COMPLETE.json')['resources']
        if end['elapsed_seconds']+exported['elapsed_seconds']>b['campaign_seconds']:raise ValueError('export resource/cumulative limit')

def domain_census(records):
    domains=[v['domain'] for r in records for k,v in r['methods'].items() if k.endswith('__domain')]
    census={k:domains.count(k) for k in ('in_domain','out_of_domain','undefined_gap')}
    if sum(census.values())!=len(domains) or len(domains)!=6*len(records):raise ValueError('domain census incomplete')
    return census

def receiving_verify(root,exporting=False):
    """Host-scoped full replay (optimizer included): qualified host, frozen
    environment and frozen source bytes. Elsewhere use portable_audit."""
    from .reconstruction import verify,assert_close
    root=Path(root);f=read(root/'freeze.json');validate_freeze(f)
    check_export(root,f,exporting);check_ledgers(root,f)
    records=verify(root);want=reduce_records(f['config'],records);want['domain_census']=domain_census(records)
    assert_close(read(root/'reduction.json'),{'development_only':f['config']['development_only'],**want})
    return {'verified':True,'mode':'host_full_replay','development_only':f['config']['development_only'],'cells':len(records),'source_identity':code_identity(f['sources'])}

def portable_audit(root):
    """Receiver-side audit on any machine running the frozen source bytes: ledgers,
    export chain, record/array hashes, regenerated data/moments/covariance
    estimates, all risks and representations recomputed from saved maps and
    factors, restart selection by saved-factor objectives, domain diagnostics and
    the full reduction. The optimizer trajectory is NOT replayed (platform BLAS
    may alter it); that remains the host replay's job."""
    from .reconstruction import audit_saved,assert_close
    root=Path(root);f=read(root/'freeze.json');validate_freeze(f,portable=True)
    check_export(root,f);check_ledgers(root,f)
    records=audit_saved(root,f);want=reduce_records(f['config'],records);want['domain_census']=domain_census(records)
    assert_close(read(root/'reduction.json'),{'development_only':f['config']['development_only'],**want})
    return {'verified':True,'mode':'portable_saved_map_audit','optimizer_replayed':False,'development_only':f['config']['development_only'],'cells':len(records),'source_identity':code_identity(f['sources'])}

def export_worker(root,out):
    parent_watch(group=True);root=Path(root);out=Path(out)
    worker_nonce(read(root/'EXPORT.START.json'))
    result=receiving_verify(root,exporting=True)
    if result['development_only']:raise ValueError('fixture scientific export forbidden')
    before=inventory(root);shutil.copytree(root,out)
    if inventory(root)!=before or inventory(out)!=before:raise ValueError('run changed while exporting')
    write_new(out/'EXPORT.CANDIDATE.json',{'deployment':VERSION,'status':'CANDIDATE_CONFIRMATION','files':before,'verification':result})

def preflight_space(target,needed):
    if shutil.disk_usage(target).free<needed:raise ValueError('insufficient free scratch for bounded operation')

def export(root,out):
    thread_gate();root=external(root);out=external(out)
    f=read(root/'freeze.json');validate_freeze(f)
    if f['config']['development_only']:raise ValueError('fixture scientific export forbidden')
    if out.exists() or out.is_relative_to(root) or root.is_relative_to(out):raise ValueError('export requires separate fresh output')
    elapsed=read(root/'ATTEMPT.END.json')['elapsed_seconds'];b=f['config']['budget'];remaining=b['campaign_seconds']-elapsed
    if remaining<=0:raise ValueError('campaign budget exhausted')
    run_bytes=tree_bytes(root)
    if 2*run_bytes>b['scratch_bytes']:raise ValueError('run plus copy exceeds scratch ceiling')
    out.parent.mkdir(parents=True,exist_ok=True);preflight_space(out.parent,run_bytes+1024**3)
    nonce=secrets.token_hex(32)
    # Single persistent reservation; a second export attempt fails on this write.
    write_new(root/'EXPORT.START.json',{'out':str(out),'remaining_seconds':remaining,'retries':0,'supervisor_pid':os.getpid(),'worker_nonce_identity':identity(nonce)})
    ledger=out.with_name(out.name+'.attempt')
    try:
        ledger.mkdir(parents=True,exist_ok=False)
        write_new(ledger/'START.json',{'source_run':str(root),'remaining_seconds':remaining,'retries':0})
        report=supervise([sys.executable,'-m',MODULE,'_export','--root',str(root),'--out',str(out)],ledger/'export.log',out,{'seconds':remaining,'rss_bytes':b['rss_bytes'],'scratch_bytes':b['scratch_bytes']},extra_paths=(root,ledger),extra_env={'PAPER1_T25_WORKER_NONCE':nonce})
        if not (out/'EXPORT.CANDIDATE.json').is_file():raise ValueError('export worker produced no candidate')
        write_new(ledger/'COMPLETE.json',report)
        write_new(out/'EXPORT.COMPLETE.json',{'status':'COMPLETE_CONFIRMATION','candidate_sha256':sha(out/'EXPORT.CANDIDATE.json'),'resources':report})
        write_new(root/'EXPORT.END.json',{'out':str(out),'resources':report})
    except BaseException as e:
        write_new(root/'EXPORT.FAILED.json',{'out':str(out),'error':str(e),'retry_allowed':False})
        if ledger.is_dir() and not (ledger/'FAILED.json').exists():write_new(ledger/'FAILED.json',{'error':str(e),'retry_allowed':False})
        raise

def diagnose(root):
    root=Path(root);f=read(root/'freeze.json');starts=list(root.glob('[0-9]*.START.json'));ends=list(root.glob('[0-9]*.END.json'))
    incomplete=[p.name for p in starts if not p.with_name(p.name.replace('.START.','.END.')).exists()]
    state='none'
    if (root/'EXPORT.START.json').exists():state='failed' if (root/'EXPORT.FAILED.json').exists() else 'complete' if (root/'EXPORT.END.json').exists() else 'interrupted_no_retry'
    return {'starts':len(starts),'ends':len(ends),'incomplete':incomplete,'reserved_cell_charges':53*len(starts),'interrupted_cell_seconds_charged':len(incomplete)*f['config']['budget']['cell_seconds'],'failed':(root/'FAILED.json').exists(),'complete':(root/'COMPLETE.json').exists(),'export':state,'retry_allowed':False}

def verify_worker(root,out):
    parent_watch(group=True);worker_nonce(read(Path(out)/'START.json'))
    write_new(Path(out)/'verified.json',receiving_verify(root))

def bounded_verify(root,out,include_reduction=False):
    """Receiving is a separately charged fresh read-only replay, never a fit retry.
    Scratch accounting covers the input run AND the receiving output."""
    thread_gate();root=external(root);out=external(out)
    if out==root or out.is_relative_to(root) or root.is_relative_to(out):raise ValueError('verification output overlaps input')
    f=read(root/'freeze.json');validate_freeze(f);b=f['config']['budget']
    if tree_bytes(root)>=b['scratch_bytes']:raise ValueError('input run alone reaches scratch ceiling')
    out.mkdir(parents=True,exist_ok=False);preflight_space(out,1024**3)
    nonce=secrets.token_hex(32)
    write_new(out/'START.json',{'operation':'receiving_replay','source_run':str(root),'reserved_replay_charges':f['enumeration']['total_charges'],'retries':0,'supervisor_pid':os.getpid(),'worker_nonce_identity':identity(nonce)})
    try:
        report=supervise([sys.executable,'-m',MODULE,'_verify','--root',str(root),'--out',str(out)],out/'verify.log',out,{'seconds':b['campaign_seconds'],'rss_bytes':b['rss_bytes'],'scratch_bytes':b['scratch_bytes']},extra_paths=(root,),extra_env={'PAPER1_T25_WORKER_NONCE':nonce})
        if read(out/'verified.json').get('verified') is not True:raise ValueError('verification worker did not verify')
        if include_reduction:write_new(out/'reduction.json',read(root/'reduction.json'))
        write_new(out/'COMPLETE.json',report)
    except BaseException as e:write_new(out/'FAILED.json',{'error':str(e),'retry_allowed':False});raise

def bounded_audit(root,out):
    """Portable receiving audit in a fresh namespace on any machine running the
    frozen source bytes; writes audit.json. Not a substitute for the host replay."""
    root=Path(root).resolve();out=external(out)
    if out==root or out.is_relative_to(root) or root.is_relative_to(out):raise ValueError('audit output overlaps input')
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'START.json',{'operation':'portable_audit','source_run':str(root),'retries':0})
    try:write_new(out/'audit.json',portable_audit(root))
    except BaseException as e:write_new(out/'FAILED.json',{'error':str(e),'retry_allowed':False});raise

def _peak_children_rss():
    import resource
    v=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(v if sys.platform=='darwin' else v*1024)

def qualify_fixtures(out):
    """Host fixture qualification: run the complete package test suite in a child
    with the frozen interpreter/environment, keep the log, record identities."""
    thread_gate();out=external(out);out.mkdir(parents=True,exist_ok=False)
    s=sources();env=environment();log=out/'fixture.log'
    with log.open('x') as stream:
        r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--basetemp',str(out/'pytest'),str(PACKAGE/'tests')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    text=log.read_text();m=re.search(r'(\d+) passed',text);failed=re.search(r'(\d+) (failed|error)',text)
    report={'exit':r.returncode,'tests_passed':int(m.group(1)) if m else 0,'tests_failed':int(failed.group(1)) if failed else 0,'code_identity':code_identity(s),'environment':env,'log_sha256':sha(log),'peak_rss_bytes':_peak_children_rss(),'host':socket.gethostname()}
    write_new(out/'fixture_report.json',report)
    if r.returncode!=0 or report['tests_failed']:raise ValueError('fixture qualification failed; see '+str(log))
    return report

BENCHMARK_NAMESPACE='paper1-t25-v1/benchmark-20260930'
BENCHMARK_SEEDS=[990001]
BENCHMARK_CELLS=[(0.0,400),(0.40,400),(0.12,1600),(0.0,3200)]

def benchmark(out):
    """Development-only production-shape timing. Fresh benchmark namespace/seed,
    disjoint from confirmation and fixture; fitted outputs are discarded and never
    written. Projects fit + in-campaign replay + export replay with margin 2."""
    import copy,psutil
    thread_gate();out=external(out);out.mkdir(parents=True,exist_ok=False)
    c=copy.deepcopy(BASE);c['namespace']=BENCHMARK_NAMESPACE;c['development_only']=True
    if set(BENCHMARK_SEEDS)&set(BASE['seeds']) or set(BENCHMARK_SEEDS)&set(FIXTURE['seeds']):raise ValueError('benchmark seed overlaps a registered namespace')
    rows=[];proc=psutil.Process()
    for seed in BENCHMARK_SEEDS:
        for theta,N in BENCHMARK_CELLS:
            t=time.monotonic();record,arrays=fit_cell(c,seed,theta,N);fit=time.monotonic()-t
            statuses=[x['status'] for k,v in record['methods'].items() if k.endswith('__restricted') for x in v['optimization']['restarts']]
            del record,arrays
            rows.append({'seed':seed,'theta':theta,'N':N,'fit_seconds':fit,'rss_bytes':proc.memory_info().rss,'restart_statuses':{k:statuses.count(k) for k in sorted(set(statuses))}})
    e=enumeration(BASE);mean=float(np.mean([r['fit_seconds'] for r in rows]));worst=float(max(r['fit_seconds'] for r in rows))
    projected=mean*e['paired_cells']*3*2
    report={'development_only':True,'scientific_endpoints_evaluated':False,'namespace':BENCHMARK_NAMESPACE,'note':'confirmation endpoints never computed; benchmark-namespace fits are discarded unviewed','code_identity':code_identity(sources()),'environment':environment(),'host':socket.gethostname(),'cells':rows,'mean_cell_fit_seconds':mean,'worst_cell_fit_seconds':worst,'projected_lifecycle_seconds_with_margin':projected,'margin':'x2 over fit+campaign replay+export replay','cell_seconds_cap':BASE['budget']['cell_seconds'],'worst_cell_within_half_cap':worst*2<=BASE['budget']['cell_seconds'],'peak_rss_bytes':max(r['rss_bytes'] for r in rows)}
    write_new(out/'benchmark_report.json',report)
    if not report['worst_cell_within_half_cap'] or projected>BASE['budget']['campaign_seconds'] or report['peak_rss_bytes']>BASE['budget']['rss_bytes']:raise ValueError('benchmark does not fit registered caps with margin')
    return report

def snapshot_record(out):
    """Attest the isolated execution snapshot: roots, code identity, read-only
    source files, no stale bytecode under the package, both Git states."""
    out=external(out);s=sources()
    writable=[r for role,root in [('EXP',ROOT),('MS',msroot())] for r in s[role+'_files'] if (root/r).stat().st_mode & 0o222]
    stale=[str(p.relative_to(ROOT)) for p in (ROOT/'experiments').rglob('*.pyc')]+[str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.pyc')]
    record={'code_identity':code_identity(s),'EXP_root':str(ROOT),'MS_root':str(msroot()),'EXP_head':s['EXP_head'],'MS_head':s['MS_head'],'EXP_status':s['EXP_status'],'MS_status':s['MS_status'],'writable_source_files':writable,'stale_bytecode':stale,'host':socket.gethostname()}
    write_new(out,record)
    if writable or stale:raise ValueError('snapshot not immutable/clean: make sources read-only and remove bytecode before qualification')
    return record
