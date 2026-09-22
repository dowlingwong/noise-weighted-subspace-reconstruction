"""Portable bounded command interface. Run from EXP with PYTHONPATH=src."""
import argparse,json,os,signal,subprocess,sys,time,uuid,shutil
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from . import protocol as p
from . import c1,cresst
from .validation import check as validate_result

def atomic_npz(path,arrays):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def event(root,e):
    with (root/'attempts.jsonl').open('a') as f:f.write(json.dumps(e,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def checked_freeze(path,ms):
    f=p.read(path);h=f.pop('freeze_id',None)
    if h!=p.identity(f):raise ValueError('freeze identity mismatch')
    f['freeze_id']=h
    if f['sources']!=p.sources(ms):raise ValueError('source/environment/commit mismatch; refreeze required')
    if f['config']['study']=='cresst':
        prep=Path(path).parent/'prepared.npz'
        if p.sha(prep)!=f['prepared_sha256']:raise ValueError('prepared quantities mutated')
    return f

def verify(root):
    root=Path(root);f=p.read(root/'freeze.json')
    canonical=p.read(p.PACKAGE/'configs'/f"{f['config']['study']}{'_fixture' if f['config']['development_only'] else ''}_protocol.yaml")
    if f['config']!=canonical or f['cells']!=p.cells(f['config'],f['stage'],f['manifest']):raise ValueError('frozen configuration/cell inventory mismatch')
    fid=f['freeze_id'];body={k:v for k,v in f.items() if k!='freeze_id'}
    if fid!=p.identity(body):raise ValueError('bad freeze')
    lines=(root/'attempts.jsonl').read_text().splitlines();events=[json.loads(l) for l in lines]
    starts={};ends={}
    for e in events:
        if e['freeze_id']!=fid:raise ValueError('mixed freezes')
        dst=starts if e['kind']=='START' else ends
        if e['attempt'] in dst:raise ValueError('duplicate attempt event')
        dst[e['attempt']]=e
    if set(starts)!=set(ends) or any(e['kind']!='COMPLETE' for e in ends.values()):raise ValueError('incomplete/failed/interrupted attempts retained and charged')
    if len({e['cell'] for e in starts.values()})!=len(starts):raise ValueError('duplicate cell attempt')
    expected={c['id']:c for c in f['cells']};results=[]
    if {e['cell'] for e in starts.values()}!=set(expected):raise ValueError('incomplete cell inventory')
    allowed={'freeze.json','attempts.jsonl','run_metadata.json','prepared.npz','stdout.log'}
    for aid,e in ends.items():
        if e['cell']!=starts[aid]['cell']:raise ValueError('cell/attempt mismatch')
        cid=e['cell']; rp=root/(cid+'.json');ap=root/(cid+'.npz')
        if p.sha(rp)!=e['result_sha256'] or p.sha(ap)!=e['arrays_sha256']:raise ValueError('result hash mismatch')
        r=p.read(rp)
        if r['freeze_id']!=fid or r['cell']!=expected[cid] or r['attempt']!=aid:raise ValueError('result provenance mismatch')
        validate_result(r,ap,f)
        if starts[aid]['reserved_fits']!=expected[cid]['cost_fits']:raise ValueError('attempt charge mismatch')
        results.append(r);allowed.update([rp.name,ap.name,cid+'.stdout',cid+'.stderr'])
    extra={q.name for q in root.iterdir()}-allowed
    if sum(v['reserved_fits'] for v in starts.values())>f['config']['budget']['max_fits']:raise ValueError('fit budget exceeded')
    if extra:raise ValueError('unexplained run inventory: '+str(extra))
    return f,results

def parser():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    for cmd in ['inventory','validate','enumerate','freeze']:
        x=sub.add_parser(cmd);x.add_argument('--config',required=True);x.add_argument('--out',required=True);x.add_argument('--manifest');x.add_argument('--data-root');x.add_argument('--exposure');x.add_argument('--stage',choices=['pilot','confirm'],default='confirm')
        if cmd=='freeze':x.add_argument('--ms-root',required=True);x.add_argument('--pilot-root')
    x=sub.add_parser('prepare');x.add_argument('--config',required=True);x.add_argument('--manifest',required=True);x.add_argument('--data-root',required=True);x.add_argument('--out',required=True)
    x=sub.add_parser('run');x.add_argument('--freeze',required=True);x.add_argument('--stage',required=True,choices=['pilot','confirm']);x.add_argument('--run-root',required=True);x.add_argument('--ms-root',required=True);x.add_argument('--data-root')
    for cmd in ['reduce','verify','export','diagnose']:
        x=sub.add_parser(cmd);x.add_argument('--run-root',required=True)
        if cmd not in ('verify','diagnose'):x.add_argument('--out',required=True)
    x=sub.add_parser('_worker');x.add_argument('--run-root',required=True);x.add_argument('--cell',required=True);x.add_argument('--attempt',required=True);x.add_argument('--data-root')
    return ap

def main():
    a=parser().parse_args()
    supplied=set(sys.argv[1:])
    allowed={
      'inventory':{'--config','--out','--data-root','--exposure'},
      'validate':{'--config','--out','--data-root','--exposure'},
      'enumerate':{'--config','--out','--manifest','--stage'},
      'freeze':{'--config','--out','--manifest','--stage','--ms-root','--pilot-root','--data-root'}}
    if a.cmd in allowed:
        unused={x for x in supplied if x.startswith('--')}-allowed[a.cmd]
        if unused:raise ValueError('unused options: '+str(unused))
    if a.cmd in ['inventory','validate','enumerate','freeze','prepare']:
        c=p.config(a.config);out=Path(a.out)
        if c['study']=='c1' and (getattr(a,'data_root',None) or getattr(a,'exposure',None)):raise ValueError('C1 takes no data options')
        if getattr(a,'pilot_root',None) and (c['study']!='c1' or a.stage!='confirm'):raise ValueError('pilot root only for C1 confirmation')
        if out.exists():raise ValueError('output exists; preserve and choose new version')
        manifest=p.read(a.manifest) if a.manifest else None
        if a.cmd in ['inventory','validate']:
            if c['study']=='c1':m={'qualification':'QUALIFIED','study':'c1','config_sha256':p.identity(c),'data':'generated independent seeds'}
            else:
                if not a.data_root or not a.exposure:raise ValueError('data root and explicit exposure audit required')
                m=cresst.qualify(a.data_root,c,p.read(a.exposure));m['config_sha256']=p.identity(c)
            p.write(out,m);return
        if manifest is None or manifest.get('config_sha256')!=p.identity(c):raise ValueError('config/manifest binding mismatch')
        if a.cmd=='prepare':
            if c['study']!='cresst':raise ValueError('prepare is for CRESST')
            out.mkdir(parents=True);prep_sources=p.sources(os.environ['MLST_MS_ROOT'])
            p.write(out/'attempt.json',{'status':'START','reserved_fits':7*len(manifest['channels']),'reserved_seconds':3600,'max_rss_bytes':c['budget']['max_rss_bytes'],'retries':0})
            import threading,psutil
            deadline=time.monotonic()+3600
            def preparation_watchdog():
                while True:
                    if time.monotonic()>deadline or psutil.Process().memory_info().rss>c['budget']['max_rss_bytes']:os._exit(90)
                    time.sleep(.1)
            threading.Thread(target=preparation_watchdog,daemon=True).start()
            report,arrays=cresst.prepare(a.data_root,c,manifest)
            if prep_sources!=p.sources(os.environ['MLST_MS_ROOT']):raise ValueError('source changed during preparation')
            cresst.check_data(a.data_root,manifest)
            atomic_npz(out/'prepared.npz',arrays)
            p.write(out/'prepared.json',{'config_sha256':p.identity(c),'manifest_sha256':p.identity(manifest),'report':report,'arrays_sha256':p.sha(out/'prepared.npz'),'sources':prep_sources})
            p.write(out/'attempt_complete.json',{'status':'COMPLETE','reserved_fits':7*len(manifest['channels'])});return
        celllist=p.cells(c,a.stage,manifest)
        if a.cmd=='enumerate':p.write(out,{'cells':celllist,'fits':sum(z['cost_fits'] for z in celllist)});return
        if c['study']=='cresst':
            if not a.data_root:raise ValueError('data root required for freeze revalidation')
            cresst.check_manifest(a.data_root,c,manifest)
        f={'config':c,'manifest':manifest,'cells':celllist,'stage':a.stage,'sources':p.sources(a.ms_root),'baseline':None}
        if c['study']=='c1' and a.stage=='confirm':
            if not a.pilot_root:raise ValueError('complete pilot required to freeze baseline')
            pf,results=verify(a.pilot_root)
            if pf['stage']!='pilot' or pf['config']!=c or pf['sources']!=f['sources']:raise ValueError('pilot scope/source mismatch')
            f['baseline']={'coefficients':c1.fit_baseline(results),'pilot_freeze':pf['freeze_id'],'pilot_results':{r['cell']['id']:p.identity(r) for r in results}}
        if c['study']=='cresst':
            # Prepared material accompanies manifest in a sibling prepared directory.
            prepared=Path(a.manifest).parent/'prepared'
            if p.read(prepared/'attempt_complete.json')['status']!='COMPLETE':raise ValueError('incomplete preparation')
            pr=p.read(prepared/'prepared.json')
            if pr['config_sha256']!=p.identity(c) or pr['manifest_sha256']!=p.identity(manifest) or pr['sources']!=f['sources'] or p.sha(prepared/'prepared.npz')!=pr['arrays_sha256']:raise ValueError('preparation binding mismatch')
            f['prepared']=pr['report'];f['prepared_sha256']=pr['arrays_sha256']
            if out.parent.joinpath('prepared.npz').exists():raise ValueError('prepared output already exists')
            out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(prepared/'prepared.npz',out.parent/'prepared.npz')
        f['freeze_id']=p.identity(f);p.write(out,f);return
    if a.cmd=='_worker':
        root=Path(a.run_root);f=p.read(root/'freeze.json');spec=next(c for c in f['cells'] if c['id']==a.cell)
        # Worker enforces lifetime/RSS independently even if the parent is SIGKILLed.
        import threading,psutil
        parent=os.getppid();deadline=time.monotonic()+f['config']['budget']['cell_seconds']
        def watchdog():
            while True:
                if os.getppid()!=parent or time.monotonic()>deadline or psutil.Process().memory_info().rss>f['config']['budget']['max_rss_bytes']:os._exit(90)
                time.sleep(.05)
        threading.Thread(target=watchdog,daemon=True).start()
        if spec['study']=='c1':result,arrays=c1.cell(spec)
        else:
            cresst.check_data(a.data_root,f['manifest'])
            with np.load(root/'prepared.npz',allow_pickle=False) as z:result,arrays=cresst.run_cell(spec,f['config'],f['manifest'],f['prepared'],z,a.data_root)
            cresst.check_data(a.data_root,f['manifest'])
        atomic_npz(root/(a.cell+'.npz'),arrays)
        p.write(root/(a.cell+'.json'),{'freeze_id':f['freeze_id'],'attempt':a.attempt,'cell':spec,'result':result});return
    if a.cmd=='run':
        f=checked_freeze(a.freeze,a.ms_root);root=Path(a.run_root)
        if f['config']['study']=='c1' and a.data_root:raise ValueError('C1 takes no data root')
        if a.stage!=f['stage']:raise ValueError('conflicting stage')
        if root.exists():raise ValueError('run root exists; no implicit retries or resume')
        if f['config']['study']=='cresst':
            if not a.data_root:raise ValueError('data root required')
            cresst.check_data(a.data_root,f['manifest']);cresst.check_manifest(a.data_root,f['config'],f['manifest'])
        # Prevent outputs inside source trees: a run must be outside both repositories.
        for source in (p.ROOT,Path(a.ms_root).resolve()):
            if root.resolve().is_relative_to(source):raise ValueError('run root must be outside source repositories')
        root.mkdir(parents=True);p.write(root/'freeze.json',f)
        p.write(root/'run_metadata.json',{'runtime_exp':str(p.ROOT),'runtime_ms':str(Path(a.ms_root).resolve()),'data_root':a.data_root,'started_unix':time.time()})
        if f['config']['study']=='cresst':shutil.copyfile(Path(a.freeze).parent/'prepared.npz',root/'prepared.npz')
        start=time.monotonic();budget=f['config']['budget'];charged=0
        for spec in f['cells']:
            if time.monotonic()-start>budget['max_elapsed_seconds']:raise ValueError('campaign time budget exhausted')
            checked_freeze(a.freeze,a.ms_root)
            if f['config']['study']=='cresst':cresst.check_data(a.data_root,f['manifest'])
            aid=str(uuid.uuid4());charged+=spec['cost_fits'];t0=time.monotonic();base={'freeze_id':f['freeze_id'],'cell':spec['id'],'attempt':aid}
            event(root,{**base,'kind':'START','reserved_fits':spec['cost_fits'],'charged_fits':charged,'time_unix':time.time()})
            proc=None;peak=0;kind='FAILED';reason='unknown'
            try:
                cmd=[sys.executable,'-m','experiments.mlst2026.upgrade_20260922','_worker','--run-root',str(root),'--cell',spec['id'],'--attempt',aid]
                if a.data_root:cmd+=['--data-root',a.data_root]
                env={**os.environ,**{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}
                def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
                signal.signal(signal.SIGTERM,interrupted)
                with (root/(spec['id']+'.stdout')).open('w') as stdout,(root/(spec['id']+'.stderr')).open('w') as stderr:
                    proc=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                    import psutil
                    peak=0
                    while proc.poll() is None:
                        try:peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
                        except psutil.NoSuchProcess:pass
                        if peak>budget['max_rss_bytes'] or time.monotonic()-t0>min(budget['cell_seconds'],budget['max_elapsed_seconds']-(t0-start)):raise RuntimeError('resource budget exceeded')
                        time.sleep(.05)
                if proc.returncode:raise RuntimeError('worker exit '+str(proc.returncode))
                checked_freeze(a.freeze,a.ms_root)
                if f['config']['study']=='cresst':cresst.check_data(a.data_root,f['manifest'])
                kind='COMPLETE';reason='ok'
            except BaseException as exc:
                reason=str(exc);kind='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
                if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            end={**base,'kind':kind,'reason':reason,'peak_rss_bytes':peak if proc else None,'elapsed_seconds':time.monotonic()-t0,'time_unix':time.time()}
            if kind=='COMPLETE':end.update(result_sha256=p.sha(root/(spec['id']+'.json')),arrays_sha256=p.sha(root/(spec['id']+'.npz')))
            event(root,end)
            if kind!='COMPLETE':raise RuntimeError('campaign stopped; attempt preserved: '+reason)
        verify(root);return
    if a.cmd=='diagnose':
        root=Path(a.run_root);f=p.read(root/'freeze.json');events=[json.loads(x) for x in (root/'attempts.jsonl').read_text().splitlines()];starts=[e for e in events if e['kind']=='START'];ends={e['attempt']:e for e in events if e['kind']!='START'}
        print(json.dumps({'status':'diagnostic_only_no_repair','reserved_fits':sum(e['reserved_fits'] for e in starts),'conservative_elapsed_charge_seconds':sum(ends.get(e['attempt'],{}).get('elapsed_seconds',f['config']['budget']['cell_seconds']) for e in starts),'unfinished_attempts':[e['attempt'] for e in starts if e['attempt'] not in ends],'orphan_result_files':[q.name for q in root.glob('*.npz') if q.stem not in {e['cell'] for e in ends.values() if e['kind']=='COMPLETE'}],'action':'retain directory; no retry/resume/promotion; versioned correction and review required'}));return
    f,results=verify(a.run_root)
    if a.cmd=='verify':print(json.dumps({'status':'complete','cells':len(results),'development_only':f['config']['development_only']}));return
    if a.cmd=='export' and (f['config']['development_only'] or f['stage']!='confirm'):raise ValueError('scientific export rejects development/pilot evidence')
    if f['config']['study']=='c1':
        reduction={'coefficients':c1.fit_baseline(results)} if f['stage']=='pilot' else c1.reduce_confirmation(results,f['baseline']['coefficients'])
    else:reduction={'equal_channel_descriptive_mean':float(np.mean([r['result']['primary_contrast']['mean'] for r in results if r['cell']['role']=='primary'])),'distinct_runs':len(set(r['result']['run'] for r in results)),'channels':{r['cell']['channel']:r['result']['primary_contrast'] for r in results if r['cell']['role']=='primary'},'all_cells':{r['cell']['id']:r['result'] for r in results},'cross_channel_scope':'descriptive only; no independent-run inference'}
    out=Path(a.out)
    if out.exists():raise ValueError('output exists; version outputs')
    if a.cmd=='reduce':p.write(out,{'status':'complete','development_only':f['config']['development_only'],'freeze_id':f['freeze_id'],'reduction':reduction});return
    out.mkdir(parents=True);shutil.copytree(a.run_root,out/'run');p.write(out/'reduction.json',{'status':'complete','development_only':False,'freeze_id':f['freeze_id'],'reduction':reduction})
    from .reporting import render
    render(out,f,results,reduction)
    p.write(out/'SHA256.json',{str(q.relative_to(out)):p.sha(q) for q in out.rglob('*') if q.is_file()})

if __name__=='__main__':
    try:main()
    except Exception as exc:print(type(exc).__name__+': '+str(exc),file=sys.stderr);sys.exit(1)
