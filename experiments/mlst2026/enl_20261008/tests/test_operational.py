"""Operational engineering fixtures only: no fitted model, pool or permanent namespace."""
import ast,hashlib,importlib,os,signal,subprocess,sys,tempfile,time,types
from pathlib import Path
import pytest
from experiments.mlst2026.enl_20261008 import deadline as d,runtime as r

def test_shared_deadline_has_separate_one_hour_cap():
    assert d.effective_deadline(100,259200,100+70*3600)==100+71*3600
    assert d.effective_deadline(100,259200,100+60*3600)==100+61*3600
    assert d.effective_deadline(100,259200,100+71*3600)==100+72*3600
    assert d.effective_deadline(100,259200,100+71.5*3600)==100+72*3600

@pytest.mark.parametrize('value',[None,float('nan'),float('inf'),True,-1])
def test_invalid_clock_refused(value):
    with pytest.raises((ValueError,TypeError)):d.effective_deadline(value,259200,1)

def test_handoff_is_exclusive_and_clock_bound(tmp_path):
    begin=time.monotonic()-10;captured=time.monotonic()-2;p=tmp_path/'phase.json'
    d.start_record(p,begin,captured)
    assert d.post_start(p,begin)==captured
    with pytest.raises(FileExistsError):d.start_record(p,begin,captured)
    with pytest.raises(ValueError):d.post_start(p,begin-1)

def generated_fixture(tmp_path,monkeypatch):
    base=tmp_path/'torch';(base/'templates').mkdir(parents=True)
    origin=base/'instantiator.py';origin.write_text('# authenticated generator fixture\n')
    template=base/'templates'/'remote_module_template.py'
    template.write_text("_TEMPLATE_PREFIX = '{assign_module_interface_cls}\\n'\n_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA = '# {args} {kwargs}\\n'\n")
    directory=tempfile.TemporaryDirectory(dir='/private/tmp' if Path('/private/tmp').is_dir() else '/tmp');p=Path(directory.name)/'_remote_module_non_scriptable.py'
    p.write_bytes(b'module_interface_cls = None\n# *args **kwargs\n')
    module=types.SimpleNamespace(__file__=str(origin),INSTANTIATED_TEMPLATE_DIR_PATH=directory.name,_TEMP_DIR=directory)
    monkeypatch.setitem(sys.modules,'torch.distributed.nn.jit.instantiator',module)
    monkeypatch.setattr(r,'BOUND',{str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in [origin,template]})
    return directory,p,origin,template

def test_exact_generated_bytes_and_memory_compile(tmp_path,monkeypatch):
    directory,p,_,_=generated_fixture(tmp_path,monkeypatch)
    try:
        expected=p.read_bytes();assert r.generated_source('_remote_module_non_scriptable',p)==expected
        code=r.GeneratedLoader('_remote_module_non_scriptable',str(p)).get_code('_remote_module_non_scriptable')
        context={};exec(code,context);assert context['module_interface_cls'] is None
    finally:directory.cleanup()

@pytest.mark.parametrize('attack',['bytes','filename','origin','symlink','template','generator','other_name','permissions'])
def test_generated_source_attacks_refused(tmp_path,monkeypatch,attack):
    directory,p,origin,template=generated_fixture(tmp_path,monkeypatch)
    name='_remote_module_non_scriptable'
    try:
        if attack=='bytes':p.write_bytes(p.read_bytes()+b'raise RuntimeError("poison")\n')
        elif attack=='filename':p=p.with_name('different.py');p.write_bytes(b'module_interface_cls = None\n# *args **kwargs\n')
        elif attack=='origin':p=tmp_path/'_remote_module_non_scriptable.py';p.write_bytes(b'module_interface_cls = None\n# *args **kwargs\n')
        elif attack=='symlink':saved=p.read_bytes();p.unlink();target=tmp_path/'payload.py';target.write_bytes(saved);p.symlink_to(target)
        elif attack=='template':template.write_text(template.read_text()+'# changed\n')
        elif attack=='generator':origin.write_text('# changed generator\n')
        elif attack=='other_name':name='_remote_module_arbitrary'
        elif attack=='permissions':os.chmod(directory.name,0o755)
        with pytest.raises((ImportError,OSError)):r.generated_source(name,p)
    finally:directory.cleanup()

def test_arbitrary_temp_source_and_sourceless_cache_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'BOUND',{})
    with pytest.raises(ImportError):r.verify_file(tmp_path/'arbitrary.py')

@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree fixture')
def test_slow_worker_and_detached_grandchild_terminated_inside_deadline(tmp_path):
    d.adopt_descendants();childfile=tmp_path/'child.pid'
    script='import os,signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,"-c","import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"],start_new_session=True); open(sys.argv[1],"w").write(str(p.pid)); time.sleep(30)'
    process=subprocess.Popen([sys.executable,'-B','-c',script,str(childfile)],start_new_session=True);tree=d.Tree(process);end=time.monotonic()+2
    try:
        with pytest.raises(TimeoutError):d.wait_tree(tree,lambda:end,grace=.5)
    finally:assert tree.terminate(end,grace=.5)
    assert time.monotonic()<=end
    pid=int(childfile.read_text());assert pid not in d.processes() or d.processes()[pid][1]=='Z'

@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree fixture')
def test_successful_leader_cannot_leave_live_orphan(tmp_path):
    d.adopt_descendants();childfile=tmp_path/'child.pid'
    script='import subprocess,sys; p=subprocess.Popen([sys.executable,"-B","-c","import time; time.sleep(30)"],start_new_session=True); open(sys.argv[1],"w").write(str(p.pid))'
    process=subprocess.Popen([sys.executable,'-B','-c',script,str(childfile)],start_new_session=True);tree=d.Tree(process);end=time.monotonic()+3
    try:
        with pytest.raises(RuntimeError,match='live descendants'):d.wait_tree(tree,lambda:end,grace=.5)
    finally:assert tree.terminate(end,grace=.5)

@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-tree fixture')
def test_interruption_stops_descendants(tmp_path):
    d.adopt_descendants();process=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)'],start_new_session=True);tree=d.Tree(process);end=time.monotonic()+2
    try:
        def interrupt():raise KeyboardInterrupt('engineered coordinator interruption')
        with pytest.raises(KeyboardInterrupt):d.wait_tree(tree,lambda:end,interrupt,grace=.5)
    finally:assert tree.terminate(end,grace=.5)

@pytest.mark.skipif(sys.platform!='linux',reason='Linux orchestration consumer fixtures')
@pytest.mark.parametrize('slow_phase',['seal','hash',None])
def test_actual_coordinator_consumer_with_engineering_workers(tmp_path,monkeypatch,slow_phase):
    """Use the consumer with injected non-scientific commands and disjoint scratch paths."""
    import json
    from experiments.mlst2026.enl_20261008 import coordinator as co,workflow as w
    real_popen=subprocess.Popen;real_wait=d.wait_tree
    monkeypatch.setenv('ENL_CONFIRMATION_AUTHORISED','yes')
    monkeypatch.setattr(w,'LEDGER',str(tmp_path/'engineering-ledger'))
    monkeypatch.setattr(w,'config',lambda mode:{'budget':{'campaign_seconds':5,'scratch_bytes':1<<20}})
    monkeypatch.setattr(w,'sources',lambda:{'engineering-fixture':'immutable'})
    monkeypatch.setattr(d,'SEAL_SECONDS',1.2)
    monkeypatch.setattr(d,'GRACE_SECONDS',.3)
    monkeypatch.setattr(d,'wait_tree',lambda tree,end,check=lambda:None:real_wait(tree,end,check,grace=.3))
    commands=[]
    supervisor="""import json,os,pathlib,time
out=pathlib.Path(os.environ['ENL_OUTPUT']);out.mkdir()
start=time.monotonic();begin=float(os.environ['ENL_CAMPAIGN_STARTED_MONOTONIC'])
pathlib.Path(os.environ['ENL_POST_SCIENCE_RECORD']).write_text(json.dumps({'campaign_start_monotonic':begin,'post_science_start_monotonic':start}))
for name in ['EXIT.json','SUPERVISOR_EXIT.json']:(out/name).write_text(json.dumps({'exit_code':0}))
(out/'ATTEMPT.json').write_text('{}');(out/'return.tgz').write_bytes(b'engineering envelope')
time.sleep(float(os.environ['ENGINEERING_SEAL_DELAY']))
"""
    finalizer="""import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]);time.sleep(float(sys.argv[3]));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pathlib.Path(sys.argv[2]).write_text(json.dumps({'archive_sha256':sha(out/'return.tgz'),'attempt_sha256':sha(out/'ATTEMPT.json'),'supervisor_exit_sha256':sha(out/'SUPERVISOR_EXIT.json'),'sources':{'engineering-fixture':'immutable'}}))
"""
    def injected(command,**kwargs):
        commands.append(command)
        if command[0]=='bash':
            kwargs['env']={**kwargs['env'],'ENGINEERING_SEAL_DELAY':'3' if slow_phase=='seal' else '.01'}
            return real_popen([sys.executable,'-B','-c',supervisor],**kwargs)
        assert command[3]=='experiments.mlst2026.enl_20261008.deadline'
        out=command[command.index('--out')+1];destination=command[command.index('--destination')+1]
        return real_popen([sys.executable,'-B','-c',finalizer,out,destination,'3' if slow_phase=='hash' else '.01'],**kwargs)
    monkeypatch.setattr(subprocess,'Popen',injected)
    out=tmp_path/'engineering-output'
    if slow_phase:
        with pytest.raises(SystemExit) as e:co.coordinate(out)
        assert e.value.code==1
    else:co.coordinate(out)
    record=json.loads(Path(str(out)+'.coordinator/COORDINATOR_EXIT.json').read_text())
    assert record['seal_export_hash_limit_seconds']==1.2
    assert record['seal_export_hash_seconds']<=1.2
    assert record['exit_code']==(1 if slow_phase else 0)
    assert len(commands)==(1 if slow_phase=='seal' else 2)
    assert record['source_unchanged'] is (slow_phase is None)

def test_recycled_pid_is_never_adopted_or_signalled(monkeypatch):
    root=100;leader=200;rows={root:(1,'S','root'),leader:(root,'S','original')}
    monkeypatch.setattr(os,'getpid',lambda:root)
    monkeypatch.setattr(d,'process_info',lambda pid:rows.get(pid))
    monkeypatch.setattr(d,'child_pids',lambda pid:{leader} if pid==root else set())
    tree=d.Tree(types.SimpleNamespace(pid=leader));assert tree.observe()=={leader}
    rows[leader]=(999,'S','replacement')
    assert tree.observe()==set()
    sent=[];monkeypatch.setattr(os,'kill',lambda pid,sig:sent.append(pid));tree.send(signal.SIGTERM)
    assert sent==[] and tree.known[leader]=='original'

def test_updated_exact_supervisor_ancestry_and_spoof_refusal(tmp_path,monkeypatch):
    from experiments.mlst2026.enl_20261008 import workflow as w
    original_path=Path;proc=tmp_path/'proc';proc.mkdir()
    mapping={'/proc':proc}
    def record(pid,parent,args):
        folder=proc/str(pid);folder.mkdir(exist_ok=True)
        (folder/'status').write_text('PPid:\t'+str(parent)+'\n')
        (folder/'cmdline').write_bytes(b'\0'.join(os.fsencode(x) for x in args)+b'\0')
        mapping[f'/proc/{pid}/status']=folder/'status';mapping[f'/proc/{pid}/cmdline']=folder/'cmdline'
    record(3001,3002,['timeout','--kill-after=60','255540'])
    record(3002,3003,[sys.executable,'-B','-m','experiments.mlst2026.enl_20261008.supervise'])
    record(3003,3004,[sys.executable,'-m','experiments.mlst2026.enl_20261008.coordinator'])
    record(3004,1,['timeout','--kill-after=60','259140'])
    monkeypatch.setattr(w,'Path',lambda value:mapping.get(str(value),original_path(value)))
    monkeypatch.setattr(os,'getppid',lambda:3001);monkeypatch.setenv('ENL_COORDINATOR_PID','3003')
    result=w.ancestry({'budget':{'run_seconds':255600}})
    assert result['timeout_seconds']==255600 and result['parent_cmd'][2]=='255540'
    record(3002,3003,[sys.executable,'-c','spoof','experiments.mlst2026.enl_20261008.supervise'])
    with pytest.raises(ValueError,match='ancestry'):w.ancestry({'budget':{'run_seconds':255600}})
