"""Adversarial regressions for INDEPENDENT_PRODUCTION_REVIEW_V1 blockers 1-8.

Development fixtures only; no production freeze, no scientific endpoint.
"""
import copy,json,os,shutil,signal,subprocess,sys,textwrap,time
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.t25_production_20260930 import workflow as w
from experiments.mlst2026.t25_production_20260930.protocol import BASE,FIXTURE

@pytest.fixture(autouse=True)
def single_thread(monkeypatch):
    for k in w.THREADS:monkeypatch.setenv(k,'1')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1')

@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    old={k:os.environ.get(k) for k in w.THREADS}
    for k in w.THREADS:os.environ[k]='1'
    base=tmp_path_factory.mktemp('t25v2');f=base/'freeze.json';w.freeze(FIXTURE,f);root=base/'attempt';w.execute(f,root)
    for k,v in old.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v
    return root

def dead(pid):
    """Gone, or a zombie awaiting a reaper (containers whose PID 1 does not reap)."""
    import psutil
    try:return psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return True

def clone(root,tmp_path,name='copy'):
    dest=tmp_path/name;shutil.copytree(root,dest);return dest

# 1. Fabricated acceptance/qualification artifacts are rejected.
def test_fabricated_evidence_rejected(tmp_path):
    s=w.sources();env=w.environment();cid=w.code_identity(s)
    acceptance={'verdict':'ACCEPTED_PRODUCTION_IMPLEMENTATION','deployment':w.VERSION,'code_identity':cid,'independent_review_path':str(tmp_path/'nope.md'),'independent_review_sha256':'not-an-actual-hash'}
    q={'host':__import__('socket').gethostname(),'execution_mode':'direct','code_identity':cid,'environment':env}
    for n in ('fixture_report','fixture_log','benchmark_report','author_record','snapshot_record'):q[n+'_sha256']='invented';q[n+'_path']=str(tmp_path/n)
    with pytest.raises(ValueError,match='evidence'):w.production_gate(acceptance,q,s,env)
    # A real file whose text does not accept this code identity is also rejected.
    review=tmp_path/'review.md';review.write_text('Verdict: NOT ACCEPTED\n')
    acceptance.update(independent_review_path=str(review),independent_review_sha256=w.sha(review))
    with pytest.raises(ValueError,match='does not accept'):w.production_gate(acceptance,q,s,env)
    frozen={'acceptance':acceptance|{'independent_review_sha256':'x'},'qualification':q,'sources':s,'environment':env}
    with pytest.raises(ValueError,match='frozen evidence hash'):w.frozen_evidence_identities(frozen)

# 2. Missing, empty or inconsistent receipts are rejected.
@pytest.mark.parametrize('mutation',[
    ('delete','ATTEMPT.START.json'),('delete','FIT_COMPLETE.json'),
    ('empty','ATTEMPT.END.json'),('empty','WORKER_COMPLETE.json'),('empty','VERIFY.START.json'),
    ('empty','VERIFY.END.json'),('empty','REDUCE.START.json'),('empty','0000.END.json'),
    ('negative','ATTEMPT.END.json'),('nan','0001.END.json'),('extra','COMPLETE.json')])
def test_ledger_mutations_rejected(completed,tmp_path,mutation):
    kind,name=mutation;root=clone(completed,tmp_path);p=root/name
    if kind=='delete':p.unlink()
    elif kind=='empty':p.write_text('{}')
    elif kind=='negative':v=w.read(p);v['elapsed_seconds']=-1.;p.write_text(json.dumps(v))
    elif kind=='nan':p.write_text(p.read_text().replace('"elapsed":','"elapsed": NaN, "x":'))
    elif kind=='extra':v=w.read(p);v['unexpected']=1;p.write_text(json.dumps(v))
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):w.receiving_verify(root)
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):w.portable_audit(root)

def test_unmodified_fixture_verifies_both_modes(completed):
    assert w.receiving_verify(completed)['mode']=='host_full_replay'
    r=w.portable_audit(completed);assert r['verified'] and r['optimizer_replayed'] is False and r['cells']==8

# 3. Leader exit with a live descendant, and signals to the supervisor.
def test_orphan_descendant_fails_and_is_killed(tmp_path):
    pidfile=tmp_path/'child.pid'
    code=f"import subprocess,sys;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);open({str(pidfile)!r},'w').write(str(p.pid))"
    budget={'seconds':20,'rss_bytes':1024**3,'scratch_bytes':1024**2}
    (tmp_path/'o').mkdir()
    with pytest.raises(ValueError,match='orphan'):w.supervise([sys.executable,'-c',code],tmp_path/'o.log',tmp_path/'o',budget)
    pid=int(pidfile.read_text());time.sleep(.3)
    assert dead(pid)

def test_sigterm_to_supervisor_kills_group(tmp_path):
    pidfile=tmp_path/'child.pid';(tmp_path/'s').mkdir()
    driver=textwrap.dedent(f"""
        import sys
        from experiments.mlst2026.t25_production_20260930 import workflow as w
        child="import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));time.sleep(60)"
        try:w.supervise([sys.executable,'-c',child],{str(tmp_path/'s.log')!r},{str(tmp_path/'s')!r},{{'seconds':60,'rss_bytes':1024**3,'scratch_bytes':1024**2}})
        except w.Interrupted:sys.exit(7)
    """)
    p=subprocess.Popen([sys.executable,'-c',driver],env=os.environ)
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():break
        time.sleep(.1)
    child=int(pidfile.read_text());p.send_signal(signal.SIGTERM);assert p.wait(20)==7;time.sleep(.3)
    assert dead(child)

# 4. Scratch accounting includes the extra (input) roots.
def test_scratch_counts_input_roots(tmp_path):
    (tmp_path/'in').mkdir();(tmp_path/'in'/'big').write_bytes(b'x'*4096);(tmp_path/'out').mkdir()
    budget={'seconds':5,'rss_bytes':1024**3,'scratch_bytes':1024}
    with pytest.raises(ValueError,match='resource cap'):
        w.supervise([sys.executable,'-c','pass'],tmp_path/'l.log',tmp_path/'out',budget,extra_paths=(tmp_path/'in',))
    (tmp_path/'in'/'link').symlink_to(tmp_path/'in'/'big')
    with pytest.raises(ValueError,match='symlink'):w.tree_bytes(tmp_path/'in')

# 5. Environment identity is process-order independent and records origins.
def test_environment_identity_stable_across_processes():
    here=w.environment()
    code="import json;from experiments.mlst2026.t25_production_20260930 import workflow as w;import scipy.linalg;print(json.dumps(w.environment(),sort_keys=True))"
    other=json.loads(subprocess.check_output([sys.executable,'-c',code],env=os.environ,text=True))
    assert other==json.loads(json.dumps(here,sort_keys=True))
    assert here['dont_write_bytecode'] is True and 'numpy' in here['import_origins']

# 6. Internal workers refuse to run without their supervisor reservation.
def test_internal_workers_refuse_without_reservation(completed,tmp_path):
    root=clone(completed,tmp_path)
    env={**os.environ};env.pop('PAPER1_T25_WORKER_NONCE',None)
    for argv in (['_cell','--freeze',str(root/'freeze.json'),'--index','0','--out',str(root/'cell-9999')],
                 ['_campaign','--root',str(root)],['_export','--root',str(root),'--out',str(tmp_path/'x')]):
        r=subprocess.run([sys.executable,'-m',w.MODULE]+argv,env=env,capture_output=True,text=True)
        assert r.returncode!=0
    assert not (root/'cell-9999').exists() and not (tmp_path/'x').exists()
    with pytest.raises(ValueError):w.worker_nonce({'worker_nonce_identity':'0'*64,'supervisor_pid':os.getpid()})

# 7. Portable audit catches tampering without the optimizer or host.
def test_portable_audit_rejects_tampered_map(completed,tmp_path):
    root=clone(completed,tmp_path);d=root/'cell-0003';a=dict(np.load(d/'arrays.npz'));a['oracle__restricted_A']=a['oracle__restricted_A']+1e-3
    (d/'arrays.npz').unlink();np.savez_compressed(d/'arrays.npz',**a)
    done=w.read(d/'COMPLETE.json');done['arrays_sha']=w.sha(d/'arrays.npz');(d/'COMPLETE.json').write_text(json.dumps(done))
    with pytest.raises(ValueError):w.portable_audit(root)

# 8. Export chain roles: candidate/complete/start/end combinations.
def export_case(tmp_path):
    c=copy.deepcopy(BASE);f={'config':c}
    root=tmp_path/'bundle';root.mkdir(parents=True)
    (root/'ATTEMPT.END.json').write_text(json.dumps({'elapsed_seconds':100.,'status':'COMPLETE'}))
    (root/'EXPORT.START.json').write_text(json.dumps({'out':'/x','remaining_seconds':c['budget']['campaign_seconds']-100.,'retries':0,'supervisor_pid':1,'worker_nonce_identity':'a'*64}))
    files=w.inventory(root)
    (root/'EXPORT.CANDIDATE.json').write_text(json.dumps({'deployment':w.VERSION,'status':'CANDIDATE_CONFIRMATION','files':files,'verification':{'verified':True,'development_only':False}}))
    res={'elapsed_seconds':5.,'peak_tree_rss_bytes':1.,'scratch_bytes':1.,'exit':0}
    (root/'EXPORT.COMPLETE.json').write_text(json.dumps({'status':'COMPLETE_CONFIRMATION','candidate_sha256':w.sha(root/'EXPORT.CANDIDATE.json'),'resources':res}))
    return f,root

def test_export_chain(tmp_path):
    f,root=export_case(tmp_path);assert w.check_export(root,f) is True
    (root/'EXPORT.COMPLETE.json').rename(tmp_path/'held')
    with pytest.raises(ValueError,match='incomplete export'):w.check_export(root,f)
    (tmp_path/'held').rename(root/'EXPORT.COMPLETE.json');(root/'EXPORT.END.json').write_text('{}')
    with pytest.raises(ValueError,match='original-run'):w.check_export(root,f)
    (root/'EXPORT.END.json').unlink();(root/'stray').write_text('x')
    with pytest.raises(ValueError,match='inventory'):w.check_export(root,f)
    (root/'stray').unlink();v=w.read(root/'EXPORT.START.json');v['retries']=1;(root/'EXPORT.START.json').write_text(json.dumps(v))
    with pytest.raises(ValueError):w.check_export(root,f)
    f2,r2=export_case(tmp_path/'b');(r2/'EXPORT.CANDIDATE.json').unlink()
    with pytest.raises(ValueError,match='without candidate'):w.check_export(r2,f2)
