"""Regressions for INDEPENDENT_PRODUCTION_REVIEW_V2 (N1, items 1, 2, 7, 8). Development fixtures only."""
import copy,json,os,shutil,socket
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.t25_production_20260930 import workflow as w
from experiments.mlst2026.t25_production_20260930 import reconstruction as rc
from experiments.mlst2026.t25_production_20260930.protocol import BASE,FIXTURE
from experiments.mlst2026.t25_production_20260930.science import reduce_records,moments

@pytest.fixture(autouse=True)
def single_thread(monkeypatch):
    for k in w.THREADS:monkeypatch.setenv(k,'1')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1')

@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    old={k:os.environ.get(k) for k in w.THREADS}
    for k in w.THREADS:os.environ[k]='1'
    base=tmp_path_factory.mktemp('t25v3');f=base/'freeze.json';w.freeze(FIXTURE,f);root=base/'attempt';w.execute(f,root)
    for k,v in old.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v
    return root

def clone(root,tmp_path):
    d=tmp_path/'copy';shutil.copytree(root,d);return d

def rewrite_record(root,index,mutate):
    """Tamper one cell record and rebuild every downstream hash and the reduction,
    as a capable adversary would."""
    d=root/f'cell-{index:04d}';r=w.read(d/'record.json');mutate(r);(d/'record.json').write_text(json.dumps(r))
    done=w.read(d/'COMPLETE.json');done['record_sha']=w.sha(d/'record.json');(d/'COMPLETE.json').write_text(json.dumps(done))
    f=w.read(root/'freeze.json');records=[w.read(root/f'cell-{i:04d}'/'record.json') for i in range(f['enumeration']['paired_cells'])]
    red=reduce_records(f['config'],records);red['domain_census']=w.domain_census(records)
    (root/'reduction.json').write_text(json.dumps({'development_only':True,**red}))
    (root/'REDUCE.END.json').write_text(json.dumps({'reduction_sha256':w.sha(root/'reduction.json')}))

def restricted_key(r):return next(k for k in sorted(r['methods']) if k.endswith('__restricted') and not k.startswith('iso'))

# N1: fit and replay share one moment function; host replay demands bitwise S.
def test_moments_shared_and_bitwise(completed,tmp_path):
    import inspect
    from experiments.mlst2026.t25_production_20260930 import science
    assert 'moments(training,N)' in inspect.getsource(science.fit_cell)
    assert 'moments(training,N)' in inspect.getsource(rc.verify)
    root=clone(completed,tmp_path);d=root/'cell-0002';a=dict(np.load(d/'arrays.npz'))
    a['S']=a['S'].copy();a['S'][0,0]=np.nextafter(a['S'][0,0],np.inf)
    (d/'arrays.npz').unlink();np.savez_compressed(d/'arrays.npz',**a)
    done=w.read(d/'COMPLETE.json');done['arrays_sha']=w.sha(d/'arrays.npz');(d/'COMPLETE.json').write_text(json.dumps(done))
    with pytest.raises(ValueError,match='bitwise'):w.receiving_verify(root)

# 7: portable audit rejects falsified optimizer metadata.
@pytest.mark.parametrize('kind',['status_flip','lss_flip','truncate','steps','backtracks'])
def test_portable_rejects_optimizer_metadata(completed,tmp_path,kind):
    root=clone(completed,tmp_path)
    def mutate(r):
        k=restricted_key(r);opt=r['methods'][k]['optimization'];x=opt['restarts'][opt['selected']]
        if kind=='status_flip':x['status']='iteration_limit' if x['status']=='stationary' else 'stationary'
        elif kind=='lss_flip':
            y=next(z for k2 in sorted(r['methods']) if k2.endswith('__restricted') for z in r['methods'][k2]['optimization']['restarts'] if z['status']=='stationary')
            y['status']='line_search_stopped'
        elif kind=='truncate':x['trajectory']=x['trajectory'][:-3]+x['trajectory'][-1:]
        elif kind=='steps':x['accepted_steps'][0]=x['accepted_steps'][0]*2 if x['accepted_steps'] else 1.
        elif kind=='backtracks':x['backtracks'][0]=x['backtracks'][0]+1 if x['backtracks'] else 1
    cell=1
    if kind=='lss_flip':
        cell=next(i for i in range(8) if any(z['status']=='stationary' for k2,v in w.read(root/f'cell-{i:04d}'/'record.json')['methods'].items() if k2.endswith('__restricted') for z in v['optimization']['restarts']))
    rewrite_record(root,cell,mutate)
    with pytest.raises(ValueError):w.portable_audit(root)
    with pytest.raises(ValueError):w.receiving_verify(root)

def test_portable_audit_without_git(completed,tmp_path,monkeypatch):
    def nogit(*a,**k):raise AssertionError('git must not be called in the portable role')
    monkeypatch.setattr(w,'git',nogit)
    assert w.portable_audit(completed)['verified']

# 2: unknown entries in a run tree.
def test_unknown_run_entry_rejected(completed,tmp_path):
    root=clone(completed,tmp_path);(root/'NOTES.json').write_text('{}')
    with pytest.raises(ValueError,match='unexpected entries'):w.portable_audit(root)
    root2=tmp_path/'c2';shutil.copytree(completed,root2);(root2/'cell-0000'/'extra.npy').write_bytes(b'0')
    with pytest.raises(ValueError,match='cell directory'):w.portable_audit(root2)

# 8: interrupted export states.
def test_interrupted_exports_rejected(completed,tmp_path):
    root=clone(completed,tmp_path);f=w.read(root/'freeze.json')
    (root/'EXPORT.START.json').write_text(json.dumps({'out':'/x','remaining_seconds':1.,'retries':0,'supervisor_pid':1,'worker_nonce_identity':'a'*64}))
    with pytest.raises(ValueError,match='interrupted export'):w.check_export(root,f)
    assert w.check_export(root,f,exporting=True) is False
    (root/'EXPORT.END.json').write_text(json.dumps({'out':'/x','resources':{'exit':0}}))
    with pytest.raises(ValueError,match='receipt invalid'):w.check_export(root,f)
    (root/'EXPORT.END.json').write_text(json.dumps({'out':'/other','resources':{'elapsed_seconds':.5,'peak_tree_rss_bytes':1,'scratch_bytes':1,'exit':0}}))
    with pytest.raises(ValueError,match='receipt invalid'):w.check_export(root,f)
    (root/'EXPORT.END.json').write_text(json.dumps({'out':'/x','resources':{'elapsed_seconds':.5,'peak_tree_rss_bytes':1,'scratch_bytes':1,'exit':0}}))
    assert w.check_export(root,f) is False
    (root/'EXPORT.START.json').unlink()
    with pytest.raises(ValueError,match='without reservation'):w.check_export(root,f)

def test_export_plumbing_end_to_end(completed,tmp_path,monkeypatch):
    """Drives the real export()/export_worker() sequence in-process with the
    verification and freeze gates stubbed (a fixture may never be exported)."""
    root=clone(completed,tmp_path)
    prod={'config':{**FIXTURE,'development_only':False}}
    monkeypatch.setattr(w,'validate_freeze',lambda f,portable=False:None)
    monkeypatch.setattr(w,'receiving_verify',lambda r,exporting=False:{'verified':True,'development_only':False,'mode':'stub'})
    monkeypatch.setattr(w,'parent_watch',lambda group=False:None)
    def fake_supervise(argv,log,out_root,budget,extra_paths=(),extra_env=None):
        monkeypatch.setenv('PAPER1_T25_WORKER_NONCE',extra_env['PAPER1_T25_WORKER_NONCE'])
        monkeypatch.setattr(os,'getppid',lambda:w.read(root/'EXPORT.START.json')['supervisor_pid'])
        w.export_worker(root,out_root);Path(log).write_text('ok')
        return {'elapsed_seconds':1.,'peak_tree_rss_bytes':1,'scratch_bytes':1,'exit':0}
    monkeypatch.setattr(w,'supervise',fake_supervise)
    f=w.read(root/'freeze.json');f['config']={**f['config'],'development_only':False};(root/'freeze.json').write_text(json.dumps(f))
    out=tmp_path/'bundle';w.export(root,out)
    assert (root/'EXPORT.END.json').is_file() and (out/'EXPORT.COMPLETE.json').is_file()
    assert w.check_export(out,prod) is True
    assert w.check_export(root,prod) is False
    with pytest.raises(FileExistsError):w.export(root,tmp_path/'second')

# 1: gate reads report contents.
def test_gate_rejects_mutable_snapshot_report(tmp_path):
    s=w.sources();env=w.environment();cid=w.code_identity(s)
    review=tmp_path/'r.md';review.write_text('ACCEPTED_PRODUCTION_IMPLEMENTATION '+cid)
    def put(name,v):p=tmp_path/name;p.write_text(json.dumps(v));return str(p),w.sha(p)
    q={'host':socket.gethostname(),'execution_mode':'direct','code_identity':cid,'environment':env}
    log=tmp_path/'fixture.log';log.write_text('31 passed');q['fixture_log_path'],q['fixture_log_sha256']=str(log),w.sha(log)
    q['fixture_report_path'],q['fixture_report_sha256']=put('fx.json',{'exit':0,'tests_passed':42,'code_identity':cid,'environment':env,'log_sha256':w.sha(log),'peak_rss_bytes':1})
    q['benchmark_report_path'],q['benchmark_report_sha256']=put('b.json',{'code_identity':cid,'environment':env,'scientific_endpoints_evaluated':False,'projected_lifecycle_seconds_with_margin':10,'worst_cell_within_half_cap':True,'peak_rss_bytes':1})
    q['author_record_path'],q['author_record_sha256']=put('a.json',{'host':'deepthought','execution_mode':'direct','protocol_resource_caps_remain':True})
    q['snapshot_record_path'],q['snapshot_record_sha256']=put('s.json',{'code_identity':cid,'EXP_root':str(w.ROOT),'MS_root':str(w.msroot()),'writable_source_files':['x.py'],'stale_bytecode':[]})
    a={'verdict':'ACCEPTED_PRODUCTION_IMPLEMENTATION','deployment':w.VERSION,'code_identity':cid,'independent_review_path':str(review),'independent_review_sha256':w.sha(review)}
    with pytest.raises(ValueError,match='not immutable'):w.production_gate(a,q,s,env)
