import copy,os,sys,json
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.t25_production_20260930 import workflow as w
from experiments.mlst2026.t25_production_20260930.protocol import BASE,FIXTURE
from experiments.mlst2026.t25_production_20260930.science import response

@pytest.fixture(autouse=True)
def single_thread(monkeypatch):
    for k in w.THREADS:monkeypatch.setenv(k,'1')

def test_exceptional_gap_is_retained_but_invalid_covariance_fails():
    result,_=response(np.eye(4),np.zeros((4,4)),2)
    assert result['domain']=='undefined_gap' and result['eta'] is None
    with pytest.raises(ValueError):response(np.eye(4),-2*np.eye(4),2)
    with pytest.raises(ValueError):response(-np.eye(4),np.zeros((4,4)),2)
    with pytest.raises(ValueError):response(np.full((4,4),np.nan),np.zeros((4,4)),2)

def test_full_whitening_changes_product_class_counterexample():
    # Rank-one product e1 maps to vec(diag(1,.3)), which has matrix rank two.
    R=np.eye(4);R[0,3]=R[3,0]=.3
    assert np.linalg.eigvalsh(R).min()>0
    mapped=R@np.array([1.,0,0,0])
    assert np.linalg.matrix_rank(mapped.reshape(2,2,order='F'))==2

def test_production_cannot_freeze_without_independent_acceptance(tmp_path):
    with pytest.raises(ValueError,match='independent production acceptance'):
        w.freeze(BASE,tmp_path/'not-created.json')
    assert not (tmp_path/'not-created.json').exists()

def test_supervisor_kills_time_and_scratch_overrun(tmp_path):
    budget={'seconds':.2,'rss_bytes':512*1024**2,'scratch_bytes':1024**2}
    with pytest.raises(ValueError,match='resource cap'):
        w.supervise([sys.executable,'-c','import time;time.sleep(10)'],tmp_path/'sleep.log',tmp_path,budget)
    budget['seconds']=5;budget['scratch_bytes']=100
    with pytest.raises(ValueError,match='resource cap'):
        w.supervise([sys.executable,'-c',f"from pathlib import Path;Path({str(tmp_path/'big')!r}).write_bytes(b'x'*4096)"],tmp_path/'disk.log',tmp_path,budget)

def test_fresh_attempt_replay_and_no_fixture_export(tmp_path):
    f=tmp_path/'freeze.json';w.freeze(FIXTURE,f);root=tmp_path/'attempt';w.execute(f,root)
    r=w.receiving_verify(root);assert r['development_only'] and r['cells']==8
    before=w.inventory(root)
    with pytest.raises(FileExistsError):w.execute(f,root)
    assert w.inventory(root)==before
    with pytest.raises(ValueError,match='fixture scientific export'):w.export(root,tmp_path/'export')
    # Recomputed reduction and receipt are both necessary, not merely COMPLETE.
    p=root/'reduction.json';v=w.read(p);v['contrasts'][0]['mean']+=1;p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='reduction receipt'):w.receiving_verify(root)
