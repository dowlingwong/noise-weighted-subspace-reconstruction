import json
import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_freeze_20261019 import prediction as p

def row(n=100, interval=(100,200)):
    return {'N_star_pred':n,'N_star_interval':interval,'checks':{'H':True},'restart_agreement':True}

def test_range_rules():
    assert p.annotate(row(),[50,100,150,200,400])['PZ2']=={'available':True,'in_range':True,'below_certificate':[50],'above_certificate':[400],'reason':None}
    assert not p.annotate(row(),[100,150,200])['PZ2']['in_range']
    assert not p.annotate(row(None,None),[50,400])['PZ2']['available']
    assert not p.annotate({**row(),'restart_agreement':False},[50,400])['PZ2']['available']
    assert not p.annotate(row(np.nan,(np.nan,200)),[50,400])['PZ2']['available']

def test_exact_prediction_wrapper(monkeypatch):
    fixed=[{'theta':t,**row()} for t in (0,.12,.25,.4)]
    def fake(S,P,i):
        assert P==1111 and i==2
        return {'fixed':fixed,'adaptive':{'A':{'root':True,'theta':.2,'row':row()},'B':{'root':False,'reason':'no root'}},'grids':{'A_grid':[56,111,222,444,888],'B_grid':[28,56,111,222,444]},'pre_fit':True}
    monkeypatch.setattr(p.population,'predictions',fake)
    r=p.predict_unit(np.eye(72),1111,2)
    assert len(r['fixed'])==4 and r['adaptive']['A']['available']
    assert not r['adaptive']['B']['available']
    assert r['geometries']['control']['grid']==[56,111,222,444,888]
    with pytest.raises(ValueError):p.predict_unit(np.eye(4),1111,2)
    with pytest.raises(np.linalg.LinAlgError):p.predict_unit(-np.eye(72),1111,2)

def test_committed_bytes(tmp_path):
    import subprocess,os
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    def git(*args):return subprocess.check_output(['git','-C',str(tmp_path),*args],env=env)
    git('init','-q');git('config','user.email','fixture@example.invalid');git('config','user.name','fixture')
    f=tmp_path/'pred.json';f.write_text('{}\n');git('add','pred.json');git('commit','-qm','fixture')
    c=git('rev-parse','HEAD').decode().strip(); assert p.committed(f,tmp_path,c)
    f.write_text('{"changed":true}')
    with pytest.raises(PermissionError):p.committed(f,tmp_path,c)
