import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.hcp_rnoise_freeze_20261019 import pilot as p

def test_source_rejects_excluded_and_changed(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'DEV',tmp_path)
    date=p.access.config()['access']['dev_dates'][0]
    with pytest.raises(PermissionError):p.source(date,{'E1':{'included':[]}}, {})
    f=tmp_path/(date+'_decimated.npy');f.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='source changed'):
        p.source(date,{'E1':{'included':[date]}},{'moments':{date:{'source_sha256':'0'*64}}})
    with pytest.raises(PermissionError):p.source('2013-06-24',{}, {})

def test_all_arms_and_ref_remedy(monkeypatch,tmp_path):
    date=p.access.config()['access']['dev_dates'][0]
    monkeypatch.setattr(p,'evidence',lambda:({},{}))
    monkeypatch.setattr(p,'committed',lambda *a,**k:'sha')
    monkeypatch.setattr(p,'PUB',tmp_path);monkeypatch.setattr(p,'OUT',tmp_path)
    pred={'units':{date:{'geometries':{'A':{'available':True,'theta':.2,'grid':[56,111]}}}}}
    (tmp_path/'DEV_PREDICTIONS.json').write_text(json.dumps(pred))
    monkeypatch.setattr(p,'source',lambda *a:({},np.eye(72)))
    from experiments.mlst2026.hcp_rnoise_20261019 import signal
    monkeypatch.setattr(signal,'build',lambda *a:object())
    calls=[]
    def fit(*args):calls.append(args);return {'D':1.,'replicate':args[-1],'N':args[3],'arm':args[5]}
    monkeypatch.setattr(p.learning,'fit_cell',fit)
    p.one_group((date,'A',0,False));assert len(calls)==12
    assert {(a[5],a[4]) for a in calls}==set(p.ARMS)
    calls.clear();p.one_group((date,'A',20,True));assert len(calls)==2 and all(a[5]=='reference' for a in calls)
    with pytest.raises(FileExistsError):p.one_group((date,'A',20,True))
