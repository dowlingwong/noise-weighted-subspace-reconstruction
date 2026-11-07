"""Adversarial public-ingestion and worker controls derived from the sealed ledger."""
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.mlst2026.hcp_xopt_20261106 import family, report, run, select
from experiments.mlst2026.hcp_xopt_20261106.config import ROOT

@pytest.fixture(scope='session')
def base():
    rows=[]
    with open(run.FITS) as f:
        for line in f:
            r=json.loads(line)
            rows.append({k:r[k] for k in ('date','replicate','geometry','N','arm','m','D')})
    return rows

@pytest.fixture
def reduce(monkeypatch,tmp_path):
    # Ancillary diagnostics are irrelevant to the ingestion->family->verdict invariant.
    for name in ('reproducibility','route_statistics','per_cell_arm_delta'):
        monkeypatch.setattr(report,name,lambda rows:{})
    def go(rows):
        p=tmp_path/'ledger.jsonl'
        with p.open('w') as f:
            for r in rows:
                f.write(json.dumps(dict(r,status='ok',original={'D':r['D']},delta={'D':0.0}))+'\n')
        try:
            return report.main(str(p),str(tmp_path/'evidence.json'))['outcome']
        except family.IncompleteLedger:
            return {'family':'INCOMPLETE'}
    return go

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_ok_never_decision(base,reduce,value):
    rows=[dict(x) for x in base]; rows[0]['D']=value
    assert reduce(rows)['family']=='INCOMPLETE'

@pytest.mark.parametrize('kind',['missing','duplicate','pz4_hole','outside34_hole'])
def test_census_never_decision(base,reduce,kind):
    rows=list(base)
    if kind=='duplicate': rows.append(rows[0])
    elif kind=='pz4_hole': rows.pop(next(i for i,r in enumerate(rows) if r['geometry']=='A' and r['arm']=='lw'))
    elif kind=='outside34_hole': rows.pop(next(i for i,r in enumerate(rows) if r['geometry']=='control' and r['arm']=='lw'))
    else: rows.pop(0)
    assert reduce(rows)['family']=='INCOMPLETE'

def test_original_control_robust(base,reduce):
    out=reduce(base); assert out['family']=='ROBUST' and out['n']==0

def test_one_finite_value_can_change_decision(base,reduce):
    rows=[dict(x) for x in base]
    i=next(i for i,r in enumerate(rows) if r['geometry']=='control' and r['arm']=='reference')
    rows[i]['D']+=1e8
    out=reduce(rows)
    assert out['family']=='SENSITIVE'
    assert any(c['cell'].startswith('control/') for c in out['changes'])

@pytest.mark.parametrize('kind',['pz4','outside','nonfinite'])
def test_family_direct_guard(base,kind):
    D=family._ingest([(*[r[k] for k in ('date','replicate','geometry','N','arm','m')],r['D']) for r in base])
    key=next(k for k in D if k[4]=='lw' and k[2]==('A' if kind=='pz4' else 'control'))
    if kind=='nonfinite': D[key]=float('nan')
    else: del D[key]
    dates=sorted({k[0] for k in D})
    with pytest.raises(family.IncompleteLedger): family.family34(D,dates)

@pytest.mark.parametrize('content',[None,'{}','not json','[]'])
def test_invalid_freeze_writes_stop(monkeypatch,tmp_path,content):
    p=tmp_path/'FREEZE.json'
    if content is not None:p.write_text(content)
    monkeypatch.setattr(run,'FREEZE_JSON',str(p))
    args=SimpleNamespace(out=str(tmp_path/'out'))
    assert run.driver(args)!=0
    assert (Path(args.out)/'STOP.json').exists()

def worker_fixture(monkeypatch,tmp_path):
    meta=next(select.scan(run.FITS)); index=tmp_path/'index.jsonl'; index.write_text(json.dumps(meta)+'\n')
    return meta,index

def test_worker_read_failure_counted(monkeypatch,tmp_path):
    _,idx=worker_fixture(monkeypatch,tmp_path)
    def bad(*a,**k):raise OSError('injected read failure')
    monkeypatch.setattr(select,'read_row',bad)
    run.worker_main(0,1,'full',str(tmp_path),100,str(idx),time.time()+100)
    r=json.loads((tmp_path/'shards/shard_00.jsonl').read_text())
    assert r['status']=='failed' and r['error']['traceback_sha256']

def test_nan_replication_guard(monkeypatch,tmp_path):
    _,idx=worker_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(select,'load_inputs',lambda d: ({},None))
    monkeypatch.setattr(run,'build',lambda *a:None)
    monkeypatch.setattr(run,'process_row',lambda *a:{'D':float('nan'),'replay':{'obj_ok':True,'obj_rel':float('nan'),'D_over_spread':float('nan'),'D_ok':False}})
    run.worker_main(0,1,'full',str(tmp_path),100,str(idx),time.time()+100)
    text=(tmp_path/'shards/shard_00.jsonl').read_text()
    assert text and json.loads(text)['status']=='failed'

def test_workroot_imports():
    import sys
    for name,mod in list(sys.modules.items()):
        if name.startswith('experiments.mlst2026.') and getattr(mod,'__file__',None):
            assert Path(mod.__file__).resolve().is_relative_to(Path(ROOT)/'EXP')
