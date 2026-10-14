"""RD-11 budget scope only; no change to statistical content."""
import json
import pytest
from experiments.mlst2026.meg_z_r1_20261014 import runner,protocol

def test_RD11_independent_budgets_and_exact_boundaries():
    cfg=protocol.load()
    assert cfg['controls']['workers']==4 and cfg['controls']['budget_hours']==6
    assert '16 workers' in cfg['d3']['D3S_budget'] and '8 h' in cfg['d3']['D3S_budget']
    r=runner.budget_projection(6*4*3600,8*16*3600)
    assert r['controls_status']==r['D3S_status']==r['status']=='PASS'
    assert r['projected_control_hours_4_workers']==6
    assert r['projected_D3S_hours_16_workers']==8
    r=runner.budget_projection(6*4*3600+1,0)
    assert r['controls_status']=='STOP' and r['D3S_status']=='PASS' and r['status']=='STOP'
    r=runner.budget_projection(0,8*16*3600+1)
    assert r['controls_status']=='PASS' and r['D3S_status']=='STOP' and r['status']=='STOP'
    assert runner.budget_projection(0,0,complete=False)['D3S_status']=='STOP'

def test_RD11_previous_measured_costs_pass_separately():
    r=runner.budget_projection(46324.60257336497,346096.387963742)
    assert r['status']=='PASS'
    assert r['projected_control_hours_4_workers']==pytest.approx(3.217,abs=.001)
    assert r['projected_D3S_hours_16_workers']==pytest.approx(6.009,abs=.001)

@pytest.mark.parametrize('hours',[3.2,6.01])
def test_campaign_guard_uses_control_budget_only(tmp_path,monkeypatch,hours):
    monkeypatch.setattr(runner,'PACKAGE',tmp_path/'kit')
    monkeypatch.setattr(runner,'reviewed',lambda *args:None)
    monkeypatch.setattr(runner,'environment',lambda:{})
    calls=[]
    monkeypatch.setattr(runner,'context',lambda *args:calls.append(True) or ({},{},{}))
    p=tmp_path/'meg_z_r1_20261014_rd11';p.mkdir()
    (p/'PREFLIGHT.json').write_text(json.dumps({'status':'STOP','D3S_status':'STOP',
        'controls_status':'PASS' if hours<=6 else 'STOP','kit_identity':'test',
        'projected_control_hours_4_workers':hours}))
    # An existing output is a sentinel: stop after guard, before any campaign work.
    out=tmp_path/'already_exists';out.mkdir()
    expected='campaign output exists' if hours<=6 else 'controls-only preflight'
    with pytest.raises(RuntimeError,match=expected):
        runner.campaign(None,None,out,'commit','test','review')
    assert bool(calls)==(hours<=6)
