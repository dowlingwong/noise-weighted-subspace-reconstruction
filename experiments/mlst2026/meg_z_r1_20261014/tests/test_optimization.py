"""Seeded equivalence against the pre-optimization implementation at dffa807."""
import json,types
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import toeplitz
from experiments.mlst2026.meg_z_r1_20261014 import controls as c,runner
def synthetic_geom(G):
    return [{'run':r,'G':G,'gaps':[{'gap':g,'start':g*300,'length':88,'stretches':[[0,88]]} for g in range(G)]} for r in range(1,7)]

baseline=types.ModuleType('baseline_controls')
baseline.__package__=c.__package__
exec((Path(__file__).parent/'BASELINE_CONTROLS.txt').read_text(),baseline.__dict__)

@pytest.mark.parametrize('law',range(1,8))
def test_seeded_generation_and_all_cells_equivalent(law):
    S=np.kron(toeplitz(.7**np.arange(20)),np.diag([2.,4.]))
    shape=c.ar_shape(S,2,20);geometry=synthetic_geom(72)
    # Include exclusion at start/middle and shortened stretches, plus final gaps.
    for R in geometry:
        R['gaps'][3]['stretches']=[[4,24],[30,88]]
        R['gaps'][-1]['stretches']=[[0,73]]
    g1,_=c.rng(law,0,0,19);g2,_=c.rng(law,0,0,19)
    got=c.generate(shape,geometry,law,g1);want=baseline.generate(shape,geometry,law,g2)
    assert g1.bit_generator.state==g2.bit_generator.state
    for role in got:
        np.testing.assert_allclose(got[role][0],want[role][0],rtol=1e-11,atol=1e-12)
        for key in got[role][1]:np.testing.assert_array_equal(got[role][1][key],want[role][1][key])
    Sref=c.reference(want['ref'][0])
    for cell in range(13):
        # C9 (RD-12e) and the run-blocked pools (RD-12d) were redefined after the optimization pass (R2-B F4, F5);
        # they are tested in test_r2b_fixes.py. Everything else stays equivalent to dffa807.
        if cell==9:continue
        a,_=c.rng(law,cell,0,19);b,_=c.rng(law,cell,0,19)
        actual=c.assess(c.apply_cell(got,Sref,shape,cell,a),Sref=Sref)
        expected=baseline.assess(baseline.apply_cell(want,Sref,shape,cell,b),Sref=Sref)
        actual.pop('run_blocked_G2S');expected.pop('run_blocked_G2S')
        assert a.bit_generator.state==b.bit_generator.state
        def compare(x,y):
            if isinstance(x,dict):
                assert x.keys()==y.keys()
                for key in x:compare(x[key],y[key])
            elif isinstance(x,(list,tuple)):
                assert len(x)==len(y)
                for xx,yy in zip(x,y):compare(xx,yy)
            elif isinstance(x,(bool,str)) or x is None:assert x==y
            else:assert x==pytest.approx(y,rel=1e-8,abs=1e-9)
        compare(actual,expected)

def test_stationary_AR10_initialization_couples_dense_draw():
    gen=np.random.default_rng(41)
    # A true multi-lag AR(10), including very small physical amplitudes.
    from scipy.signal import lfilter
    a=np.array([.5,.12,-.06,.03,0,0,0,0,0,.01])
    impulse=lfilter([1.],np.r_[1.,-a],np.r_[1.,np.zeros(10000)])
    r=np.array([impulse[:len(impulse)-i]@impulse[i:] for i in range(20)])*1e-24
    shape=c.ar_shape(np.kron(toeplitz(r),np.diag([1.,3.])),2,20)
    xi=gen.normal(size=(40,88,2))
    want=np.einsum('ij,gjc->gic',shape['Lt'],xi,optimize=True)
    np.testing.assert_allclose(c.temporal_draw(shape,xi),want,rtol=1e-10,atol=1e-25)

def test_full_preflight_measures_all_D3S_even_when_controls_over_cap(tmp_path,monkeypatch):
    from experiments.mlst2026.meg_z_r1_20261014 import population,signal
    monkeypatch.setattr(signal,'temporal_bases',lambda:(np.eye(10)[:,:3],np.eye(10)[:,3:6],{}))
    monkeypatch.setattr(runner,'environment',lambda:{})
    monkeypatch.setattr(runner,'verify_kit',lambda:{})
    monkeypatch.setattr(runner,'context',lambda *args:({s:{'population':np.eye(240)} for s in runner.DEV},{'subjects':{s:{'runs':[]} for s in runner.DEV}},{}))
    monkeypatch.setattr(c,'replicate',lambda *args:{'subject':runner.DEV[args[2]],'seconds':2.})
    monkeypatch.setattr(c,'generate',lambda *args:{'ref':(np.zeros((300,240)),{})})
    monkeypatch.setattr(c,'reference',lambda *args:np.eye(240))
    # Avoid cost of real population solves; check exact loop coverage and STOP.
    calls=[]
    monkeypatch.setattr(population,'general_noise',lambda *args:calls.append(args[-1]))
    shapes={s:{'population':np.eye(240),'fit':{}} for s in runner.DEV}
    monkeypatch.setattr(runner,'context',lambda *args:(shapes,{'subjects':{s:{'runs':[]} for s in runner.DEV}},{}))
    out=tmp_path/'preflight.json'
    with pytest.raises(SystemExit):runner.preflight(None,None,out,full_timing=True)
    result=json.loads(out.read_text())
    assert calls==[62]*4 and all(r['complete'] for r in result['D3S_timings'])
    assert result['status']=='STOP'
