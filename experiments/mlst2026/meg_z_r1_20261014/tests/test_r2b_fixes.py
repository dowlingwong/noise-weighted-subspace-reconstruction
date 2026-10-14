"""R2-B review fixes (MS reviews/megz_r2_implementation_review_20261014_01; DESIGN rev 2.3, RD-12)."""
import hashlib,json,types
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.meg_z_r1_20261014 import controls as c,gates as g,windows as w,runner,gate_cli,protocol
from experiments.mlst2026.meg_z_r1_20261014.partition import assign_roles
K=protocol.PACKAGE
THR4=4*2.06604

def test_onset_locked_mean_unbiased_on_sub03_geometry():
    """F1: a 4x mean added to the continuous eval band stretch at offsets 30-39 (onset-locked, position 3) reads its
    true energy in the eval slot on the exact sub-03 geometry (restart tiling read about 4.6 here)."""
    runs=json.loads((K/'fixtures/DEV_GEOMETRY.json').read_text())['subjects']['sub-03']['runs']
    plans=[w.window_plan(R['gaps'],R['run']) for R in runs];E=[]
    for rep in range(3):
        rng=np.random.default_rng([1203,rep]);ys=[rng.standard_normal((R['G'],88,24)) for R in runs]
        base=c.cut_generated({'runs':ys},plans);S=c.reference(base['ref'][0]);_,_,L=g.coordinates(S)
        u=rng.standard_normal(240);u*=np.sqrt(THR4/np.sum((u@L)**2));profile=u.reshape(10,24)
        for R,y in zip(runs,ys):
            for gap,role in enumerate(assign_roles(len(R['gaps']),R['run'])):
                if role=='eval':y[gap,30:40]+=profile
        rec=c.cut_generated({'runs':ys},plans);rec['ref']=base['ref']
        r=c.assess(rec,False,Sref=S)
        E.append(next(s['E_cr'] for s in r['G3']['slots'] if s['role']=='eval' and s['position']==3))
    assert abs(np.mean(E)-THR4)<1.0 and min(E)>6.5

def test_fixed_grid_budget_audit():
    """F1: fixed-grid pools on the exact dev geometry still meet every budget; T = 10 and m_hi = 768 unchanged."""
    from experiments.mlst2026.meg_z_r1_20261014.metadata_census_audit import budget_audit
    r=budget_audit()
    assert r['pass'] and r['T']==10 and r['m_hi']==768
    assert r['budgets_with_margin']==pytest.approx({'train':1760.,'cal':844.8,'eval':550.})
    got={s:tuple(v['pools'][k] for k in ('train','cal','eval')) for s,v in r['subjects'].items()}
    assert got=={'sub-01':(2204,1088,1080),'sub-02':(2115,1084,1069),'sub-03':(1972,999,991),'sub-04':(1954,987,1000)}

def test_truncated_final_gap_keeps_full_band_mirror():
    """F6: a final eval gap cut short by the recording end keeps rposition = 7 - position; g3 does not raise."""
    geom=[{'gap':k,'start':100*k,'length':88,'stretches':[[0,88]]} for k in range(144)]
    geom.append({'gap':144,'start':14400,'length':50,'stretches':[[0,50]]})
    assert assign_roles(145,1)[-1]=='eval' and w.band_slots(10)==8
    plan=w.window_plan(geom,1);last=[r for r in plan['eval'] if r['gap']==144]
    assert [r['position'] for r in last]==[0,1,2,3,4] and [r['rposition'] for r in last]==[7,6,5,4,3]
    rows=plan['eval'];Z=np.random.default_rng(6).normal(size=(len(rows),12))
    meta={k:np.array([r[k] for r in rows]) for k in ('position','rposition','run','block')}
    assert len(g.g3({'eval':(Z,meta)})['slots'])==8

def synthetic_geom(G=48):
    return [{'run':r,'G':G,'gaps':[{'gap':k,'start':k*300,'length':88,'stretches':[[0,88]]} for k in range(G)]} for r in range(1,7)]

def test_run_blocked_pools_exclude_reference():
    """F4 / RD-12d: run-blocked G2-S pools train + cal + eval only; the reference role cannot move it."""
    gen,_=c.rng(2,0,0,3);rec=c.generate(c.identity_shape(2,88),synthetic_geom(),2,gen);S=c.reference(rec['ref'][0])
    before=c.assess(rec,False,Sref=S)['run_blocked_G2S']
    X,m=rec['ref'];X=X.copy();X[m['run']==1]*=10.;rec2=dict(rec);rec2['ref']=(X,m)
    after=c.assess(rec2,False,Sref=S)['run_blocked_G2S']
    for run in before:
        for key in ('m_eval/m_train','g_bottom50','g_middle40','g_top10','physical_trace_ratio'):
            assert after[run][key]==pytest.approx(before[run][key],rel=1e-12)

def test_C9_jitter_on_continuous_stretch_spills_into_positions_1_2():
    """F5 / RD-12e: the jittered profile is added to the continuous eval stretch, so delays spill into positions 1-2
    and nothing reaches position 3 or later (0.2 s + 9 samples < 30 samples)."""
    C=2;d=10*C;n_gaps=40
    meta={'position':np.tile(np.arange(8),n_gaps),'rposition':7-np.tile(np.arange(8),n_gaps),'run':np.ones(8*n_gaps,int),
          'block':np.repeat(np.arange(n_gaps),8)//12,'gap':np.repeat(np.arange(n_gaps),8),'start':np.arange(8*n_gaps)*10}
    X=np.zeros((8*n_gaps,d));rec={ro:(X.copy(),meta) for ro in ('ref','train','cal','eval')}
    out=c.apply_cell(rec,np.eye(d),{'C':C,'L':88},9,np.random.default_rng(5))[('eval')][0]
    energy=np.array([[np.sum(out[(meta['gap']==k)&(meta['position']==p)]**2) for p in range(8)] for k in range(n_gaps)])
    assert np.all(energy[:,3:]==0) and np.sum(energy[:,1]>0)>0 and np.sum(energy[:,2]>0)>0
    for k in range(n_gaps):
        s=np.concatenate([out[(meta['gap']==k)&(meta['position']==p)].reshape(10,C) for p in range(3)])
        nz=np.flatnonzero(np.any(s!=0,axis=1));assert len(nz) and nz[0]<=19   # onset delay <= 0.2 s (18.3 samples)

def test_G1_retest_records_first_files(tmp_path,monkeypatch):
    """F7: the once-only retest records the first derived file and summary before they are rewritten."""
    h5=tmp_path/'sub-01.h5';h5.write_bytes(b'first derived file');summary=tmp_path/'sub-01_summary.json'
    summary.write_text('{"pool_counts": {"evaluation": 600}}');first_sha=hashlib.sha256(b'first derived file').hexdigest()
    drops=[]
    def subject(path):
        return {'G1':{'pass':bool(drops)},'excluded':not drops,'derived_file_sha256':gate_cli.sha_file(path)}
    def process(raw,sub,derived,expected,drop):
        drops.append(list(drop));h5.write_bytes(b'retest derived file');summary.write_text('{}')
    sd=types.SimpleNamespace(attrs={},C=2,T=2,channels=['a','b'],ref=types.SimpleNamespace(S=np.diag([5.,1.,5.,1.])))
    monkeypatch.setattr(gate_cli,'subject',subject);monkeypatch.setattr(gate_cli,'process_subject',process)
    monkeypatch.setattr(gate_cli,'load_subject',lambda path,cache=False:sd)
    r=gate_cli.with_retest(tmp_path,'sub-01','/raw',['a','b'])
    assert drops==[['a']] and r['G1']['pass'] and r['G1_retest']['count']==1
    assert r['G1_retest']['first']['derived_file_sha256']==first_sha
    assert r['G1_retest']['first_files']=={'derived_file_sha256':first_sha,
        'summary_sha256':hashlib.sha256(b'{"pool_counts": {"evaluation": 600}}').hexdigest(),'summary':{'pool_counts':{'evaluation':600}}}

def test_preflight_refuses_existing_output_before_any_input(tmp_path,monkeypatch):
    """F8: a preflight never overwrites returned evidence; it stops before reading any input."""
    out=tmp_path/'PREFLIGHT.json';out.write_text('{}');touched=[]
    monkeypatch.setattr(runner,'environment',lambda:touched.append('env'))
    with pytest.raises(RuntimeError,match='exists'):runner.preflight('/no-data','/no-gates',out)
    assert not touched and out.read_text()=='{}'

def test_controls_cli_requires_explicit_preflight():
    """F8: the campaign names its controls-only preflight explicitly."""
    with pytest.raises(SystemExit):
        runner.main(['controls','--derived','d','--gates','g','--out','o','--reviewed-commit','c','--reviewed-identity','i','--review','r'])
