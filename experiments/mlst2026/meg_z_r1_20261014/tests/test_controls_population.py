import numpy as np
import pytest,json
from experiments.mlst2026.meg_z_r1_20261014 import controls as c,protocol,population as p,signal,runner
from experiments.mlst2026.meg_z_20261013.signal import SignalModel,orth

def synthetic_geom(G=145):
    return [{'run':r,'G':G,'gaps':[{'gap':g,'start':g*300,'length':88,'stretches':[[0,88]]} for g in range(G)]} for r in range(1,7)]

def test_AR10_Yule_Walker_lags_and_stationary_covariance():
    from scipy.linalg import toeplitz
    S=np.kron(toeplitz(.7**np.arange(20)),np.diag([2.,4.]));shape=c.ar_shape(S,2,20)
    assert shape['fit']['order']==10 and shape['fit']['lags_11_19']==list(range(11,20))
    assert np.allclose(shape['fit']['coefficients'],[.7]+[0]*9,atol=1e-10)
    assert max(abs(x) for x in shape['fit']['relative_errors'])<1e-10
    assert np.trace(shape['Lc']@shape['Lc'].T)==pytest.approx(2)
    assert np.allclose(shape['Lt']@shape['Lt'].T,3*toeplitz(.7**np.arange(88)))

def test_exact_prescribed_seed_and_distinct_cells():
    a,entropy=c.rng(4,7,2,17);b,_=c.rng(4,7,2,17)
    assert entropy==[20261014,4,7,2,17] and a.bit_generator.__class__.__name__=='PCG64DXSM'
    assert np.array_equal(a.normal(size=10),b.normal(size=10));x,_=c.rng(4,8,2,17);assert not np.array_equal(x.normal(size=10),b.normal(size=10))

@pytest.mark.parametrize('law',range(1,8))
def test_laws_generation_finite_guard_reset_and_windows(law):
    shape=c.identity_shape(2,88);gen,_=c.rng(law,0,0,0);rec=c.generate(shape,synthetic_geom(36),law,gen)
    assert all(np.isfinite(X).all() for X,m in rec.values())
    for ro,(X,m) in rec.items():
        assert X.shape[1]==20
        assert np.all(m['rposition']==7-m['position'])
        assert len(X)==len(m['gap'])
    # Run starts are independent: seeded generation alone never carries a previous run's final gap.
    assert np.any(rec['train'][1]['run']==1) and np.any(rec['eval'][1]['run']==4)

@pytest.mark.parametrize('cell',range(13))
def test_cells_only_modify_eval_with_fixed_metric_and_group_definition(cell):
    gen=np.random.default_rng(24);d=20;X=gen.normal(size=(64,d));m={'position':np.arange(64)%8,'rposition':7-np.arange(64)%8,'run':np.ones(64,int),'block':np.arange(64)//8,'gap':np.arange(64),'start':np.arange(64)*10}
    rec={ro:(X.copy(),m) for ro in ('ref','train','cal','eval')};S=np.diag(np.arange(1,d+1));shape=c.identity_shape(2,88)
    out=c.apply_cell(rec,S,shape,cell,gen)
    for ro in ('ref','train','cal'):assert out[ro] is rec[ro]
    assert np.array_equal(rec['eval'][0],X)
    if cell==0:assert np.array_equal(out['eval'][0],X)
    elif cell==1:assert np.allclose(out['eval'][0],X*np.sqrt(1.5))
    elif cell in (7,8):
        delta=out['eval'][0]-X;u=delta[0];assert np.sum(u*u/np.diag(S))==pytest.approx((4 if cell==7 else 2)*2.06604)
        assert np.allclose(delta[m['position']!=0],0)
    assert np.isfinite(out['eval'][0]).all()

def test_35_claims_exact_and_CP_thresholds():
    claims=c.claims();assert len(claims)==35
    assert len([x for x in claims if x[0]=='G2 detection'])==18
    assert c.claim_bound(22,800,False)['pass'];assert not c.claim_bound(23,800,False)['pass']
    assert c.claim_bound(745,800,True)['pass'];assert not c.claim_bound(744,800,True)['pass']
    assert c.claim_bound(22,800,False)['one_sided_bound']<=.05
    assert c.claim_bound(745,800,True)['one_sided_bound']>=.90

def test_incomplete_campaign_summary_refused():
    with pytest.raises(RuntimeError):c.summarize([])

def test_source_H5_sha_refusal_before_open(tmp_path):
    (tmp_path/'sub-01.h5').write_bytes(b'no HDF5');gate={'subjects':{'sub-01':{'derived_file_sha256':'0'*64}}};(tmp_path/'gates.json').write_text(json.dumps(gate))
    with pytest.raises(RuntimeError,match='pre-read'):c.source_moments(tmp_path,tmp_path/'gates.json')

def model_fixture():
    gen=np.random.default_rng(921);C,T=4,7
    A=orth(gen.normal(size=(C,2)));B=orth(gen.normal(size=(T,3)));D=orth(gen.normal(size=(T,3))-B@(B.T@gen.normal(size=(T,3))))
    D=orth(D-B@(B.T@D));return SignalModel(A,B,D,.25,np.array([8.,4.,2.,8.,4.,2.]))

def test_primary_general_noise_identity_and_bracket():
    m=model_fixture();d=m.d;S=np.eye(d);g=np.random.default_rng(928)
    primary=p.primary(m,S,g,0);r=p.general_noise(m,S,S,np.random.default_rng(928),0)
    assert r['objective_gap_M']==pytest.approx(primary['a_best'],abs=1e-8)
    assert r['delta']==pytest.approx(0,abs=1e-8)
    assert r['cU']==pytest.approx(primary['cU'],rel=1e-8)
    assert r['cF']==pytest.approx(primary['cF'],rel=1e-8)
    if primary['N_star_pred'] is not None:
        n=primary['N_star_pred'];assert np.allclose(primary['N_star_bracket'],[(n+1)*1.013-1,(n+1)*1.14-1])
    assert len(primary['prestated_PZ2_miss_reasons'])==3

def test_D3S_50_estimates_descriptive(monkeypatch):
    seen=[]
    monkeypatch.setattr(p,'general_noise',lambda model,S,N,g,n_random:seen.append(S.copy()) or {'delta':.1,'trace_N_sim_over_d':1.02})
    r=p.d3s(None,np.eye(3),lambda g:np.eye(3),np.random.default_rng(1),1.03)
    assert len(seen)==50 and r['n_estimates']==50 and r['never_gate'] and r['observed_m_train']==1.03
    with pytest.raises(ValueError):p.d3s(None,None,None,None,1,n_estimates=49)

def test_B_dagger_unshrunk_metric():
    m=model_fixture();r=p.b_dagger(m.second_moment(),np.eye(m.d),np.eye(m.T),np.eye(m.C),2,3)
    assert r['lambda_min']==pytest.approx(1) and r['B_dagger']>=0

def test_campaign_requires_review_before_any_input(tmp_path):
    with pytest.raises((RuntimeError,FileNotFoundError,KeyError)):
        runner.campaign('/no-data','/no-gates',tmp_path/'out','0'*40,'0'*64,tmp_path/'missing-review')
    assert not (tmp_path/'out').exists()
