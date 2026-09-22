import numpy as np
import pytest
from experiments.mlst2026.upgrade_20260922 import c1,cresst as c

def test_mapback_sign_and_wrong_symmetric_space():
    a=np.diag([3.,1.]);eps=1e-5;e=np.array([[0,eps],[eps,0]])
    r,z=c1.response(a,e,1)
    assert abs(z['deltaP'][1,0]/eps+.5)<1e-12
    assert abs(z['P_hat'][1,0]/eps+.5)<1e-8
    ss,si=c1.powers(np.eye(2)+e);_,u=c1.eig(si@a@si)
    wrong=u[:,:1]@u[:,:1].T
    assert abs(wrong[1,0]/eps+1)<1e-8
    assert np.linalg.norm(a@np.linalg.inv(np.eye(2)+e)@z['basis']-z['basis']@(z['basis'].T@a@np.linalg.inv(np.eye(2)+e)@z['basis']))<1e-10

def test_domain_spd_commuting_and_direction():
    a=np.diag([9.,7,5,3,1,1]);lam=np.diag(a)
    for eta in [.01,.03,.1,.3]:
        e=np.zeros((6,6));e[3,4]=e[4,3]=eta*2/9
        r,_=c1.response(a,e,4);assert (r['domain']=='in_domain')==(eta<=.1)
    with pytest.raises(ValueError):c1.response(a,-np.eye(6),4)
    r,_=c1.response(a,np.diag([.001,.002,.003,.004,.005,.006]),4);assert r['displacement']<1e-12
    near=c1.direction('near',lam,np.random.default_rng(1));far=c1.direction('far',lam,np.random.default_rng(1))
    assert c1.response(a,near*.001,4)[0]['prediction']>c1.response(a,far*.001,4)[0]['prediction']

def test_sample_reference_nested_actual_covariance():
    result,z=c1.cell({'d':8,'spectrum':'S1','stage':'development_pilot','seed':0})
    rows=[r for r in result['records'] if r['part']=='estimated']
    assert len({r['training_sha256'] for r in rows})==1
    for r in rows:
        prefix=r['array_prefix'];assert r['prefix_sha256']==c1.digest(z['calibration_pool'][:r['m']])
        assert np.array_equal(z[prefix+'_A'],z['sample_A'])
        si=c1.powers(z['true_covariance'])[1]
        assert np.allclose(z[prefix+'_E'],si@z[prefix+'_effective_covariance']@si-np.eye(8))
    with pytest.raises(ValueError):c1.fit_baseline([{'cell':{'stage':'confirm'},'result':result}])

def test_support_functional_and_endpoints():
    d=16;t,q=c.support(d,4);assert np.linalg.matrix_rank(t)==d-1
    assert np.linalg.norm(t@np.ones(d))<1e-12
    assert not np.allclose(q.T@np.ones(d),0) # range is not zero-sum
    h=np.zeros(d);h[5:12]=np.array([.2,.5,.9,1,.7,.4,.1]);pulses=np.tile(h,(8,1))
    t,q,b,read=c.template(pulses,4,(5,12),.001)
    y=np.random.default_rng(5).normal(size=(3,2,d-1));clean=np.zeros_like(y);amp=np.zeros((3,2))
    ident=c.endpoints(y,clean,amp,read,(np.zeros(d-1),np.eye(d-1)),np.eye(d-1))
    zero=c.endpoints(y,clean,amp,read,(np.zeros(d-1),np.zeros((d-1,d-1))),np.eye(d-1))
    assert ident['amplitude_mse'].mean()>0 and zero['amplitude_mse'].mean()==0
    raw=np.random.default_rng(4).normal(size=(8,d));q0=t.T@q@read
    assert abs(q0.sum())<1e-10
    assert np.allclose((raw@t.T@q)@read,raw@q0)
    # Deliberate matched-filter degeneracy: projection containing h preserves readout.
    h=b[:,:1];metric=np.diag(np.linspace(1,2,d-1));cov=np.linalg.inv(metric);proj=c.gls(h,cov)
    matched=metric@h[:,0]/(h[:,0]@metric@h[:,0])
    assert np.allclose(y@proj.T@matched,y@matched)
    # Area readout generally is not that invariant matched filter.
    assert not np.allclose(y@proj.T@read,y@read)

def test_known_covariance_risk_identity():
    rng=np.random.default_rng(2);d=3;cov=np.diag([1,2,3.]);m=np.diag([2,.5,1]);b=np.array([[.5,.1,0],[0,.7,0],[.2,0,.1]])
    noise=rng.normal(size=(100000,d))@np.linalg.cholesky(cov).T;s=np.ones((100000,d));recon=(s+noise)@b.T+.2
    risk=lambda e:np.einsum('ni,ij,nj->n',e,m,e).mean()
    assert abs(risk(recon-s)-risk(recon-s-noise)-c.correction(b,cov,m))<.08
    with pytest.raises(ValueError):c.correction(b,cov,m,linear=False)

def test_fixed_affine_fairness_and_block_fallback():
    rng=np.random.default_rng(3);train=rng.normal(size=(32,8));cov=np.diag(np.arange(1,9));basis=rng.normal(size=(8,3));pairs=c.maps(train,cov,basis,3)
    for name in ['pca','pca_gls','weighted_gls']:
        a,b=pairs[name];assert np.linalg.matrix_rank(b)==3;assert np.allclose(a+b@train.mean(0),train.mean(0))
    inp,clean,amp=c.bank(rng.normal(size=(10,8)),basis,2.,1)
    assert np.allclose(inp-clean,(inp-clean)[:,0:1,:])
    r=c.block_interval(np.arange(19.),1,0);assert r['interval'] is None and r['scope']=='descriptive'

def test_qr_transport_descent_counterexample():
    x=np.ones((2,1));u=np.array([[1.],[0.]]);old=np.array([[1.]])
    e=u.T@x;unc=x@e.T@np.linalg.inv(e@e.T);new,rotation=np.linalg.qr(unc);gamma=rotation@e
    loss=lambda a,b:float(np.sum((x-a@b)**2))
    assert loss(new,gamma)<=loss(unc,e)+1e-12<=loss(u,e)+1e-12<=loss(u,old)+1e-12
    assert loss(u,gamma)>loss(u,old) # old appendix incorrectly used transported coefficient here

def test_unequal_block_sizes_consistent_estimate():
    values=np.r_[np.zeros(20),np.ones(21)*100]
    r=c.block_interval(values,2,0)
    assert r['blocks']==20 and r['mean']==values.mean() and r['interval'] is not None
