import numpy as np
from numpy.testing import assert_allclose
from experiments.mlst2026.t25_production_20260930.science import *
from experiments.mlst2026.t25_production_20260930.protocol import BASE,FIXTURE,enumeration

def test_gradient_gls_and_vectorization():
    r=rng('gradient-fixture');C,T,kc,kt=3,4,2,2;d=C*T
    u,v=qr(r.normal(size=(C,kc))),qr(r.normal(size=(T,kt)))
    z=r.normal(size=(d,d));W=z@z.T+np.eye(d);x=r.normal(size=(60,d));S=x.T@x/60
    val,gc,gt=objective_gradient(u,v,W,S);a,b=r.normal(size=u.shape),r.normal(size=v.shape);h=1e-6
    fd=(objective_gradient(u+h*a,v+h*b,W,S)[0]-objective_gradient(u-h*a,v-h*b,W,S)[0])/(2*h)
    assert_allclose(fd,np.sum(gc*a)+np.sum(gt*b),rtol=1e-7)
    B=np.kron(v,u);A=map_for(B,W);res=x-x@A.T
    assert_allclose(np.einsum('ni,ij,nj->',res,W,res)/len(x),np.trace(W@S)-val,atol=1e-9)
    coeff=r.normal(size=(kc,kt));assert_allclose(B@coeff.reshape(-1,order='F'),(u@coeff@v.T).reshape(-1,order='F'))

def test_exact_affine_risk_sigma_points():
    r=rng('affine-fixture');d=5;z=r.normal(size=(d,d));Cs=z@z.T;z=r.normal(size=(d,d));sig=z@z.T+np.eye(d)
    A=r.normal(size=(d,d));b=r.normal(size=d);W=np.linalg.inv(sig);ls=np.linalg.cholesky(Cs);ln=np.linalg.cholesky(sig)
    # exact second moments, independent Cartesian signal/noise sigma points
    s=np.r_[np.sqrt(d)*ls.T,-np.sqrt(d)*ls.T];n=np.r_[np.sqrt(d)*ln.T,-np.sqrt(d)*ln.T]
    error=(s[:,None,:]+n[None,:,:])@A.T+b-s[:,None,:]
    exact=np.einsum('abi,ij,abj->',error,W,error)/(4*d*d)
    assert_allclose(risks(A,b,Cs,sig,W)['clean'],exact,rtol=1e-13)

def test_solver_white_control_fixed_physical_class():
    c=FIXTURE;g,sig,L,Li,cal,x=make_data(c,900001,0.)
    S=np.cov(x,rowvar=False,bias=True);initial=starts(S,g.C,g.T,g.k_c,g.k_t,10)
    B,d,fac=structured(np.eye(g.d),S,initial,40)
    u,v=fac[d['selected']];assert_allclose(B,np.kron(v,u))
    assert d['selected']==int(np.argmax([a['objective'] for a in d['restarts']]))
    for row in d['restarts']:assert np.min(np.diff(row['trajectory']))>=-1e-10
    # full physical metric must still yield a literal product decoder
    W=np.linalg.inv(np.cov(cal,rowvar=False)+.1*np.eye(g.d));B,d,fac=structured(W,S,initial,40)
    u,v=fac[d['selected']];assert_allclose(B,np.kron(v,u));A=map_for(B,W)
    assert_allclose(A@A,A,atol=1e-10);assert np.linalg.matrix_rank(A)==g.k
    iso=unrestricted(np.eye(g.d),S,g.k);oracle=unrestricted(np.linalg.inv(np.eye(g.d)),S,g.k)
    assert_allclose(iso@iso.T,oracle@oracle.T)

def test_geometry_calibration_and_nested_data():
    c=FIXTURE;g,sigma,L,Li,pool,x=make_data(c,900001,.3)
    assert_allclose(np.linalg.eigvalsh(g.K(.3)),np.sort(g.mu),atol=1e-10)
    covs,d=estimate(pool[:32],g.C,g.T)
    assert_allclose(d['kron']['trace_channel'],g.C,atol=1e-12)
    assert min(np.linalg.eigvalsh(covs['full']))>0;assert min(np.linalg.eigvalsh(covs['kron']))>0
    _,_,_,_,p2,x2=make_data(c,900001,0.)
    assert_allclose(pool,p2);assert digest(x)!=digest(x2)
    assert digest(pool)!=digest(make_data(c,900002,.3)[4])
    # the oracle population map's propagated noise is exactly rank
    B=L@top(g.K(.3),g.k);A=map_for(B,Li@Li)
    assert_allclose(risks(A,np.zeros(g.d),L@g.K(.3)@L,sigma,Li@Li)['noise'],g.k,atol=1e-10)

def test_allocation_contrasts_profiles():
    e=enumeration(BASE);assert e['total_charges']==30536;assert e['unique_component_records']==638976
    methods={v+'__'+cl:{'clean':1.} for v in ['iso','oracle']+[f'{a}_{m}' for m in BASE['m'] for a in ('full','kron')] for cl in ('restricted','unrestricted')}
    assert len(contrast_values(methods,BASE['m']))==36
    p=profile([{'N':1,'ci':[-2,-1]},{'N':2,'ci':[1,2]},{'N':3,'ci':[-2,-1]}]);assert p['nonmonotone'];assert p['adjacent_brackets']==[[1,2],[2,3]]
    assert profile([{'N':1,'ci':[0,0]},{'N':2,'ci':[-1,1]}])['signs']==['?','?']

def test_population_oracle_and_structured_white_limits():
    c=FIXTURE;g=geometry(c);K=g.K(0.);sigma=covariance(c);L,Li=powers(sigma)
    # Exact population product optimum supplied as one valid start.
    uc=qr(powers(.4**np.abs(np.arange(g.C)[:,None]-np.arange(g.C)))[0]@g.U_c0)
    ut=qr(powers(.7**np.abs(np.arange(g.T)[:,None]-np.arange(g.T)))[0]@g.U_t0)
    S=L@(K+np.eye(g.d))@L;B,_,_=structured(Li@Li,S,[(uc,ut)],150)
    expected=L@top(K,g.k)
    assert_allclose(qr(Li@B)@qr(Li@B).T,qr(Li@expected)@qr(Li@expected).T,atol=1e-8)
    init=starts(K+np.eye(g.d),g.C,g.T,g.k_c,g.k_t,42)
    b1,_,_=structured(np.eye(g.d),K+np.eye(g.d),init,150)
    b2,_,_=structured(np.linalg.inv(np.eye(g.d)),K+np.eye(g.d),init,150)
    assert_allclose(b1@b1.T,b2@b2.T,atol=1e-12)
    refs=population_references(c);assert refs[0]['approximation_lower']==0;assert refs[0]['feasible_upper']==0
    assert refs[1]['approximation_lower']<=refs[1]['feasible_upper']
