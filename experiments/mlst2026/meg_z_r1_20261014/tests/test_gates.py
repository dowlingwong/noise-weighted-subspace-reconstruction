import numpy as np
import pytest
from experiments.mlst2026.meg_z_r1_20261014 import gates as g

def test_role_covariance_unshrunk_divisor_n_and_shift_invariant():
    X=np.arange(36).reshape(12,3);S=g.reference(X)
    assert np.allclose(S,(X-X.mean(0)).T@(X-X.mean(0))/12)
    assert np.allclose(S,g.reference(X+[10,20,30]))

def test_g2_own_eigenvectors_and_five_statistics():
    gen=np.random.default_rng(23);d=20;S=np.diag(np.linspace(1,100,d));X=gen.normal(size=(12000,d))*np.sqrt(np.diag(S));Y=gen.normal(size=(12000,d))*np.sqrt(np.diag(S))
    null=g.role_g2(X,Y,S);assert null['pass'];assert len(null['generalised_eigenvalues'])==d
    Y[:,18:]*=np.sqrt(1.5);drift=g.role_g2(X,Y,S);assert not drift['pass'] and drift['g_top10']>1.25
    assert drift['m_eval/m_train']<1.25

def test_g2_physical_ratio_separate_and_no_inflation_correction():
    qt=np.ones(20)*1.03;qe=qt.copy();r=g.g2_diagonal(qt,qe,1,1.4,details=False)
    assert r['m_train']==pytest.approx(1.03) and not r['pass'] and r['physical_trace_ratio']==1.4
    assert all(r[k]==1 for k in ('m_eval/m_train','g_bottom50','g_middle40','g_top10'))

def test_cluster_formula_exact():
    gen=np.random.default_rng(22);Z=gen.normal(size=(88,9));keys=np.repeat(np.arange(8),11)
    r=g.cluster_energy(Z,keys);mu=Z.mean(0);u=np.stack([(Z[keys==k]-mu).sum(0) for k in range(8)])
    assert r['E_cr']==pytest.approx(mu@mu-8/7*np.sum(u*u)/88**2)
    assert r['naive_debias']==pytest.approx(mu@mu-np.sum((Z-mu)**2)/(88*87))

@pytest.mark.parametrize('n,B',[(29,8),(88,7)])
def test_cluster_minima(n,B):
    Z=np.ones((n,9));keys=np.arange(n)%B;assert not g.cluster_energy(Z,keys)['evaluated']

@pytest.mark.parametrize('a',[0.,.5])
def test_cluster_independent_and_ar05_MC(a):
    gen=np.random.default_rng(int(320+a*1000));n=33*11;d=24;energies=[];naives=[];fractions=[]
    for _ in range(160):
        X=gen.normal(size=(33,11,d))
        for j in range(1,11):X[:,j]=a*X[:,j-1]+np.sqrt(1-a*a)*X[:,j]
        r=g.cluster_energy(X.reshape(n,d),np.repeat(np.arange(33),11));energies.append(r['E_cr']);naives.append(r['naive_debias']);fractions.append(r['cluster_term']/r['naive_term'])
    f=1+2/11*sum((11-l)*a**l for l in range(1,11))
    assert abs(np.mean(energies))<.08
    assert abs(np.mean(fractions)-f)<.12
    assert abs(np.mean(naives)-(f-1)*d/n)<.08

def test_g3_strict_threshold_screen_and_dedup():
    gen=np.random.default_rng(81);Z=gen.normal(size=(352,12));meta={'position':np.zeros(352,int),'rposition':np.full(352,7),'run':np.ones(352,int),'block':np.repeat(np.arange(32),11)}
    r=g.g3({'eval':(Z,meta),'cal':(Z,meta),'train':(Z,meta)});assert len(r['slots'])==3 and r['pass']
    locked=Z-Z.mean(0);locked[:,0]+=np.sqrt(4*2.06604)
    r=g.g3({'eval':(locked,meta),'cal':(Z,meta),'train':(Z,meta)});assert r['slot_fail']

def test_once_only_retest_and_failure_exclusion():
    S=np.diag([2e7,1.,2e7,1.]);calls=[]
    def rebuild(i):calls.append(i);return np.diag([1.,1e12])
    _,channels,r=g.g1_with_retest(S,2,2,['loud','quiet'],rebuild)
    assert calls==[0] and channels==['quiet'] and r['retest']['count']==1 and r['excluded']

def test_e1_dev_and_budget_rules():
    ok={s:{k:{'pass':True} for k in ('G1','G2','G3')} for s in ('a','b','c','d')};ok['d']['G3']['pass']=False
    assert not g.e1(ok)['stop'];ok['c']['G2']['pass']=False;assert g.e1(ok)['stop']
    assert g.budget_eval(500)['pass'] and g.budget_eval(499)['excluded']
    assert g.choose_m_hi({'a':1085,'b':1081,'c':1016,'d':1004})==768
    assert g.choose_m_hi(dict.fromkeys('abcd',600))==512
    with pytest.raises(ValueError):g.choose_m_hi(dict.fromkeys('abcd',500))

def test_n_min_per_cell_upper80_SD_rule_and_no_pass_branch():
    """R2-B F3 / RD-12c: registered power rule per ref-arm cell; remedies once, then STOP and route A."""
    from scipy.stats import t
    assert g.sd_upper(1,3)==pytest.approx(2.117,abs=.002)
    assert g.sd_upper(1,4)==pytest.approx(1.728,abs=.002)
    r=g.n_min(.1,4,[3],1);assert r['n_min']==8 and r['pass']
    sd=np.array([.5,.9,.2]);pred=np.array([2.,2.6,1.]);F=3
    want=next(n for n in range(8,13) if np.all(t.ppf(1-.05/(2*F),n-1)*sd*g.sd_multiplier(4)/np.sqrt(n)<.5*np.abs(pred)))
    r=g.n_min(sd,4,pred,F);assert r['pass'] and r['n_min']==want and want>8
    blocked=g.n_min([.1,5.],4,[3,3],2);assert not blocked['pass'] and blocked['failing_cells']==[1]
    with pytest.raises(ValueError):g.n_min([.1,.2],4,[3],1)
    first=g.n_min(10,4,[.001],35);assert not first['pass'] and not first['stop'] and 'remedy_required' in first
    final=g.n_min(10,4,[.001],35,after_remedy=True)
    assert final['stop'] and final['route']=='A for the confirmation stage' and final['confirmation_access'] is False

def test_g4_compares_unshrunk_reference():
    S=np.kron(np.diag([1.,2.]),np.diag([3.,1.]));r=g.g4(S,2,2)
    assert r['kron_vs_unshrunk']['relative_frobenius']<1e-9

def test_retired_LW_descriptive_does_not_gate():
    gen=np.random.default_rng(8);X=gen.normal(size=(200,8))*np.arange(1,9);Y=gen.normal(size=(100,8))*np.arange(1,9)
    r=g.retired_descriptives(X,Y,g.reference(X),20)
    assert r['descriptive'] and r['reference']=='Ledoit-Wolf full' and 'pass' not in r
