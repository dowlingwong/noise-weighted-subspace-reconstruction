import numpy as np
from experiments.mlst2026.hcp_rnoise_20261019.signal import build
from experiments.mlst2026.hcp_rnoise_20261019.population import primary,general_noise

def test_population_certificate_and_general_noise():
    # Fabricated isotropic covariance, fixed header-derived synthetic fields.
    S=np.eye(72)*1e-24;gen=np.random.Generator(np.random.PCG64DXSM(19))
    m=build(S,0.)
    zero=primary(m,S,gen,n_random=0)
    assert zero['B_cert']==0 and abs(zero['a_best'])<1e-7
    assert 'N_star_bracket' not in zero
    m=build(S,.4)
    row=primary(m,S,gen,n_random=0)
    assert row['B_cert']>0 and row['a_best']+1e-7>=row['B_cert']
    varied=general_noise(m,S,1.2*S,gen,n_random=0)
    assert abs(varied['trace_N_sim_over_d']-1.2)<1e-12
    assert varied['descriptive'] is True and 'delta' in varied


def test_fabricated_learning_uses_block_draws_and_shared_reference_score():
    from experiments.mlst2026.hcp_rnoise_20261019 import controls,gates,protocol
    from experiments.mlst2026.hcp_rnoise_20261019.learning import fit_cell
    u=next(u for u in protocol.split()['units'] if u['split']=='dev')
    gen,_=controls.rng(2,99,0,999);shape=controls.synthetic_shape()
    rec=controls.generate(shape,u,2,gen);S=gates.reference(rec['ref'][0]);m=build(S,.12)
    row=fit_cell(rec,m,S,56,256,'reference',0,999)
    assert len(set(row['sampled_training_indices']))==56
    assert np.isfinite(row['D']) and np.isfinite(row['U']['clean_ref']) and np.isfinite(row['F']['clean_ref'])

def test_d3s_uses50_reference_draws_and_fixed_physical_models(monkeypatch):
    from experiments.mlst2026.hcp_rnoise_20261019 import population,controls,protocol
    from experiments.mlst2026.meg_z_r1_20261014 import population as r1
    u=next(u for u in protocol.split()['units'] if u['split']=='dev')
    shape=controls.synthetic_shape();S=shape['population'];m0=build(S,0.);m4=build(S,.4)
    expected={id(m0):m0.variances.copy(),id(m4):m4.variances.copy()};seen=[];draws=[]
    def generator(shape,unit,law,gen):
        X=gen.normal(size=(200,72))*1e-12;draws.append(X)
        return {'ref':(X,{})}
    def general(model,Sref,Nphysical,gen,n_random=62):
        assert np.array_equal(model.variances,expected[id(model)])
        seen.append((id(model),float(np.trace(Sref))))
        return {'trace_N_sim_over_d':1.1,'delta':0.,'descriptive':True}
    monkeypatch.setattr(controls,'generate',generator)
    monkeypatch.setattr(r1,'general_noise',general)
    out=population.d3s_summary([m0,m4],shape,u,0,1.05)
    assert out['n_estimates']==50 and len(draws)==50 and len(seen)==100
    assert out['trace_N_quantiles']==[1.1,1.1,1.1]
    assert out['stride_T_Wishart_heuristic']['arithmetic_count']==606
    assert out['stride_T_Wishart_heuristic']['not_ESS'] is True
