"""DESIGN 2: primary old general_row under S_ref; D3-S is descriptive only."""
import numpy as np
from experiments.mlst2026.meg_z_20261013 import population as old
from experiments.mlst2026.meg_z_20261013.common import symmetric
PZ2_REASONS=['whitened noise scale c != 1','reference-estimation tilt: larger N*','run-mixture non-Gaussianity of pooled training role']

def primary(model,Sref,gen,n_random=62):
    row=old.general_row(model.second_moment(),Sref,model.C,model.T,2,3,model.theta,model.A,model.B1,gen,n_random)
    n=row['N_star_pred'];row['N_star_bracket']=None if n is None else [(n+1)*c-1 for c in (1.013,1.14)]
    row['prestated_PZ2_miss_reasons']=PZ2_REASONS
    return row

def general_noise(model,Sref,Nphysical,gen,n_random=62):
    """General M=K+N_sim path. Opposite noise signs in risk/learner objectives.

    a_hat is the nominal K+I approximation contrast under this same metric.
    delta=a_hat-D_W uses the actual K+N learner and K-N risk.
    N* is the descriptive S4.3 analogue recomputed with M (no gate).
    """
    nominal=primary(model,Sref,gen,n_random)
    _,A=old.spd_powers(Sref);K=symmetric(A@model.second_moment()@A);N=symmetric(A@Nphysical@A);M=K+N
    C,T=model.C,model.T;kc,kt=2,3;k=6
    cls=old.GeneralProductClass(A,M,C,T,kc,kt)
    S4=(model.second_moment()+Nphysical).reshape(C,T,C,T,order='F')
    top=lambda z,r:np.linalg.eigh(symmetric(z))[1][:,-r:]
    starts=[(top(np.einsum('ctdt->cd',S4),kc),top(np.einsum('ctcs->ts',S4),kt)),(model.A,model.B1)]
    starts += [(old.core.orth(gen.normal(size=(C,kc))),old.core.orth(gen.normal(size=(T,kt)))) for _ in range(n_random)]
    found=[cls.refine(cls.chart(c,t)) for c,t in starts];pt,info=max(found,key=lambda v:v[1]['value'])
    fo=cls.first_order(pt);Q,_=np.linalg.qr(cls.basis(pt));PF=Q@Q.T
    ell,U=np.linalg.eigh(M);order=np.argsort(ell)[::-1];ell=ell[order];U=U[:,order[:k]];PU=U@U.T
    a=float(np.sum(ell[:k])-info['value']);DW=float(np.trace((PU-PF)@(K-N)));delta=nominal['a_best']-DW
    cU=old.core.unrestricted_coefficient(ell,k);cF=fo['cF_gaussian'];B=cF-cU+a
    return {'a_hat':nominal['a_best'],'objective_gap_M':a,'D_W':DW,'delta':delta,'N_star_pred':float((cU-cF)/a-1) if a>1e-9 and B<0 else None,
        'cU':float(cU),'cF':float(cF),'trace_N_sim_over_d':float(np.trace(N)/len(N)),
        'H_min_eig':fo['H_min_eig'],'descriptive':True,'n_starts':len(starts)}

def d3s(model,source_cov,reference_draw,gen,m_train,n_estimates=50,n_random=62):
    if n_estimates!=50: raise ValueError('D3-S requires exactly 50 reference estimates')
    models=model if isinstance(model,(list,tuple)) else [model]
    rows=[]
    for estimate in range(50):
        S=reference_draw(gen)
        rows.append([general_noise(m,S,source_cov,gen,n_random) for m in models] if len(models)>1 else general_noise(models[0],S,source_cov,gen,n_random))
    return {'descriptive':True,'never_gate':True,'observed_m_train':float(m_train),'n_estimates':50,'rows':rows}

b_dagger=old.b_dagger
