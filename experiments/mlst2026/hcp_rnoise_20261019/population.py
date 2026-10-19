"""D3 certificate/general population path and R1 D3-S without borrowed inflation bracket."""
import numpy as np
from experiments.mlst2026.meg_z_20261013 import population as old
from experiments.mlst2026.meg_z_20261013.estimators import kronecker_flipflop_from_S
from experiments.mlst2026.meg_z_r1_20261014.population import general_noise, d3s
from .protocol import load
from .readings import pz1
from . import controls
from .signal import build
from .gates import reference


def primary(model,Sref,gen,n_random=None):
    n_random=load()['signal']['population_random_starts'] if n_random is None else n_random
    St,Sc,_=kronecker_flipflop_from_S(Sref,model.C,model.T)
    cert=old.b_dagger(model.second_moment(),Sref,St,Sc,2,3)
    row=old.general_row(model.second_moment(),Sref,model.C,model.T,2,3,model.theta,model.A,model.B1,gen,n_random,B_cert=cert['B_dagger'])
    row['B_dagger']=cert;row['prestated_PZ2_miss_reasons']=load()['inference']['miss_reasons']
    return row


class SubjectPopulation:
    """Duck-typed for R1 theta_for_target; all signal choices precede samples."""
    def __init__(self,Sref,date_index): self.Sref=Sref;self.date_index=date_index
    def row(self,theta,n_random,stream,crosscheck=False):
        # Stable stream IDs use SHA256, never Python's randomized hash().
        import hashlib
        key=int.from_bytes(hashlib.sha256(stream.encode()).digest()[:4],'little')
        gen,_=controls.rng(0,key,self.date_index,0)
        model=build(self.Sref,theta);row=primary(model,self.Sref,gen,n_random)
        row['restart_agreement']=old.restart_agreement(row,.75)
        return row


def predictions(Sref,P,date_index=0):
    from .windows import grids
    cfg=load()['signal'];sp=SubjectPopulation(Sref,date_index);gg=grids(P)
    rows=[sp.row(t,cfg['population_random_starts'],f'fixed-{t}') for t in cfg['theta_grid']]
    rule={'theta_bracket':cfg['theta_search_bracket'],'search_random_starts':cfg['theta_search_random_starts'],'xtol':cfg['theta_search_xtol']}
    adaptive={name:old.theta_for_target(sp,gg[name],rule,cfg['population_random_starts'],name) for name in ('A','B')}
    return {'fixed':rows,'adaptive':adaptive,'grids':gg,'pre_fit':True,'PZ1':pz1(rows)}


def d3s_summary(models,shape,unit_metadata,date_index,m_train):
    cfg=load()['reference'];gen,_=controls.rng(2,cfg['D3S_reference_seed_cell'],date_index,0)
    def draw(g): return reference(controls.generate(shape,unit_metadata,2,g)['ref'][0])
    result=d3s(models,shape['population'],draw,gen,m_train,n_estimates=cfg['D3S_estimates'])
    rows=[r for estimate in result['rows'] for r in (estimate if isinstance(estimate,list) else [estimate])]
    result['trace_N_quantiles']=np.quantile([r['trace_N_sim_over_d'] for r in rows],[.025,.5,.975]).tolist()
    # An arithmetic independent-Wishart comparator, explicitly not a calibrated range.
    from .windows import plan
    p=plan(unit_metadata['n_times'],unit_metadata['sfreq_hz']);T=p['T'];d=shape['population'].shape[0]
    n=sum((b['end']-b['active_start'])//T for b in p['blocks'] if b['role']=='ref')
    result['stride_T_Wishart_heuristic']={'arithmetic_count':n,'not_ESS':True,'mean_scale':n/(n-d-1) if n>d+1 else None}
    return result
