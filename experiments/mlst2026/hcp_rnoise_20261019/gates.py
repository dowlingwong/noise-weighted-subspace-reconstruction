"""Carried G1/G2/G4/G5; new whole-unit screen, budget and E1. No G3."""
import numpy as np
from experiments.mlst2026.meg_z_r1_20261014.gates import reference, coordinates, groups, role_g2, g4, n_min as _r1_n_min
from experiments.mlst2026.meg_z_20261013.gates import g1_spd, g5_mardia
from .protocol import load
from .preprocess import artifact_screen


def choose_m_hi(pools):
    c=load()['budgets']
    if len(pools)!=12:
        raise ValueError('all12 pre-gate dev pools required')
    for m in c['m_high_candidates']:
        if all(v>=c['margin']*m for v in pools.values()):
            return m
    raise ValueError('no calibration budget meets10% margin')


def budget(roles):
    cfg=load(); b=cfg['budgets']; minimum=cfg['windows']['minimum_role_blocks']
    counts={r:len(v[0]) for r,v in roles.items()}
    blocks={r:len(np.unique(v[1]['block'])) for r,v in roles.items()}
    ok=(counts['cal']>=b['margin']*512 and counts['eval']>=b['margin']*b['evaluation']
        and counts['train']>=b['margin']*b['N_max'] and all(v>=minimum for v in blocks.values()))
    return {'pass':bool(ok),'counts':counts,'blocks':blocks}


def assess(roles, *, native_artifact=None, details=True, qualification=True, date_index=0):
    cfg=load(); C=cfg['geometry']['C']; T=cfg['windows']['T']
    S=reference(roles['ref'][0]); first=g1_spd(S,C,T,cfg['gates']['G1'])
    result={'G1':first,'budget':budget(roles),'G3':{'applicable':False},'artifact':native_artifact,
            'reference_n':len(roles['ref'][0])}
    if not first['pass']:
        result['G2']={'pass':False,'evaluated':False,'reason':'G1 failure, no retest'}
        return result
    result['G2']=role_g2(roles['train'][0],roles['eval'][0],S,details)
    if details:
        result['G2_cal']=role_g2(roles['train'][0],roles['cal'][0],S,False)
        if qualification:
            from .descriptives import lag_covariance,block_bootstrap
            _,_,L=coordinates(S);result['G4']=g4(S,C,T)
            result['G5']={**g5_mardia(roles['train'][0]@L),'descriptive':True,'IID_z_not_inferential':True}
            result['within_block_lag_covariance']=lag_covariance(roles,S)
            result['whole_block_bootstrap']=block_bootstrap(roles,S,date_index)
        X=np.concatenate([roles[r][0] for r in ('train','cal','eval')])
        thirds=np.concatenate([roles[r][1]['third'] for r in ('train','cal','eval')])
        result['scan_third_transport']={str(t):role_g2(X[thirds!=t],X[thirds==t],S,False) for t in range(3)}
        band=cfg['gates']['G2']['band']
        for row in [result['G2_cal'],*result['scan_third_transport'].values()]:
            row['G2S_pass']=all(band[0]<=row[k]<=band[1] for k in
                               ('m_eval/m_train','g_bottom50','g_middle40','g_top10','physical_trace_ratio'))
    return result


def e1(results):
    cfg=load()['gates']['E1']
    if set(results)!=set(load()['access']['dev_dates']):
        raise ValueError('all12 dev outcomes required; no silent deletion')
    included=[d for d,r in sorted(results.items()) if all(isinstance(r.get(g),dict) and r[g].get('pass') is True for g in cfg['criteria'])]
    return {'included':included,'minimum':cfg['dev_minimum'],'total':12,'pass':len(included)>=cfg['dev_minimum'],
            'status':'PASS' if len(included)>=cfg['dev_minimum'] else 'STOP','no_replacements':True}

def n_min(pilot_sd,n_dev,predicted,family,alpha=.05,after_remedy=False,*,cell_ids=None,expected_cells=None):
    """Unchanged R1 formula, finite and complete paired pilot-cell contract."""
    import numbers
    sd=np.atleast_1d(np.asarray(pilot_sd,float));pred=np.atleast_1d(np.asarray(predicted,float))
    integer=lambda v:isinstance(v,numbers.Integral) and not isinstance(v,(bool,np.bool_))
    ids=[] if cell_ids is None else list(cell_ids)
    expected=[] if expected_cells is None else list(expected_cells)
    if (not ids or not expected or len(ids)!=sd.size or len(set(ids))!=len(ids)
            or len(set(expected))!=len(expected) or set(ids)!=set(expected)):
        raise ValueError('explicit complete frozen pilot cell identities required')
    if (sd.ndim!=1 or sd.size==0 or sd.shape!=pred.shape
            or not np.isfinite(sd).all() or not np.isfinite(pred).all() or np.any(sd<0)
            or not integer(n_dev) or not 9<=n_dev<=12 or not integer(family) or family<sd.size
            or not np.isfinite(alpha) or alpha!=load()['inference']['family_alpha'] or not isinstance(after_remedy,bool)):
        raise ValueError('nonempty finite paired cells,nonnegative SDs,E1-valid pilot and positive family required')
    return _r1_n_min(sd,n_dev,pred,family,alpha,after_remedy)
