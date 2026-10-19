"""Carried G1/G2/G4/G5; new whole-unit screen, budget and E1. No G3."""
import numpy as np
from experiments.mlst2026.meg_z_r1_20261014.gates import reference, coordinates, groups, role_g2, g4, n_min
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


def assess(roles, *, native_artifact=None, details=True):
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
        _,_,L=coordinates(S); result['G4']=g4(S,C,T)
        result['G5']={**g5_mardia(roles['train'][0]@L),'descriptive':True,'IID_z_not_inferential':True}
        X=np.concatenate([roles[r][0] for r in ('train','cal','eval')])
        thirds=np.concatenate([roles[r][1]['third'] for r in ('train','cal','eval')])
        result['scan_third_transport']={str(t):role_g2(X[thirds!=t],X[thirds==t],S,False) for t in range(3)}
    return result


def e1(results):
    cfg=load()['gates']['E1']
    if set(results)!=set(load()['access']['dev_dates']):
        raise ValueError('all12 dev outcomes required; no silent deletion')
    included=[d for d,r in sorted(results.items()) if all(r.get(g,{}).get('pass') is True for g in cfg['criteria'])]
    return {'included':included,'minimum':cfg['dev_minimum'],'total':12,'pass':len(included)>=cfg['dev_minimum'],
            'status':'PASS' if len(included)>=cfg['dev_minimum'] else 'STOP','no_replacements':True}
