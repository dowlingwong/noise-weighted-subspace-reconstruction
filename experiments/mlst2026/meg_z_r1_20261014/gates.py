"""DESIGN 2–5: unshrunk role metric, G2-R/S, clustered G3 and E1."""
import numpy as np
from scipy.stats import chi2,t
from experiments.mlst2026.meg_z_20261013.gates import g1_spd,noisiest_channel
from experiments.mlst2026.meg_z_20261013.estimators import moment_stats,kronecker_flipflop_from_S,whitened_error,nearest_kronecker
from .protocol import load

def reference(X): return moment_stats(X).S

def groups(d):
    a=d//2;b=int(.9*d)
    return {'g_bottom50':slice(0,a),'g_middle40':slice(a,b),'g_top10':slice(b,d)}

def coordinates(S):
    w,Q=np.linalg.eigh((S+S.T)/2)
    if w[0]<=0: raise np.linalg.LinAlgError('reference not SPD')
    return w,Q,Q/np.sqrt(w)

def role_g2(X,Y,S,details=True):
    w,Q,L=coordinates(S);Z=(X-X.mean(0))@L;V=(Y-Y.mean(0))@L
    qt=np.mean(Z*Z,axis=0);qe=np.mean(V*V,axis=0)
    return g2_diagonal(qt,qe,float(np.mean(np.sum((X-X.mean(0))**2,axis=1))),
        float(np.mean(np.sum((Y-Y.mean(0))**2,axis=1))),V,details)

def g2_diagonal(qt,qe,trace_train,trace_eval,Zeval=None,details=True):
    cfg=load()['gates']['G2'];d=len(qt);mt=float(qt.mean());me=float(qe.mean())
    vals={'m_train':mt,'m_eval/m_train':me/mt,'physical_trace_ratio':trace_eval/trace_train}
    vals.update({name:float(qe[g].sum()/qt[g].sum()) for name,g in groups(d).items()})
    vals['pass']=all(cfg['band'][0]<=v<=cfg['band'][1] for v in vals.values())
    vals['group_train_ref']={name:float(qt[g].mean()) for name,g in groups(d).items()}
    if details and Zeval is not None:
        cov=Zeval.T@Zeval/len(Zeval);ev=np.linalg.eigvalsh((cov+cov.T)/2);gamma=d/len(Zeval)
        edges=[max(1-np.sqrt(gamma),0)**2,(1+np.sqrt(gamma))**2]
        vals['generalised_eigenvalues']=ev.tolist();vals['extremes']=[float(ev[0]),float(ev[-1])]
        vals['MP_edges']=list(map(float,edges));vals['MP_edge_ratios']=[float(edges[0]/ev[0]),float(ev[-1]/edges[1])]
        vals['retired_mean_whitened_eigenvalue']=me
        vals['retired_spread_gate_pass']=bool(vals['MP_edge_ratios'][0]<=2 and vals['MP_edge_ratios'][1]<=2)
    return vals

def cluster_energy(Z,keys):
    """Input whitened rows; naive divisor n-1 matches review 02/03."""
    n=len(Z);_,inv=np.unique(keys,axis=0,return_inverse=True);B=int(inv.max()+1) if n else 0
    if n<30 or B<8: return {'n':n,'clusters':B,'evaluated':False,'reason':'n < 30 or clusters < 8'}
    mu=Z.mean(0);Y=Z-mu;u=np.zeros((B,Z.shape[1]));np.add.at(u,inv,Y)
    naive=float(np.sum(Y*Y)/(n*(n-1)));cr=float(B/(B-1)*np.sum(u*u)/(n*n));raw=float(mu@mu)
    return {'n':n,'clusters':B,'evaluated':True,'raw':raw,'naive_term':naive,'cluster_term':cr,
        'naive_debias':raw-naive,'E_cr':raw-cr}

def g3(roles):
    rows=[];num=den=0.;energies=[]
    for role,(Z,meta) in roles.items():
        for pos in sorted(set(meta['position'])):
            sel=meta['position']==pos
            row=cluster_energy(Z[sel],np.stack([meta['run'][sel],meta['block'][sel]],1));row.update(role=role,position=int(pos),rposition=int(meta['rposition'][sel][0]))
            # rposition is a relabeling of these same sets, never a second test.
            if len(set(meta['rposition'][sel]))!=1: raise ValueError('rposition is not a position mirror')
            if row['evaluated']:
                energies.append(row['E_cr'])
                if role in ('eval','cal'):num+=row['cluster_term'];den+=row['naive_term']
            rows.append(row)
    fh=num/den if den>0 else None;worst=max(energies) if energies else None
    slot_fail=worst is None or worst>=2.06604;screen=fh is None or fh>2.64
    return {'slots':rows,'max_E_cr':worst,'f_hat':fh,'slot_fail':bool(slot_fail),
        'screen_flag':bool(screen),'pass':not(slot_fail or screen),'alignments_deduplicated':True}

def g1_with_retest(S,C,T,channels,rebuild,proc_history_empty=True):
    cfg=load()['gates']['G1']['unchanged_limits'];first=g1_spd(S,C,T,cfg,proc_history_empty)
    if first['pass']:return S,channels,{'G1':first,'retest':None}
    drop=noisiest_channel(S,C,T,channels);new_channels=[v for i,v in enumerate(channels) if i!=drop['index']]
    second_S=rebuild(drop['index'])
    second=g1_spd(second_S,C-1,T,cfg,proc_history_empty)
    return second_S,new_channels,{'G1':second,'retest':{'first':first,'dropped':drop,'count':1},'excluded':not second['pass']}

def e1(results,extra_excluded=()):
    included=[s for s,r in results.items() if s not in extra_excluded and all(r[g]['pass'] for g in ('G1','G2','G3'))]
    return {'included':included,'stop':len(included)<3,'minimum':3}

def choose_m_hi(pools):
    for c in (1024,768,512):
        if len(pools)==4 and all(v>=1.1*c for v in pools.values()):return c
    raise ValueError('no calibration budget meets 10% margin')

def sd_upper(sd,n_dev):
    if n_dev<3:raise ValueError('E1 needs >=3 dev')
    return float(sd*np.sqrt((n_dev-1)/chi2.ppf(.2,n_dev-1)))

def n_min(pilot_sd,n_dev,predicted,family,alpha=.05):
    """Simultaneous power rule at the returned n, using one-sided 80% SD upper limit."""
    upper=sd_upper(pilot_sd,n_dev)
    for n in range(8,13):
        half=t.ppf(1-alpha/(2*family),n-1)*upper/np.sqrt(n)
        if np.all(half<.5*np.abs(predicted)):return {'n_min':n,'sd_upper':upper,'half_width':float(half),'pass':True}
    return {'n_min':None,'sd_upper':upper,'pass':False,'stop':True}

def budget_eval(n):return {'pass':n>=500,'excluded':n<500,'required':500}

def g4(S,C,T):
    St,Sc,info=kronecker_flipflop_from_S(S,C,T);K=np.kron(St,Sc)
    return {'descriptive':True,'kron_vs_unshrunk':whitened_error(K,S),'nkp_relative_distance_sample':nearest_kronecker(S,C,T)[2],'flipflop':info}


def retired_descriptives(Xref,Xeval,Sref,n_eff):
    """Retired LW mean/MP-spread statistics are descriptive, never gates."""
    from experiments.mlst2026.meg_z_20261013.estimators import MomentStats,ledoit_wolf_from_stats
    from experiments.mlst2026.meg_z_20261013.gates import g2_stationarity
    Y=Xref-Xref.mean(0);q=np.sum(Y*Y,axis=1)
    st=MomentStats(len(Xref),Xref.mean(0),Sref,float(q@q),n_eff)
    lw,info=ledoit_wolf_from_stats(st)
    r=g2_stationarity(lw,Xeval,[.8,1.25],2.)
    return {'descriptive':True,'reference':'Ledoit-Wolf full','shrinkage':info['shrinkage'],
        'mean_whitened_eigenvalue':r['mean_whitened_eigenvalue'],'MP_edge_ratios':[r['mp_lower_over_min'],r['max_over_mp_upper']],
        'retired_spread_pass':r['checks']['spread']}
