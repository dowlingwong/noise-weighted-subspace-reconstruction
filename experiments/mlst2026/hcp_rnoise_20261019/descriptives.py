"""Qualification-only dependence descriptions. Never gates, CIs, or ESS claims."""
import numpy as np
from experiments.mlst2026.meg_z_r1_20261014.gates import coordinates,g2_diagonal
from .protocol import load

BOOTSTRAP_REPLICATES=200
STATISTICS=('m_train','m_eval/m_train','g_bottom50','g_middle40','g_top10','physical_trace_ratio')


def lag_covariance(roles,Sref):
    """Lag one EXISTING nonoverlapping window, pairing within original blocks."""
    _,_,L=coordinates(Sref);out={}
    for role in ('train','cal','eval'):
        X,meta=roles[role];Z=X@L;Z=Z-Z.mean(axis=0)
        left=[];right=[]
        for b in np.unique(meta['block']):
            ids=np.flatnonzero(meta['block']==b)
            if 'start' in meta:
                ids=ids[np.argsort(meta['start'][ids])]
                if len(ids)>1 and not np.all(np.diff(meta['start'][ids])==load()['windows']['T']):
                    raise ValueError('lag pairs must use adjacent original grid windows')
            left.extend(ids[:-1]);right.extend(ids[1:])
        if not left:
            out[role]={'available':False,'n_pairs':0};continue
        cov=Z[np.asarray(left)].T@Z[np.asarray(right)]/len(left)
        lag0=Z.T@Z/len(Z);scale=np.linalg.norm(lag0);trace=np.trace(lag0)
        out[role]={'available':bool(scale>0 and trace>0),'n_pairs':len(left),
                   'lag_windows':1,'lag_samples':load()['windows']['T'],'covariance_whitened':cov.tolist(),
                   'frobenius_over_lag0':float(np.linalg.norm(cov)/scale) if scale>0 else None,
                   'trace_over_lag0':float(np.trace(cov)/trace) if trace>0 else None}
    return {'descriptive':True,'crosses_blocks':False,'covariance_center':'pooled role mean in fixed Sref coordinates','roles':out}


def _block_moments(X,meta,L):
    blocks=np.unique(meta['block']);ids=[np.flatnonzero(meta['block']==b) for b in blocks]
    if len(blocks)<2 or len(set(map(len,ids)))!=1:
        raise ValueError('bootstrap requires equal complete blocks')
    Z=X@L
    return {'blocks':blocks,'n':np.array([len(i) for i in ids]),
            'sumZ':np.stack([Z[i].sum(axis=0) for i in ids]),
            'squareZ':np.stack([(Z[i]*Z[i]).sum(axis=0) for i in ids]),
            'sumX':np.stack([X[i].sum(axis=0) for i in ids]),
            'squareX':np.array([np.sum(X[i]*X[i]) for i in ids])}


def _moments(m,draw):
    n=m['n'][draw].sum();mz=m['sumZ'][draw].sum(axis=0)/n;mx=m['sumX'][draw].sum(axis=0)/n
    return m['squareZ'][draw].sum(axis=0)/n-mz*mz, float(m['squareX'][draw].sum()/n-mx@mx)


def block_bootstrap(roles,Sref,date_index=0):
    """Conditional on Sref; resample full blocks WITH replacement, not windows.

    Learning retains its separately specified without-replacement cyclic draw.
    Here every resampled block contributes all original windows unchanged.
    """
    _,_,L=coordinates(Sref);moments={r:_block_moments(*roles[r],L) for r in ('train','cal','eval')}
    entropy=[20261019,9,103,int(date_index),0]
    gen=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))
    values={r:[] for r in ('cal','eval')};draw_ids={r:[] for r in moments};failed=0
    for _ in range(BOOTSTRAP_REPLICATES):
        drawn={r:gen.integers(len(v['blocks']),size=len(v['blocks'])) for r,v in moments.items()}
        for r,idx in drawn.items():draw_ids[r].append(moments[r]['blocks'][idx].tolist())
        qt,pt=_moments(moments['train'],drawn['train'])
        valid=True;current={}
        for r in values:
            qe,pe=_moments(moments[r],drawn[r]);row=g2_diagonal(qt,qe,pt,pe,details=False)
            current[r]=[row[k] for k in STATISTICS]
            valid=valid and np.isfinite(current[r]).all()
        if not valid:failed+=1
        else:
            for r in values:values[r].append(current[r])
    summaries={}
    if failed==0:
        for r,rows in values.items():
            a=np.asarray(rows)
            summaries[r]={k:{'sd':float(a[:,j].std(ddof=1)),'quantiles_025_50_975':np.quantile(a[:,j],[.025,.5,.975]).tolist()}
                          for j,k in enumerate(STATISTICS)}
    return {'descriptive':True,'available':failed==0,'conditional_on_fixed_Sref':True,
            'not_calibrated_CI':True,'not_ESS':True,'replicates':BOOTSTRAP_REPLICATES,'failed_replicates':failed,
            'seed_entropy':entropy,'blocks_per_role':{r:len(v['blocks']) for r,v in moments.items()},
            'whole_block_draw_ids':draw_ids,'first_replicate_statistics':{r:dict(zip(STATISTICS,v[0])) for r,v in values.items() if v},'summaries':summaries}
