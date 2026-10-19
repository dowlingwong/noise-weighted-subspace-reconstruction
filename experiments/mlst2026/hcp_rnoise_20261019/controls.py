"""Continuous-block AR10 surrogates and exact24-claim controls. No campaign on import."""
import math
import numpy as np
from scipy.linalg import toeplitz, solve_toeplitz
from scipy.stats import beta
from experiments.mlst2026.meg_z_r1_20261014.controls import temporal_draw, gap_filter, scale_group
from .protocol import load
from .windows import plan, cut
from .gates import reference, coordinates, groups, assess


def rng(law, cell, date_index, replicate):
    entropy=[20261019,int(law),int(cell),int(date_index),int(replicate)]
    if any(v<0 for v in entropy):
        raise ValueError('seed indices must be nonnegative')
    return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy))),entropy


def fit_shape(spatial_cov, temporal_lags):
    """Inputs are future ref-active sample moments; never fit across guards.

    This accepts arrays only. H0–H3 passes fabricated covariances. No loader reads
    real dev moments or samples in the candidate CLI.
    """
    C=load()['geometry']['C']; T=load()['windows']['T']
    Sc=np.asarray(spatial_cov,float); r=np.asarray(temporal_lags,float)
    if Sc.shape!=(C,C) or r.shape!=(20,) or not np.isfinite(Sc).all() or not np.isfinite(r).all():
        raise ValueError('C x C spatial covariance and20 finite lag moments required')
    if not np.allclose(Sc,Sc.T,rtol=1e-10,atol=0) or np.trace(Sc)<=0:
        raise ValueError('invalid spatial moments')
    Sc=Sc/(np.trace(Sc)/C); Lc=np.linalg.cholesky(Sc)
    a=solve_toeplitz((r[:10],r[:10]),r[1:11]); innovation=float(r[0]-a@r[1:11]); roots=np.roots(np.r_[1.,-a])
    if innovation<=0 or not np.isfinite(innovation) or np.max(np.abs(roots))>=1:
        raise ValueError('STOP nonstationary AR10 or nonpositive innovation; no correction')
    fitted=np.empty(20); fitted[:11]=r[:11]
    for k in range(11,20): fitted[k]=sum(a[j-1]*fitted[k-j] for j in range(1,11))
    population=np.kron(toeplitz(r[:T]),Sc); ev,Q=np.linalg.eigh(population)
    return {'C':C,'Lc':Lc,'Lt':np.linalg.cholesky(toeplitz(r[:10])),
            'ar_coefficients':a,'innovation_variance':innovation,'population':population,'population_Q':Q,
            'fit':{'order':10,'coefficients':a.tolist(),'innovation_variance':innovation,'max_root_modulus':float(np.max(abs(roots))),
                   'observed_lags_11_19':r[11:].tolist(),'fitted_lags_11_19':fitted[11:].tolist(),
                   'relative_errors':((fitted[11:]-r[11:])/r[0]).tolist()}}


def ref_moments(data, window_plan):
    """Centred divisor-n spatial/lag moments, strictly inside active ref blocks."""
    x=np.asarray(data,float); C=load()['geometry']['C']
    if x.shape!=(C,window_plan['n_decimated']):
        raise ValueError('channel/time shape differs')
    chunks=[x[:,b['active_start']:b['end']] for b in window_plan['blocks'] if b['role']=='ref']
    if not chunks or any(z.shape[1]<20 for z in chunks):
        raise ValueError('ref lag geometry too short')
    pooled=np.concatenate(chunks,axis=1); mean=pooled.mean(axis=1); z=pooled-mean[:,None]; Sc=z@z.T/z.shape[1]
    lags=[]
    for lag in range(20):
        products=[np.sum((v[:,lag:]-mean[:,None])*(v[:,:v.shape[1]-lag]-mean[:,None])) for v in chunks]
        lags.append(sum(products)/(C*sum(v.shape[1]-lag for v in chunks)))
    return Sc,np.asarray(lags)


def identity_shape():
    C=load()['geometry']['C'];d=C*load()['windows']['T']
    return {'C':C,'Lc':np.eye(C),'population':np.eye(d),'population_Q':np.eye(d),'fit':{'law':'identity IID'}}


def synthetic_shape(rho=.7):
    """Declared timing/fixture covariance, unrelated to a measured scan."""
    C=load()['geometry']['C']; Sc=toeplitz(.35**np.arange(C))
    return fit_shape(Sc, 1e-24*float(rho)**np.arange(20))


def generate(shape, unit_metadata, law, gen):
    if law not in load()['controls']['laws']:
        raise ValueError('unknown law')
    p=plan(unit_metadata['n_times'],unit_metadata['sfreq_hz']); C=shape['C']; width=p['block_samples']; nb=len(p['blocks'])
    if nb<1:
        raise ValueError('no full blocks')
    xi=gen.normal(size=(nb,width,C))
    a={1:0.,2:0.,3:.25,4:.5,5:.75,6:0.,7:0.}[law]
    y=gap_filter(temporal_draw(shape,xi)@shape['Lc'].T,a)
    # Place only retained full blocks. Edges/guards never enter windows or moments.
    data=np.zeros((C,p['n_decimated']))
    for b,z in zip(p['blocks'],y): data[:,b['start']:b['end']]=z.T
    rec=cut(data,p)
    if law in (6,7):
        popQ=shape['population_Q']; V=popQ[:,int(.9*popQ.shape[0]):]
        targets=range(3) if law==6 else range(nb); factors={}
        for key in targets:
            s=.2 if law==6 else .1
            factors[key]=(float(np.exp(gen.normal(-s*s/2,s))),float(np.exp(gen.normal(-.1**2/2,.1))))
        for role,(X,meta) in rec.items():
            X=X.copy(); keycol=meta['third'] if law==6 else meta['block']
            for key in np.unique(keycol):
                select=keycol==key; top,global_var=factors[int(key)]
                X[select]=scale_group(X[select],V,top)*math.sqrt(global_var)
            rec[role]=(X,meta)
    return rec


def apply_cell(rec, Sref, shape, cell):
    if cell not in load()['controls']['cells']:
        raise ValueError('no stimulus/G3 cells in empty-room controls')
    C=load()['geometry']['C']; T=load()['windows']['T']; d=C*T
    X,meta=rec['eval']; X=X.copy(); _,Q,_=coordinates(Sref)
    if cell==1: X*=math.sqrt(1.5)
    elif cell in (2,3,4):
        group=groups(d)[{2:'g_top10',3:'g_bottom50',4:'g_middle40'}[cell]]
        X=scale_group(X,Q[:,group],1.5)
    elif cell in (5,6):
        S4=Sref.reshape(C,T,C,T,order='F'); v=np.linalg.eigh(np.einsum('ctdt->cd',S4)/T)[1][:,0 if cell==5 else -1]
        z=X.reshape(-1,T,C); X=(z+(math.sqrt(8)-1)*(z@v)[:,:,None]*v).reshape(-1,d)
    elif cell in (10,11): X=scale_group(X,Q[:,:1] if cell==10 else Q[:,-1:],8)
    elif cell==12: X=scale_group(X,shape['population_Q'][:,-1:],1.5)
    out=dict(rec); out['eval']=(X,meta); return out


def replicate(shape, unit_metadata, law, cell, date_index, replicate_index, *, details=False, timing=False):
    gen,entropy=rng(law,99 if timing else cell,date_index,replicate_index)
    actual=identity_shape() if law==1 else shape
    rec=generate(actual,unit_metadata,law,gen)
    try:
        S=reference(rec['ref'][0]); changed=apply_cell(rec,S,actual,cell)
        result=assess(changed,details=details)
    except np.linalg.LinAlgError as exc:
        result={'G1':{'pass':False},'G2':{'pass':False,'evaluated':False},'numerical_failure':type(exc).__name__}
    return {**result,'law':law,'cell':cell,'date_index':date_index,'replicate':replicate_index,
            'seed_entropy':entropy,'bit_generator':'PCG64DXSM'}


def event(row, claim_type):
    """F2: a failed reference is null exclusion and detection NON-success."""
    g1=row['G1']['pass'] is True
    if claim_type=='null': return (not g1) or row['G2']['pass'] is not True
    if claim_type=='detection': return g1 and row['G2']['pass'] is False and row['G2'].get('evaluated',True)
    raise ValueError('unknown claim type')


def claim_bound(count, n, claim_type):
    cfg=load()['controls']; a=cfg['alpha_per_claim']
    if not 0<=count<=n or n<=0:
        raise ValueError('invalid CP counts')
    if claim_type=='null':
        bound=1. if count==n else float(beta.ppf(1-a,count+1,n-count))
        passed=bound<=cfg['null_upper_max']; direction='upper'
    elif claim_type=='detection':
        bound=0. if count==0 else float(beta.ppf(a,count,n-count+1))
        passed=bound>=cfg['detection_lower_min']; direction='lower'
    else: raise ValueError('unknown claim type')
    return {'count':int(count),'n':int(n),'bound':bound,'direction':direction,
            'pass':bool(n==cfg['pooled_n'] and passed),'alpha':a}


def summarize(rows):
    cfg=load()['controls']
    keys=[(v['law'],v['cell'],v['date_index'],v['replicate']) for v in rows]
    expected={(l,c,d,r) for l in cfg['laws'] for c in cfg['cells'] for d in range(cfg['shapes']) for r in range(cfg['replicates_per_shape'])}
    if len(set(keys))!=len(keys) or set(keys)!=expected:
        raise ValueError('STOP incomplete/duplicate/unexpected controls; no selective pooling')
    groups_={}
    for row in rows: groups_.setdefault((row['law'],row['cell']),[]).append(row)
    claims=[]
    for claim in cfg['claim_tuples']:
        selected=groups_[(claim['law'],claim['cell'])]
        claims.append({**claim,**claim_bound(sum(event(row,claim['type']) for row in selected),len(selected),claim['type'])})
    desc={f'L{l}/C{c}':{'n':len(rr),'G1_failure_count':sum(v['G1']['pass'] is not True for v in rr),
                           'G2_exclusion_count':sum(event(v,'null') for v in rr),
                           'G2_detection_count':sum(event(v,'detection') for v in rr)} for (l,c),rr in groups_.items()}
    return {'status':'PASS' if all(c['pass'] for c in claims) else 'STOP','claims':claims,'descriptive':desc,
            'failure_rule':cfg['failure_rule']}
