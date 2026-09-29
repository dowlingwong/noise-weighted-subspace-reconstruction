"""Finite-sample cumulant diagnostic primitives; no implicit data selection."""
import numpy as np
from scipy.stats import beta,binom
THRESHOLD=1e-12

def standardize(x):
    x=np.asarray(x)
    if np.iscomplexobj(x):raise ValueError('complex inputs require explicit real embedding')
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('finite real matrix required')
    n,p=x.shape
    if p<1 or n<p+2:raise ValueError('N must be at least p+2')
    centered=x-x.mean(0);s=centered.T@centered/n;w,q=np.linalg.eigh(s)
    if w[-1]<=0 or w[0]<=THRESHOLD*w[-1]:raise ValueError('rank/conditioning gate failed; no regularization')
    return centered@(q/np.sqrt(w))@q.T,{'N':n,'p':p,'relative_threshold':THRESHOLD,'condition':float(w[-1]/w[0])}

def statistics(x,algorithm='auto'):
    u,meta=standardize(x);n,p=u.shape
    if algorithm=='auto':algorithm='tensor' if p*p<=n else 'gram'
    if algorithm=='tensor':
        third=np.einsum('ia,ib,ic->abc',u,u,u,optimize=True)/n;b1=float(np.sum(third**2))
    elif algorithm=='gram':
        b1=0.
        for start in range(0,n,128):b1+=float(np.sum((u[start:start+128]@u.T)**3))/n**2
    else:raise ValueError('unknown algorithm')
    return {**meta,'b1':b1,'b2':float(np.mean(np.sum(u*u,axis=1)**2)),'marginal_skewness':np.mean(u**3,axis=0).tolist(),'marginal_excess':(np.mean(u**4,axis=0)-3).tolist(),'algorithm':algorithm}

def complex_real(z):
    z=np.asarray(z)
    if not np.iscomplexobj(z) or z.ndim!=2:raise ValueError('complex interior bins required')
    out=np.empty((len(z),2*z.shape[1]));out[:,0::2]=np.sqrt(2)*z.real;out[:,1::2]=np.sqrt(2)*z.imag
    return out

def local_statistic(x,lengths):
    if sum(lengths)!=len(x) or any(type(n)!=int or n<=0 for n in lengths):raise ValueError('block ledger does not cover all rows')
    if len(lengths)<2:return {'status':'unavailable','reason':'fewer than two blocks'}
    rows=[];start=0
    for n in lengths:
        try:rows.append(statistics(x[start:start+n]))
        except ValueError as e:return {'status':'unavailable','reason':str(e),'block_index':len(rows),'lengths':lengths}
        start+=n
    return {'status':'available','b2':float(sum(n*r['b2'] for n,r in zip(lengths,rows))/len(x)),'blocks':rows,'lengths':lengths}

def excess_ratio(pooled,pooled_null,local,local_null):
    pooled_null=np.asarray(pooled_null);local_null=np.asarray(local_null)
    if pooled_null.ndim!=1 or local_null.ndim!=1 or min(len(pooled_null),len(local_null))<2 or not np.isfinite(pooled_null).all() or not np.isfinite(local_null).all() or not np.isfinite([pooled,local]).all():return None
    ex=pooled-np.mean(pooled_null);std=np.std(pooled_null,ddof=1)
    if std<=0 or ex<=std:return None
    return float((local-np.mean(local_null))/ex)

def null_b2(gen,n,p,B=999):
    if n<p+2 or B!=999:raise ValueError('registered null size/rank required')
    # Batched eigensystems implement the same centered divisor-N statistic.
    x=gen.normal(size=(B,n,p));x-=x.mean(1,keepdims=True)
    s=np.einsum('bni,bnj->bij',x,x)/n;w,q=np.linalg.eigh(s)
    if np.any(w[:,0]<=THRESHOLD*w[:,-1]):raise ValueError('null rank failure')
    u=np.einsum('bni,bij->bnj',x,q)/np.sqrt(w)[:,None,:]
    return np.mean(np.sum(u*u,axis=2)**2,axis=1)

def plus_one(observed,bank):
    bank=np.asarray(bank)
    if bank.ndim!=1 or len(bank)!=999 or not np.isfinite(bank).all() or not np.isfinite(observed):raise ValueError('invalid Monte Carlo bank')
    return float((1+np.sum(bank>=observed))/(len(bank)+1))

def holm(slots):
    if not slots or len(slots)>32:raise ValueError('fixed unit allocation out of range')
    if len(set(slots))!=len(slots):raise ValueError('duplicate units')
    p={k:(1. if v is None else float(v)) for k,v in slots.items()}
    if any(not np.isfinite(v) or v<0 or v>1 for v in p.values()):raise ValueError('invalid p')
    order=sorted(p,key=lambda k:(p[k],k));adj={};previous=0.
    for i,k in enumerate(order):previous=max(previous,min(1.,(len(order)-i)*p[k]));adj[k]=previous
    return {k:{'raw_p':p[k],'adjusted_p':adj[k],'rejected':bool(adj[k]<=.05),'evaluable':slots[k] is not None} for k in slots}

def binomial_interval(k,n,alpha=.05):
    return [0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]

def calibration(k,n=200,profiles=1):
    if n!=200 or profiles<1 or not 0<=k<=n:raise ValueError('invalid control allocation')
    p=float(binom.sf(k-1,n,.05))
    return {'rejections':k,'trials':n,'excess_type1_tail':p,'failed':bool(p<=.01/profiles),'binomial_ci':binomial_interval(k,n),'scope':'finite iid reference simulation; not model adequacy'}

def matched_control(gen,n,p,kind,u=None):
    e=gen.normal(size=(n,p))
    if kind=='S0':return e
    if kind=='S1':return e*np.sqrt(2/gen.gamma(3,size=n))[:,None]
    if kind!='S2':raise ValueError('unknown control')
    u=np.asarray(u)
    if u.shape!=(p,) or not np.isclose(u@u,1):raise ValueError('fixed unit template required')
    eta=.5;prob=.02
    normal=e+(np.sqrt(1-eta)-1)*(e@u)[:,None]*u
    return normal+np.sqrt(eta/(prob*(1-prob)))*(gen.binomial(1,prob,n)-prob)[:,None]*u

def classify(evaluable,rejected,power_lower):
    if not evaluable:return 'not evaluable'
    if rejected:return 'excess kurtosis relative to qualified Gaussian reference'
    if power_lower is not None and power_lower>=.8:return 'no excess detected at demonstrated sensitivity to the specified alternative'
    return 'no excess detected; limited or uncertain sensitivity'
