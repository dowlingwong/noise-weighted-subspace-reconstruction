"""Covariance-only response in true-whitened coordinates."""
import hashlib
import numpy as np
from scipy.stats import t

TOL = 1e-10

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()

def rng_for(*parts):
    return np.random.default_rng(int.from_bytes(hashlib.sha256('/'.join(map(str,parts)).encode()).digest()[:8], 'little'))

def eig(a):
    if not np.isfinite(a).all() or not np.allclose(a,a.T,atol=1e-12):
        raise ValueError('nonfinite or nonsymmetric operator')
    w,q=np.linalg.eigh((a+a.T)/2)
    return w[::-1],q[:,::-1]

def powers(c):
    w,q=eig(c)
    if w[-1]<=0: raise ValueError('covariance is not SPD')
    return (q*np.sqrt(w))@q.T,(q/np.sqrt(w))@q.T

def effective(raw):
    w,q=eig(raw); floor=1e-8*np.trace(raw)/len(raw)
    if floor<=0: raise ValueError('nonpositive covariance scale')
    return (q*np.maximum(w,floor))@q.T, float(floor), int(np.sum(w<floor))

def response(a,e,k):
    """Right invariant span of A S^-1 = S^1/2 (S^-1/2 A S^-1/2) S^-1/2."""
    lam,q=eig(a); d=len(a)
    if not 0<k<d or lam[k-1]-lam[k]<=0 or lam[-1]<-TOL:
        raise ValueError('rank, PSD or gap invalid')
    ss,si=powers(np.eye(d)+e)
    mu,u=eig(si@a@si)
    v=np.linalg.qr(ss@u[:,:k])[0]
    p=q[:,:k]@q[:,:k].T; ph=v@v.T
    eq=q.T@e@q
    x=-lam[k:,None]*eq[k:,:k]/(lam[None,:k]-lam[k:,None])
    dpq=np.zeros_like(a); dpq[k:,:k]=x; dpq[:k,k:]=x.T
    dp=q@dpq@q.T
    eps=float(np.linalg.norm(e,2)); gap=float(lam[k-1]-lam[k]); eta=eps*lam[0]/gap
    # A 64-ulp allowance addresses representation of an exact requested boundary;
    # the raw eta is always saved. It is not a scientific tolerance adjustment.
    inside=bool(eta<=.1+64*np.spacing(.1))
    rem=float(np.linalg.norm(ph-p-dp,2))
    if np.linalg.norm(eq[k:,:k],2)<1e-12:
        ar=np.diag(lam[:k]);ad=np.diag(lam[k:]);sr=np.eye(k)+eq[:k,:k];sd=np.eye(d-k)+eq[k:,k:]
        ir=powers(sr)[1];idd=powers(sd)[1]
        ordered=bool(np.linalg.eigvalsh(ir@ar@ir)[0]>np.linalg.eigvalsh(idd@ad@idd)[-1])
        ordering='block_preserved' if ordered else 'block_exchanged_or_tied'
    else:ordering='coupled_positive_sorted_gap' if mu[k-1]>mu[k] else 'coupled_tied'
    rr={'ordering_check':ordering,'epsilon':eps,'eta':float(eta),'gap':gap,'lambda1':float(lam[0]),
        'domain':'in_domain' if inside else 'out_of_domain','lambda_min_I_E':float(np.linalg.eigvalsh(np.eye(d)+e)[0]),
        'perturbed_gap':float(mu[k-1]-mu[k]),'displacement':float(np.linalg.norm(ph-p,2)),
        'prediction':float(np.linalg.norm(dp,2)),'remainder':rem,
        'bound':44*float(eta)**2 if inside else None,
        'bound_slack':44*float(eta)**2-rem if inside else None,
        'bound_violation':bool(inside and rem>44*eta**2+TOL),'tolerance':TOL}
    arr={'A':a,'E':e,'lambda':lam,'basis':v,'P':p,'P_hat':ph,'deltaP':dp}
    return rr,arr

def spectrum(name,d):
    if name=='S1': return np.r_[9,7,5,3,np.ones(d-4)]
    if name=='S2': return np.r_[9,7,5,1.25,np.ones(d-4)]
    if name=='S3': return np.r_[12,8,4,2,np.linspace(1.8,1,d-4)]
    raise ValueError('unknown spectrum')

def direction(name,lam,rng,k=4):
    d=len(lam); e=np.zeros((d,d))
    if name in ('near','far'):
        coef=lam[k:,None]/(lam[None,:k]-lam[k:,None])
        j,i=np.unravel_index(np.argmax(coef) if name=='near' else np.argmin(coef),coef.shape)
        e[j+k,i]=e[i,j+k]=1
    elif name=='dense':
        z=rng.normal(size=(d,d)); e=(z+z.T)/2
    elif name=='block':
        z=rng.normal(size=(d,d)); e=(z+z.T)/2; e[:k,k:]=0;e[k:,:k]=0
    else: raise ValueError('unknown direction')
    return e/np.linalg.norm(e,2)

def cell(spec):
    d=spec['d']; name=spec['spectrum']; stage=spec['stage']; seed=spec['seed']; k=4
    key=('upgrade-c1-v1',stage,seed,d,name)
    lam=spectrum(name,d); q=np.linalg.qr(rng_for(*key,'basis').normal(size=(d,d)))[0]
    a=(q*lam)@q.T; arrays={'population_A':a}; rows=[]
    for family in ('near','far','dense','block'):
        e0=q@direction(family,lam,rng_for(*key,'direction',family))@q.T
        for eta in (.01,.03,.1,.3):
            e=e0*(eta*(lam[k-1]-lam[k])/lam[0]); r,z=response(a,e,k)
            tag=f'{family}_{eta}'; arrays.update({tag+'_'+n:v for n,v in z.items()})
            rows.append({**r,'part':'controlled','direction':family,'requested_eta':eta,'perturbation_sha256':digest(e),'array_prefix':tag})
    # Independent physical AR(1) calibration; true whitening is symmetric.
    sigma=.6**np.abs(np.arange(d)[:,None]-np.arange(d)[None,:]); ss,si=powers(sigma)
    clean=(rng_for(*key,'clean').normal(size=(4*d,d))*np.sqrt(lam-1))@q.T
    noisy=clean+rng_for(*key,'training-noise').normal(size=(4*d,d))
    physical=noisy@ss; centered=physical-physical.mean(axis=0)
    sample=si@(centered.T@centered/(len(centered)-1))@si
    pool=rng_for(*key,'calibration').normal(size=(64*d,d))@ss
    arrays.update(training=physical,calibration_pool=pool,true_covariance=sigma,sample_A=sample)
    for m in (4*d,16*d,64*d):
        raw=np.cov(pool[:m],rowvar=False)
        lag=np.array([np.diag(raw,j).mean() for j in range(d)])
        for method in ('empirical','shrinkage','toeplitz'):
            estimate=raw if method=='empirical' else (.9*raw+.1*np.eye(d)*np.trace(raw)/d if method=='shrinkage' else lag[np.abs(np.arange(d)[:,None]-np.arange(d)[None,:])])
            cov,floor,nfloor=effective(estimate); e=si@cov@si-np.eye(d)
            r,z=response(sample,e,k); tag=f'{method}_{m}'
            arrays.update({tag+'_'+n:v for n,v in z.items()}); arrays[tag+'_raw_covariance']=estimate; arrays[tag+'_effective_covariance']=cov
            rows.append({**r,'part':'estimated','method':method,'m':m,'array_prefix':tag,'reference':'fixed_sample_operator','training_sha256':digest(physical),'prefix_sha256':digest(pool[:m]),'floor':floor,'floored_eigenvalues':nfloor})
    return {'records':rows,'conditions':25,'references':2,'training_sha256':digest(physical)},arrays

def fit_baseline(results):
    groups={}
    for c in results:
        if c['cell']['stage'] not in ('pilot','development_pilot'): raise ValueError('baseline requires pilot only')
        key=f"{c['cell']['d']}/{c['cell']['spectrum']}"
        groups.setdefault(key,[]).extend(r for r in c['result']['records'] if r['part']=='controlled' and r['direction']!='block' and r['requested_eta']<=.1)
    return {key:float(sum(r['epsilon']*r['displacement'] for r in rows)/sum(r['epsilon']**2 for r in rows)) for key,rows in groups.items()}

def reduce_confirmation(results,coeff):
    by_seed={}; estimated={}; controls=[]; stress=[]
    for c in results:
        if c['cell']['stage'] not in ('confirm','development_confirm'):raise ValueError('not confirmation')
        key=f"{c['cell']['d']}/{c['cell']['spectrum']}"
        for r in c['result']['records']:
            if r['part']=='estimated':
                group=f"{key}/{r['method']}/{r['m']}"; estimated.setdefault(group,[]).append(r);continue
            if r['direction']=='block': controls.append(r);continue
            if r['requested_eta']>.1: stress.append(r);continue
            score=(abs(coeff[key]*r['epsilon']-r['displacement'])-abs(r['prediction']-r['displacement']))/r['requested_eta']
            by_seed.setdefault(c['cell']['seed'],[]).append(score)
    values=np.array([np.mean(by_seed[s]) for s in sorted(by_seed)])
    ci=None
    if len(values)>1:
        half=t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values));ci=[float(values.mean()-half),float(values.mean()+half)]
    return {'primary':'MAE_norm_only - MAE_first_order; equal cells within independent seed','seed_scores':dict(zip(map(str,sorted(by_seed)),values.tolist())), 'mean':float(values.mean()),'ci95_student_seed':ci,'predictive_advantage':bool(ci is not None and ci[0]>0), 'commuting_control_max':max(r['displacement'] for r in controls),'stress_count':len(stress),'estimated':{k:{'n':len(v),'in_domain':sum(r['domain']=='in_domain' for r in v),'mean_displacement':float(np.mean([r['displacement'] for r in v])),'mean_absolute_predictor_error':float(np.mean([abs(r['prediction']-r['displacement']) for r in v])),'bound_violations':sum(r['bound_violation'] for r in v)} for k,v in estimated.items()}}
