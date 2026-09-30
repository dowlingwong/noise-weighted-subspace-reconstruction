"""Fixed physical product-class fitting; Fortran vec c+C*t throughout.

No outcome files are imported. T20S is used only for its registered algebraic
population constructor; C1 response is audited code reuse, not archived outcomes.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.stats import t
from experiments.mlst2026.successor_20260913.t20s.generator import make_geometry, partial_trace_cap
from experiments.mlst2026.upgrade_20260922.c1 import response as _response


def response(a,e,k):
    """Retain primary results when ONLY the reference sample eigengap is zero."""
    from experiments.mlst2026.upgrade_20260922.c1 import eig, TOL
    lam,_=eig(a)
    if not 0<k<len(a) or lam[-1]<-TOL:raise ValueError('rank or PSD invalid')
    eig(e); powers(np.eye(len(a))+e)
    gap=float(lam[k-1]-lam[k])
    if gap<=0:
        return {'domain':'undefined_gap','gap':gap,'epsilon':float(np.linalg.norm(e,2)),
                'eta':None,'bound':None,'prediction':None,'displacement':None,'remainder':None,
                'reason':'reference retained/discarded eigenspace not uniquely defined'},{}
    return _response(a,e,k)


def rng(*key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256('/'.join(map(str,key)).encode()).digest()[:8],'little'))


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()


def symmetric(a):
    return (a+a.T)/2


def powers(a):
    w,q=np.linalg.eigh(symmetric(a))
    if not np.isfinite(w).all() or w[0]<=0: raise ValueError('non-SPD matrix')
    return (q*np.sqrt(w))@q.T,(q/np.sqrt(w))@q.T


def qr(a):
    q,r=np.linalg.qr(a)
    if np.min(np.abs(np.diag(r)))<1e-12: raise ValueError('rank loss')
    return q*np.where(np.diag(r)<0,-1.,1.)


def top(a,k):
    return np.linalg.eigh(symmetric(a))[1][:,-k:]


def geometry(c):
    return make_geometry(C=c['C'],T=c['T'],k_c=c['kc'],k_t=c['kt'],spec=c['spectrum'],seed=c['geometry_seed'])


def covariance(c):
    C,T=c['C'],c['T']
    sc=c['rho_c']**np.abs(np.arange(C)[:,None]-np.arange(C))
    st=c['rho_t']**np.abs(np.arange(T)[:,None]-np.arange(T))
    return np.kron(st,sc)


def regularize(a):
    scale=float(np.trace(a)/len(a))
    if not np.isfinite(scale) or scale<=0: raise ValueError('nonpositive covariance scale')
    shrunk=.9*symmetric(a)+.1*scale*np.eye(len(a))
    w,q=np.linalg.eigh(shrunk); floor=1e-8*scale
    return (q*np.maximum(w,floor))@q.T,{'scale':scale,'raw_min':float(np.linalg.eigvalsh(symmetric(a))[0]),'shrunk_min':float(w[0]),'floor':floor,'floor_count':int(np.sum(w<floor))}


def estimate(pool,C,T):
    n=len(pool); x=pool-pool.mean(0); raw=x.T@x/(n-1)
    full,fd=regularize(raw)
    z=x.reshape(n,C,T,order='F'); v=float(np.trace(raw)/(C*T))
    ac=np.einsum('nct,ndt->cd',z,z)/((n-1)*T)/v
    at=np.einsum('nct,ncs->ts',z,z)/((n-1)*C)
    sc,cd=regularize(ac); st,td=regularize(at)
    return {'full':full,'kron':np.kron(st,sc)}, {'prefix_sha':digest(pool),'mean_sha':digest(pool.mean(0)),'full':fd,'kron':{'channel':cd,'time':td,'trace_channel':float(np.trace(sc)),'trace_time':float(np.trace(st)),'iterations':0}}


def map_for(B,W):
    return B@np.linalg.solve(B.T@W@B,B.T@W)


def objective_gradient(uc,ut,W,S,H=None):
    B=np.kron(ut,uc); H=W@S@W if H is None else H; Q=B.T@W@B; M=B.T@H@B
    qi=np.linalg.inv(Q); value=float(np.trace(qi@M))
    G=2*(H@B@qi-W@B@qi@M@qi)
    C,kc=uc.shape; T,kt=ut.shape
    tensor=G.reshape(C,T,kc,kt,order='F')
    gc=np.einsum('ctab,tb->ca',tensor,ut)
    gt=np.einsum('ctab,ca->tb',tensor,uc)
    return value,gc,gt


def starts(S,C,T,kc,kt,seed):
    s4=S.reshape(C,T,C,T,order='F')
    first=(top(np.einsum('ctdt->cd',s4),kc),top(np.einsum('ctcs->ts',s4),kt))
    result=[first]; gen=rng(seed,'starts')
    for _ in range(3):result.append((qr(gen.normal(size=(C,kc))),qr(gen.normal(size=(T,kt)))))
    return result


def structured(W,S,initial,max_iter=150,tol=1e-8):
    scale=float(np.trace(W@S)); rows=[]; bases=[]; H=W@S@W
    if scale<=0:raise ValueError('zero training energy')
    for u0,v0 in initial:
        u,v=u0.copy(),v0.copy(); history=[]; status='iteration_limit'; gradnorm=None; next_step=1.; accepted_steps=[]; backtracks=[]
        for step in range(max_iter):
            val,gc,gt=objective_gradient(u,v,W,S,H)
            gc=(gc-u@(u.T@gc))/scale; gt=(gt-v@(v.T@gt))/scale
            norm2=float(np.sum(gc*gc)+np.sum(gt*gt)); gradnorm=np.sqrt(norm2)
            history.append(val)
            if gradnorm<=tol: status='stationary';break
            alpha=next_step; accepted=False
            for backtrack in range(30):
                un,vn=qr(u+alpha*gc),qr(v+alpha*gt)
                new=objective_gradient(un,vn,W,S,H)[0]
                if new/scale>=val/scale+1e-4*alpha*norm2:
                    u,v=un,vn;accepted=True;accepted_steps.append(alpha);backtracks.append(backtrack);next_step=min(2*alpha,1024.);break
                alpha*=.5
            if not accepted:status='line_search_stopped';break
        value,gc,gt=objective_gradient(u,v,W,S,H)
        gradnorm=float(np.sqrt(np.sum(((gc-u@(u.T@gc))/scale)**2)+np.sum(((gt-v@(v.T@gt))/scale)**2)))
        if gradnorm<=tol:status='stationary'
        history.append(value)
        rows.append({'objective':value,'gradient_norm':gradnorm,'status':status,'trajectory':history,'accepted_steps':accepted_steps,'backtracks':backtracks})
        bases.append((u,v))
    index=int(np.argmax([r['objective'] for r in rows])); u,v=bases[index]
    return np.kron(v,u),{'selected':index,'restarts':rows},bases


def unrestricted(W,S,k):
    ws,wi=powers(W)
    return qr(wi@top(ws@S@ws,k))


def risks(A,b,Cs,Sigma,W0):
    I=np.eye(len(A)); E=A-I
    def tr(M,C):return float(np.trace(M@C))
    approximation=tr(W0,E@Cs@E.T); noise=tr(W0,A@Sigma@A.T); offset=float(b@W0@b)
    return {'signal':approximation,'noise':noise,'offset':offset,'clean':approximation+noise+offset,
        'euclidean_clean':float(np.trace(E@Cs@E.T+A@Sigma@A.T)+b@b),
        'observed':tr(W0,E@(Cs+Sigma)@E.T)+offset}


def representation(B,Li,truth):
    q=qr(Li@B); sv=np.linalg.svd(truth.T@q,compute_uv=False)
    return {'max_angle_degrees':float(np.degrees(np.arccos(np.clip(sv.min(),0,1)))),
            'projector_frobenius_squared':float(np.linalg.norm(q@q.T-truth@truth.T,'fro')**2)}


def make_data(c,seed,theta):
    g=geometry(c); sigma=covariance(c); L,Li=powers(sigma)
    key=(c['namespace'],seed)
    calibration=rng(*key,'calibration').normal(size=(max(c['m']),g.d))@L
    clean=rng(*key,theta,'signal').normal(size=(max(c['N']),g.d))@(g.V(theta)*np.sqrt(g.mu)).T@L
    noise=rng(*key,theta,'noise').normal(size=(max(c['N']),g.d))@L
    return g,sigma,L,Li,calibration,clean+noise


def moments(training,N):
    """The ONLY training-moment computation; fit and every replay call this so
    the BLAS kernel (syrk for same-buffer A.T@A) is identical."""
    x=training[:N]; mu=x.mean(0); xc=x-mu
    return x,mu,xc.T@xc/N


def fit_cell(c,seed,theta,N):
    g,sigma,L,Li,pool,training=make_data(c,seed,theta)
    x,mu,S=moments(training,N)
    K=g.K(theta); Cs=L@K@L; W0=Li@Li; truth=top(K,g.k)
    estimates={'iso':np.eye(g.d),'oracle':sigma}; covdiag={}
    for m in c['m']:
        covs,diag=estimate(pool[:m],g.C,g.T)
        for name,cov in covs.items():estimates[f'{name}_{m}']=cov
        covdiag[str(m)]=diag
    initial=starts(S,g.C,g.T,g.k_c,g.k_t,(c['namespace'],seed,theta,N))
    results={}; arrays={'S':S,'mean':mu,'Sigma':sigma,'Cs':Cs,'K':K,'training':x,'calibration':pool}
    sample=Li@S@Li
    for name,cov in estimates.items():
        W=np.linalg.inv(cov); arrays[name+'_cov']=cov
        B,diagnostic,factors=structured(W,S,initial,c['max_iter'],c['gradient_tolerance'])
        Bu=unrestricted(W,S,g.k)
        for cls,basis in [('restricted',B),('unrestricted',Bu)]:
            tag=name+'__'+cls; A=map_for(basis,W); b=(np.eye(g.d)-A)@mu
            arrays[tag+'_B']=basis;arrays[tag+'_A']=A;arrays[tag+'_b']=b
            result=risks(A,b,Cs,sigma,W0)|representation(basis,Li,truth)
            result['training_loss']=float(np.trace(W@S)-np.trace(np.linalg.solve(basis.T@W@basis,basis.T@W@S@W@basis)))
            if cls=='restricted':
                result['optimization']=diagnostic
                rr=[]
                for i,(uc,ut) in enumerate(factors):
                    arrays[tag+f'_uc_{i}']=uc;arrays[tag+f'_ut_{i}']=ut
                    ar=map_for(np.kron(ut,uc),W);br=(np.eye(g.d)-ar)@mu
                    rr.append(risks(ar,br,Cs,sigma,W0)['clean'])
                result['restart_clean_risks']=rr
                result['restart_risk_spread']=float(max(rr)-min(rr))
            results[tag]=result
        if name not in ('iso','oracle'):
            e=Li@cov@Li-np.eye(g.d)
            r,_=response(sample,e,g.k); results[name+'__domain']=r
    return {'seed':seed,'theta':theta,'N':N,'training_sha':digest(x),'calibration_sha':digest(pool),'covariance':covdiag,'methods':results},arrays


def contrast_values(methods,ms):
    variants=['iso','oracle']+[f'{arm}_{m}' for m in ms for arm in ('full','kron')]
    R=lambda v,cl:methods[v+'__'+cl]['clean']
    D={v:R(v,'restricted')-R(v,'unrestricted') for v in variants}
    out={'class/'+v:d for v,d in D.items()}
    for v in variants:
        if v=='oracle':continue
        for cl in ('restricted','unrestricted'):out[f'metric/{v}/{cl}']=R(v,cl)-R('oracle',cl)
        out['interaction/'+v]=D[v]-D['oracle']
    for m in ms:out[f'covariance_class/{m}']=D[f'kron_{m}']-D[f'full_{m}']
    for arm in ('full','kron'):
        for a,b in zip(ms,ms[1:]):out[f'calibration_class/{arm}/{a}-{b}']=D[f'{arm}_{b}']-D[f'{arm}_{a}']
    return out


def profile(rows):
    signs=[('-' if r['ci'][1]<0 else '+' if r['ci'][0]>0 else '?') for r in rows]
    brackets=[[a['N'],b['N']] for a,b,sa,sb in zip(rows,rows[1:],signs,signs[1:]) if sa!='?' and sb!='?' and sa!=sb]
    decided=[s for s in signs if s!='?']
    changes=sum(a!=b for a,b in zip(decided,decided[1:]))
    return {'signs':signs,'adjacent_brackets':brackets,'nonmonotone':changes>1,'no_adjacent_crossing':not brackets}


def reduce_records(c,records):
    groups={};expected={(s,th,n) for s in c['seeds'] for th in c['theta'] for n in c['N']}
    got={(r['seed'],r['theta'],r['N']) for r in records}
    if got!=expected or len(got)!=len(records):raise ValueError('missing/duplicate cells')
    for r in records:
        contrasts=contrast_values(r['methods'],c['m'])
        for name,val in contrasts.items():groups.setdefault((r['theta'],r['N'],name),[]).append(val)
    family=len(groups);rows=[]
    for (theta,N,name),values in sorted(groups.items()):
        v=np.asarray(values); half=float(t.ppf(1-.05/(2*family),len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
        mean=float(v.mean());rows.append({'zero_standard_error':bool(v.std(ddof=1)==0),'theta':theta,'N':N,'contrast':name,'mean':mean,'ci':[mean-half,mean+half],'seeds':len(v)})
    profiles={}
    for th in c['theta']:
        for v in ['iso','oracle']+[f'{arm}_{m}' for m in c['m'] for arm in ('full','kron')]:
            rr=sorted([r for r in rows if r['theta']==th and r['contrast']=='class/'+v],key=lambda r:r['N'])
            profiles[f'{th}/{v}']=profile(rr)
    limits=[]
    for r in records:
        for v in ['iso','oracle']+[f'{arm}_{m}' for m in c['m'] for arm in ('full','kron')]:
            rr=r['methods'][v+'__restricted']; uu=r['methods'][v+'__unrestricted']; opt=rr['optimization']
            limited=opt['restarts'][opt['selected']]['status']!='stationary' or rr['restart_risk_spread']>=abs(rr['clean']-uu['clean'])
            limits.append({'seed':r['seed'],'theta':r['theta'],'N':r['N'],'metric':v,'optimization_limited':limited})
    return {'optimization_limits':limits,'mechanism_rule':'learner contrasts retained; no class/covariance mechanism attribution where selected fit is nonstationary or restart risk spread reaches absolute class contrast','zero_SE_rule':'degenerate observed-seed interval; not proof of population certainty; zero is undecided','family_size':family,'interval':'approximate Bonferroni Student-t over independent seeds','contrasts':rows,'profiles':profiles}


def population_references(c):
    g=geometry(c);rows=[]
    B=np.kron(g.U_t0,g.U_c0)
    for theta in c['theta']:
        K=g.K(theta);U=top(K,g.k);pt=partial_trace_cap(U,g.C,g.T,g.k_c,g.k_t)
        lower=g.Delta_k*pt['k_minus_cap']; upper=float(np.trace(U.T@K@U)-np.trace(B.T@K@B))
        if abs(lower)<1e-10:lower=0.
        if abs(upper)<1e-10:upper=0.
        if lower>upper+1e-9 or lower<0:raise ValueError('population certificate inconsistent')
        spectrum_error=float(np.max(np.abs(np.linalg.eigvalsh(K)-np.sort(g.mu))))
        if spectrum_error>1e-10:raise ValueError('registered spectrum invariant failed')
        rows.append({'theta':theta,'spectrum_max_error':spectrum_error,'approximation_lower':lower,'feasible_upper':upper,'upper_reference':'fixed theta0 product factors in true-whitened coordinates; not fitted optimum','partial_trace':pt})
    return rows
