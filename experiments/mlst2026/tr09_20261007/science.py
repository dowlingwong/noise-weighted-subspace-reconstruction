"""Accepted physical-coordinate T25 learners on paired sparse-latent records."""
from . import guard
guard.assert_direct()
import numpy as np
from scipy.stats import t
from experiments.mlst2026.t25_production_20260930 import science as t25

def data(c,seed,theta,p):
    g=t25.geometry(c); sigma=t25.covariance(c); L,Li=t25.powers(sigma)
    key=(c['namespace'],seed,theta); n=max(c['N'])
    amplitude=t25.rng(*key,'amplitude').normal(size=(n,g.d))
    activity=t25.rng(*key,'activity').uniform(size=(n,g.k))
    z=amplitude.copy(); z[:,:g.k]*=(activity<p)/np.sqrt(p)
    clean=z@(g.V(theta)*np.sqrt(g.mu)).T@L
    noise=t25.rng(*key,'noise').normal(size=(n,g.d))@L
    return g,sigma,L,Li,clean+noise

def fit(c,seed,theta,N,p):
    g,sigma,L,Li,pool=data(c,seed,theta,p)
    x,mean,S=t25.moments(pool,N); W=Li@Li; Cs=L@g.K(theta)@L
    initial=t25.starts(S,g.C,g.T,g.k_c,g.k_t,(c['namespace'],seed,theta,N,'paired-starts'))
    B,diag,factors=t25.structured(W,S,initial,150,1e-8)
    Bu=t25.unrestricted(W,S,g.k)
    arrays={'mean':mean,'S':S,'Cs':Cs,'Sigma':sigma,'W':W}
    results={}
    for label,basis in [('U',Bu),('F',B)]:
        A=t25.map_for(basis,W); b=(np.eye(g.d)-A)@mean
        arrays['A_'+label]=A;arrays['b_'+label]=b
        results[label]=t25.risks(A,b,Cs,sigma,W)
    rr=[]
    for i,(uc,ut) in enumerate(factors):
        A=t25.map_for(np.kron(ut,uc),W);b=(np.eye(g.d)-A)@mean
        arrays[f'A_restart_{i}']=A;arrays[f'b_restart_{i}']=b
        rr.append(t25.risks(A,b,Cs,sigma,W)['clean'])
    D=results['F']['clean']-results['U']['clean'];spread=max(rr)-min(rr)
    row={'seed':seed,'theta':theta,'N':N,'p':p,'R_U':results['U']['clean'],'R_F':results['F']['clean'],'D':D,
         'risks':results,'optimization':diag,'restart_clean_risks':rr,'restart_spread':spread,
         'optimization_limited':diag['restarts'][diag['selected']]['status']!='stationary' or spread>=abs(D),
         'selected_equals_hosvd':bool(np.linalg.norm(arrays['A_F']-arrays['A_restart_0'],'fro')<=1e-6),
         'training_sha':t25.digest(x),'seconds':None}
    guard.assert_loaded_modules()
    return row,arrays

def reduce(c,records):
    expected={(s,th,n,p) for s in c['seeds'] for th in c['theta'] for n in c['N'] for p in c['p']}
    lookup={(r['seed'],r['theta'],r['N'],r['p']):r for r in records}
    if set(lookup)!=expected or len(records)!=len(expected):raise ValueError('missing/duplicate/foreign cells')
    families={'primary':[],'secondary':[]}
    for th in c['theta']:
      for n in c['N']:
       for p in c['p']:
        if p==1:continue
        for endpoint in ['D','R_U','R_F']:
         vals=[lookup[s,th,n,p][endpoint]-lookup[s,th,n,1.][endpoint] for s in c['seeds']]
         groups=['secondary']+(['primary'] if p==.1 and n in [400,1000] and endpoint=='D' else [])
         for family in groups:
          size=4 if family=='primary' else len(c['theta'])*len(c['N'])*(len(c['p'])-1)*3
          sd=float(np.std(vals,ddof=1));mu=float(np.mean(vals));half=float(t.ppf(1-.05/(2*size),len(vals)-1)*sd/np.sqrt(len(vals)))
          ci=[mu-half,mu+half];sign='negative' if ci[1]<0 else 'positive' if ci[0]>0 else 'undecided'
          families[family].append(dict(theta=th,N=n,p=p,endpoint=endpoint,values=vals,mean=mu,sd=sd,ci=ci,decision=sign,zero_SE=sd==0,family_size=size,seeds=len(vals)))
    primary=families['primary'];decisions=[r['decision'] for r in primary]
    reading='replicated' if len(primary)==4 and all(x=='negative' for x in decisions) else 'did_not_replicate_in_positive_cells' if 'positive' in decisions else 'not_resolved_at_this_replication'
    return dict(families=families,reading=reading,optimization_limited_count=sum(r['optimization_limited'] for r in records),hosvd_agreement_count=sum(r['selected_equals_hosvd'] for r in records),cell_count=len(records),interval='approximate Bonferroni Student-t, independent seeds; zero-SE not population certainty')
