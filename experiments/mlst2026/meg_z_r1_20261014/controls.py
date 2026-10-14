"""DESIGN 6/CONFIG controls: exact band geometry, L1–L7, C0–C12, 35 claims.

Preflight uses the same complete replicate function as the campaign. No controls
execute on import. Each replicate has an independent prescribed SeedSequence.
"""
import json,math,time
from pathlib import Path
import numpy as np
from scipy.linalg import toeplitz,solve_toeplitz
from scipy.stats import beta
from experiments.mlst2026.meg_z_20261013.common import sha_file,atomic_json
from .protocol import PACKAGE,DEV,load
from .windows import window_plan
from .gates import reference,coordinates,role_g2,g3,groups

def rng(law,cell,subject,replicate):
    entropy=[20261014,int(law),int(cell),int(subject),int(replicate)]
    return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy))),entropy

def source_moments(derived,gates_path):
    """Only the sealed T20 reference moment dataset is read; SHA before/after."""
    import h5py
    sealed=json.loads(Path(gates_path).read_text());sources={};hashes={}
    for sub in DEV:
        path=Path(derived)/f'{sub}.h5';want=sealed['subjects'][sub]['derived_file_sha256'];before=sha_file(path)
        if before!=want:raise RuntimeError(f'STOP {sub} pre-read SHA mismatch')
        try:
            with h5py.File(path,'r') as h:
                S=h['reference']['second_moment'][()].astype(float)
                n=int(h['reference']['n'][()])
                assert S.shape==(480,480) and n>1
                sources[sub]=S
        finally:
            after=sha_file(path)
            if after!=want:raise RuntimeError(f'STOP {sub} post-read SHA mismatch')
        hashes[sub]={'before':before,'after':after,'sealed':want,'dataset':'reference/second_moment (centred divisor n)'}
    return sources,hashes

def ar_shape(S,C=24,T=20,L=88):
    S4=S.reshape(C,T,C,T,order='F');St=np.einsum('ctcs->ts',S4)/C;Sc=np.einsum('ctdt->cd',S4)/T
    Sc=Sc/(np.trace(Sc)/C)
    empirical=np.array([np.diag(St,k).mean() for k in range(T)])
    a=solve_toeplitz((empirical[:10],empirical[:10]),empirical[1:11]);innovation=float(empirical[0]-a@empirical[1:11])
    roots=np.roots(np.r_[1,-a]);
    if innovation<=0 or np.max(np.abs(roots))>=1:raise RuntimeError('STOP nonstationary AR(10); no correction searched')
    r=np.empty(max(L,20));r[:11]=empirical[:11]
    for lag in range(11,len(r)):r[lag]=sum(a[j-1]*r[lag-j] for j in range(1,11))
    cov_t=toeplitz(r[:L]);Lt=np.linalg.cholesky(cov_t);Lc=np.linalg.cholesky(Sc)
    pop=np.kron(toeplitz(r[:10]),Sc);ev,V=np.linalg.eigh(pop)
    return {'Lt':Lt,'Lc':Lc,'population':pop,'population_Q':V,'C':C,'L':L,
        'fit':{'order':10,'coefficients':a.tolist(),'innovation_variance':innovation,'max_root_modulus':float(np.max(np.abs(roots))),
        'lags_11_19':list(range(11,20)),'observed':empirical[11:20].tolist(),'fitted':r[11:20].tolist(),
        'relative_errors':((r[11:20]-empirical[11:20])/empirical[0]).tolist(),'source':'unshrunk all-window T20 moments; temporal Toeplitz partial trace; spatial partial trace trace C'}}

def identity_shape(C,L):
    return {'Lt':np.eye(L),'Lc':np.eye(C),'population':np.eye(10*C),'population_Q':np.eye(10*C),'C':C,'L':L,'fit':{'law':'identity IID'}}

def cut_generated(Y,plans,T=10):
    out={}
    for role in ('ref','train','cal','eval'):
        xx=[];mm={k:[] for k in ('position','rposition','run','block','gap','start')}
        for R,yr,plan in zip(Y['geometry'],Y['runs'],plans):
            for w in plan[role]:
                xx.append(yr[w['gap'],w['offset']:w['offset']+T].reshape(-1))
                for k in mm:mm[k].append(w[k])
        out[role]=(np.asarray(xx),{k:np.asarray(v,dtype=int) for k,v in mm.items()})
    return out

def generate(shape,geometry,law,gen):
    a={1:0.,2:0.,3:.25,4:.5,5:.75,6:0.,7:0.}[law];ys=[]
    for R in geometry:
        # Stationary AR(10) covariance at each gap, independently drawn before gap AR.
        xi=gen.normal(size=(R['G'],shape['L'],shape['C']))
        y=np.einsum('ij,gjc->gic',shape['Lt'],xi,optimize=True)@shape['Lc'].T
        for g in range(1,R['G']):y[g]=a*y[g-1]+math.sqrt(1-a*a)*y[g]
        ys.append(y)
    plans=[window_plan(R['gaps'],R['run']) for R in geometry]
    rec=cut_generated({'geometry':geometry,'runs':ys},plans)
    if law in (6,7):
        d=shape['C']*10;V=shape['population_Q'][:,int(.9*d):];factors={}
        for role,(X,meta) in rec.items():
            X=X.copy();keys=meta['run'] if law==6 else meta['run']*1000+meta['block']
            for key in sorted(set(keys)):
                if int(key) not in factors:
                    factors[int(key)]=(math.exp(gen.normal(0,.2 if law==6 else .1)),math.exp(gen.normal(0,.1)))
                st,sg=factors[int(key)];idx=keys==key;Z=X[idx]
                X[idx]=math.sqrt(sg)*(Z+(math.sqrt(st)-1)*(Z@V)@V.T)
            rec[role]=(X,meta)
    return rec

def scale_group(X,V,factor):return X+(math.sqrt(factor)-1)*(X@V)@V.T

def apply_cell(rec,Sref,shape,cell,gen):
    X,meta=rec['eval'];X=X.copy();w,Q,L=coordinates(Sref);C=shape['C'];d=C*10
    if cell==1:X*=math.sqrt(1.5)
    elif cell in (2,3,4):
        sl=groups(d)[{2:'g_top10',3:'g_bottom50',4:'g_middle40'}[cell]];X=scale_group(X,Q[:,sl],1.5)
    elif cell in (5,6):
        S4=Sref.reshape(C,10,C,10,order='F');v=np.linalg.eigh(np.einsum('ctdt->cd',S4)/10)[1][:,0 if cell==5 else -1]
        z=X.reshape(-1,10,C);z=z+(math.sqrt(8)-1)*(z@v)[:,:,None]*v;X=z.reshape(-1,d)
    elif cell in (7,8,9):
        u=gen.normal(size=d);u*=math.sqrt((2 if cell==8 else 4)*2.06604/np.sum((u@L)**2))
        sel=meta['position']==0
        if cell!=9:X[sel]+=u
        else:
            # C9: same physical CxT locked profile shifted by independent Uniform[0,.2] onset latency.
            # Linear interpolation on the rational decimated clock, zero outside its support.
            profile=u.reshape(10,C);times=np.arange(10)*12/1100
            for i in np.flatnonzero(sel):
                delay=gen.uniform(0,.2)
                X[i]+=np.stack([np.interp(times-delay,times,profile[:,c],left=0,right=0) for c in range(C)],1).reshape(-1)
    elif cell in (10,11):X=scale_group(X,Q[:,0:1] if cell==10 else Q[:,-1:],8)
    elif cell==12:
        # Loudest physical direction is the top eigenvector of generator population covariance.
        X=scale_group(X,shape['population_Q'][:,-1:],1.5)
    elif cell!=0:raise ValueError('cell outside C0–C12')
    out=dict(rec);out['eval']=(X,meta);return out

def effective_reference_count(meta,T=10):
    count=0
    for run,gap in set(zip(meta['run'],meta['gap'])):
        starts=np.sort(meta['start'][(meta['run']==run)&(meta['gap']==gap)])
        for seq in np.split(starts,np.flatnonzero(np.diff(starts)!=1)+1):
            if len(seq):count+=(len(seq)+T-1)//T
    return count

def assess(rec,details=True,Sref=None):
    from .gates import g2_diagonal
    Sref=reference(rec['ref'][0]) if Sref is None else Sref
    w,Q,L=coordinates(Sref)
    whitened={ro:(X@L,meta) for ro,(X,meta) in rec.items()}
    def compare(ro):
        Z=whitened['train'][0];V=whitened[ro][0];Z=Z-Z.mean(0);V=V-V.mean(0)
        X=rec['train'][0];Y=rec[ro][0]
        r=g2_diagonal(np.mean(Z*Z,axis=0),np.mean(V*V,axis=0),
            float(np.mean(np.sum((X-X.mean(0))**2,axis=1))),float(np.mean(np.sum((Y-Y.mean(0))**2,axis=1))),V,details and ro=='eval')
        r['G2S_pass']=all(.8<=r[k]<=1.25 for k in ('m_eval/m_train','g_bottom50','g_middle40','g_top10','physical_trace_ratio'))
        return r
    G2=compare('eval');G2cal=compare('cal')
    from .gates import retired_descriptives
    G2['retired_LW']=retired_descriptives(rec['ref'][0],rec['eval'][0],Sref,effective_reference_count(rec['ref'][1]))
    G3=g3({ro:v for ro,v in whitened.items() if ro!='ref'})
    allX=np.concatenate([v[0] for v in rec.values()]);allZ=np.concatenate([v[0] for v in whitened.values()]);allrun=np.concatenate([v[1]['run'] for v in rec.values()])
    run_drift={}
    for run in range(1,7):
        sel=allrun==run;Z=allZ[~sel];V=allZ[sel];Z=Z-Z.mean(0);V=V-V.mean(0)
        X=allX[~sel];Y=allX[sel]
        r=g2_diagonal(np.mean(Z*Z,axis=0),np.mean(V*V,axis=0),float(np.mean(np.sum((X-X.mean(0))**2,axis=1))),float(np.mean(np.sum((Y-Y.mean(0))**2,axis=1))),details=False)
        r['G2S_pass']=all(.8<=r[k]<=1.25 for k in ('m_eval/m_train','g_bottom50','g_middle40','g_top10','physical_trace_ratio'))
        run_drift[str(run)]=r
    return {'G2':G2,'G2_cal':G2cal,'G3':G3,'run_blocked_G2S':run_drift,'reference_n':len(rec['ref'][0])}

def replicate(law,cell,subject_index,replicate_index,shapes,geometry,details=True):
    gen,entropy=rng(law,cell,subject_index,replicate_index);sub=DEV[subject_index]
    t0=time.perf_counter();shape=identity_shape(shapes[sub]['C'],shapes[sub]['L']) if law==1 else shapes[sub]
    rec=generate(shape,geometry['subjects'][sub]['runs'],law,gen);S=reference(rec['ref'][0]);rec=apply_cell(rec,S,shape,cell,gen)
    result=assess(rec,details,Sref=S);result.update(law=law,cell=cell,subject=sub,replicate=replicate_index,seed_entropy=entropy,bit_generator='PCG64DXSM',seconds=time.perf_counter()-t0)
    return result

def claims():
    out=[]
    out += [('G2 null',l,0,'G2',False) for l in range(1,7)]
    out += [('G3 slot null',l,0,'slot',False) for l in (1,2,3,4,6)]
    out += [('G3 screen null',l,0,'screen',False) for l in (1,2,3)]
    out += [('G2 detection',l,c,'G2',True) for l in (1,2,4) for c in range(1,7)]
    out += [('G3 detection',l,7,'slot',True) for l in (1,2,4)]
    assert len(out)==35
    return out

def event(row,which):
    return (not row['G2']['pass']) if which=='G2' else row['G3']['slot_fail' if which=='slot' else 'screen_flag']

def claim_bound(count,n,detect):
    alpha=load()['controls']['alpha_per_claim']
    bound=float(beta.ppf(alpha,count,n-count+1)) if detect and count else (0. if detect else (1. if count==n else float(beta.ppf(1-alpha,count+1,n-count))))
    return {'count':count,'n':n,'one_sided_bound':bound,'direction':'lower' if detect else 'upper','pass':n==800 and (count>=745 if detect else count<=22)}

def summarize(rows):
    expected={(l,c,s,r) for l in range(1,8) for c in range(13) for s in DEV for r in range(200)}
    keys=[(v['law'],v['cell'],v['subject'],v['replicate']) for v in rows]
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise RuntimeError('STOP incomplete or duplicate control return')
    claims_out=[]
    for label,l,c,which,detect in claims():
        selected=[v for v in rows if v['law']==l and v['cell']==c];cnt=sum(event(v,which) for v in selected)
        claims_out.append({'claim':label,'law':l,'cell':c,**claim_bound(cnt,800,detect)})
    desc={}
    for l in range(1,8):
        for c in range(13):
            rr=[r for r in rows if r['law']==l and r['cell']==c]
            item={'n':800,'G2_exclusion_rate':sum(not r['G2']['pass'] for r in rr)/800,
                'G3_slot_exceedance_rate':sum(r['G3']['slot_fail'] for r in rr)/800,
                'G3_screen_flag_rate':sum(r['G3']['screen_flag'] for r in rr)/800,
                'G3_subject_exclusion_rate':sum(not r['G3']['pass'] for r in rr)/800,
                'run_blocked_G2S_any':sum(any(not z['G2S_pass'] for z in r['run_blocked_G2S'].values()) for r in rr)/800,
                'run_blocked_G2S_run6':sum(not r['run_blocked_G2S']['6']['G2S_pass'] for r in rr)/800}
            if c==0 and l in (2,3,4,5):item['generalised_extreme_quantiles']=np.quantile([r['G2']['extremes'] for r in rr],[.025,.5,.975],axis=0).tolist()
            desc[f'L{l}/C{c}']=item
    return {'claims':claims_out,'descriptive_rates':desc,'status':'PASS' if all(v['pass'] for v in claims_out) else 'STOP','failure_rule':load()['controls']['failure_rule']}
