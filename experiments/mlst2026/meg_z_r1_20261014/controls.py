"""DESIGN 6/CONFIG controls: exact band geometry, L1–L7, C0–C12, 35 claims.

Preflight uses the same complete replicate function as the campaign. No controls
execute on import. Each replicate has an independent prescribed SeedSequence.
"""
import json,math,time
from pathlib import Path
import numpy as np
from scipy.linalg import toeplitz,solve_toeplitz
from scipy.stats import beta
from scipy.signal import lfilter
from numpy.lib.stride_tricks import sliding_window_view
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
    return {'Lt':Lt,'Lc':Lc,'ar_coefficients':a,'innovation_variance':innovation,'population':pop,'population_Q':V,'C':C,'L':L,
        'fit':{'order':10,'coefficients':a.tolist(),'innovation_variance':innovation,'max_root_modulus':float(np.max(np.abs(roots))),
        'lags_11_19':list(range(11,20)),'observed':empirical[11:20].tolist(),'fitted':r[11:20].tolist(),
        'relative_errors':((r[11:20]-empirical[11:20])/empirical[0]).tolist(),'source':'unshrunk all-window T20 moments; temporal Toeplitz partial trace; spatial partial trace trace C'}}

def identity_shape(C,L):
    return {'Lt':np.eye(L),'Lc':np.eye(C),'population':np.eye(10*C),'population_Q':np.eye(10*C),'C':C,'L':L,'fit':{'law':'identity IID'}}

def cut_generated(Y,plans,T=10):
    out={}
    for role in ('ref','train','cal','eval'):
        xx=[];mm={k:[] for k in ('position','rposition','run','block','gap','start')}
        for yr,plan in zip(Y['runs'],plans):
            rows=plan[role]
            if not rows:continue
            indices=np.asarray([[w['gap'],w['offset']] for w in rows],dtype=int)
            # View axes are gap, offset, channel, time; preserve time-major vec.
            view=sliding_window_view(yr,T,axis=1)
            xx.append(view[indices[:,0],indices[:,1]].transpose(0,2,1).reshape(len(rows),-1))
            for k in mm:mm[k].append(np.fromiter((w[k] for w in rows),dtype=int,count=len(rows)))
        out[role]=(np.concatenate(xx) if xx else np.asarray([]),
                   {k:np.concatenate(v) if v else np.empty(0,dtype=int) for k,v in mm.items()})
    return out

def temporal_draw(shape,xi):
    """Same stationary Cholesky coupling, then exact AR(10) innovations.

    Preserve the initial ten stationary samples and all PCG64DXSM draws/order.
    Yule-Walker makes subsequent Cholesky rows the AR recursion, up to roundoff.
    """
    if shape.get('fit',{}).get('law')=='identity IID':return xi
    if 'ar_coefficients' not in shape or xi.shape[1]<=10:
        return np.einsum('ij,gjc->gic',shape['Lt'],xi,optimize=True)
    den=np.r_[1.,-shape['ar_coefficients']]
    initial=np.einsum('ij,gjc->gic',shape['Lt'][:10,:10],xi[:,:10],optimize=True)
    forcing=xi*np.sqrt(shape['innovation_variance'])
    boundary=toeplitz(den[:10],np.r_[1.,np.zeros(9)])
    forcing[:,:10]=np.einsum('ij,gjc->gic',boundary,initial,optimize=True)
    return lfilter([1.],den,forcing,axis=1)

def gap_filter(y,a):
    if a==0:return y
    scale=math.sqrt(1-a*a)
    forcing=y*scale
    forcing[0]=y[0]  # stationary run start, not a zero-state attenuated draw
    return lfilter([1.],[1.,-a],forcing,axis=0)

def generate(shape,geometry,law,gen):
    a={1:0.,2:0.,3:.25,4:.5,5:.75,6:0.,7:0.}[law];ys=[]
    for R in geometry:
        # Stationary AR(10) covariance at each gap, independently drawn before gap AR.
        xi=gen.normal(size=(R['G'],shape['L'],shape['C']))
        y=gap_filter(temporal_draw(shape,xi)@shape['Lc'].T,a)
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
        if cell!=9:X[meta['position']==0]+=u
        else:
            # C9 (DESIGN rev 2.3, RD-12e; R2-B F5): the same physical CxT locked profile is added to the continuous
            # eval band stretch at band offset 0 + U[0, 0.2 s] (linear interpolation on the decimated clock, zero
            # outside its support) before windows are cut, so delays spill into positions 1-2. Cutting is linear,
            # so each fixed-grid window receives its slice of that stretch (offset = T * position, RD-12a).
            # One delay per eval gap, drawn in (run, gap) order.
            profile=u.reshape(10,C);dt=12/1100;times=np.arange(10)*dt;grid=np.arange(shape['L'])*dt
            stretch={}
            for key in sorted(set(zip(meta['run'].tolist(),meta['gap'].tolist()))):
                delay=gen.uniform(0,.2)
                stretch[key]=np.stack([np.interp(grid-delay,times,profile[:,c],left=0,right=0) for c in range(C)],1)
            for i,(r,g,pos) in enumerate(zip(meta['run'].tolist(),meta['gap'].tolist(),meta['position'].tolist())):
                X[i]+=stretch[(r,g)][10*pos:10*pos+10].reshape(-1)
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
    # Run-blocked descriptive (DESIGN rev 2.3, RD-12d; R2-B F4): run r vs all other runs, pooling train + cal + eval
    # windows only (no reference, no guard), as gate_cli does in R3. Pooled-run moments via sufficient sums.
    run_drift={};d=Sref.shape[0]
    counts=np.zeros(6,dtype=int);zs=np.zeros((6,d));zss=np.zeros((6,d));xs=np.zeros((6,d));xss=np.zeros(6)
    for role,(X,meta) in rec.items():
        if role=='ref':continue
        Z=whitened[role][0];runs,starts,n=np.unique(meta['run'],return_index=True,return_counts=True)
        idx=runs-1;counts[idx]+=n
        zs[idx]+=np.add.reduceat(Z,starts,axis=0);zss[idx]+=np.add.reduceat(Z*Z,starts,axis=0)
        xs[idx]+=np.add.reduceat(X,starts,axis=0);xss[idx]+=np.add.reduceat(np.sum(X*X,axis=1),starts)
    for run in range(1,7):
        i=run-1;nt=int(counts.sum()-counts[i]);ne=int(counts[i])
        ts=zs.sum(0)-zs[i];tss=zss.sum(0)-zss[i];tx=xs.sum(0)-xs[i];txss=xss.sum()-xss[i]
        qt=tss/nt-(ts/nt)**2;qe=zss[i]/ne-(zs[i]/ne)**2
        pt=txss/nt-np.sum((tx/nt)**2);pe=xss[i]/ne-np.sum((xs[i]/ne)**2)
        r=g2_diagonal(qt,qe,float(pt),float(pe),details=False)
        r['G2S_pass']=all(.8<=r[k]<=1.25 for k in ('m_eval/m_train','g_bottom50','g_middle40','g_top10','physical_trace_ratio'))
        run_drift[str(run)]=r
    from .gates import g1_spd
    G1=g1_spd(Sref,rec['ref'][0].shape[1]//10,10,load()['gates']['G1']['unchanged_limits'])
    return {'G1':G1,'G2':G2,'G2_cal':G2cal,'G3':G3,'run_blocked_G2S':run_drift,'reference_n':len(rec['ref'][0])}

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
            item={'n':800,'G1_limit_failure_rate':sum(not r['G1']['pass'] for r in rr)/800,'G2_exclusion_rate':sum(not r['G2']['pass'] for r in rr)/800,
                'G3_slot_exceedance_rate':sum(r['G3']['slot_fail'] for r in rr)/800,
                'G3_screen_flag_rate':sum(r['G3']['screen_flag'] for r in rr)/800,
                'G3_subject_exclusion_rate':sum(not r['G3']['pass'] for r in rr)/800,
                'run_blocked_G2S_any':sum(any(not z['G2S_pass'] for z in r['run_blocked_G2S'].values()) for r in rr)/800,
                'run_blocked_G2S_run6':sum(not r['run_blocked_G2S']['6']['G2S_pass'] for r in rr)/800}
            if c==0 and l in (2,3,4,5):item['generalised_extreme_quantiles']=np.quantile([r['G2']['extremes'] for r in rr],[.025,.5,.975],axis=0).tolist()
            desc[f'L{l}/C{c}']=item
    return {'claims':claims_out,'descriptive_rates':desc,'status':'PASS' if all(v['pass'] for v in claims_out) else 'STOP','failure_rule':load()['controls']['failure_rule']}
