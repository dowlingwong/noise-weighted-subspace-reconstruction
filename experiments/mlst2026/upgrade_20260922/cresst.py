"""Strict release adapter and fixed-affine semisynthetic methods.

No legacy endpoint or adaptive template selector is reused. Release I/O is
reused only after checking its permissive boolean and raw-width boundaries.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from .c1 import powers, eig, rng_for, digest
from noise_geometry.cresst.loader import load_cresst_release

ROLES=('D','C_ref','C_fit','F','V','E')

def load(root,split='train',width=512):
    root=Path(root); f=pd.read_csv(root/f'features_{split}.csv')
    required={'run','channel','hours','noise','clean'}
    if not required<=set(f): raise ValueError('missing metadata: '+str(required-set(f)))
    if f[list(required)].isna().any().any():raise ValueError('missing metadata values')
    for key in ('noise','clean'):
        s=f[key].astype(str).str.strip().str.lower()
        if not s.isin(['true','false','1','0','1.0','0.0']).all():raise ValueError('unknown boolean')
    for key in ('run','channel','hours'):
        if not np.isfinite(pd.to_numeric(f[key],errors='raise')).all():raise ValueError('nonfinite metadata')
    p=root/f'X_{split}.npy'
    with p.open('rb') as stream: header=stream.read(6)
    if header!=b'\x93NUMPY' and p.stat().st_size!=len(f)*width*4:raise ValueError('raw byte count/width')
    x,f,meta=load_cresst_release(root,split=split,mmap=True)
    if x.shape!=(len(f),width) or x.dtype!=np.dtype('<f4'):raise ValueError('width/dtype mismatch')
    if f.duplicated(['run','channel','hours']).any():raise ValueError('duplicate acquisition identity')
    # Stream validation, not a release-sized float64 materialization.
    seen=set()
    for start in range(0,len(x),256):
        block=x[start:start+256]
        if not np.isfinite(block).all():raise ValueError('nonfinite waveform')
        for row in block:
            h=hashlib.sha256(row.tobytes()).hexdigest()
            if h in seen:raise ValueError('duplicate waveform; qualify release before allocating roles')
            seen.add(h)
    for key in ('noise','clean'):
        f[key]=f[key].astype(str).str.strip().str.lower().isin(['true','1','1.0'])
    f['record_id']=[f'{split}:{int(i)}' for i in f.release_row_index]
    return x,f,meta

def allocate(f):
    """Single shared chronological axis, time ties never split across roles."""
    result={}
    for (run,ch),g in f.groupby(['run','channel'],sort=True):
        g=g.sort_values(['hours','release_row_index']);times=np.sort(g.hours.unique()); cuts=np.quantile(times,[.15,.30,.50,.70,.80])
        if len(set(cuts))<5:continue
        role=np.searchsorted(cuts,g.hours.to_numpy(),side='right')
        slots={r:g.iloc[np.where(role==i)[0]].release_row_index.astype(int).tolist() for i,r in enumerate(ROLES)}
        result[f'{run}/{ch}']={'run':float(run),'channel':float(ch),'cuts_hours':cuts.tolist(),'roles':slots}
    return result

def support(d,pre):
    if not 1<=pre<d:raise ValueError('pretrigger rank')
    b=np.r_[np.ones(pre)/pre,np.zeros(d-pre)]; t=np.eye(d)-np.ones((d,1))*b
    q=null_space(b[None,:]) # range(T)=ker(b^T), not generally 1-perp
    if not np.allclose(q@q.T@t,t,atol=1e-12):raise ValueError('support mismatch')
    return t,q

def template(pulses,pre,roi,dt):
    """Frozen trigger-aligned mean; no outcome-dependent shift search."""
    d=pulses.shape[1]; t,q=support(d,pre)
    h=(pulses@t.T).mean(axis=0); peak=np.max(h)
    if peak<=0:raise ValueError('nonpositive template polarity')
    h/=peak
    a,b=roi
    if not pre<=a<b<=d or dt<=0:raise ValueError('ROI/sample spacing')
    read=np.zeros(d);read[a:b]=dt;read[:pre]=-dt*(b-a)/pre
    normal=read@h
    if normal<=0:raise ValueError('nonpositive template area')
    read/=normal; h0=q.T@h; rq=q.T@read
    modes=[]
    for z in (np.gradient(h),np.gradient(np.gradient(h))):
        z=t@z;z-=h*(read@z);z=q.T@z
        if np.linalg.norm(z)<=1e-12:raise ValueError('degenerate shape mode')
        modes.append(z*.1*np.linalg.norm(h0)/np.linalg.norm(z))
    basis=np.column_stack([h0,*modes])
    if np.linalg.matrix_rank(basis)!=3 or not np.allclose(rq@basis,[1,0,0],atol=1e-10):raise ValueError('signal basis/readout')
    # Bounded z in [-1,1]^2, waveform positivity is not assumed for bipolar detector tails.
    for z1 in (-1,1):
        for z2 in (-1,1):
            shape=q@(basis@np.array([1,z1,z2]))
            if np.max(np.abs(shape))>2 or np.linalg.norm(shape)>2*np.linalg.norm(h):raise ValueError('shape validity gate')
    return t,q,basis,rq

def covariance(y,kind):
    """Support shrinkage. PSD model is stationary full-coordinate circular covariance,
    transformed to support by caller, then shrunk on the actual support."""
    raw=np.cov(y,rowvar=False); p=len(raw)
    if kind=='identity':return np.eye(p)
    if kind=='full':return .9*raw+.1*np.eye(p)*np.trace(raw)/p
    raise ValueError('unsupported covariance')

def psd_covariance(raw,t,q):
    # Demean each frequency's ensemble coefficient; explicit circular stationarity.
    z=raw@t.T; z=z-z.mean(axis=0)
    power=np.mean(np.abs(np.fft.rfft(z,axis=1))**2,axis=0)/z.shape[1]
    ac=np.fft.irfft(power,n=z.shape[1]); ix=np.arange(z.shape[1]); full=ac[(ix[:,None]-ix[None,:])%len(ix)]
    c=q.T@t@full@t.T@q;return .9*c+.1*np.eye(len(c))*np.trace(c)/len(c)

def gls(b,c):
    if np.linalg.matrix_rank(b)!=b.shape[1]:raise ValueError('basis rank deficient')
    ci=np.linalg.solve(c,b);return b@np.linalg.solve(b.T@ci,ci.T)

def maps(train,c,basis,k):
    n,p=train.shape
    if n<4*k or not 0<k<p:raise ValueError('inadmissible rank/training size')
    mean=train.mean(axis=0); z=train-mean; w,u=eig(z.T@z/(n-1));ss,si=powers(c)
    if w[k-1]<=1e-12:raise ValueError('effective PCA rank')
    _,v=eig(si@(z.T@z/(n-1))@si)
    ep=u[:,:k]@u[:,:k].T; pg=gls(u[:,:k],c);wp=gls(ss@v[:,:k],c)
    pairs={'identity':(np.zeros(p),np.eye(p)),'zero':(np.zeros(p),np.zeros((p,p)))}
    for name,b in [('pca',ep),('pca_gls',pg),('weighted_gls',wp)]:pairs[name]=(mean-b@mean,b)
    pairs['fixed_template']=(np.zeros(p),gls(basis[:,:1],c))
    # This uses exactly the constructed injected modes: explicitly oracle.
    pairs['oracle_joint_template']=(np.zeros(p),gls(basis,c))
    return pairs

def bank(noise,basis,scale,seed,fixed=False):
    """One deterministic bank outside the method loop; replicas retain raw record axis."""
    rng=rng_for('cresst-bank-v1',seed); n=len(noise)
    levels=np.array([1,2,4,8.])*scale
    z=rng.uniform(-1,1,size=(n,4,2));z[:]=0 if fixed else z
    shape=basis[:,0][None,None,:]+z@basis[:,1:].T
    clean=levels[None,:,None]*shape
    return noise[:,None,:]+clean,clean,np.broadcast_to(levels,(n,4))

def endpoints(inputs,clean,amplitudes,read,mapping,metric):
    a,b=mapping; recon=inputs@b.T+a
    ahat=recon@read # Never replace by inputs @ read.
    err=recon-clean;res=recon-inputs
    return {'amplitude_mse':((ahat-amplitudes)**2).mean(axis=1),
            'clean_euclidean':np.sum(err**2,axis=-1).mean(axis=1),
            'clean_common_metric':np.einsum('nld,de,nle->nl',err,metric,err).mean(axis=1),
            'observed_common_metric':np.einsum('nld,de,nle->nl',res,metric,res).mean(axis=1),
            'bias_by_amplitude':(ahat-amplitudes).mean(axis=0), 'amplitude_mean':ahat.mean(axis=0)}

def correction(b,c,m,*,linear=True):
    if not linear:raise ValueError('fixed affine map required; nonlinear selection unsupported')
    return float(2*np.trace(m@b@c)-np.trace(m@c))

def block_interval(values,block_length,seed,channels=1):
    """Paired raw-record contrasts; Bonferroni simultaneous intervals across channels."""
    n=len(values)//block_length
    blocks=np.array([np.mean(values[i*block_length:(i+1)*block_length]) for i in range(n)])
    # Include trailing records in last block, never discard outcomes.
    if n and n*block_length<len(values):blocks[-1]=np.mean(values[(n-1)*block_length:])
    if not n:return {'mean':float(np.mean(values)),'blocks':0,'interval':None,'scope':'descriptive'}
    out={'mean':float(np.mean(values)),'unique_records':len(values),'blocks':n,'block_means':blocks.tolist(),'scope':'channel_conditional' if n>=20 else 'descriptive','interval':None}
    if n>=20:
        r=rng_for('block-bootstrap',seed); sizes=np.full(n,block_length);sizes[-1]+=len(values)-n*block_length
        indices=r.integers(0,n,size=(9999,n));draws=(blocks[indices]*sizes[indices]).sum(axis=1)/sizes[indices].sum(axis=1)
        alpha=.05/channels;out['interval']=np.quantile(draws,[alpha/2,1-alpha/2]).tolist()
    return out

def qualify(root,c,exposure):
    from .protocol import sha
    s=c['scientific_settings']; x,f,meta=load(root,s['split'],s['width']); alloc=allocate(f); selected={};rejected={}
    if not isinstance(exposure,dict):raise ValueError('explicit channel exposure audit required')
    for name,ch in alloc.items():
        audit=exposure.get(name,{})
        if not isinstance(audit,dict):raise ValueError('exposure entry requires status, evidence, sample_spacing_seconds and acquisition_terms')
        exp=audit.get('status','uncertain')
        if not c['development_only'] and (not audit.get('evidence') or not audit.get('acquisition_terms') or audit.get('time_axis_verified') is not True or audit.get('sample_spacing_seconds') not in (.00064,.00128)):
            rejected[name]='missing exposure/acquisition/sample-spacing qualification';continue
        if exp not in ('exposed','uncertain','audited_untouched'):raise ValueError('unknown exposure class')
        noise={r:[i for i in ids if bool(f.iloc[i]['noise']) and bool(f.iloc[i]['clean'])] for r,ids in ch['roles'].items()}
        pulses=[i for i in ch['roles']['D'] if not bool(f.iloc[i]['noise']) and bool(f.iloc[i]['clean'])]
        # Eligibility uses metadata only; no outcome-based channel or record selection.
        if min(map(len,noise.values()))<s['minimum_noise_roles'] or len(pulses)<s['minimum_template_pulses']:
            rejected[name]='insufficient role counts';continue
        if len(selected)>=s['channel_cap']:rejected[name]='metadata sorted cap';continue
        if exp=='uncertain' and not c['development_only']:rejected[name]='exposure audit unresolved';continue
        selected[name]={**ch,'noise_roles':noise,'template_ids':pulses,'exposure':exp,
                        'sample_spacing_seconds':audit.get('sample_spacing_seconds',s['dt_seconds']),
                        'audit':audit,'scope':'replication' if exp=='exposed' else 'channel-adapted held-out time segment'}
    if not selected:raise ValueError('BLOCKED: no eligible channels after metadata/exposure gates')
    paths=[Path(root)/f'X_{s["split"]}.npy',Path(root)/f'features_{s["split"]}.csv']
    hashes={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
    if sum(len(ids) for ch in selected.values() for ids in ch['roles'].values())>c['budget']['max_unique_records']:raise ValueError('unique-record budget exceeded')
    return {'exposure_audit':exposure,'qualification':'QUALIFIED','qualification_scope':'adapter and metadata only; prepared D/V gate still mandatory','files':hashes,'channels':selected,'rejected':rejected,'official_source':'https://www.origins-cluster.de/odsl/dark-matter-data-center/available-datasets/cresst','description':'https://arxiv.org/html/2508.03078v1','sample_spacing_source':'512-bin downsampling from 8192/16384 at 25 kHz requires channel-specific confirmation','source_meta':{k:v for k,v in meta.items() if k not in ('source_traces','source_features')}}

def check_manifest(root,c,manifest):
    exposure=manifest['exposure_audit']
    rebuilt=qualify(root,c,exposure)
    expected={k:v for k,v in manifest.items() if k!='config_sha256'}
    if rebuilt!=expected:raise ValueError('manifest allocation/metadata/exposure mismatch')

def check_data(root,manifest):
    from .protocol import sha
    for name,info in manifest['files'].items():
        p=Path(root)/name
        if sha(p)!=info['sha256'] or p.stat().st_size!=info['bytes']:raise ValueError('data hash mismatch')

def prepare(root,c,manifest):
    """Only D/C/F/V rows accessed numerically. E metadata allocation is frozen already."""
    check_data(root,manifest);check_manifest(root,c,manifest);s=c['scientific_settings'];x,f,_=load(root,s['split'],s['width']);arrays={};report={}
    for name,ch in manifest['channels'].items():
        ids=ch['noise_roles'];tag=hashlib.sha256(name.encode()).hexdigest()[:12]
        t,q,basis,read=template(np.asarray(x[ch['template_ids'][:s['template_cap']]],float),s['pretrigger'],s['roi'],ch['sample_spacing_seconds'])
        cref=np.asarray(x[ids['C_ref'][:s['calibration_cap']]],float)@t.T@q
        ref=covariance(cref,'full'); metric=np.linalg.inv(ref)
        of=metric@basis[:,0]/(basis[:,0]@metric@basis[:,0]);scale=float(np.std(cref@of,ddof=1))
        if not scale>0:raise ValueError('invalid reference scale')
        rawfit=np.asarray(x[ids['C_fit'][:s['calibration_cap']]],float)
        cov=psd_covariance(rawfit,t,q);powers(cov)
        trainnoise=np.asarray(x[ids['F'][:128]],float)@t.T@q
        train,_,_=bank(trainnoise,basis,scale,'F/'+name)
        # One injection per unique F record: frozen cycling of amplitude levels.
        train=train[np.arange(len(train)),np.arange(len(train))%4]
        v=np.asarray(x[ids['V'][:512]],float)@t.T@q
        vv=v@read;center=vv-vv.mean();den=center@center
        correlations=[float(center[:-lag]@center[lag:]/den) if den else 0. for lag in range(1,min(33,len(v)//2))]
        significant=[i+1 for i,r in enumerate(correlations) if abs(r)>2/np.sqrt(len(v))]
        block=max(1,2*max(significant,default=0))
        vmaps=maps(train,cov,basis,3);vin,vs,va=bank(v,basis,scale,'V/'+name)
        pe=endpoints(vin,vs,va,read,vmaps['pca_gls'],metric)['amplitude_mse'];we=endpoints(vin,vs,va,read,vmaps['weighted_gls'],metric)['amplitude_mse']
        contrast=we-pe;nb=min(s['evaluation_cap'],len(ids['E']))//block
        # Conservative planning width, no claim of power or practical equivalence.
        blocks=[contrast[i:i+block].mean() for i in range(0,len(contrast),block) if len(contrast[i:i+block])==block]
        width=float(3.92*np.std(blocks,ddof=1)/np.sqrt(nb)) if len(blocks)>1 and nb else None
        report[name]={'tag':tag,'scale':scale,'block_length':block,'validation_autocorrelation':correlations,'planned_E_blocks':nb,'expected_interval_width_from_V':width,'purpose':'effect estimation; no practical threshold','inference_scope':'channel_conditional' if nb>=20 else 'descriptive_insufficient_blocks','training_ids':ids['F'][:128],'calibration_ids':ids['C_fit'][:s['calibration_cap']],'template_ids':ch['template_ids'][:s['template_cap']],'reference_ids':ids['C_ref'][:s['calibration_cap']],'validation_ids':ids['V'][:512],'evaluation_ids':ids['E'][:s['evaluation_cap']]}
        # Freeze all adaptation, including paired training and exact effective covariance.
        for key,a in {'T':t,'Q':q,'basis':basis,'readout':read,'C_ref':ref,'train':train,'training_noise':trainnoise,'raw_C_fit':rawfit,'primary_covariance':cov,'validation_contrast':contrast}.items():arrays[tag+'_'+key]=a
    return report,arrays

def run_cell(spec,c,manifest,prepared,arrays,root):
    s=c['scientific_settings'];name=spec['channel'];p=prepared[name];tag=p['tag'];get=lambda key:arrays[tag+'_'+key]
    t,q,basis,read=get('T'),get('Q'),get('basis'),get('readout');ref=get('C_ref');metric=np.linalg.inv(ref)
    raw=get('raw_C_fit');raw=raw[:max(2,int(len(raw)*spec['cal_fraction']))]
    cov=psd_covariance(raw,t,q) if spec['covariance']=='psd' else covariance(raw@t.T@q,spec['covariance']);powers(cov)
    x,_,_=load(root,s['split'],s['width']);noise=np.asarray(x[p['evaluation_ids']],float)@t.T@q
    train=get('train')[:spec['n']]
    if spec['fixed_shape']:
        # Rebuild same F noise bank with fixed shape; same paired methods within cell.
        fnoise=np.asarray(x[p['training_ids'][:spec['n']]],float)@t.T@q
        tr,_,_=bank(fnoise,basis,p['scale'],'F/'+name,True);train=tr[np.arange(len(tr)),np.arange(len(tr))%4]
    pairs=maps(train,cov,basis,spec['rank']);inputs,clean,amps=bank(noise,basis,p['scale'],'E/'+name,spec['fixed_shape'])
    return summarize(spec,manifest,p,inputs,clean,amps,read,pairs,metric,noise,basis,ref,cov)

def summarize(spec,manifest,p,inputs,clean,amps,read,pairs,metric,noise,basis,ref,cov):
    name=spec['channel']
    metrics={};outarrays={'inputs':inputs,'clean':clean,'amplitudes':amps,'effective_covariance':cov,'C_ref':ref,'evaluation_ids':np.array(p['evaluation_ids']),'readout':read}
    for method,mapping in pairs.items():
        e=endpoints(inputs,clean,amps,read,mapping,metric)
        a,b=mapping;zero=(noise@b.T+a)@read
        metrics[method]={k:v.tolist() for k,v in e.items()};metrics[method]['noise_only_mean']=float(zero.mean());metrics[method]['noise_only_sd']=float(zero.std(ddof=1));metrics[method]['actual_rank']=int(np.linalg.matrix_rank(b));metrics[method]['slope']=float(np.polyfit(amps[0],e['amplitude_mean'],1)[0])
        outarrays[method+'_a']=a;outarrays[method+'_B']=b
    direct=np.linalg.solve(ref,basis[:,0]);direct/=basis[:,0]@direct
    direct_mse=((inputs@direct-amps)**2).mean(axis=1)
    metrics['direct_filter']={'amplitude_mse':direct_mse.tolist(),'noise_only_sd':float(np.std(noise@direct,ddof=1)),'noise_only_mean':float(np.mean(noise@direct))}
    comparisons={}
    for method,e in metrics.items():
        value=float(np.mean(e['amplitude_mse'])); comparisons[method]={}
        for reference in ('identity','zero','pca_gls','direct_filter'):
            denominator=float(np.mean(metrics[reference]['amplitude_mse']));comparisons[method][reference]={'mse_difference':value-denominator,'mse_ratio':value/denominator if denominator>0 else None,'denominator_mse':denominator}
    diagnostic={method:{'measured_clean_minus_observed':float(np.mean(metrics[method]['clean_common_metric'])-np.mean(metrics[method]['observed_common_metric'])),'reference_covariance_correction':correction(b,ref,metric),'scope':'diagnostic only: measured noise mean/stationarity and reference covariance uncertain'} for method,(a,b) in pairs.items()}
    contrast=np.array(metrics['weighted_gls']['amplitude_mse'])-np.array(metrics['pca_gls']['amplitude_mse'])
    return {'comparisons':comparisons,'risk_identity_diagnostic':diagnostic,'run':manifest['channels'][name]['run'],'channel':manifest['channels'][name]['channel'],'metrics':metrics,'direct_filter_amplitude_mse':direct_mse.tolist(),'primary_contrast':block_interval(contrast,p['block_length'],spec['id'],len(manifest['channels'])),'paired_bank_sha256':digest(inputs),'unique_noise_records':len(noise),'block_length':p['block_length'],'units':'voltage-equivalent normalized pulse area','scope':manifest['channels'][name]['scope']},outarrays
