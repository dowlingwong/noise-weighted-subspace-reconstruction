"""Validate scientific records from saved arrays at the receiving boundary."""
import numpy as np
from . import c1,cresst

def close(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and not isinstance(a,bool):return np.isfinite(a) and np.isfinite(b) and bool(np.isclose(a,b,rtol=1e-9,atol=1e-10))
    return a==b

def check(record,path,freeze):
    spec=record['cell'];r=record['result']
    with np.load(path,allow_pickle=False) as z:
        if any(not np.isfinite(z[k]).all() for k in z.files):raise ValueError('nonfinite saved arrays')
        if spec['study']=='c1':
            if len(r['records'])!=25 or r['conditions']!=25 or r['references']!=2:raise ValueError('C1 record inventory')
            tags=set()
            for row in r['records']:
                tag=row['array_prefix']
                if tag in tags:raise ValueError('duplicate condition')
                tags.add(tag);computed,arr=c1.response(z[tag+'_A'],z[tag+'_E'],4)
                if any(not np.allclose(z[tag+'_'+key],value,rtol=1e-9,atol=1e-10) for key,value in arr.items()):raise ValueError('saved C1 matrix mismatch')
                if any(not close(row[k],v) for k,v in computed.items()):raise ValueError('C1 result disagrees with source arrays')
                if row['part']=='controlled':
                    if tag!=f"{row['direction']}_{row['requested_eta']}" or not np.isclose(computed['eta'],row['requested_eta'],rtol=1e-9):raise ValueError('direction/eta labels mismatch')
                if row['part']=='estimated':
                    si=c1.powers(z['true_covariance'])[1]
                    if not np.allclose(z[tag+'_E'],si@z[tag+'_effective_covariance']@si-np.eye(spec['d'])) or row['training_sha256']!=c1.digest(z['training']):raise ValueError('effective covariance/training identity mismatch')
                    if not np.array_equal(z[tag+'_A'],z['sample_A']) or row['prefix_sha256']!=c1.digest(z['calibration_pool'][:row['m']]):raise ValueError('sample reference/prefix mismatch')
            expected={f'{d}_{eta}' for d in ('near','far','dense','block') for eta in (.01,.03,.1,.3)}|{f'{method}_{m}' for method in ('empirical','shrinkage','toeplitz') for m in (4*spec['d'],16*spec['d'],64*spec['d'])}
            if tags!=expected:raise ValueError('wrong C1 condition set')
        else:
            from pathlib import Path
            from .protocol import sha
            prep=Path(path).parent/'prepared.npz'
            if sha(prep)!=freeze['prepared_sha256']:raise ValueError('prepared source hash mismatch')
            p=freeze['prepared'][spec['channel']];tag=p['tag']
            with np.load(prep,allow_pickle=False) as prepared:
                get=lambda key:prepared[tag+'_'+key]
                t,q,basis,read,ref=get('T'),get('Q'),get('basis'),get('readout'),get('C_ref')
                raw=get('raw_C_fit');raw=raw[:max(2,int(len(raw)*spec['cal_fraction']))]
                cov=cresst.psd_covariance(raw,t,q) if spec['covariance']=='psd' else cresst.covariance(raw@t.T@q,spec['covariance'])
                train=get('train')[:spec['n']]
                if spec['fixed_shape']:
                    tr,_,_=cresst.bank(get('training_noise')[:spec['n']],basis,p['scale'],'F/'+spec['channel'],True)
                    train=tr[np.arange(len(tr)),np.arange(len(tr))%4]
                pairs=cresst.maps(train,cov,basis,spec['rank'])
                # noise is a selected measured-record value; raw release hashes and IDs
                # bind its acquisition. Derivations from that value are rechecked here.
                noise=z['inputs'][:,0,:]-z['clean'][:,0,:]
                inp,clean,amp=cresst.bank(noise,basis,p['scale'],'E/'+spec['channel'],spec['fixed_shape'])
                expected,arrays=cresst.summarize(spec,freeze['manifest'],p,inp,clean,amp,read,pairs,np.linalg.inv(ref),noise,basis,ref,cov)
                # Floating subtraction may differ in the final bank hash. Validate
                # the original bank numerically and its stored byte hash separately.
                expected['paired_bank_sha256']=c1.digest(z['inputs'])
                if set(z.files)!=set(arrays) or any(not np.allclose(z[k],v,rtol=1e-10,atol=1e-12) for k,v in arrays.items()):raise ValueError('CRESST array/method inventory or reconstruction map mismatch')
                if not close(r,expected):raise ValueError('CRESST headline/endpoints disagree with recomputation')
