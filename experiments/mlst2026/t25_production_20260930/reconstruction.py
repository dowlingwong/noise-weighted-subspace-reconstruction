"""Exact seed regeneration, all-factor replay and saved-map risk verification."""
from pathlib import Path
import numpy as np
from .protocol import BASE,FIXTURE,enumeration
from .science import *
from .workflow import read,sha,identity,verify_source

def assert_close(got,want):
    if isinstance(want,dict):
        if set(got)!=set(want):raise ValueError('record inventory changed')
        for k in want:assert_close(got[k],want[k])
    elif isinstance(want,list):
        if len(got)!=len(want):raise ValueError('record length changed')
        for a,b in zip(got,want):assert_close(a,b)
    elif isinstance(want,(float,np.floating)):
        if not np.isclose(got,want,rtol=1e-8,atol=1e-9):raise ValueError('numerical reconstruction mismatch')
    elif got!=want:raise ValueError('record identity changed')


def verify(root,require_complete=True,replay_optimizer=True):
    from .science import response,starts
    root=Path(root)
    if (root/'FAILED.json').exists() or (require_complete and not (root/'COMPLETE.json').is_file()):raise ValueError('failed/incomplete attempt')
    f=read(root/'freeze.json')
    if replay_optimizer:verify_source(f)
    c=f['config']
    if c not in (BASE,FIXTURE):raise ValueError('unknown config')
    if f['config_identity']!=identity(c) or f['enumeration']!=enumeration(c):raise ValueError('freeze inventory mismatch')
    from .science import population_references,structured
    assert_close(f['population_references'],population_references(c))
    records=[]
    expected_dirs={f'cell-{i:04d}' for i in range(len(f['enumeration']['cells']))}
    if {p.name for p in root.glob('cell-*')}!=expected_dirs:raise ValueError('incomplete cell inventory')
    for i,cell in enumerate(f['enumeration']['cells']):
        if not (root/f'{i:04d}.START.json').is_file() or not (root/f'{i:04d}.END.json').is_file():raise ValueError('interrupted attempt')
        assert_close(read(root/f'{i:04d}.START.json')['cell'],cell)
        dest=root/f'cell-{i:04d}';done=read(dest/'COMPLETE.json')
        if done!={'arrays_sha':sha(dest/'arrays.npz'),'record_sha':sha(dest/'record.json')}:raise ValueError('tampered record/arrays')
        r=read(dest/'record.json');a=dict(np.load(dest/'arrays.npz',allow_pickle=False))
        seed,th,N=cell['seed'],cell['theta'],cell['N'];g,sigma,L,Li,pool,x=make_data(c,seed,th);x=x[:N];mu=x.mean(0);S=(x-mu).T@(x-mu)/N;K=g.K(th);Cs=L@K@L;truth=top(K,g.k);W0=Li@Li
        for name,want in {'training':x,'calibration':pool,'S':S,'mean':mu,'Sigma':sigma,'K':K,'Cs':Cs}.items():
            if not np.allclose(a[name],want,rtol=1e-10,atol=1e-10):raise ValueError('data/moment mismatch '+name)
        if (r['seed'],r['theta'],r['N'],r['training_sha'],r['calibration_sha'])!=(seed,th,N,digest(x),digest(pool)):raise ValueError('seed/data identities')
        covs={'iso':np.eye(g.d),'oracle':sigma};cd={}
        for m in c['m']:
            cc,dd=estimate(pool[:m],g.C,g.T);cd[str(m)]=dd
            covs.update({f'{n}_{m}':z for n,z in cc.items()})
        assert_close(r['covariance'],cd);methods={};expected_arrays=set(('S','mean','Sigma','Cs','K','training','calibration'))
        for name,cov in covs.items():
            expected_arrays.add(name+'_cov')
            if not np.allclose(a[name+'_cov'],cov):raise ValueError('covariance differs')
            W=np.linalg.inv(cov)
            for cls in ('restricted','unrestricted'):
                tag=name+'__'+cls;B=a[tag+'_B'];A=map_for(B,W);b=(np.eye(g.d)-A)@mu
                expected_arrays.update(tag+'_'+s for s in ('B','A','b'))
                if not np.allclose(a[tag+'_A'],A) or not np.allclose(a[tag+'_b'],b):raise ValueError('map mismatch')
                if np.linalg.matrix_rank(B)!=g.k:raise ValueError('rank mismatch')
                row=risks(A,b,Cs,sigma,W0)|representation(B,Li,truth)
                row['training_loss']=float(np.trace(W@S)-np.trace(np.linalg.solve(B.T@W@B,B.T@W@S@W@B)))
                if cls=='restricted':
                    opt=r['methods'][tag]['optimization'];rr=[];objectives=[]
                    initial=starts(S,g.C,g.T,g.k_c,g.k_t,(c['namespace'],seed,th,N))
                    if len(opt['restarts'])!=4 or len(initial)!=4:raise ValueError('restart inventory')
                    if replay_optimizer:
                        replayB,replayOpt,replayFactors=structured(W,S,initial,c['max_iter'],c['gradient_tolerance'])
                        assert_close(opt,replayOpt)
                        if not np.allclose(B,replayB,atol=1e-9):raise ValueError('optimizer replay mismatch')
                    for j in range(4):
                        uc,ut=a[tag+f'_uc_{j}'],a[tag+f'_ut_{j}']
                        if replay_optimizer and (not np.allclose(uc,replayFactors[j][0],rtol=1e-8,atol=1e-9) or not np.allclose(ut,replayFactors[j][1],rtol=1e-8,atol=1e-9)):raise ValueError('restart factor replay mismatch')
                        expected_arrays.update([tag+f'_uc_{j}',tag+f'_ut_{j}'])
                        if not np.allclose(uc.T@uc,np.eye(g.k_c)) or not np.allclose(ut.T@ut,np.eye(g.k_t)):raise ValueError('factor orthogonality')
                        val,gc,gt=objective_gradient(uc,ut,W,S);objectives.append(val)
                        scale=np.trace(W@S);gn=np.sqrt(np.sum(((gc-uc@(uc.T@gc))/scale)**2)+np.sum(((gt-ut@(ut.T@gt))/scale)**2))
                        assert_close(opt['restarts'][j]['objective'],val);assert_close(opt['restarts'][j]['gradient_norm'],gn)
                        trajectory=opt['restarts'][j]['trajectory']
                        if not np.isfinite(trajectory).all() or np.min(np.diff(trajectory))<-1e-8:raise ValueError('invalid optimization trajectory')
                        if opt['restarts'][j]['status']=='stationary' and gn>c['gradient_tolerance']*1.001:raise ValueError('false stationarity')
                        ar=map_for(np.kron(ut,uc),W);br=(np.eye(g.d)-ar)@mu;rr.append(risks(ar,br,Cs,sigma,W0)['clean'])
                    selected=int(np.argmax(objectives))
                    if opt['selected']!=selected or not np.allclose(B,np.kron(a[tag+f'_ut_{selected}'],a[tag+f'_uc_{selected}'])):raise ValueError('class/selection mismatch')
                    row.update(optimization=opt,restart_clean_risks=rr,restart_risk_spread=float(max(rr)-min(rr)))
                else:
                    from .science import unrestricted
                    oracle=unrestricted(W,S,g.k)
                    if not np.allclose(qr(B)@qr(B).T,oracle@oracle.T,atol=1e-8):raise ValueError('unrestricted fit mismatch')
                methods[tag]=row
            if name not in ('iso','oracle'):methods[name+'__domain']=response(Li@S@Li,Li@cov@Li-np.eye(g.d),g.k)[0]
        if set(a)!=expected_arrays:raise ValueError('unexpected/missing arrays')
        assert_close(r['methods'],methods);records.append(r)
    return records



def audit_saved(root,f):
    """Portable receiver audit: everything in verify() except optimizer replay and
    the host environment comparison."""
    return verify(root,require_complete=True,replay_optimizer=False)
