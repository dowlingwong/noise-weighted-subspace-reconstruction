"""E-NL saved-pool/model algebra. Never selects models by truth risk."""
import hashlib, json, random, time
from pathlib import Path
import numpy as np
from scipy.stats import t
from experiments.mlst2026.p1nl_dev_20261006.generator import SparseLatentModel, rng
from experiments.mlst2026.p1nl_dev_20261006 import bayes
from experiments.mlst2026.p1nl_dev_20261006.tr07_pilot import models, baselines, diagnostics


def digest(a):
    a=np.ascontiguousarray(a); return hashlib.sha256(a.tobytes()).hexdigest()


def locations(c):
    return [(int(k),N,p) for k,Ns in c['arms'].items() for N in Ns for p in c['p']]


def order(c):
    seeds=list(c['seeds']); random.Random(20261008).shuffle(seeds)
    result=[]
    for seed in seeds:
        ps=list(c['p']); random.Random(models.key_int(c['namespace'],seed,'p-order')).shuffle(ps)
        for p in ps:
            loc=[(k,N,pp) for k,N,pp in locations(c) if pp==p]
            random.Random(models.key_int(c['namespace'],seed,p,'location-order')).shuffle(loc)
            result.extend((seed,*row) for row in loc)
    return result


def raw_pools(c,seed):
    d=c['geometry']['C']*c['geometry']['T']; out={}
    for role,n in [('train',max(N for Ns in c['arms'].values() for N in Ns)),('evaluation',c['eval_records']),('representation',c['representation_records'])]:
        key=(c['namespace'],seed,role)
        out[role+'_G']=rng(*key,'latent').normal(size=(n,d))
        out[role+'_U']=rng(*key,'activity').random(size=(n,c['n_sparse']))
        out[role+'_E']=rng(*key,'noise').normal(size=(n,d))
    return out


def data(pool,role,m):
    Z=pool[role+'_G'].copy(); Z[:,:m.n_sparse]*=(pool[role+'_U']<m.p)/np.sqrt(m.p)
    S=(Z*np.sqrt(m.mu))@m.V.T
    return S+pool[role+'_E'],S,Z


def save_model(path,model):
    np.savez_compressed(path,**{k:v.detach().numpy() for k,v in model.state_dict().items()})


def load_model(torch,c,k,path):
    model=models.build(torch,c['geometry']['C']*c['geometry']['T'],k,c['hidden'],c['activation'])
    with np.load(path,allow_pickle=False) as a:
        state=model.state_dict()
        if set(a.files)!=set(state): raise ValueError('checkpoint state census mismatch')
        for key in state:
            if a[key].shape!=tuple(state[key].shape) or not np.isfinite(a[key]).all(): raise ValueError('checkpoint shape/nonfinite mismatch')
        model.load_state_dict({key:models.tensor(torch,a[key]) for key in state})
    model.eval();return model


def validation(torch,model,X,Li,batch):
    """Loss and full gradient at the saved validation-best checkpoint, chunked."""
    model.zero_grad(); total=0.
    for i in range(0,len(X),batch):
        a=models.tensor(torch,X[i:i+batch]); w=models.tensor(torch,Li)
        loss=(((a-model(a))@w)**2).sum()/len(X)
        total+=float(loss.detach());loss.backward()
    norm=float(np.sqrt(sum(float((p.grad.detach()**2).sum()) for p in model.parameters())))
    model.zero_grad()
    return total,norm


def interference(f,m,c):
    amps=np.array(c['interference_amplitudes'])/np.sqrt(m.p); zero=f(np.zeros((1,m.d)))[0]
    response=[]
    for a in range(m.n_sparse):
        y=np.outer(amps*np.sqrt(m.mu[a]),m.V[:,a]); response.append((f(y)-zero)@m.V[:,:m.n_sparse]/np.sqrt(m.mu[:m.n_sparse]))
    response=np.array(response); slope=(response*amps[None,:,None]).sum(1)/(amps**2).sum()
    return {'amplitudes':amps.tolist(),'response':response.tolist(),'slope':slope.tolist(),
            'off_diagonal_pairs':[[a,b] for a in range(m.n_sparse) for b in range(m.n_sparse) if a!=b and abs(slope[a,b])>=c['interference_threshold']]}


def references(m,k):
    ref=bayes.references(m,k)
    ref['quadrature_convergence']=[bayes.sparse_mmse(float(lam),m.p)[1] for lam in m.mu[:m.n_sparse]]
    ref['sparse_oracle']=float(np.asarray(ref['bayes_per_coordinate'])[:m.n_sparse].sum()+m.mu[m.n_sparse:].sum())
    return ref


def closure_values(shrinkage,risk,bayes_risk,sparse_oracle):
    result={}
    for name,target in [('full_bayes',bayes_risk),('sparse_oracle',sparse_oracle)]:
        denom=shrinkage-target
        result[name]={'numerator':shrinkage-risk,'denominator':denom,'value':None if denom<=.001 else (shrinkage-risk)/denom}
    return result

def evaluate(torch,ae,m,k,c,pool):
    Y,S,_=data(pool,'evaluation',m)
    Sh=models.predict(torch,ae,m.physical(Y),c['eval_chunk'])@m.Li
    ev=diagnostics.mc_risk(Sh,S,m)
    Yrep,_,Zrep=data(pool,'representation',m)
    Shrep=models.predict(torch,ae,m.physical(Yrep),c['eval_chunk'])@m.Li
    code=models.code(torch,ae,m.physical(Yrep),c['eval_chunk'])
    rec=diagnostics.recovery(Shrep,code,Zrep[:,:m.n_sparse],m,.5)
    recovered=sum(v is not None and v>=c['recovery_threshold'] for v in rec['output_r2'])
    f=lambda y:models.predict(torch,ae,m.physical(y),c['eval_chunk'])@m.Li
    inter=interference(f,m,c)
    refs=references(m,k); risk=ev['risk']
    closure=closure_values(refs['shrinkage'],risk,refs['bayes'],refs['sparse_oracle'])
    return {'R_AE':risk,'mcse':ev['mcse'],'per_coordinate':ev['per_coordinate'].tolist(),'per_coordinate_se':ev['per_coordinate_se'].tolist(),
            'groups':diagnostics.group_sums(ev['per_coordinate'],m,k),'recovery':rec,'recovered_output_count':recovered,
            'recovered_code_count':sum(v is not None and v>=c['recovery_threshold'] for v in rec['code_r2']),
            'interference':inter,'signature':recovered>k and bool(inter['off_diagonal_pairs']),
            'leakage_k':diagnostics.leakage(Sh,k),'leakage_6':diagnostics.leakage(Sh,m.n_sparse),
            'references':{name:float(refs[name]) for name in ['projector','shrinkage','full_wiener','bayes','sparse_oracle','bayes_error_estimate']},
            'quadrature_convergence':refs['quadrature_convergence'],'closures':closure,'sparse_numerator':float(np.asarray(refs['shrinkage_per_coordinate'])[:m.n_sparse].sum()-ev['per_coordinate'][:m.n_sparse].sum())},ev['per_sample']


def fit(c,seed,k,N,p,pool,out):
    torch=models.import_torch();m=SparseLatentModel(c['geometry'],c['theta'],p,c['n_sparse']);cfg={**c,'k':k}
    Y,_,_=data(pool,'train',m);Y=Y[:N];X=m.physical(Y);nval=int(.2*N);ntrain=N-nval
    Ap,bp,info_p=baselines.fitted_projector(Y[:ntrain],k);As,bs,info_s=baselines.fitted_shrinkage(Y[:ntrain],k)
    stem=f'cell_{seed}_{k}_{N}_{p}';mean=Y[:ntrain].mean(0);cov=(Y[:ntrain]-mean).T@(Y[:ntrain]-mean)/ntrain
    ell,V=np.linalg.eigh(cov)
    np.savez_compressed(out/(stem+'_linear.npz'),Ap=Ap,bp=bp,As=As,bs=bs,mean=mean,cov=cov,eigenvalues=ell[::-1],eigenvectors=V[:,::-1])
    rec={'seed':seed,'k':k,'N':N,'p':p,'train_digest':digest(Y[:ntrain]),'validation_digest':digest(Y[ntrain:]),'n_train':ntrain,'n_validation':nval,
         'R_proj':baselines.affine_risk(Ap,bp,m.K),'R_shrink':baselines.affine_risk(As,bs,m.K),'proj_info':info_p,'shrink_info':info_s,
         'baseline_groups':{'projector':diagnostics.group_sums(diagnostics.affine_per_coordinate(Ap,bp,m),m,k),'shrinkage':diagnostics.group_sums(diagnostics.affine_per_coordinate(As,bs,m),m,k)},'restarts':[]}
    best=None
    for r in range(c['recipe']['restarts']):
        begin=time.monotonic();init=models.key_int(c['namespace'],seed,k,N,p,'init',r)
        ae,fitrec=models.train(torch,cfg,X[:ntrain],X[ntrain:],m.Li,init)
        row={'index':r,'init_key':init,'fit':fitrec,'train_seconds':time.monotonic()-begin,'checkpoint':None}
        if ae is not None:
            name=stem+f'_restart{r}.npz';save_model(out/name,ae);row['checkpoint']=name
            loss,grad=validation(torch,ae,X[ntrain:],m.Li,c['recipe']['batch']);row['validation_gradient_norm']=grad
            if not np.isclose(loss,fitrec['best_validation_loss'],rtol=1e-9,atol=1e-9):raise ValueError('validation-best checkpoint mismatch')
            ev_start=time.monotonic();ev,per=evaluate(torch,ae,m,k,c,pool);row['evaluation_seconds']=time.monotonic()-ev_start;row['evaluation']=ev
            # Retain paired per-draw differences, even though primary means use exact affine risks.
            Yev,Sev,_=data(pool,'evaluation',m)
            ep=((Yev@Ap.T+bp-Sev)**2).sum(1);es=((Yev@As.T+bs-Sev)**2).sum(1)
            np.savez_compressed(out/(stem+f'_evaluation{r}.npz'),AE=per,projector=ep,shrinkage=es)
            row['paired_mcse']={name:float((per-v).std(ddof=1)/np.sqrt(len(per))) for name,v in [('projector',ep),('shrinkage',es)]}
            if best is None or fitrec['best_validation_loss']<rec['restarts'][best]['fit']['best_validation_loss']:best=r
        rec['restarts'].append(row)
    if best is None:
        (out/(stem+'_failed.json')).write_text(json.dumps(rec,indent=2,allow_nan=False))
        raise RuntimeError('all restarts failed; preserve partial attempt; no retry')
    rec['selected']=best;rec['optimization_limited']=rec['restarts'][best]['fit']['stopped']=='max_epochs'
    rec['evaluation']=rec['restarts'][best]['evaluation'];rec['R_AE']=rec['evaluation']['R_AE']
    rec['contrasts']={'projector':rec['R_AE']-rec['R_proj'],'shrinkage':rec['R_AE']-rec['R_shrink']}
    # Reference probe matrices, deterministic and independent of model selection.
    rec['reference_interference']={name:interference(f,m,c) for name,f in [('projector',lambda y:y@Ap.T+bp),('shrinkage',lambda y:y@As.T+bs),('bayes',lambda y:bayes.posterior_mean(m,y))]}
    return rec


def interval(values,family,alpha):
    values=np.array(values,dtype=float)
    if len(values)<2 or not np.isfinite(values).all():raise ValueError('invalid seed contrast')
    mean=float(values.mean());sd=float(values.std(ddof=1));half=float(t.isf(alpha/(2*family),len(values)-1)*sd/np.sqrt(len(values)))
    lo,hi=mean-half,mean+half
    return {'mean':mean,'sd':sd,'ci':[lo,hi],'zero_SE':sd==0,'decision':'negative' if hi<0 else 'positive' if lo>0 else 'undecided','values':values.tolist(),'family_size':family,'df':len(values)-1}


def crossing(Ns,decisions):
    brackets=[[a,b] for a,b,x,y in zip(Ns,Ns[1:],decisions,decisions[1:]) if x=='positive' and y=='negative']
    state='decided_bracket' if brackets else 'advantage_at_smallest_N' if decisions[0]=='negative' else 'never_negative' if 'negative' not in decisions and 'undecided' not in decisions else 'undecided_gap_or_no_adjacent_bracket'
    return {'brackets':brackets,'state':state,'decisions':decisions}


def reduce(c,rows):
    expected={(s,k,N,p) for s in c['seeds'] for k,N,p in locations(c)}
    indexed={(r['seed'],r['k'],r['N'],r['p']):r for r in rows}
    if len(indexed)!=len(rows) or set(indexed)!=expected:raise ValueError('complete unique seed/cell census required')
    if len(c['seeds'])==1 and c['namespace']=='paper1-enl-dev/resource-full-20261008':
        return {'kind':'ENGINEERING_ONLY','cells':len(rows),'restarts':sum(len(r['restarts']) for r in rows),'inferential_intervals':False,'scope':'Bounded resource measurement only; no scientific gate.'}
    summaries=[];primary=[]
    for k,N,p in locations(c):
        rs=[indexed[(s,k,N,p)] for s in c['seeds']]
        summary={'k':k,'N':N,'p':p,'optimization_limited':sum(r['optimization_limited'] for r in rs),'descriptive_signature_count':sum(r['evaluation']['signature'] for r in rs)}
        decisions=[]
        for label in ['projector','shrinkage']:
            v=interval([r['contrasts'][label] for r in rs],c['primary_family'],c['risk_alpha']);v.update(k=k,N=N,p=p,baseline=label);primary.append(v);decisions.append(v['decision'])
        summary['beats_both']=decisions==['negative','negative'];summary['signature_label_allowed']=summary['beats_both'] and summary['descriptive_signature_count']>=c['signature_seeds'];summaries.append(summary)
    if len(primary)!=c['primary_family']:raise ValueError('primary family mismatch')
    crossings={}
    for k,Ns in c['arms'].items():
        if len(Ns)<2:continue
        for p in c['p']:
            separate={label:crossing(Ns,[next(x['decision'] for x in primary if x['k']==int(k) and x['N']==N and x['p']==p and x['baseline']==label) for N in Ns]) for label in ['projector','shrinkage']}
            joint=[b for b in separate['projector']['brackets'] if b in separate['shrinkage']['brackets']]
            crossings[f'{k}/{p}']={'by_baseline':separate,'joint_brackets':joint}
    return {'primary':primary,'levels':summaries,'crossings':crossings,'seeds':len(c['seeds']),'cells':len(rows),'primary_family':c['primary_family'],'optimization_limited':sum(r['optimization_limited'] for r in rows),'scope':'Prescribed fitted learners; descriptive signature is not a mechanism theorem.'}
