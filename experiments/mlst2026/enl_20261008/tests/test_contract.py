"""Boundary and algebra challenges; no confirmation draws or namespace claim."""
import copy,json,os
from pathlib import Path
import numpy as np
import pytest
from experiments.mlst2026.enl_20261008 import workflow as w,science as s

def test_grid_keys_and_resource_contract():
 c=w.config('confirmation');w.validate(c,'confirmation')
 assert len(s.order(c))==432 and len(set(s.order(c)))==432
 assert sum(len(v) for v in c['arms'].values())*len(c['p'])*2==36
 assert c['recipe']['restarts']*len(s.order(c))==1728
 assert c['budget']['campaign_seconds']==259200 and c['budget']['run_seconds']==252000
 assert set(c['seeds']).isdisjoint(w.config('fixture')['seeds'])
 assert s.models.key_int(c['namespace'],c['seeds'][0],3,3200,.1,'init',0)!=s.models.key_int(c['namespace'],c['seeds'][0],6,3200,.1,'init',0)

def test_confirmation_refusal_precedes_numerical_and_output(tmp_path,monkeypatch):
 monkeypatch.delenv('ENL_CONFIRMATION_AUTHORISED',raising=False)
 monkeypatch.setattr(w,'numerical',lambda:pytest.fail('numerical imports reached before refusal'))
 with pytest.raises(ValueError,match='authorization'):w.run('confirmation',tmp_path/'never')
 assert not (tmp_path/'never').exists()
 monkeypatch.setenv('ENL_CONFIRMATION_AUTHORISED','yes')
 with pytest.raises(ValueError,match='exact freeze'):w.run('confirmation',tmp_path/'never')

def test_source_tamper_and_cache_refusal(tmp_path,monkeypatch):
 fake=tmp_path/'root';p=fake/'experiments';p.mkdir(parents=True);(p/'bad.pyc').write_bytes(b'not executable')
 monkeypatch.setattr(w,'ROOT',fake)
 with pytest.raises(ValueError,match='bytecode'):w.no_cache()

def test_common_latents_noise_nested_and_probe_separation():
 c=w.config('fixture');seed=c['seeds'][0];pool=s.raw_pools(c,seed)
 m1=s.SparseLatentModel(c['geometry'],c['theta'],1.,c['n_sparse']);mp=s.SparseLatentModel(c['geometry'],c['theta'],.1,c['n_sparse'])
 y1,x1,z1=s.data(pool,'train',m1);yp,xp,zp=s.data(pool,'train',mp)
 mask=pool['train_U']<.1
 assert np.allclose(zp[:,:c['n_sparse']],z1[:,:c['n_sparse']]*mask/np.sqrt(.1),rtol=1e-14,atol=1e-14)
 assert np.allclose(y1-x1,yp-xp)
 assert s.digest(y1[:32])!=s.digest(y1[32:64])
 assert s.digest(pool['evaluation_G'])!=s.digest(pool['representation_G'])
 expected=mp.sample_keyed(64,c['namespace'],seed,'train')
 assert np.array_equal(expected[0],yp) and np.array_equal(expected[2],zp)

def test_independent_affine_risk_and_gaussian_bayes():
 c=w.config('fixture');m=s.SparseLatentModel(c['geometry'],.25,1.,4);pool=s.raw_pools(c,c['seeds'][0]);Y,_,_=s.data(pool,'train',m)
 A,b,_=s.baselines.fitted_shrinkage(Y,3)
 independent=np.linalg.norm((A-np.eye(m.d))@(m.V*np.sqrt(m.mu)),'fro')**2+np.linalg.norm(A,'fro')**2+b@b
 assert abs(independent-s.baselines.affine_risk(A,b,m.K))<1e-10
 refs=s.references(m,3);assert refs['bayes']==pytest.approx(refs['full_wiener'],abs=1e-10)
 assert all(x['error_estimate']==0 for x in refs['quadrature_convergence'])

def test_zero_se_undecided_and_bonferroni():
 assert s.interval([0.]*24,36,.05)['decision']=='undecided'
 assert s.interval([.1]*24,36,.05)['zero_SE'] and s.interval([.1]*24,36,.05)['ci']==[.1,.1]
 assert s.interval([-1.]*24,36,.05)['decision']=='negative'
 assert s.interval([1.]*24,36,.05)['decision']=='positive'
 a=s.interval(np.arange(24),36,.05);b=s.interval(np.arange(24),1,.05)
 assert a['ci'][0]<b['ci'][0] and a['df']==23 and a['family_size']==36
 with pytest.raises(ValueError):s.interval([1.,float('nan')],36,.05)

def test_crossing_no_interpolation_and_nonmonotonic_gaps():
 Ns=[400,1600,3200,12800]
 assert s.crossing(Ns,['positive','negative','positive','negative'])['brackets']==[[400,1600],[3200,12800]]
 assert s.crossing(Ns,['positive','undecided','negative','negative'])['brackets']==[]
 assert s.crossing(Ns,['negative']*4)['state']=='advantage_at_smallest_N'
 assert s.crossing(Ns,['positive']*4)['state']=='never_negative'

def test_primary_census_and_descriptive_signature_does_not_change_risk():
 c=w.config('fixture');rows=[{'seed':seed,'k':k,'N':N,'p':p,'contrasts':{'projector':-1.,'shrinkage':-2.},'optimization_limited':True,'evaluation':{'signature':True}} for seed,k,N,p in s.order(c)]
 v=s.reduce(c,rows);assert v['primary_family']==18 and len(v['primary'])==18
 with pytest.raises(ValueError,match='unique'):s.reduce(c,rows+rows[:1])
 with pytest.raises(ValueError,match='unique'):s.reduce(c,rows[:-1])
 for row in rows:row['evaluation']['signature']=False
 assert [r['ci'] for r in s.reduce(c,rows)['primary']]==[r['ci'] for r in v['primary']]

def test_affine_interference_and_bayes_diagonal():
 c=w.config('fixture');m=s.SparseLatentModel(c['geometry'],.25,.1,4)
 I=s.interference(lambda y:y,m,c)
 assert np.allclose(I['slope'],np.eye(4)) and not I['off_diagonal_pairs']
 assert np.shape(I['response'])==(4,8,4)
 b=s.interference(lambda y:s.bayes.posterior_mean(m,y),m,c)
 assert not b['off_diagonal_pairs'] and np.all(np.diag(b['slope'])<1)

def test_unsafe_manifest_refuses_before_archive(tmp_path):
 out=tmp_path/'return';out.mkdir();(out/'SHA256SUMS').write_text('0'*64+'  ../escape\n')
 with pytest.raises(ValueError):w.verify_receipt(out)
 link=out/'linked';link.symlink_to(out/'SHA256SUMS')
 with pytest.raises(ValueError,match='symlink'):w.verify_receipt(out)


def test_closure_total_numerator_undefined_and_unclipped():
 x=s.closure_values(10.,2.,5.,9.)
 assert x['full_bayes']['value']==1.6 and x['sparse_oracle']['value']==8.
 assert s.closure_values(10.,12.,5.,9.)['full_bayes']['value']==-.4
 assert s.closure_values(10.,2.,10.,11.)['full_bayes']['value'] is None
 assert s.closure_values(10.,2.,10.,11.)['sparse_oracle']['value'] is None


def test_single_seed_resource_has_no_scientific_intervals():
 c=w.config('resource');rows=[{'seed':seed,'k':k,'N':N,'p':p,'restarts':[{}]} for seed,k,N,p in s.order(c)]
 v=s.reduce(c,rows);assert v['kind']=='ENGINEERING_ONLY' and v['cells']==2 and v['restarts']==2 and v['inferential_intervals'] is False
