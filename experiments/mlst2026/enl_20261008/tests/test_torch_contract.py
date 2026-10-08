import numpy as np
import pytest
pytest.importorskip('torch')
from experiments.mlst2026.enl_20261008 import science as s,workflow as w

def test_checkpoint_reloading_selection_and_saved_pool_audit(tmp_path):
 w.run('fixture',tmp_path/'fixture')
 w.verify_receipt(tmp_path/'fixture');rows=w.audit(tmp_path/'fixture')
 assert len(rows)==18 and all(len(r['restarts'])==2 for r in rows)
 for r in rows:
  valid=[i for i,x in enumerate(r['restarts']) if x['checkpoint'] is not None]
  assert r['selected']==min(valid,key=lambda i:r['restarts'][i]['fit']['best_validation_loss'])

def test_gradient_matches_directional_difference():
 torch=s.models.import_torch();c=w.config('fixture');m=s.SparseLatentModel(c['geometry'],.25,1.,4)
 model=s.models.build(torch,m.d,2,[8,6],'silu');pool=s.raw_pools(c,c['seeds'][0]);Y,_,_=s.data(pool,'train',m);X=m.physical(Y[:16])
 loss,norm=s.validation(torch,model,X,m.Li,16)
 p=next(model.parameters());before=p.detach().clone();eps=1e-6
 def value():
  a=s.models.tensor(torch,X);r=(a-model(a))@s.models.tensor(torch,m.Li);return float((r*r).sum(1).mean().detach())
 a=s.models.tensor(torch,X);r=(a-model(a))@s.models.tensor(torch,m.Li);((r*r).sum(1).mean()).backward();g=float(p.grad[0,0])
 with torch.no_grad():p[0,0]+=eps
 plus=value()
 with torch.no_grad():p[0,0]-=2*eps
 minus=value()
 with torch.no_grad():p.copy_(before)
 assert (plus-minus)/(2*eps)==pytest.approx(g,rel=1e-5,abs=1e-6) and norm>=0


def test_all_restart_failure_is_preserved(tmp_path,monkeypatch):
 c=w.config('fixture');pool=s.raw_pools(c,c['seeds'][0])
 monkeypatch.setattr(s.models,'train',lambda *args:(None,{'failed':'engineered nonfinite validation','epochs_run':1,'validation_history':[]}))
 with pytest.raises(RuntimeError,match='all restarts failed'):s.fit(c,c['seeds'][0],2,32,.1,pool,tmp_path)
 failed=list(tmp_path.glob('*_failed.json'));assert len(failed)==1
 assert len(w.read(failed[0])['restarts'])==2


def test_nonfinite_validation_fails_restart():
 c=w.config('fixture');torch=s.models.import_torch();X=np.full((8,12),np.nan)
 model,rec=s.models.train(torch,{**c,'k':2},X,X,np.eye(12),17)
 assert model is None and rec['failed']=='non-finite validation loss'


def test_validation_tie_chooses_first_restart(tmp_path,monkeypatch):
 c=w.config('fixture');torch=s.models.import_torch();pool=s.raw_pools(c,c['seeds'][0])
 def fake(torch,cfg,Xtr,Xva,Li,key):
  model=s.models.build(torch,12,2,c['hidden'],'silu')
  with torch.no_grad():
   for param in model.parameters():param.fill_(0.)
  val,_=s.validation(torch,model,Xva,Li,16)
  return model,{'best_validation_loss':val,'best_epoch':0,'epochs_run':2,'stopped':'max_epochs','validation_history':[val,val]}
 monkeypatch.setattr(s.models,'train',fake)
 row=s.fit(c,c['seeds'][0],2,32,.1,pool,tmp_path)
 assert row['selected']==0 and row['restarts'][0]['fit']['best_validation_loss']==row['restarts'][1]['fit']['best_validation_loss']


def test_clean_module_entrypoint_installs_runtime_without_recursive_import():
 import os,subprocess,sys
 script="import runpy; w=runpy.run_module('experiments.mlst2026.enl_20261008.workflow',run_name='enl_cli_fixture'); w['numerical'](); w['loaded_identity'](w['sources']()); print('CLI_NUMERICAL_OK')"
 result=subprocess.run([sys.executable,'-B','-c',script],env=dict(os.environ),capture_output=True,text=True,timeout=120)
 assert result.returncode==0,result.stderr
 assert 'CLI_NUMERICAL_OK' in result.stdout
