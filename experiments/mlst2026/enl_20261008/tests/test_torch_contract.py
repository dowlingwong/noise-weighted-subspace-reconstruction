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
