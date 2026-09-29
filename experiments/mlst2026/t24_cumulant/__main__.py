"""Fail-closed T24 mathematical fixture workflow, no raw-data adapters."""
import argparse,json,sys,time,os
from pathlib import Path
import numpy as np
from . import core
from experiments.mlst2026.t25_interaction.workflow import write_new,read,sha,sources,verify_source
from experiments.mlst2026.t25_interaction.science import rng
p=argparse.ArgumentParser(description='T24 v2 candidate: synthetic core only; production remains blocked')
s=p.add_subparsers(dest='command',required=True)
q=s.add_parser('inventory');q.add_argument('--out',required=True);q.add_argument('--data-root',required=True)
q=s.add_parser('enumerate');q.add_argument('--inventory',required=True);q.add_argument('--out',required=True)
q=s.add_parser('freeze');q.add_argument('--out',required=True);q.add_argument('--fixture',action='store_true')
q=s.add_parser('run');q.add_argument('--freeze',required=True);q.add_argument('--run-root',required=True)
for name in ('verify','reduce','export'):
 q=s.add_parser(name);q.add_argument('--run-root',required=True)
 if name!='verify':q.add_argument('--out',required=True)
a=p.parse_args()
attempt_root=None
def check_freeze(f):
 if set(f)!={'version','status','N','p','B','sources'} or (f['version'],f['status'],f['N'],f['p'],f['B'])!=('t24/2','fixture_only',64,3,999):raise ValueError('closed fixture schema mismatch')
 verify_source(f)

def verify(root):
 root=Path(root);f=read(root/'freeze.json');check_freeze(f)
 if (root/'FAILED.json').exists() or read(root/'START.json')!={'development_only':True,'statistic_charges':1000,'retries':0}:raise ValueError('failed/interrupted ledger')
 done=read(root/'COMPLETE.json')
 if set(done)!={'arrays_sha','summary_sha'}:raise ValueError('completion ledger schema')
 if done['arrays_sha']!=sha(root/'arrays.npz') or done['summary_sha']!=sha(root/'summary.json'):raise ValueError('tampered output')
 arr=dict(np.load(root/'arrays.npz',allow_pickle=False));summary=read(root/'summary.json')
 if set(arr)!={'x','null'}:raise ValueError('array inventory')
 expected_x=rng('t24-v2-fixture','observed').normal(size=(64,3));expected_null=core.null_b2(rng('t24-v2-fixture','null'),64,3)
 if not np.array_equal(arr['x'],expected_x) or not np.array_equal(arr['null'],expected_null):raise ValueError('fixture draw identity')
 expected={'development_only':True,'statistics':core.statistics(arr['x']),'p':core.plus_one(core.statistics(arr['x'])['b2'],arr['null']),'scope':'iid mathematical fixture, not instrument evidence'}
 if expected!=summary:raise ValueError('summary reconstruction mismatch')
 return summary
try:
 if a.command=='inventory':
  root=Path(a.data_root);rows=[]
  if root.is_dir():
   for f in sorted(root.iterdir()):rows.append({'name':f.name,'directory':f.is_dir(),'bytes':f.stat().st_size if f.is_file() else None})
  write_new(a.out,{'version':'t24/2','root':str(root.resolve()),'root_exists':root.is_dir(),'top_level_inventory_only':rows,'qualification':'BLOCKED','reasons':['raw adapters, exact unit/basis ledger, temporal Gaussian models and leakage predicate unqualified'],'scientific_data_read':False})
 elif a.command=='enumerate':
  inv=read(a.inventory)
  write_new(a.out,{'qualification':'BLOCKED','inventory_sha':sha(a.inventory),'actual_units':None,'minimum_units':9,'max_units':32,'B':999,'R0':200,'R1':200,'R2':200,'statistic_evaluations_per_profile':1+999+600*(1+999),'caps':{'rss_bytes':4*1024**3,'scratch_bytes':8*1024**3,'unit_seconds':1800,'campaign_seconds':14400,'N':4096,'p':32,'threads':1,'retries':0},'reason':'actual profile/unit/dependence ledger unavailable; no resource feasibility claim'})
 elif a.command=='freeze':
  if not a.fixture:raise ValueError('PRODUCTION BLOCKED: actual data, bases, temporal model, leakage, full controls and allocation require independent acceptance; no override')
  write_new(a.out,{'version':'t24/2','status':'fixture_only','N':64,'p':3,'B':999,'sources':sources()})
 elif a.command=='run':
  f=read(a.freeze);check_freeze(f)
  for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
   if os.environ.get(key)!='1':raise ValueError('one thread required: '+key)
  root=Path(a.run_root).resolve();exp=Path(__file__).resolve().parents[3]
  ms=Path(os.environ.get('MLST_MS_ROOT',str(exp.parent/'arXiv-v1-polished-extended-preprint'))).resolve()
  if root.is_relative_to(exp) or root.is_relative_to(ms):raise ValueError('run root must be outside both sources')
  root.mkdir(parents=True,exist_ok=False);attempt_root=root;write_new(root/'freeze.json',f);write_new(root/'START.json',{'development_only':True,'statistic_charges':1000,'retries':0})
  x=rng('t24-v2-fixture','observed').normal(size=(64,3));null=core.null_b2(rng('t24-v2-fixture','null'),64,3)
  st=core.statistics(x);np.savez_compressed(root/'arrays.npz',x=x,null=null)
  write_new(root/'summary.json',{'development_only':True,'statistics':st,'p':core.plus_one(st['b2'],null),'scope':'iid mathematical fixture, not instrument evidence'})
  verify_source(f);write_new(root/'COMPLETE.json',{'arrays_sha':sha(root/'arrays.npz'),'summary_sha':sha(root/'summary.json')});verify(root)
 elif a.command=='verify':print(json.dumps(verify(a.run_root)))
 elif a.command=='reduce':write_new(a.out,verify(a.run_root))
 elif a.command=='export':
  verify(a.run_root);raise ValueError('fixture scientific export forbidden; instrument implementation not accepted')
except Exception as e:
 if attempt_root is not None and not (attempt_root/'FAILED.json').exists():write_new(attempt_root/'FAILED.json',{'error':str(e),'retry_allowed':False,'development_only':True})
 print(str(e),file=sys.stderr);sys.exit(2)
