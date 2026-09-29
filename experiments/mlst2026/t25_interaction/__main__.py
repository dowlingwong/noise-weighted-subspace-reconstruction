import argparse,json,sys
from pathlib import Path
from . import workflow as w
from .protocol import config,enumeration
p=argparse.ArgumentParser(description='T25 candidate: fixtures only until production acceptance')
s=p.add_subparsers(dest='command',required=True)
for name in ('inventory','enumerate','freeze'):
 q=s.add_parser(name);q.add_argument('--config',required=True);q.add_argument('--out',required=True)
q=s.add_parser('run');q.add_argument('--freeze',required=True);q.add_argument('--run-root',required=True)
q=s.add_parser('_worker');q.add_argument('--freeze',required=True);q.add_argument('--index',type=int,required=True);q.add_argument('--out',required=True)
for name in ('verify','diagnose'):
 q=s.add_parser(name);q.add_argument('--run-root',required=True)
for name in ('reduce','export'):
 q=s.add_parser(name);q.add_argument('--run-root',required=True);q.add_argument('--out',required=True)
a=p.parse_args()
try:
 if a.command=='inventory':
  c=config(a.config);w.write_new(a.out,{'study':'t25','development_only':c['development_only'],'source':w.sources(),'allocation':enumeration(c),'production_status':'BLOCKED pending implementation and host acceptance'})
 elif a.command=='enumerate':w.write_new(a.out,enumeration(config(a.config)))
 elif a.command=='freeze':w.freeze(config(a.config),a.out)
 elif a.command=='run':w.run(a.freeze,a.run_root)
 elif a.command=='_worker':w.worker(a.freeze,a.index,a.out)
 elif a.command=='verify':print(json.dumps({'verified_cells':len(w.verify(a.run_root))}))
 elif a.command=='reduce':w.reduce(a.run_root,a.out)
 elif a.command=='export':w.export(a.run_root,a.out)
 elif a.command=='diagnose':
  root=Path(a.run_root);starts=list(root.glob('*.START.json'));ends=list(root.glob('*.END.json'))
  print(json.dumps({'starts':len(starts),'ends':len(ends),'reserved_charges':sum(w.read(q)['reserved_charges'] for q in starts),'missing_END_charge_seconds':(len(starts)-len(ends))*w.read(root/'freeze.json')['config']['budget']['cell_seconds'],'retry_allowed':False}))
except Exception as e:
 print(str(e),file=sys.stderr);sys.exit(2)
