import argparse,json,sys
from pathlib import Path
from . import workflow as w
from .protocol import config,enumeration

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inventory','enumerate','freeze','run','verify','reduce','export','diagnose','audit','qualify','benchmark','snapshot','_campaign','_cell','_export','_verify'])
    for name in ('config','freeze','root','out','acceptance','qualification'):p.add_argument('--'+name)
    p.add_argument('--index',type=int);a=p.parse_args()
    if a.command=='inventory':w.write_new(a.out,{'deployment':w.VERSION,'sources':w.sources(),'environment':w.environment()})
    elif a.command=='enumerate':w.write_new(a.out,enumeration(config(a.config)))
    elif a.command=='freeze':w.freeze(config(a.config),a.out,a.acceptance,a.qualification)
    elif a.command=='run':w.execute(a.freeze,a.root)
    elif a.command=='verify':w.bounded_verify(a.root,a.out)
    elif a.command=='reduce':w.bounded_verify(a.root,a.out,include_reduction=True)
    elif a.command=='export':w.export(a.root,a.out)
    elif a.command=='diagnose':print(json.dumps(w.diagnose(a.root),indent=2))
    elif a.command=='_campaign':w.worker_campaign(a.root)
    elif a.command=='_cell':w.worker_cell(a.freeze,a.index,a.out)
    elif a.command=='_export':w.export_worker(a.root,a.out)
    elif a.command=='qualify':w.qualify_fixtures(a.out)
    elif a.command=='benchmark':w.benchmark(a.out)
    elif a.command=='snapshot':w.snapshot_record(a.out)
    elif a.command=='audit':w.bounded_audit(a.root,a.out)
    elif a.command=='_verify':w.verify_worker(a.root,a.out)

if __name__=='__main__':
    try:main()
    except Exception as e:print(type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(2)
