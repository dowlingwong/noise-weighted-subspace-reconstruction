"""Guarded dev-only D3--D5 launcher. No learning or confirmation commands."""
import argparse
import json
from . import access, pipeline
from experiments.mlst2026.hcp_rnoise_20261019.runner import environment


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify-kit')
    for command in ('decode-dev','gates','seal','fit-generator','cost','controls','summarize-controls'):
        p=sub.add_parser(command);p.add_argument('--dates',nargs='+',required=True)
        p.add_argument('--ms',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True)
        if command=='fit-generator':p.add_argument('--seal-commit',required=True)
        if command=='controls':p.add_argument('--workers',type=int,default=16)
    for command in ('confirmation','predict','fit','pilot','register'):
        sub.add_parser(command,help='prohibited in D0--D6')
    args=parser.parse_args(argv)
    if args.command=='verify-kit':result=access.verify()
    elif args.command in ('confirmation','predict','fit','pilot','register'):
        raise PermissionError('STOP outside dev-only D0--D6 authorization')
    else:
        units=access.authorize(args.dates,args.ms,args.review);environment()
        if args.command in ('decode-dev','gates'):result=pipeline.qualify(args.dates,args.ms,args.review,args.out)
        elif args.command=='seal':result=pipeline.seal(units,args.out)
        elif args.command=='fit-generator':result=pipeline.fit_generator(args.ms,args.seal_commit,args.out)
        elif args.command=='cost':result=pipeline.cost(units,args.out)
        elif args.command=='controls':result=pipeline.campaign(units,args.out,args.workers)
        elif args.command=='summarize-controls':result=pipeline.summarize(args.out)
    print(json.dumps(pipeline.clean(result),indent=2,allow_nan=False))
    if result.get('status')=='STOP':raise SystemExit(2)


if __name__=='__main__':main()
