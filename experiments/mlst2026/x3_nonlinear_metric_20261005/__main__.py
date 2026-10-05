"""X3 (nonlinear metric check) command line.

    python -m experiments.mlst2026.x3_nonlinear_metric_20261005 fixture --out WORK
    python -m experiments.mlst2026.x3_nonlinear_metric_20261005 production --out WORK --acceptance A --qualification Q

Host qualification steps: identity, snapshot, qualify, benchmark.  Receiving: audit.
"""
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['fixture', 'production', 'identity', 'references', 'torch-check', 'snapshot', 'qualify',
                                       'benchmark', 'freeze', 'run', 'export', 'audit', '_campaign', '_cell'])
    for name in ('config', 'freeze', 'root', 'out', 'acceptance', 'qualification'):
        p.add_argument('--' + name)
    p.add_argument('--index', type=int)
    a = p.parse_args()
    from . import workflow as w
    if a.command == 'fixture':
        print(json.dumps(w.fixture_mode(a.out), indent=2))
    elif a.command == 'production':
        print(json.dumps(w.production_mode(a.out, a.acceptance, a.qualification), indent=2))
    elif a.command == 'identity':
        print(w.code_identity(w.source_files()))
    elif a.command == 'references':
        from .protocol import BASE
        print(json.dumps(w.science.population_references(BASE), indent=2))
    elif a.command == 'torch-check':
        from .protocol import BASE
        import torch
        ok = torch.__version__.split('+')[0] == BASE['torch_version']
        print(json.dumps({'torch': torch.__version__, 'registered': BASE['torch_version'], 'ok': ok}))
        if not ok:
            raise SystemExit(3)
    elif a.command == 'snapshot':
        w.snapshot_record(a.out)
    elif a.command == 'qualify':
        w.qualify_fixtures(a.out)
    elif a.command == 'benchmark':
        print(json.dumps({k: v for k, v in w.benchmark(a.out).items() if k != 'environment'}, indent=2))
    elif a.command == 'freeze':
        from .protocol import config
        w.freeze(config(a.config), a.out, a.acceptance, a.qualification)
    elif a.command == 'run':
        w.execute(a.freeze, a.root)
    elif a.command == 'export':
        w.export(a.root, a.out)
    elif a.command == 'audit':
        print(json.dumps(w.audit(a.root, a.out), indent=2))
    elif a.command == '_campaign':
        w.worker_campaign(a.root)
    elif a.command == '_cell':
        w.worker_cell(a.freeze, a.index, a.out)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e), file=sys.stderr)
        sys.exit(2)
