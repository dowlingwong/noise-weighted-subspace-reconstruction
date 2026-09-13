#!/usr/bin/env python3
"""Repository-based remote handoff. Freeze locally; preflight/execute on the server.

No archive is necessary when both full repositories are present. File hashes,
not a branch name or a self-referential commit ID, define the tested snapshot.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

EXP = Path(__file__).resolve().parents[4]
ROUND = EXP / 'experiments/mlst2026/successor_20260913'
sys.path.insert(0, str(EXP))
PROTOCOLS = {p: f'PROTOCOL_{p.upper()}_v2.json' for p in ('t19s','t20s','t21s','t22s')}
PROTOCOLS['cresst_s'] = 'PROTOCOL_CRESST_S_v1.json'
TESTS = [f'tests/test_successor_20260913_{p}.py' for p in ('common', *PROTOCOLS)] + ['tests/test_successor_remote.py']
SELECTIONS = ('t19s','t20s','t21s','t22s','cresst-archive','cresst-calibration')
THREAD_KEYS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREAD_KEYS:
    os.environ[key] = '1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def commit(root):
    return subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()


def matrix_records():
    from experiments.mlst2026.successor_20260913.common.protocol import load_protocol
    out = {}
    for pkg, name in PROTOCOLS.items():
        path = ROUND / pkg / name
        p = load_protocol(path)
        runner = importlib.import_module(f'experiments.mlst2026.successor_20260913.{pkg}.runner')
        matrix = runner.build_matrix(p, False)
        out[pkg] = {'protocol_sha256': digest(path), 'matrix_sha256': matrix.matrix_hash(),
                    'cells': len(matrix.cells), 'fits': sum(c['n_fits'] for c in matrix.cells)}
    return out


def freeze(ms, manifest):
    from experiments.mlst2026.successor_20260913.make_handoff import CRESST_ARCHIVE_FILES, ARCHIVE_INPUTS
    exp_paths = {EXP / 'pyproject.toml'}
    for folder in ('src','experiments','tests'):
        exp_paths.update(p for p in (EXP / folder).rglob('*.py') if '__pycache__' not in p.parts)
    exp_paths.update(ROUND / p / n for p,n in PROTOCOLS.items())
    exp_paths.update(ROUND.glob('*/requirements.txt'))
    exp_paths.add(ROUND / 'remote/requirements-tested.txt')
    for folder in ARCHIVE_INPUTS:
        paths = [p for p in (EXP / folder).rglob('*') if p.is_file()]
        if not paths:
            raise ValueError(f'Historical exposure inputs missing: {folder}')
        exp_paths.update(paths)
    ms_paths = {ms / p for p in ('main.tex','supplement.tex','evidence/MANIFEST.json',
                                 'scripts/verify_evidence.py','scripts/evidence_root.py')}
    ms_paths.update(ms / 'evidence/data' / p for p in CRESST_ARCHIVE_FILES)
    files = {kind: {str(p.relative_to(root)): digest(p) for p in sorted(paths)}
             for kind,root,paths in [('experiment',EXP,exp_paths),('manuscript',ms,ms_paths)]}
    obj = {'schema': 'mlst-repository-handoff/1', 'files': files, 'matrices': matrix_records(),
           'source_commits_at_freeze': {'experiment': commit(EXP),'manuscript': commit(ms)},
           'note': 'Commit IDs are historical; content hashes are authoritative. No production evidence implied.'}
    write(manifest,obj)
    print(f'Frozen {sum(map(len,files.values()))} input files into {manifest}')


def verify(ms, manifest):
    obj = json.loads(Path(manifest).read_text())
    if obj.get('schema') != 'mlst-repository-handoff/1':
        raise ValueError('Unrecognized handoff schema')
    for kind,root in [('experiment',EXP),('manuscript',ms)]:
        for rel,want in obj['files'][kind].items():
            path = (root / rel).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != want:
                raise ValueError(f'Input missing or changed: {kind}/{rel}')
    if matrix_records() != obj['matrices']:
        raise ValueError('Production matrix differs from frozen handoff')
    return obj


def environment():
    import numpy, scipy, pytest, h5py
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'hostname': socket.gethostname(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
            'pytest': pytest.__version__, 'h5py': h5py.__version__, 'threads': {k:os.environ[k] for k in THREAD_KEYS}}


def command(argv, logfile, env=None):
    print('Running: ' + ' '.join(map(str,argv)), flush=True)
    with Path(logfile).open('w') as log:
        result = subprocess.run(list(map(str,argv)), cwd=EXP, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command exited {result.returncode}; see {logfile}')


def validate_test_report(path):
    tree = ET.parse(path)
    suites = [node for node in tree.getroot().iter('testsuite')]
    totals = {k:sum(int(n.get(k,0)) for n in suites) for k in ('tests','failures','errors','skipped')}
    if totals['tests'] == 0 or any(totals[k] for k in ('failures','errors','skipped')):
        raise ValueError(f'Test acceptance failed: {totals}')
    return totals


def require_external_out(ms, out):
    out = out.resolve()
    for root in (EXP.resolve(),ms.resolve()):
        if out.is_relative_to(root):
            raise ValueError('Run/preflight outputs must be outside both source repositories')
    return out


@contextmanager
def single_launcher(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root / '.launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another launcher owns this run root') from exc
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def preflight(ms, manifest, out):
    verify(ms,manifest)
    out.mkdir(parents=True,exist_ok=True)
    receipt = out / 'PREFLIGHT.json'
    receipt.unlink(missing_ok=True)  # a failed rerun must not retain an old green receipt
    env = dict(os.environ, CRESST_S_ARCHIVE_ROOT=str(ms / 'evidence/data'))
    command([sys.executable,ms/'scripts/verify_evidence.py','--require-complete'],out/'evidence.log',env)
    test_started = time.monotonic()
    command([sys.executable,'-m','pytest',*TESTS,'-q','--durations=10',f'--junitxml={out / "tests.xml"}'],out/'tests.log',env)
    test_wall_s = time.monotonic() - test_started
    totals = validate_test_report(out/'tests.xml')
    for pkg,proto in PROTOCOLS.items():
        command([sys.executable,'-m',f'experiments.mlst2026.successor_20260913.{pkg}',
                 'enumerate','--protocol',ROUND/pkg/proto,'--out',out/'matrices'/pkg],out/f'{pkg}-enumerate.log',env)
    verify(ms,manifest)
    write(receipt,{'schema':'mlst-preflight/1','manifest_sha256':digest(manifest),
                   'environment':environment(),'tests':totals,'development_test_wall_s':test_wall_s,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                   'mode':'single host, one launcher per run root, one BLAS thread',
                   'limitations':['No cross-node filesystem locking certification','No production execution',
                                  'No raw CRESST pulse adapter validation','No real-data T22 matched stage']})
    print(f'Preflight passed: {totals}; receipt {receipt}')


def phase_plan(selection):
    if selection == 't19s':
        return 't19s', [('run',[]),('reduce',[])]
    if selection == 't20s':
        return 't20s', [('run',['--families','algebraic']),('run',['--families','pilot']),
                       ('reduce',['--allow-partial']),('run',['--families','confirmation']),('reduce',[])]
    if selection == 't21s':
        return 't21s', [('run',['--roles','pilot','algebraic']),('reduce',['--allow-partial']),
                       ('run',['--roles','confirmation']),('reduce',[])]
    if selection == 't22s':
        return 't22s', [('run',['--families','sim_pilot','control']),('reduce',['--allow-partial']),
                       ('run',['--families','sim_confirm']),('reduce',[])]
    if selection == 'cresst-archive':
        return 'cresst_s', [('run',['--roles','control','algebraic']),('reduce',['--allow-partial'])]
    if selection == 'cresst-calibration':
        return 'cresst_s', [('run',['--families','calib_pilot']),('reduce',['--allow-partial']),
                           ('run',['--families','calib_confirm','calib_control']),('reduce',['--allow-partial'])]
    raise ValueError('Unsupported selection: full CRESST robustness and real T22 require separate data preparation')


def check_retry_limit(out, pkg):
    """At most two retries per cell; never lift the frozen fit cap."""
    from collections import Counter
    from experiments.mlst2026.successor_20260913.common.ledger import ExecutionLedger
    from experiments.mlst2026.successor_20260913.common.protocol import load_protocol
    protocol = load_protocol(ROUND / pkg / PROTOCOLS[pkg])
    ledger = ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=protocol["budget"]["estimator_fits_cap"],
                             package=pkg, protocol_sha256=protocol["_sha256"])
    attempts = list(ledger.attempts().values())
    complete = {a["cell_id"] for a in attempts if a["state"] == "completed"}
    counts = Counter(a["cell_id"] for a in attempts)
    exhausted = [cell for cell,n in counts.items() if n >= 3 and cell not in complete]
    if exhausted:
        raise ValueError(f"Two retries exhausted for {exhausted}; investigate without increasing the cap")


def execute(ms, manifest, receipt, out, selection):
    verify(ms,manifest)
    checked = json.loads(receipt.read_text())
    if checked.get('manifest_sha256') != digest(manifest) or checked.get('environment') != environment():
        raise ValueError('Preflight is missing/stale or belongs to another environment; run it on this server')
    if checked.get('schema') != 'mlst-preflight/1' or checked.get('tests',{}).get('tests',0) == 0:
        raise ValueError('Invalid preflight receipt')
    pkg,phases = phase_plan(selection)
    with single_launcher(out):
        identity = {'manifest_sha256':digest(manifest),'selection':selection}
        state = out/'LAUNCH.json'
        if state.exists() and json.loads(state.read_text()) != identity:
            raise ValueError('This run root belongs to a different selection/snapshot')
        write(state,identity)
        stamp = time.strftime('%Y%m%dT%H%M%S',time.gmtime())
        logs = out/'logs'/f'{stamp}-{os.getpid()}'
        logs.mkdir(parents=True)
        write(logs/'environment.json',environment())
        for i,(action,extra) in enumerate(phases):
            if action == "run":
                check_retry_limit(out,pkg)
            # Existing reducers enforce frozen pilot provenance on resume.
            argv=[sys.executable,'-m',f'experiments.mlst2026.successor_20260913.{pkg}',action,
                  '--protocol',ROUND/pkg/PROTOCOLS[pkg],'--out',out,*extra]
            if pkg == 'cresst_s': argv += ['--archive-root',ms/'evidence/data']
            command(argv,logs/f'{i:02d}-{action}.log')
            if action == 'reduce':
                import shutil
                shutil.copy2(out/'REDUCED.json',logs/f'{i:02d}-REDUCED.json')
        print(f'Execution/reduction finished: {out}. Results still require scientific review.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['freeze','verify','preflight','execute'])
    ap.add_argument('--ms',required=True,type=Path,help='Full manuscript repository on this machine')
    ap.add_argument('--manifest',type=Path,default=ROUND/'remote/MANIFEST.json')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--receipt',type=Path)
    ap.add_argument('--selection',choices=SELECTIONS)
    a=ap.parse_args();a.ms=a.ms.resolve()
    try:
        if a.action=='freeze':freeze(a.ms,a.manifest)
        elif a.action=='verify': verify(a.ms,a.manifest); print('All frozen files and matrices verified')
        else:
            if a.out is None:raise ValueError('--out is required')
            out=require_external_out(a.ms,a.out)
            if a.action=='preflight':preflight(a.ms,a.manifest,out)
            else:
                if a.receipt is None or a.selection is None:raise ValueError('--receipt and --selection are required')
                execute(a.ms,a.manifest,a.receipt,out,a.selection)
    except (ValueError,RuntimeError,OSError) as exc:
        print(f'REFUSED: {exc}',file=sys.stderr);return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
