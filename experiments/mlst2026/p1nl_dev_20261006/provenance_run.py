"""Run a development module with a bound executed-source record (review finding R5).

    python -m experiments.mlst2026.p1nl_dev_20261006.provenance_run --out DIR -- <module> [args...]

Before the run: copies every file of this development package and every accepted T25 file it imports
into DIR/source_snapshot/, records SHA-256 of each, argv, cwd, host, platform, Python and the installed
distributions.  Imports ``<module>`` and, still before running it, hashes and snapshots every
in-repository module that the import loaded (including those outside this package).  Runs
``<module>.main(args)`` in-process.  After the run: re-hashes all of those, records every in-repository
module actually loaded, and checks that no file hashed before the run changed during it.  Scope: a
module first imported lazily *during* the run can only be hashed after it; such modules are listed
separately (``loaded_during_run_only``).
Writes DIR/PROVENANCE.json.  The module's own outputs go wherever its arguments say (normally DIR/run).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def package_files():
    return sorted(p for p in PKG.rglob('*') if p.is_file() and p.suffix in ('.py', '.json')
                  and '__pycache__' not in p.parts)


def loaded_repo_files():
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if f and Path(f).is_absolute() and Path(f).is_file():
            p = Path(f).resolve()
            if p.is_relative_to(ROOT.resolve()) and p.suffix == '.py':
                out[str(p.relative_to(ROOT.resolve()))] = sha(p)
    return dict(sorted(out.items()))


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit('usage: provenance_run --out DIR -- <module> [args...]')
    i = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv[:i])
    module, margs = argv[i + 1], argv[i + 2:]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    snap = out / 'source_snapshot'
    before = {}
    for p in package_files():
        rel = p.relative_to(ROOT)
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, snap / rel)
        before[str(rel)] = sha(p)
    rec = {'module': module, 'module_args': margs, 'argv': sys.argv, 'cwd': os.getcwd(),
           'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'host': socket.gethostname(),
           'platform': platform.platform(), 'python': sys.version,
           'env': {k: os.environ.get(k) for k in ('PYTHONPATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                  'MKL_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')},
           'distributions': sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()),
           'package_files_before': before}
    mod = importlib.import_module(module)
    pre = loaded_repo_files()
    for rel in pre:
        if rel not in before:
            dest = snap / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
    rec['loaded_repo_modules_before_run'] = pre
    t0 = time.time()
    mod.main(margs)
    rec['seconds'] = time.time() - t0
    rec['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    loaded = loaded_repo_files()
    rec['loaded_repo_modules'] = loaded
    for rel, h in loaded.items():
        if rel not in before and rel not in pre:      # lazily imported during the run: post-run bytes
            dest = snap / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
    rec['loaded_during_run_only'] = sorted(set(loaded) - set(pre))
    rec['package_files_after'] = {str(p.relative_to(ROOT)): sha(p) for p in package_files()}
    pre_after = {rel: sha(ROOT / rel) for rel in pre}
    rec['unchanged_during_run'] = rec['package_files_after'] == before and pre_after == pre
    rec['snapshot_files'] = {str(p.relative_to(snap)): sha(p) for p in sorted(snap.rglob('*')) if p.is_file()}
    (out / 'PROVENANCE.json').write_text(json.dumps(rec, indent=1))
    if not rec['unchanged_during_run']:
        raise SystemExit('package files changed during the run')


if __name__ == '__main__':
    main()
