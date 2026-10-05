"""Fail-closed check that the T25 production code X2 reuses is the accepted code.

The accepted T25 code identity ``01f2f82f...`` is defined by the T25 addendum and
``t25_production_20260930/workflow.py::code_identity`` as the SHA-256 of the JSON
of a *source closure*: every EXP Python file, the T25 package JSON configs,
``pyproject.toml``, and four MS files (03_theory.tex, app_main_proofs.tex and the
T25 protocol and addendum).  Recomputing it over today's trees would NOT give
``01f2f82f``: EXP has gained the t24_descriptive package and MS rewrote
03_theory.tex after the freeze, and every later X2/X3 file would enter the glob.
So X2 checks two things instead, before it imports any T25 module:

1. the shipped copy of the accepted closure (``t25_accepted_sources.json``,
   copied verbatim from the returned T25 ``freeze.json``) hashes to the accepted
   identity; and
2. every in-repository module X2 actually loads -- the T25 package and its
   transitive in-repo imports -- is byte-identical to that closure.

Check 2 is repeated after import (``assert_loaded_modules``) against
``sys.modules``, so an import added later cannot slip past.  Any mismatch raises
``AcceptedCodeMismatch`` and nothing runs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
EXP_ROOT = PACKAGE.parents[2]
ACCEPTED_T25_IDENTITY = '01f2f82f9144dd1833e577bf1830707417197dd61c1c1580eba984c712304762'
CLOSURE_FILE = PACKAGE / 't25_accepted_sources.json'
#: T25 files X2 imports directly; their transitive in-repo imports are caught by
#: assert_loaded_modules().
T25_DIRECT = [
    'experiments/mlst2026/t25_production_20260930/__init__.py',
    'experiments/mlst2026/t25_production_20260930/protocol.py',
    'experiments/mlst2026/t25_production_20260930/reconstruction.py',
    'experiments/mlst2026/t25_production_20260930/science.py',
    'experiments/mlst2026/t25_production_20260930/workflow.py',
    'experiments/mlst2026/successor_20260913/t20s/generator.py',
    'experiments/mlst2026/upgrade_20260922/c1/__init__.py',
]


class AcceptedCodeMismatch(RuntimeError):
    pass


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def closure_identity(closure: dict) -> str:
    body = {k: closure[k] for k in ('EXP_files', 'MS_files')}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_closure() -> dict:
    closure = json.loads(CLOSURE_FILE.read_text())
    if closure.get('accepted_identity') != ACCEPTED_T25_IDENTITY:
        raise AcceptedCodeMismatch('closure file names a different accepted identity')
    got = closure_identity(closure)
    if got != ACCEPTED_T25_IDENTITY:
        raise AcceptedCodeMismatch(f'shipped T25 closure hashes to {got}, not {ACCEPTED_T25_IDENTITY}')
    return closure


def assert_files(relatives, closure=None) -> dict:
    closure = closure or load_closure()
    out = {}
    for rel in relatives:
        want = closure['EXP_files'].get(rel)
        if want is None:
            raise AcceptedCodeMismatch(f'{rel} is not part of the accepted T25 closure')
        path = EXP_ROOT / rel
        if not path.is_file():
            raise AcceptedCodeMismatch(f'{rel} is missing')
        got = sha(path)
        if got != want:
            raise AcceptedCodeMismatch(f'{rel} differs from the accepted T25 bytes ({got[:12]} != {want[:12]})')
        out[rel] = got
    return out


def assert_direct() -> dict:
    """Run before importing any T25 module."""
    return assert_files(T25_DIRECT)


#: The complete set of accepted T25-closure files a run loads (measured with a
#: fresh import of science.py and workflow.py).  It enters X2's own code identity;
#: assert_loaded_modules() additionally checks whatever is actually loaded.
T25_REUSED = [
    'experiments/__init__.py',
    'experiments/mlst2026/__init__.py',
    'experiments/mlst2026/successor_20260913/__init__.py',
    'experiments/mlst2026/successor_20260913/t20s/__init__.py',
    'experiments/mlst2026/successor_20260913/t20s/generator.py',
    'experiments/mlst2026/t25_production_20260930/__init__.py',
    'experiments/mlst2026/t25_production_20260930/protocol.py',
    'experiments/mlst2026/t25_production_20260930/reconstruction.py',
    'experiments/mlst2026/t25_production_20260930/science.py',
    'experiments/mlst2026/t25_production_20260930/workflow.py',
    'experiments/mlst2026/upgrade_20260922/__init__.py',
    'experiments/mlst2026/upgrade_20260922/c1/__init__.py',
    'experiments/paper1/__init__.py',
    'experiments/paper1/t20_population_accounting.py',
    'experiments/synthetic/__init__.py',
    'experiments/synthetic/nfpa_confirmatory/__init__.py',
    'experiments/synthetic/nfpa_confirmatory/config.py',
    'experiments/synthetic/nfpa_confirmatory/covariance.py',
    'experiments/synthetic/nfpa_confirmatory/generator.py',
    'experiments/synthetic/nfpa_confirmatory/methods.py',
    'src/noise_geometry/__init__.py',
    'src/noise_geometry/filters/__init__.py',
    'src/noise_geometry/filters/optimal.py',
    'src/noise_geometry/metrics/__init__.py',
    'src/noise_geometry/metrics/reconstruction.py',
    'src/noise_geometry/noise/__init__.py',
    'src/noise_geometry/noise/covariance.py',
    'src/noise_geometry/noise/psd.py',
    'src/noise_geometry/noise/synthetic.py',
    'src/noise_geometry/noise/whitening.py',
    'src/noise_geometry/subspace/__init__.py',
    'src/noise_geometry/subspace/angles.py',
    'src/noise_geometry/subspace/pca.py',
]
#: Top-level packages that must be imported from this repository, never from an
#: installed copy elsewhere (which the byte check could not see).
REPO_PACKAGES = ('experiments', 'noise_geometry')


def assert_repo_origin() -> None:
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in REPO_PACKAGES:
            continue
        f = getattr(module, '__file__', None)
        if f and not Path(f).resolve().is_relative_to(EXP_ROOT):
            raise AcceptedCodeMismatch(f'{name} is imported from {f}, outside {EXP_ROOT}')


def loaded_repo_modules() -> list[str]:
    """Repository-relative paths of every loaded module file under EXP_ROOT."""
    rels = set()
    for module in list(sys.modules.values()):
        f = getattr(module, '__file__', None)
        if not f or not Path(f).is_absolute() or not Path(f).is_file():
            continue                     # e.g. torch._classes reports a bare relative name
        p = Path(f).resolve()
        if p.suffix != '.py' or not p.is_relative_to(EXP_ROOT):
            continue
        rels.add(str(p.relative_to(EXP_ROOT)))
    return sorted(rels)


def assert_loaded_modules(own_package: Path = PACKAGE) -> dict:
    """Every loaded in-repo module is either this package's own file or an
    accepted T25-closure file with identical bytes."""
    closure = load_closure()
    assert_repo_origin()
    own = str(own_package.relative_to(EXP_ROOT))
    reused = [r for r in loaded_repo_modules() if not r.startswith(own + '/')]
    # tests and pytest helpers are not part of a run
    reused = [r for r in reused if '/tests/' not in r and not r.startswith('tests/')
              and Path(r).name != 'conftest.py']
    return assert_files(reused, closure)


def report() -> dict:
    closure = load_closure()
    return {'accepted_t25_identity': ACCEPTED_T25_IDENTITY, 'closure_EXP_head': closure['EXP_head'],
            'closure_MS_head': closure['MS_head'], 'direct_files': assert_files(T25_DIRECT, closure)}
