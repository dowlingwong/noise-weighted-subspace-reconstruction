"""Fresh seed namespace: derived by SHA-256 from 'y19-confirm-20261012', disjoint from earlier studies."""
import hashlib
import json
from pathlib import Path

from experiments.mlst2026.y19_20261012 import predictions as P, runner, benchmark

EXP = Path(__file__).resolve().parents[4]
OWN = Path(__file__).resolve().parents[1]
M = EXP / 'experiments' / 'mlst2026'


def test_seed_derivation():
    want = [int.from_bytes(hashlib.sha256(f'y19-confirm-20261012/seed/{i}'.encode()).digest()[:8], 'little') % 10 ** 12
            for i in range(24)]
    assert P.seeds() == want
    assert len(set(want)) == 24
    proto = json.loads((OWN / 'protocol.json').read_text())
    assert proto['seeds'] == want and proto['namespace'] == 'y19-confirm-20261012'


def _prior_seeds():
    s = set(range(900000, 901000)) | set(range(3000, 3200)) | set(range(4000, 4400))          # T20-S namespaces
    s |= set(range(202609290100, 202609290124)) | {900001, 900002}                           # T25
    for name in ('confirmation.json', 'fixture.json', 'benchmark.json'):                      # TR09
        p = M / 'tr09_20261007' / name
        if p.is_file():
            s |= set(json.loads(p.read_text()).get('seeds', []))
    for name in ('confirmation.json', 'fixture.json', 'resource.json'):                       # E-NL
        s |= set(json.loads((M / 'enl_20261008' / name).read_text()).get('seeds', []))
    s |= set(json.loads((M / 't25_production_20260930' / 'production.json').read_text())['seeds'])
    return s


def test_disjoint_from_prior_studies():
    prior = _prior_seeds()
    assert len(prior) > 1000
    assert not set(P.seeds()) & prior
    assert not set(runner.smoke_protocol()['seeds']) & (prior | set(P.seeds()))


def test_namespace_unused_elsewhere():
    hits = []
    for p in (EXP / 'experiments').rglob('*'):
        if (p.suffix not in ('.py', '.json', '.md', '.sh') or OWN in p.parents or '__pycache__' in p.parts
                or any(part.startswith('y19_20261012_return') for part in p.parts)):
            continue
        try:
            if 'y19-confirm-20261012' in p.read_text(errors='ignore'):
                hits.append(str(p))
        except OSError:
            pass
    assert hits == []
    others = {'y19-smoke-fixture', benchmark.NAMESPACE, 'paper1-t25-v1/confirm', 'paper1-tr09-v1/confirm',
              'paper1-enl-v1/confirm', 'paper1-tr09-dev'}
    assert P.NAMESPACE not in others
