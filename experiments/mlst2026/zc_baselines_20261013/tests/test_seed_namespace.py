"""Fresh seed namespace 'zc-confirm-20261013': SHA-256 derivation, disjoint from every earlier study."""
import hashlib
import json
import re
from pathlib import Path

from experiments.mlst2026.zc_baselines_20261013 import design, runner

EXP = Path(__file__).resolve().parents[4]
OWN = Path(__file__).resolve().parents[1]
M = EXP / 'experiments' / 'mlst2026'
EARLIER_NAMESPACES = {'paper1-t25-v1/confirm', 'paper1-t25-v1/fixture', 'paper1-tr09-v1/confirm', 'paper1-enl-v1/confirm',
                      'paper1-tr09-dev', 'y19-confirm-20261012', 'y19-smoke-fixture', 'y19-benchmark-nonG2'}


def test_seed_derivation():
    want = [int.from_bytes(hashlib.sha256(f'zc-confirm-20261013/seed/{i}'.encode()).digest()[:8], 'little') % 10 ** 12
            for i in range(24)]
    assert design.seeds() == want and len(set(want)) == 24
    proto = json.loads((OWN / 'protocol.json').read_text())
    assert proto['seeds'] == want and proto['namespace'] == 'zc-confirm-20261013'


def _prior_seeds():
    s = set(range(900000, 901000)) | set(range(3000, 3200)) | set(range(4000, 4400))          # T20-S namespaces
    s |= set(range(202609290100, 202609290124)) | {900001, 900002}                           # T25
    for name in ('confirmation.json', 'fixture.json', 'benchmark.json'):                      # TR09
        p = M / 'tr09_20261007' / name
        if p.is_file():
            s |= set(json.loads(p.read_text()).get('seeds', []))
    for name in ('confirmation.json', 'fixture.json', 'resource.json'):                       # E-NL
        p = M / 'enl_20261008' / name
        if p.is_file():
            s |= set(json.loads(p.read_text()).get('seeds', []))
    s |= set(json.loads((M / 't25_production_20260930' / 'production.json').read_text())['seeds'])
    y19 = json.loads((M / 'y19_20261012' / 'protocol.json').read_text())                     # Y19
    s |= set(y19['seeds'])
    s |= {int.from_bytes(hashlib.sha256(f'{ns}/seed/{i}'.encode()).digest()[:8], 'little') % 10 ** 12
          for ns in ('y19-smoke-fixture', 'y19-benchmark-nonG2') for i in range(24)}
    return s


def test_disjoint_from_prior_studies():
    prior = _prior_seeds()
    assert len(prior) > 1000
    conf = set(design.seeds())
    assert not conf & prior
    dev = set(design.seeds(design.DEV_NAMESPACE, 24))
    smoke = set(design.smoke_protocol()['seeds'])
    assert not dev & (prior | conf) and not smoke & (prior | conf | dev)


def test_namespaces_distinct():
    ours = {design.NAMESPACE, design.DEV_NAMESPACE, design.SMOKE_NAMESPACE}
    assert len(ours) == 3 and not ours & EARLIER_NAMESPACES
    assert runner.load_protocol()['dev_namespace'] == design.DEV_NAMESPACE


ARTEFACT_SUFFIXES = {'.json', '.jsonl', '.npz', '.h5', '.hdf5', '.txt', '.log', '.csv'}
METRIC_DIRS = {'oracle', 'kron256', 'kron1024', 'full256', 'full1024'}


def _allowed(p: Path, root: Path) -> bool:
    """The kit itself and approved return paths may carry the confirmation namespace."""
    rel = p.relative_to(root)
    return OWN in p.parents or p == OWN or any(part.startswith('zc_baselines_20261013_return') for part in rel.parts)


def namespace_artefacts(root: Path, namespace: str = design.NAMESPACE, seeds=None) -> list:
    """Artefacts (not mentions) of the confirmation namespace under `root`:
    (i) data/log files (JSON, NPZ, H5, TXT, LOG, ...) carrying a "namespace": "<ns>" field, the namespace bytes
        (binary containers) or one of the confirmation seed integers;
    (ii) any records/ tree with the ZC record layout records/theta<th>/N<N>/<metric>/seed<ii>.json;
    (iii) any other .py file that ASSIGNS the namespace (x = '<ns>' or 'namespace': '<ns>'); comparisons such as
        `!=` or `not in` (disjointness assertions in other kits) are not artefacts.
    The kit directory and *_return* paths are exempt."""
    seeds = design.seeds() if seeds is None else seeds
    ns = re.escape(namespace)
    field = re.compile(r'"namespace"\s*:\s*"' + ns + '"')
    seed_re = re.compile(r'(?<!\d)(' + '|'.join(str(x) for x in seeds) + r')(?!\d)')
    assign = re.compile(r"(?<![=!<>])=\s*['\"]" + ns + r"['\"]|['\"]namespace['\"]\s*:\s*['\"]" + ns + r"['\"]")
    layout = re.compile(r'records/theta\d+\.\d+/N\d+/([^/]+)/seed\d+\.json$')
    hits = []
    for p in root.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or _allowed(p, root):
            continue
        rel = p.relative_to(root).as_posix()
        m = layout.search(rel)
        if m and m.group(1) in METRIC_DIRS:
            hits.append(('zc-record-layout', rel))
            continue
        try:
            if p.suffix in ARTEFACT_SUFFIXES:
                data = p.read_bytes()
                if p.suffix in ('.npz', '.h5', '.hdf5'):
                    if namespace.encode() in data:
                        hits.append(('binary-namespace', rel))
                    continue
                text = data.decode('utf-8', errors='ignore')
                if field.search(text):
                    hits.append(('namespace-field', rel))
                elif seed_re.search(text):
                    hits.append(('seed-integer', rel))
            elif p.suffix == '.py' and namespace in (text := p.read_text(errors='ignore')) and assign.search(text):
                hits.append(('py-assignment', rel))
        except OSError:
            pass
    return hits


def test_no_confirmation_namespace_artefacts():
    assert namespace_artefacts(EXP / 'experiments') == []


def test_artefact_detector_is_strict(tmp_path):
    ns, seeds = design.NAMESPACE, design.seeds()
    root = tmp_path / 'experiments'
    (root / 'other').mkdir(parents=True)
    (root / 'other' / 'test_x.py').write_text(f"assert proto['namespace'] != '{ns}'\nfor p in ('a', '{ns}'): pass\n")
    (root / 'other' / 'notes.json').write_text(json.dumps({'prior_namespaces': [ns]}))
    assert namespace_artefacts(root) == []                                   # mentions are not artefacts
    (root / 'a').mkdir()
    (root / 'a' / 'START.json').write_text(json.dumps({'namespace': ns}, indent=1))
    (root / 'b').mkdir()
    (root / 'b' / 'RUN_LOG.txt').write_text(f'seed {seeds[5]} done\n')
    (root / 'c' / 'records' / 'theta0.25' / 'N400' / 'oracle').mkdir(parents=True)
    (root / 'c' / 'records' / 'theta0.25' / 'N400' / 'oracle' / 'seed00.json').write_text('{}')
    (root / 'd').mkdir()
    (root / 'd' / 'arrays.npz').write_bytes(b'PK..' + ns.encode() + b'..')
    (root / 'e').mkdir()
    (root / 'e' / 'cfg.py').write_text(f"NAMESPACE = '{ns}'\n")
    (root / 'f').mkdir()
    (root / 'f' / 'proto.py').write_text(f"P = {{'namespace': '{ns}'}}\n")
    kinds = sorted(k for k, _ in namespace_artefacts(root))
    assert kinds == ['binary-namespace', 'namespace-field', 'py-assignment', 'py-assignment', 'seed-integer',
                     'zc-record-layout']
    ret = root / 'zc_baselines_20261013_return' / 'records' / 'theta0.25' / 'N400' / 'oracle'
    ret.mkdir(parents=True)
    (ret / 'seed00.json').write_text(json.dumps({'namespace': ns}))
    assert len(namespace_artefacts(root)) == 6                               # approved return path is exempt


def test_benchmark_and_smoke_never_use_confirmation_namespace():
    from experiments.mlst2026.zc_baselines_20261013 import benchmark
    assert benchmark.dev_protocol()['namespace'] == 'zc-dev-20261013'
    assert design.smoke_protocol()['namespace'] == 'zc-smoke-fixture'
