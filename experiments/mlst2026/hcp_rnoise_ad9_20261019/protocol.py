"""Exact registered AD9 configuration and complete qualification ledger checks."""
import json
from pathlib import Path
from experiments.mlst2026.meg_z_20261013.runner import kit_hashes
from experiments.mlst2026.hcp_rnoise_freeze_20261019.identity import verify as inherited_verify

PACKAGE = Path(__file__).resolve().parent


def load():
    return json.loads((PACKAGE / 'protocol_frozen.json').read_bytes())


def verify(expected=None):
    actual = kit_hashes(PACKAGE)
    if actual != json.loads((PACKAGE / 'KIT_SHA256.json').read_bytes()) or (expected is not None and actual != expected):
        raise PermissionError('STOP AD9 kit differs from registered implementation')
    if inherited_verify() != json.loads((PACKAGE / 'INHERITED_KIT_SHA256.json').read_bytes()):
        raise PermissionError('STOP inherited F0-F3 implementation changed')
    return actual


def cohort(qualification, config=None):
    """Report all 60 outcomes before any confirmatory inference; never replace."""
    config = load() if config is None else config
    dates = config['access']['confirmation_analysis_dates']
    if len(dates) != 60 or len(set(dates)) != 60 or set(qualification) != set(dates):
        raise ValueError('STOP complete fixed 60-date qualification ledger required')
    criteria = ('G1', 'G2', 'artifact', 'budget')
    included, excluded = [], {}
    for d in dates:
        row = qualification[d]
        if any(type(row.get(k)) is not bool for k in criteria):
            raise ValueError('STOP explicit gate booleans required')
        failures = [k for k in criteria if not row[k]]
        if failures:
            excluded[d] = failures
        else:
            included.append(d)
    passed = len(included) >= config['n_min']
    return {'status': 'PASS' if passed else 'STOP', 'confirmatory_claim_allowed': passed,
            'included': included, 'excluded': excluded, 'n_included': len(included),
            'n_min': config['n_min'], 'replacement': False,
            'reason': None if passed else 'fewer than n_min passing dates; no confirmatory claim'}
