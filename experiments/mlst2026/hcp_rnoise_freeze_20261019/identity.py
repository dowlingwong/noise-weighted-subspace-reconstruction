"""Verify this additive candidate and every inherited frozen identity."""
import json
from pathlib import Path
from experiments.mlst2026.meg_z_20261013.runner import kit_hashes
from experiments.mlst2026.hcp_rnoise_dev_20261019.access import verify as dev_verify

PACKAGE=Path(__file__).resolve().parent


def verify(expected=None):
    actual=kit_hashes(PACKAGE)
    if actual!=json.loads((PACKAGE/'KIT_SHA256.json').read_text()):
        raise PermissionError('STOP additive implementation identity mismatch')
    if expected is not None and actual!=expected:
        raise PermissionError('STOP implementation differs from registered freeze kit')
    dev_verify()
    return actual
