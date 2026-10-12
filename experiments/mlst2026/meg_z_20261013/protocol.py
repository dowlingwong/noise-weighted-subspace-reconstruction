"""Protocol loading.  ``protocol_dev.json`` holds the development defaults (NOT FROZEN).  At the D4 freeze a
separate ``protocol.json`` with "frozen": true is written into this directory (new kit hashes, freeze commit);
only that file can authorise a confirmation run (runner preflight)."""
from __future__ import annotations

import json
from pathlib import Path

from .common import PACKAGE

PROTOCOL_DEV = PACKAGE / 'protocol_dev.json'
PROTOCOL_FROZEN = PACKAGE / 'protocol.json'


def load_protocol(path=None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    if PROTOCOL_FROZEN.is_file():
        return json.loads(PROTOCOL_FROZEN.read_text())
    return json.loads(PROTOCOL_DEV.read_text())


def is_frozen(proto: dict) -> bool:
    return bool(proto.get('frozen') is True and str(proto.get('status', '')).startswith('FROZEN'))


def arms(proto: dict) -> list[dict]:
    return list(proto['arms'])
