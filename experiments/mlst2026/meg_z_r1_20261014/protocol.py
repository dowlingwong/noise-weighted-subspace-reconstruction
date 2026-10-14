"""Exact DESIGN rev 2.1 CONFIG contract and frozen-protocol confirmation guard."""
import json
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
CONFIG_SHA='da2e4de3777589ef5c204355461dce2cb1bb613823d5ce08c6035efa3c640c1b'
PARTITION_SHA='809672165ee76d009f478eef0180478fe281834273aa01ef56f265d8e80fb8d7'
DEV=tuple(f'sub-{i:02}' for i in range(1,5))
def load(): return json.loads((PACKAGE/'protocol_r1.json').read_text())
def guard(subjects,protocol_path=None):
    if all(s in DEV for s in subjects): return
    if not all(s in tuple(f'sub-{i:02}' for i in range(1,17)) for s in subjects): raise SystemExit('unknown subject')
    frozen=PACKAGE/'protocol.json'
    if protocol_path is not None or not frozen.is_file(): raise SystemExit('confirmation requires the frozen kit protocol.json')
    p=json.loads(frozen.read_text())
    if p.get('frozen') is not True or not str(p.get('status','')).startswith('FROZEN'): raise SystemExit('protocol is not frozen')
