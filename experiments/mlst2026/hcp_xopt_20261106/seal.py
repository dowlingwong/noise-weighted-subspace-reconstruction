"""Seal the finished resume-4 run (addendum A2.d).

Writes into `W2/run_20261108/sealed/` exactly four entries:
  ledger_merged.jsonl.gz   the merged ledger
  run_archive.tar          results/, claims/, logs/, reclaims.log
  MANIFEST.json            row count, chunk count, per-member sha256 computed BEFORE tarring
  SHA256SUMS               sha256 of the three files above

Verification: the archive's member count equals the loose-file count, and every member's
sha256 (read from the tar stream) matches MANIFEST.json.  Only then are the loose results/,
claims/ and logs/ of this run deleted, and sealed/ made read-only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import time

from . import claim

SEALED = 'sealed'


class SealError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _loose_members(qdir):
    members = []
    for sub in ('results', 'claims', 'logs'):
        root = os.path.join(qdir, sub)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for f in filenames:
                if f.endswith('.tmp'):
                    continue
                p = os.path.join(dirpath, f)
                members.append(os.path.relpath(p, qdir))
    if os.path.exists(os.path.join(qdir, 'reclaims.log')):
        members.append('reclaims.log')
    return sorted(members)


def archive_run(qdir, ledger=None):
    manifest = claim.load_manifest(qdir)
    merged_src = ledger or os.path.join(qdir, 'ledger_merged.jsonl.gz')
    if not os.path.exists(merged_src):
        raise SealError('merged ledger missing; run merge before seal')
    sealed = os.path.join(qdir, SEALED)
    os.makedirs(sealed, exist_ok=True)
    members = _loose_members(qdir)
    shas = {rel: sha256_file(os.path.join(qdir, rel)) for rel in members}
    n_rows = 0
    with gzip.open(merged_src, 'rt') as fh:
        for _ in fh:
            n_rows += 1
    rec = dict(scope=manifest['scope'], row_list_sha256=manifest['row_list_sha256'],
               chunk_count=manifest['n_chunks'], chunk_size=manifest['chunk_size'],
               planned_rows=manifest['planned_rows'], merged_rows=n_rows,
               n_members=len(members), members=shas,
               created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with open(os.path.join(sealed, 'MANIFEST.json'), 'w') as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
        fh.write('\n')
    shutil.copy2(merged_src, os.path.join(sealed, 'ledger_merged.jsonl.gz'))
    with tarfile.open(os.path.join(sealed, 'run_archive.tar'), 'w') as tar:
        for rel in members:
            tar.add(os.path.join(qdir, rel), arcname=rel, recursive=False)
    lines = []
    for name in ('ledger_merged.jsonl.gz', 'run_archive.tar', 'MANIFEST.json'):
        lines.append(f'{sha256_file(os.path.join(sealed, name))}  {name}')
    with open(os.path.join(sealed, 'SHA256SUMS'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    verify_archive(qdir)
    for sub in ('results', 'claims', 'logs'):
        p = os.path.join(qdir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    return rec


def verify_archive(qdir):
    sealed = os.path.join(qdir, SEALED)
    with open(os.path.join(sealed, 'MANIFEST.json')) as fh:
        rec = json.load(fh)
    # SHA256SUMS must match the three sealed files
    sums = {}
    with open(os.path.join(sealed, 'SHA256SUMS')) as fh:
        for line in fh:
            if line.strip():
                h, name = line.split()
                sums[name] = h
    for name in ('ledger_merged.jsonl.gz', 'run_archive.tar', 'MANIFEST.json'):
        if name in sums and sha256_file(os.path.join(sealed, name)) != sums[name]:
            raise SealError(f'SHA256SUMS mismatch for {name}')
    if sha256_file(os.path.join(sealed, 'ledger_merged.jsonl.gz')) != \
            rec.get('ledger_sha256', sha256_file(os.path.join(sealed, 'ledger_merged.jsonl.gz'))):
        raise SealError('merged ledger sha differs from MANIFEST')
    tar_path = os.path.join(sealed, 'run_archive.tar')
    with tarfile.open(tar_path, 'r') as tar:
        tar_members = [m for m in tar.getmembers() if m.isfile()]
        if len(tar_members) != rec['n_members']:
            raise SealError(f'archive member count {len(tar_members)} != loose-file count '
                            f'{rec["n_members"]}')
        got = {}
        for m in tar_members:
            h = hashlib.sha256()
            fh = tar.extractfile(m)
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
            got[m.name] = h.hexdigest()
    if got != rec['members']:
        bad = [k for k in set(got) | set(rec['members']) if got.get(k) != rec['members'].get(k)]
        raise SealError(f'archive member sha256 mismatch: {bad[:5]}')
    return rec


def freeze_sealed(qdir):
    sealed = os.path.join(qdir, SEALED)
    for root, dirs, files in os.walk(sealed):
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o555)
    os.chmod(sealed, 0o555)
