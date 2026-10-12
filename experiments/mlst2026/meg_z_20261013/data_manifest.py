"""D0: download plan, subject split and DATA_MANIFEST.json for OpenNeuro ds000117 (raw MEG only).

Facts checked on 12 Oct 2026 against the OpenNeuroDatasets/ds000117 git mirror (tags 1.0.0 ... 2.0.0):
* latest snapshot **2.0.0** (2026-01-13; BIDS 1.8.0; DOI 10.18112/openneuro.ds000117.v2.0.0);
  dataset_description.json "License": **"CC0"** (F12: CC0, not CC BY 4.0; cite Wakeman & Henson 2015 and the
  snapshot DOI 10.18112/openneuro.ds000117.v2.0.0);
* participants.tsv lists sub-01 ... sub-16 plus sub-emptyroom; the README section "Relationship of Subject
  Numbering" states that the OpenNeuro version is the "FTP subset of N=16" analysed set (of the 19 subjects
  of Wakeman & Henson 2015, original subject_01 and subject_06 were removed for bad EEG data and subject_19
  for poor EEG and fMRI data), numbered sub-01 ... sub-16.  **These 16 are the analysed subjects.**
* per subject: sub-XX/ses-meg/meg/sub-XX_ses-meg_task-facerecognition_run-0{1..6}_meg.fif (raw, ~860 MB
  each, 1100 Hz, 306 MEG + 70 EEG; not MaxFiltered), _events.tsv (onset s from the first datapoint, value,
  trial_type; stimulus codes 5,6,7,13,14,15,17,18,19; button codes 256 (btnL), 4096 (btnR)), _channels.tsv
  (status good/bad), _meg.json; sub-XX/ses-meg/sub-XX_ses-meg_scans.tsv; sub-XX_ses-meg_coordsystem.json.
  The SSS calibration/crosstalk files are not needed (no MaxFilter) and are not downloaded.
* the selection is 416 subject files (96 raw runs) + 28 sub-emptyroom files; 103 annexed files, 84.7 GB.
* git-annex keys of the large files are MD5E-s<size>--<md5>; ``annex_expectations`` reads them from a
  blob-less clone of the mirror so that every downloaded byte is checked against a second source.

Subject split (fixed here, before any download; runbook §6.1): development = the 4 lowest-numbered analysed
subjects (sub-01 ... sub-04); confirmation = the other 12 (sub-05 ... sub-16).

Commands (EXP root, PYTHONPATH=src):
    python -m experiments.mlst2026.meg_z_20261013.data_manifest split
    python -m experiments.mlst2026.meg_z_20261013.data_manifest plan --target /ceph/dwong/meg_z/raw/ds000117
    python -m experiments.mlst2026.meg_z_20261013.data_manifest annex --mirror DIR --out ANNEX.json
    python -m experiments.mlst2026.meg_z_20261013.data_manifest download --target DIR [--subjects dev|all]
    python -m experiments.mlst2026.meg_z_20261013.data_manifest manifest --root DIR --out DATA_MANIFEST.json [--annex ANNEX.json]
    python -m experiments.mlst2026.meg_z_20261013.data_manifest verify --root DIR --manifest DATA_MANIFEST.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

DATASET = 'ds000117'
SNAPSHOT = '2.0.0'
MIRROR_URL = 'https://github.com/OpenNeuroDatasets/ds000117.git'
ANALYSED_SUBJECTS = [f'sub-{i:02d}' for i in range(1, 17)]
SUBJECT_LIST_SOURCE = ('ds000117 README (snapshot 2.0.0 and 1.1.0), section "Relationship of Subject Numbering relative '
                       'to other versions of Dataset": "4. Current OpenNeuro subset N=16 used for (BIDS)" = the FTP '
                       'subset of N=16 (original subject_01 and subject_06 removed because of bad EEG data, subject_19 '
                       'because of poor EEG and fMRI data); participants.tsv lists exactly sub-01 ... sub-16 (+ '
                       'sub-emptyroom).')
N_DEV = 4
DEV_SUBJECTS = ANALYSED_SUBJECTS[:N_DEV]
CONFIRMATION_SUBJECTS = ANALYSED_SUBJECTS[N_DEV:]
RUNS = [1, 2, 3, 4, 5, 6]
EMPTYROOM = 'sub-emptyroom'
ESSENTIAL = ['dataset_description.json', 'participants.tsv', 'participants.json', 'README', 'CHANGES']


def split() -> dict:
    return {'analysed': ANALYSED_SUBJECTS, 'dev': DEV_SUBJECTS, 'confirmation': CONFIRMATION_SUBJECTS,
            'rule': 'development = the 4 lowest-numbered analysed subjects; confirmation = the other 12; '
                    'fixed in code before any download (runbook §6.1, design E5)',
            'source': SUBJECT_LIST_SOURCE}


def run_stem(sub: str, run: int) -> str:
    return f'{sub}/ses-meg/meg/{sub}_ses-meg_task-facerecognition_run-{run:02d}'


def subject_files(sub: str) -> list[str]:
    files = []
    for r in RUNS:
        stem = run_stem(sub, r)
        files += [stem + '_meg.fif', stem + '_events.tsv', stem + '_channels.tsv', stem + '_meg.json']
    files += [f'{sub}/ses-meg/{sub}_ses-meg_scans.tsv', f'{sub}/ses-meg/meg/{sub}_ses-meg_coordsystem.json']
    return files


def include_patterns(subjects=None, emptyroom: bool = True) -> list[str]:
    subjects = ANALYSED_SUBJECTS if subjects is None else list(subjects)
    unknown = sorted(set(subjects) - set(ANALYSED_SUBJECTS))
    if unknown:
        raise ValueError(f'not an analysed subject: {unknown}')
    pats = [f for s in subjects for f in subject_files(s)]
    if emptyroom:
        pats.append(EMPTYROOM)            # directory prefix: every file under sub-emptyroom/
    return pats


def plan_download(target: str, subjects=None, tag: str = SNAPSHOT, emptyroom: bool = True) -> dict:
    inc = include_patterns(subjects, emptyroom)
    cli = ['openneuro-py', 'download', f'--dataset={DATASET}', f'--tag={tag}', f'--target-dir={target}']
    cli += [f'--include={p}' for p in inc]
    return {'dataset': DATASET, 'tag': tag, 'target_dir': target, 'include': inc, 'n_patterns': len(inc),
            'python_api': f"openneuro.download(dataset='{DATASET}', tag='{tag}', target_dir='{target}', include=<include>, "
                          "verify_hash='auto', verify_size=True)",
            'cli': ' '.join(cli), 'split': split()}


def download(target: str, subjects=None, tag: str = SNAPSHOT, emptyroom: bool = True) -> dict:
    import openneuro
    plan = plan_download(target, subjects, tag, emptyroom)
    openneuro.download(dataset=DATASET, tag=tag, target_dir=target, include=plan['include'],
                       verify_hash='auto', verify_size=True)
    return plan


# ----------------------------------------------------------------------------------------------- annex keys
ANNEX_RE = re.compile(r'(?P<backend>[A-Z0-9]+E?)-s(?P<size>\d+)--(?P<hash>[0-9a-f]+)')


def parse_annex_target(target: str) -> dict | None:
    m = ANNEX_RE.search(Path(target.strip()).name)
    if not m:
        return None
    return {'backend': m['backend'], 'size': int(m['size']), 'hash': m['hash']}


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def annex_expectations(mirror: str, tag: str = SNAPSHOT, subjects=None) -> dict:
    """Expected size + MD5 (annexed files) or git blob SHA-1 (regular files) for every file the download
    plan selects, read from a blob-less clone of the OpenNeuroDatasets mirror:
        git clone --depth 1 --no-checkout --branch <tag> MIRROR_URL <mirror>"""
    import os
    import shutil
    git_exe = shutil.which('git')                    # F15: the host's git, wherever it lives on PATH
    if git_exe is None:
        raise SystemExit('git not found on PATH')
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')

    def git(*a):
        return subprocess.check_output([git_exe, '-C', mirror, *a], env=env)
    rows = git('ls-tree', '-r', '--full-tree', tag).decode().splitlines()
    tree = {}
    for line in rows:
        meta, path = line.split('\t', 1)
        mode, kind, obj = meta.split()
        tree[path] = (mode, obj)
    subjects = ANALYSED_SUBJECTS if subjects is None else subjects
    want = [p for s in subjects for p in subject_files(s)] + sorted(p for p in tree if p.startswith(EMPTYROOM + '/'))
    want += [p for p in ESSENTIAL if p in tree]
    out, missing = {}, []
    for p in want:
        if p not in tree:
            missing.append(p)
            continue
        mode, obj = tree[p]
        if mode == '120000':                                 # symlink -> annex object
            key = parse_annex_target(git('cat-file', '-p', obj).decode())
            out[p] = {'annex': key}
        else:
            out[p] = {'git_blob_sha1': obj}
    return {'mirror': MIRROR_URL, 'tag': tag, 'files': out, 'missing_in_mirror': missing}


# ----------------------------------------------------------------------------------------------- manifest
def file_hashes(path: Path) -> dict:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    size = path.stat().st_size
    blob = hashlib.sha1(b'blob %d\0' % size)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            sha.update(block)
            md5.update(block)
            blob.update(block)
    return {'size': size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest(), 'git_blob_sha1': blob.hexdigest()}


def expected_files(root: Path, subjects=None, emptyroom: bool = True) -> list[str]:
    subjects = ANALYSED_SUBJECTS if subjects is None else subjects
    files = [p for s in subjects for p in subject_files(s)]
    if emptyroom and (root / EMPTYROOM).is_dir():
        files += sorted(str(p.relative_to(root)) for p in (root / EMPTYROOM).rglob('*') if p.is_file())
    files += [p for p in ESSENTIAL if (root / p).is_file()]
    return files


def build_manifest(root: str, subjects=None, annex: dict | None = None, tag: str = SNAPSHOT, emptyroom: bool = True) -> dict:
    root = Path(root)
    files, missing, mismatch = {}, [], []
    for rel in expected_files(root, subjects, emptyroom):
        p = root / rel
        if not p.is_file():
            missing.append(rel)
            continue
        h = file_hashes(p)
        exp = (annex or {}).get('files', {}).get(rel)
        if exp is not None:
            if 'annex' in exp and exp['annex']:
                k = exp['annex']
                ok = h['size'] == k['size'] and (k['backend'] != 'MD5E' or h['md5'] == k['hash'])
                h['annex_check'] = 'ok' if ok else 'MISMATCH'
            elif 'git_blob_sha1' in exp:
                h['annex_check'] = 'ok' if h['git_blob_sha1'] == exp['git_blob_sha1'] else 'MISMATCH'
            if h.get('annex_check') == 'MISMATCH':
                mismatch.append(rel)
        files[rel] = h
    desc = json.loads((root / 'dataset_description.json').read_text()) if (root / 'dataset_description.json').is_file() else {}
    changes = (root / 'CHANGES').read_text().splitlines()[:3] if (root / 'CHANGES').is_file() else []
    return {'dataset': DATASET, 'snapshot_tag': tag, 'snapshot_doi': desc.get('DatasetDOI'), 'license': desc.get('License'),
            'changes_head': changes, 'root': str(root), 'split': split(), 'n_files': len(files),
            'total_bytes': int(sum(f['size'] for f in files.values())), 'missing': missing,
            'second_source_mismatch': mismatch,
            'second_source': (annex or {}).get('mirror'), 'files': files,
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'attribution': 'Wakeman, D.G. & Henson, R.N. (2015). A multi-subject, multi-modal human neuroimaging dataset. '
                           'Sci. Data 2:150001, doi:10.1038/sdata.2015.1; OpenNeuro ds000117 (' + str(desc.get('DatasetDOI')) + ')'}


def verify(manifest: dict, root: str) -> dict:
    root = Path(root)
    bad = []
    for rel, h in manifest['files'].items():
        p = root / rel
        if not p.is_file():
            bad.append({'file': rel, 'problem': 'missing'})
            continue
        sha = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1 << 22), b''):
                sha.update(block)
        if sha.hexdigest() != h['sha256']:
            bad.append({'file': rel, 'problem': 'sha256 differs'})
    return {'ok': not bad, 'checked': len(manifest['files']), 'problems': bad}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['split', 'plan', 'annex', 'download', 'manifest', 'verify'])
    ap.add_argument('--target')
    ap.add_argument('--root')
    ap.add_argument('--mirror')
    ap.add_argument('--annex')
    ap.add_argument('--manifest')
    ap.add_argument('--out')
    ap.add_argument('--subjects', default='all', help="'all', 'dev', 'confirmation' or comma list")
    ap.add_argument('--no-emptyroom', action='store_true')
    a = ap.parse_args(argv)
    subs = {'all': ANALYSED_SUBJECTS, 'dev': DEV_SUBJECTS, 'confirmation': CONFIRMATION_SUBJECTS}.get(a.subjects)
    subs = subs if subs is not None else a.subjects.split(',')
    if a.command == 'split':
        out = split()
    elif a.command == 'plan':
        out = plan_download(a.target, subs, emptyroom=not a.no_emptyroom)
    elif a.command == 'annex':
        out = annex_expectations(a.mirror, subjects=subs)
    elif a.command == 'download':
        out = download(a.target, subs, emptyroom=not a.no_emptyroom)
    elif a.command == 'manifest':
        annex = json.loads(Path(a.annex).read_text()) if a.annex else None
        out = build_manifest(a.root, subs, annex, emptyroom=not a.no_emptyroom)
    else:
        out = verify(json.loads(Path(a.manifest).read_text()), a.root)
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text + '\n')
        print(json.dumps({k: out[k] for k in out if k not in ('files',)}, indent=1)[:4000])
    else:
        print(text)
    if a.command == 'verify' and not out['ok']:
        sys.exit(2)
    if a.command == 'manifest' and (out['missing'] or out['second_source_mismatch']):
        sys.exit(2)


if __name__ == '__main__':
    main()
