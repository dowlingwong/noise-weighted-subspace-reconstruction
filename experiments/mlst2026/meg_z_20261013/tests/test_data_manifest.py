"""D0: subject split, include list, manifest building and verification (no download)."""
import hashlib
import json

import pytest

from experiments.mlst2026.meg_z_20261013 import data_manifest as M
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol


def test_split_fixed_before_download():
    sp = M.split()
    assert sp['analysed'] == [f'sub-{i:02d}' for i in range(1, 17)]
    assert sp['dev'] == ['sub-01', 'sub-02', 'sub-03', 'sub-04'] and sp['confirmation'] == [f'sub-{i:02d}' for i in range(5, 17)]
    assert not set(sp['dev']) & set(sp['confirmation'])
    proto = load_protocol()
    assert proto['subjects']['dev'] == sp['dev'] and proto['subjects']['confirmation'] == sp['confirmation']


def test_include_patterns_and_plan():
    inc = M.include_patterns()
    assert len(inc) == 16 * (6 * 4 + 2) + 1 and inc[-1] == 'sub-emptyroom'
    assert 'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-06_meg.fif' in inc
    assert not any('acq-' in p or 'proc-sss' in p for p in inc)                 # no SSS calibration/crosstalk files
    plan = M.plan_download('/x', M.DEV_SUBJECTS)
    assert '--tag=2.0.0' in plan['cli'] and '--dataset=ds000117' in plan['cli']
    with pytest.raises(ValueError):
        M.include_patterns(['sub-17'])


def test_manifest_and_verify(tmp_path):
    root = tmp_path / 'ds'
    files = M.subject_files('sub-01')
    for i, rel in enumerate(files):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f'content {i}'.encode())
    (root / 'dataset_description.json').write_text(json.dumps({'License': 'CC0', 'DatasetDOI': 'doi:x'}))
    annex = {'mirror': 'm', 'files': {}}
    for rel in files[:3]:
        data = (root / rel).read_bytes()
        annex['files'][rel] = {'annex': {'backend': 'MD5E', 'size': len(data), 'hash': hashlib.md5(data).hexdigest()}}
    annex['files'][files[3]] = {'git_blob_sha1': M.git_blob_sha1((root / files[3]).read_bytes())}
    annex['files'][files[4]] = {'annex': {'backend': 'MD5E', 'size': 1, 'hash': '0' * 32}}      # deliberate mismatch
    man = M.build_manifest(root, ['sub-01'], annex, emptyroom=False)
    assert man['missing'] == [] and man['license'] == 'CC0' and man['split']['dev'][0] == 'sub-01'
    assert man['second_source_mismatch'] == [files[4]]
    assert all(man['files'][f]['annex_check'] == 'ok' for f in files[:4])
    assert M.verify(man, root)['ok']
    (root / files[0]).write_bytes(b'tampered')
    v = M.verify(man, root)
    assert not v['ok'] and v['problems'][0]['file'] == files[0]
    man2 = M.build_manifest(root, ['sub-01', 'sub-02'], None, emptyroom=False)
    assert len(man2['missing']) == len(M.subject_files('sub-02'))


def test_parse_annex_key():
    t = '../../../.git/annex/objects/MV/XV/MD5E-s860225132--bdbee2944afbeb3943d75628cb18aef5.fif/MD5E-s860225132--bdbee2944afbeb3943d75628cb18aef5.fif'
    k = M.parse_annex_target(t)
    assert k == {'backend': 'MD5E', 'size': 860225132, 'hash': 'bdbee2944afbeb3943d75628cb18aef5'}
    assert M.parse_annex_target('plain text') is None
