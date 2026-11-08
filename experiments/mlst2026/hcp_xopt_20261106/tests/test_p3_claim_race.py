"""p3 claim-queue race test (resume 5).

The run lost 36 deepthought2 + 4 deepthought workers to a FileNotFoundError in the claim
creation window: a second process saw the freshly created claim directory before claim.json
existed, read None, treated it as stale, renamed the directory away, and the creator's
os.replace then failed.  Fails before the fix (the window was treated as stale), passes after.
"""
import os
from pathlib import Path

from experiments.mlst2026.hcp_xopt_20261106 import claim


def _mkqueue(tmp_path):
    qdir = str(tmp_path / 'q')
    rows = [dict(offset=i, date='d', geometry='A', N=56, arm='reference', m=None,
                 replicate=i, date_index=0, theta=0.0) for i in range(2)]
    manifest = claim.prepare(qdir, 'full', rows, chunk_size=1)
    return qdir, claim.chunk_ids(manifest)


def test_claim_creation_window_is_busy_never_stale(tmp_path):
    qdir, ids = _mkqueue(tmp_path)
    cid = ids[0]
    os.mkdir(Path(qdir, 'claims', cid))          # claim directory exists, claim.json not yet
    res = claim.claim(qdir, cid, host='h', pid=1)
    assert res['action'] == 'busy', res
    # the creating process can still publish its claim.json safely
    assert claim._write_claim(qdir, cid, dict(host='creator', pid=2, start=0.0, heartbeat=1e18)) is True
    assert claim.still_owner(qdir, cid, 'creator', 2)


def test_genuinely_stale_claim_still_reclaimed(tmp_path):
    qdir, ids = _mkqueue(tmp_path)
    cid = ids[1]
    assert claim.claim(qdir, cid, host='dead', pid=9)['action'] == 'claimed'
    p = Path(qdir, 'claims', cid, 'claim.json')
    import json, time
    rec = json.loads(p.read_text())
    rec['heartbeat'] = time.time() - 3 * 3600
    p.write_text(json.dumps(rec))
    assert claim.claim(qdir, cid, host='new', pid=10)['action'] == 'reclaimed'


def test_write_claim_returns_false_if_directory_vanished(tmp_path):
    qdir, ids = _mkqueue(tmp_path)
    cid = ids[0]
    assert claim.claim(qdir, cid, host='h', pid=1)['action'] == 'claimed'
    import shutil
    shutil.rmtree(Path(qdir, 'claims', cid))
    assert claim._write_claim(qdir, cid, dict(host='h', pid=1, start=0.0, heartbeat=0.0)) is False
