import tarfile, hashlib, pathlib, json, collections
W = pathlib.Path('/home/dwong/mlst-successor-closeout-20260917')
T = W/'MLST_SUCCESSOR_TRANSFER_20260917.tar'
def sh(fobj):
    h = hashlib.sha256()
    for b in iter(lambda: fobj.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
# inventories from the return stage (addendum supersedes)
inv = {}
E = pathlib.Path('/home/dwong/mlst-successor-audit-20260916-envloss')
for f in ('RETURN_INVENTORY.sha256', 'RETURN_INVENTORY_ADDENDUM.sha256'):
    for line in (E/f).read_text().splitlines():
        if line and not line.startswith('#'): h, p = line.split('  ', 1); inv[p] = h
lines, stats, bad = [], collections.Counter(), []
with tarfile.open(T, 'r:') as tf:
    for m in tf:
        if not m.isfile(): continue
        h = sh(tf.extractfile(m))
        src = pathlib.Path('/' + m.name)
        s = sh(open(src, 'rb')) if src.exists() else None
        stats['members'] += 1; stats['bytes'] += m.size
        if s != h: bad.append(('member!=source', m.name))
        if '/' + m.name in inv:
            stats['in_return_inventory'] += 1
            if inv['/' + m.name] != h: stats['changed_since_return_inventory'] += 1; bad.append(('member!=return_inventory', m.name))
        lines.append(f'{h}  {m.name}\n')
(W/'TRANSFER_CONTENTS.sha256').write_text(''.join(lines))
tar_sha = sh(open(T, 'rb'))
res = {'tar': str(T), 'tar_sha256': tar_sha, 'tar_bytes': T.stat().st_size, **stats,
       'mismatches': bad[:50], 'n_mismatches': len(bad),
       'status': 'TRANSFER_VERIFIED' if not bad else 'TRANSFER_MISMATCH_REVIEW'}
(W/'TRANSFER_VERIFICATION.json').write_text(json.dumps(res, indent=2) + '\n')
print(json.dumps({k: v for k, v in res.items() if k != 'mismatches'}, indent=1))
