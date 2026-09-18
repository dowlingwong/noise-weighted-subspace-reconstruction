# Environment record — T22 interpreter deletion, rebuild, and mechanism

Primary incident record (inventoried, unchanged): `mlst-successor-audit-20260916-envloss/INCIDENT_2026-09-16_env_deletion.md`.
This file **adds and corrects**; it does not replace that record.

## Correction: the 14:06 deletion was blocked, not interrupted

The incident record described "an interrupted `rm -rf`". The evidence supports a different mechanism.

* `/home/dwong` is **nfs4**. Deleting a file that a process holds open renames it to a hidden `.nfs…` file,
  the containing directory cannot be removed (`Directory not empty`), and the hidden file is removed when the
  holder releases it. Reproduced on this mount on 2026-09-17 (see below).
* The surviving directories were exactly `numpy`, `scipy`, `h5py` and their `.libs`. Those are the packages whose
  modules the running T22 launcher and worker had loaded. `pytest`, `pluggy`, the `.dist-info` directories and
  `bin/` were not held and were fully deleted.
* In the quarantined tree, 34 of those directories have mtime **2026-09-16 23:44:45–46**, the second the T22
  launcher exited after its final reduction. Nothing else ran in that tree then: it had been renamed away at
  22:24:37. The mtime change is the NFS client removing the hidden files as production released them.

**Implications**
1. The deletion did not stop on its own. It could not remove files held open by production.
2. **The T22 launcher and worker held original-environment module files open until they exited at 23:44:45.**
   This is positive evidence that confirmation cells ran on original-environment modules through to the end.
   It still cannot exclude a module first imported lazily after the 22:24:37 swap, which would have resolved to
   the rebuilt tree at identical versions.
3. The quarantined venv holds no files now. Only its directory skeleton remains, recorded with mtimes in
   `damaged_venv_skeleton_with_mtimes.txt`.

### Mechanism reproduction (2026-09-17 01:40, same mount)

```
open held.so; rm held.so        -> .nfs000000000013519000000956 appears
rmdir <dir>                     -> Directory not empty
release handle                  -> hidden file gone; rmdir succeeds
mount type: nfs4
```

## Stage attribution

| Stage | Environment | Evidence |
|---|---|---|
| All 727 T22 cells | original | ledger pids 2933410 (pilot/control), 3050746 (690 confirmation, started 2026-09-13 21:11); open-file release at 23:44:45 |
| T22 partial reduction and freeze | original | 2026-09-13 21:11 |
| T19, T20, T21, CRESST — every stage | original | completed 2026-09-13 |
| **T22 final reduction** | **rebuilt** | spawned 2026-09-16 23:43:39 from the rebuilt `bin/python` |
| Post-run validation re-reductions | rebuilt | T22 copy; T19 cross-environment check |

## Compatibility evidence

* `invocation_environments_vs_receipt.txt`: every launch-time `environment.json` compared against the receipt.
* The rebuilt-environment preflight's environment dict equals T22's receipt (same file).
* `rebuilt_venv_uv_pip_freeze.txt`, `rebuilt_venv_all_files.sha256`, `rebuilt_venv_pyvenv.cfg`: the exact rebuilt
  environment.
* `base_interpreter.sha256`: the intact base interpreter both environments used.
* Cross-environment reproduction: T19's original-environment `REDUCED.json` reproduces identically under the
  rebuilt environment (`CROSS_ENV_COMPATIBILITY_T19.txt` in the 16 Sep records).
* The original wheels could not be byte-compared; their metadata and files were deleted.

## Earlier interpreter path (2026-09-13, before the final receipt)

Launch-time `environment.json` files show that five invocations at 13:17–13:23 UTC on 13 Sep ran from
`/ceph/dwong/mlst-successor-py312-20260913/bin/python`. All later invocations ran from
`/home/dwong/mlst-successor-py312-20260913/bin/python`. The early records differ from the receipt **only
in `executable`**: the Python build string, numpy, scipy, h5py, pytest, platform, hostname and thread
limits are identical (`invocation_environments_vs_receipt.txt`). T22's receipt was created at 13:33:09 UTC,
after those invocations.

Completed cells attributed by attempt start time to the early invocations:

| Selection | Early `/ceph`-path interpreter | Later `/home`-path interpreter |
|---|---|---|
| T19-S | **20 completed** (all carried pilot cells) + 1 failed | 116 completed |
| T20-S | **14 completed** + 1 failed | 950 completed |
| T21-S | **310 completed** + 1 failed | 2,189 completed |
| T22-S | 0 completed (1 interrupted, the historical orphan) | 727 completed |
| CRESST archive | **21 completed (all)** | 0 |
| CRESST calibration | 0 | 25 completed |

The `/ceph` interpreter no longer exists, so its installed files cannot be byte-compared with either
later environment. **Disclosure:** 365 completed cells across T19, T20, T21 and CRESST archive came from
a same-version interpreter installed at a different path. Every affected selection's reduction was
re-run on a copy under the later environment and reproduced, apart from relocated path fields. For T19
that includes the pilot freeze built from those 20 cells, re-checked under the rebuilt environment.
