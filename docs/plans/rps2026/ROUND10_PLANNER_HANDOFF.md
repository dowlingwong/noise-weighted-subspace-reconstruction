# Round 10 → planner handoff

*From the remote execution agent. Round 10 is finished and published. This is
the short version; the full execution record is
`docs/plans/rps2026/ROUND10_REMOTE_REPORT.md` on the branch named below.*

## Result in one line

**E10b and E11 both completed with every gate passing, and the evidence branch
is published and verified.**

## Where to find it

| What | Where |
|---|---|
| **Branch (published)** | `round10/remote-corrections` @ `81d3fd8aabd62a06bfe38f2e5992e13a09bc07f8` |
| Evidence commit | `64df3c7b5da61f48aaaebbf31c0e8daf963d7c67` |
| Remote | `git@github.com:dowlingwong/noise-weighted-subspace-reconstruction.git` |
| **Worktree** | `/ceph/dwong/noise-weighted-subspace-reconstruction` on host **`bms2`** |
| Out-of-tree runtime | `/home/dwong/round10_runtime_20260819_180515` (source of truth was copied into the repo; this directory is scratch and may be deleted) |
| Halted-attempt record | `round10/remote-corrections-halted-20260819` @ `41cb5fb` (published, preserve or delete at your discretion) |

To pick it up:

```bash
git fetch origin round10/remote-corrections
git switch round10/remote-corrections     # or: git worktree add ../r10 round10/remote-corrections
```

Note the worktree currently sits on `bms2`, **not** `deepthought` where Rounds 8
and 9 ran. `origin` on that checkout was repointed from HTTPS to SSH (transport
only; there are no HTTPS credentials on these hosts).

### Evidence paths on the branch

```text
results/rps2026/E10b_calibrated_estimated_covariance/   10 seed shards + raw.csv, summary.json, manifest.json, shard_registry.json
results/rps2026/E11_corrected_angles/                   30 seed shards + raw.csv, summary.json, manifest.json, shard_registry.json
results/rps2026/Round10_remote/                         archive audits before/after, full pytest log, machine preflight, runner logs
results/rps2026/Round10_smoke/                          implementation check only — NOT evidence
docs/plans/rps2026/ROUND10_REMOTE_REPORT.md             full execution record
```

Manifest summary digests, if you want to pin them:

```text
E10b  1846242438133ce52c1dd4adf2d78f34cfc34570c4748ac3e71f50f26b0d496b
E11   d4f9d004430f5bc9cf55a347c33bced1448190d1e1c9cc8678b1841c22bd6170
```

## What the numbers say

### E10b — the calibration fixed the diagnostics, not the conclusions

| Diagnostic | E10 (Round 9) | E10b |
|---|---:|---:|
| product scale ratio vs oracle | 0.0103 | **0.9991** |
| Kronecker relative Frobenius error | 0.9897 | **0.0365** |
| oracle-whitening residual | 2.197 | **0.0034** |
| channel / temporal shape error | 0.028163 / 0.019673 | 0.028163 / 0.019673 |

The shape errors are unchanged to every printed digit — they were already
trace-normalized and therefore already scale-free. The held-out contrasts are
also unchanged from Round 9's E10 (`test_normalized_chi2` −0.0191221,
`test_raw_mse` +0.0034178), which is the predicted invariance of NFPA's fit and
reconstruction under a global rescaling of both whitening factors.

**Planner-relevant:** the Round 9 limitation that E10's diagnostics were
uninterpretable is now resolved. The scientific claims that rested on the
contrasts did not move, so nothing that cited E10's contrasts needs revisiting;
anything that cited E10's *diagnostics* should now cite E10b's.

### E11 — the angle defect was exactly as scoped

NFPA's legacy and v2 angles are **identical** in all three cohorts; only
Iso-MPCA moves, by ~0.8°. This confirms the blast radius recorded in the Round 9
report: the defective branch of `methods.basis_whitened` was reachable only by
Iso-MPCA.

| Cohort | n | v2 contrast NFPA − Iso | 95% CI |
|---|---:|---:|---|
| `frozen_0_9` | 10 | −23.8156 | [−24.2204, −23.3930] |
| `replication_10_29` | 20 | −23.7078 | [−24.0765, −23.3396] |
| `post_hoc_pooled_0_29` | 30 | −23.7437 | [−24.0264, −23.4590] |

Round 9 reported −24.62 / −24.44 under the legacy convention. Direction,
significance and cohort ordering are unchanged; only the magnitude shifts by
~0.8–0.9°.

**Planner-relevant:** any manuscript text quoting an `angle_to_cw_pca` number
needs the v2 value substituted. The summary states explicitly that
interpretation must use **only** the `_v2` columns; the `_legacy` columns exist
solely to reproduce the archived diagnostic and are invalid as subspace angles
for Iso-MPCA. The pooled cohort is still `post_hoc_pooled_0_29` and must not be
described as a thirty-seed confirmatory experiment.

## Integrity, briefly

- Complete repository suite passed **before** either run: exit 0, 417 passed,
  9 skipped, 426 collected. (The previous dispatch halted here; the two
  `gwpy`-dependent GWOSC tests are now `importorskip`-guarded and skip.)
- All 15 authoritative archive digests recorded before and after; the after file
  is byte-identical to the before file. Nothing archived was modified.
- Both manifests: `status = complete`, `summary_regeneration_audit.passed = true`,
  freeze `048f1bd96acf`, `source_tree_dirty_at_start = false`.
- Gates recomputed from the authoritative E1/E9 tables, not read from stored
  columns: `worst_stored_vs_recomputed_disagreement = 0.0` in both, worst
  reproduction error 6.02e-15 (E10b) and 2.80e-14 (E11) against a 1e-9 tolerance.
- Neither run was interrupted, so no `--resume` was used and no `in_progress`
  manifest exists.
- Wall times: E10b 1.00 h (cap 3 h), E11 3.32 h (cap 4 h). CPU-only; `bms2` has
  no GPU.

All eight required terminal fields are true.

## What the planner still owns

1. **Manuscript substitution.** Replace legacy angle figures with v2, and E10
   diagnostics with E10b. The contrasts do not change.
2. **Merge decision.** `round10/remote-corrections` is not merged anywhere. It
   branches from `76aaff6`, which descends from `ef1f785` on the mainline.
3. **The halted branch.** `round10/remote-corrections-halted-20260819` exists
   only to preserve the record of the 2026-08-19 stop. Delete it once the
   report has been read, if you want the branch list tidy.
4. **Round 9's open item, still open.** `methods.basis_whitened` was
   deliberately left unfixed so E9's rows stayed comparable with archived
   Block A. E11 supplies the corrected values alongside the legacy ones rather
   than rewriting the archive, so the underlying function is still defective for
   Iso-MPCA. Decide whether to fix it and what that implies for the archived
   `angle_to_cw_pca_*` columns.

Nothing on the remote side is outstanding.
