# Round 9 remote report

*Execution record for the overnight run on `deepthought`, 2026-08-18/19.
Scientific interpretation and every manuscript edit remain desk work.*

## Identifiers

| Field | Value |
|---|---|
| `handoff_branch` | `codex/rps-round9-overnight-handoff` |
| `handoff_content_commit` | `ff50d7184a2ab66275f01d076e9750ae8c367107` |
| Work branch | `round9/remote-evidence` |
| Start commit | `ff50d7184a2ab66275f01d076e9750ae8c367107` |
| Phase 1 evidence commit | `aa30b66872330d59a1c84aa8dd530a0a9b2b4f7a` |
| Implementation freeze commit | `51711aa37f2d2749ca46934983358cc57868e2f9` |
| Freeze follow-up (no science changed) | `391cd44c9a66cd01a8231a52b10c0cdbac228572` |
| `scientific_thresholds_changed` | `false` |
| `archived_E1_overwritten` | `false` |
| `manuscript_edited` | `false` |

The runners record the freeze follow-up commit as `implementation_freeze_commit`
in every manifest, because that was HEAD when the ten- and twenty-seed runs
executed. It changed resumability, evaluation caching and wall-clock caps only.

## Machine and environment

`deepthought`, Linux 4.18.0-553.120.1.el8_10, 40 cores, 503 GB RAM
(460 GB available), `/ceph` 318 TB free, load average 4.18 at preflight, no
interactive users. Python 3.12.2, NumPy 2.4.6, SciPy 1.17.1 via
`.venv/bin/python`. All four GPUs were held by this account's own
`llama-server` processes; no GPU work was launched, and no other user's
process or session was touched.

## Blocking infrastructure failure: no Git credentials

`git fetch origin codex/rps-round9-overnight-handoff` → exit 128,
`could not read Username for 'https://github.com'`. The runbook's Phase 0
emits `ROUND9_HANDOFF_FETCH_FAILED` here.

Recorded as an infrastructure gate failure, not silently passed. The provenance
requirement the gate protects was satisfied against the local object store: the
pinned content commit exists by exact SHA and is an ancestor of the fetched
remote-tracking ref, and `round9/remote-evidence` descends from it. The run
continued because nothing scientific depends on network access.

**Consequence: `git push -u origin round9/remote-evidence` cannot be executed
from this machine.** The branch is committed locally and needs pushing from an
authenticated session. This is the one runbook deliverable not completed.

## E6 — variant verification

Verified, not recomputed. The tracked destination was absent from this branch's
ancestry; the artifact lives on the unmerged Round 8 side branch `8ee9eb1`,
which diverges from the Round 9 line at `27fcfc5`. Recovered from the Git
object store by `scripts/round9_verify_e6.py`.

- `excluding_dc.cv_inverse_J` = 0.6883439500348468 → rounds to **0.69** ✓
- `all_bands.cv_inverse_J` = 0.6914499950682564 (not the gated statistic)
- `collinearity_supported: false` ✓
- 328 bands before DC exclusion ✓
- boxcar window, median average, 512 calibration windows ✓
- source SHA-256 = destination SHA-256 = `7f85a6b5b7ebd14cf9324444c46be13e22da5fc25c399409e268d921fb63a433` ✓

No `E6_VARIANT_MISMATCH`.

## E7 — state and gates

**Complete on arrival. Not relaunched, not duplicated.** Both arms
(`t_mse_s20260818`, `t_chi2_s20260818`, stamp `20260817_195350`) held
`checkpoint_00025000.pt` and a 50-evaluation log.

All four Round 8 gates pass. Configs differ in exactly one field: `train.loss`.
Both report `model_params = 569352` and seed 20260818. Wall times 4.98 h and
4.93 h against an 8 h cap. `chi2_of` was not run.

Paired `t_chi2 − t_mse`, sign agrees with round 6 on all four metrics; every
round-9 interval excludes zero:

| Metric | Round 9 | 95% CI | Round 6 |
|---|---:|---|---:|
| `val_chi2` | −1.169e-04 | [−1.330e-04, −1.017e-04] | −1.148e-04 |
| `val_mse` | −57.41 | [−76.15, −39.37] | −57.18 |
| `train_chi2` | −1.036e-04 | [−1.183e-04, −8.879e-05] | −9.943e-05 |
| `train_mse` | −19.86 | [−37.54, −1.71] | −14.26 |

Round 6's `train_mse` interval included zero; round 9's does not.

Gate-1 limitation: the seed-specific screen logs were not retained, so the
resolved data-root *string* is unavailable for these arms. Data identity rests
on both arms reporting an identical window partition (10067 / 7046 / 3017).

## E8 — archived-fit gate and both semantic conditions

Reproduction gate: all 50 (seed, method) rows reproduce the archived
`test_normalized_chi2`, maximum relative error **6.356e-15** against a 1e-9
tolerance. The refits are the archived fits.

Two conditions, separately labelled, never pooled:

- `primary_alpha_0p1` — `post_hoc_exploratory`, config hash
  `98001a62855cb9a5` (matches the archive). This is the post-hoc completion
  of the **dropped** Block H, not a predeclared analysis.
- `separable_alpha_0` — `delayed_predeclared_no_numeric_threshold`, config
  hash `09f7dbd811814c55`. This is the delayed execution of H3.

In both conditions all seven NFPA-minus-Iso-MPCA contrasts are negative
(favouring NFPA), every 95% interval excludes zero, and all 10 seeds agree in
sign. Full tables are in the run log and in each condition's `summary.json`.

Separable-condition planted-factor means: NFPA channel 0.0349, temporal 0.7677,
joint 0.7683; Iso-MPCA 1.021, 1.280, 1.351. **The channel subspace is recovered
almost exactly while temporal recovery is far weaker** — worth the desk's
attention rather than being averaged into a single "recovery" claim.

Method-awareness is explicit: CW-PCA and IsoPCA report the joint projector error
only, because an unrestricted rank-6 subspace identifies no channel or temporal
axis; OF reports amplitude decoding only. Those fields are **null**, not zero.

The original protocol supplied no quantitative recovery threshold, so no binary
H3 verdict is derived here and none should be read into these numbers.

## E9 — replication and pooling decision

Seeds 10–29 under the unchanged frozen configuration, written only to
`E9_blockA_replication/`. The replication summary was produced before any
pooled estimate.

| Contrast | Seeds 0–9 | Seeds 10–29 | Sign agrees |
|---|---:|---:|:--:|
| `test_normalized_chi2` | −0.0192011 | −0.0190460 | yes |
| `test_raw_mse` | +0.00341426 | +0.00341312 | yes |
| `angle_to_cw_pca_mean_deg` (secondary) | −24.6215 | −24.4378 | yes |

The opposite signs of the two primary contrasts are the metric reversal itself.
Both agree between subsets and no interval includes zero, so
`pooling_permitted: true`; `pooled_summary.json` shows all three sets side by
side. The 30-seed aggregate is labelled `post_hoc_pooled` and is **not** a
30-seed confirmatory experiment.

## E10 — estimator diagnostics and paired contrasts

Iso-MPCA control reproduced its archived metrics to 6.022e-15.

Degradation, estimated minus archived oracle NFPA: `normalized_chi2`
+7.905e-05 [+6.301e-05, +9.635e-05]; `raw_mse` +3.548e-06
[+1.154e-06, +6.692e-06]. Both resolved, both tiny — roughly 0.008% relative on
a held-out chi-square near 1.015.

Under estimated whitening, NFPA − Iso-MPCA is `test_normalized_chi2`
−0.0191221 and `test_raw_mse` +0.00341781, against oracle values −0.0192011
and +0.00341426. **The reversal survives estimated whitening essentially
intact.**

Diagnostics respect the reciprocal Kronecker-factor scale non-identifiability:
shape errors are trace-normalized (channel 0.0282, temporal 0.0197), the scale
is reported once globally, and the full-covariance quantities come from
Kronecker trace/Frobenius identities rather than dense (C·T)² products. The
scale ratio (0.0103), the Kronecker Frobenius error (0.9897) and the
oracle-whitening residual (2.197) are dominated by that non-identified global
scale — `E[Sigma_c_hat] = Sigma_c · tr(Sigma_t)/T` — and not by estimation
error. NFPA's fit and reconstruction are invariant to that global factor.

This is one fixed condition: a measured sensitivity check, not a general
covariance-robustness boundary.

## Failures, deviations and skipped dependents

1. **`ROUND9_HANDOFF_FETCH_FAILED`** — no Git credentials; fetch and push both
   unavailable. Provenance verified locally instead. Push outstanding.
2. **Pre-existing defect in `methods.basis_whitened`**, found while
   implementing E8 and deliberately **not** fixed: it unstacks a
   `(C·T, rank)` basis with `B.reshape(rank, C, T, order="F")` instead of
   `B.T.reshape(...)`. Reached only by Iso-MPCA; feeds only
   `angle_to_cw_pca_*`. Reconstruction metrics, the E8 gate and both E9
   primary contrasts are unaffected. Fixing it would redefine an archived Block
   A column and break E9's comparability with seeds 0–9, so it is pinned by a
   test, annotated in E9's summary, and left to the desk.
3. **Wall-clock caps raised** (E8 2h→3h per condition, E9 4h→6h), justified from
   the archived 432 s/seed end-to-end rate and this run's 488 s one-seed gate.
   Never lowered; no scientific budget reduced.
4. No experiment failed and no dependent was skipped.

## Push attempt

```
$ git push -u origin round9/remote-evidence
fatal: could not read Username for 'https://github.com': terminal prompts disabled
exit 128
```

The branch is committed locally at `386fd6b49cc4449bffcf0cb858a0cb174cde7444`
and descends from the pinned handoff content commit. It has **not** been
pushed. Nothing else was pushed either; `main` was not touched and nothing was
force-pushed.

## Final machine state

No Round 9 process, no screen session, no stale `run.pid`. `nvidia-smi`
reports no compute processes at all: this account's pre-existing
`llama-server` processes exited on their own during the run and were never
touched by it.
Archived E1 digest re-checked at the end: `e9778eb2d663…`, unchanged from
preflight. `results/rps2026` is 1.1 MB. `git diff --check` clean. 49 tests pass.
Evidence audit: **61/61**.

Nothing under the TIDMAD raw-data tree was modified; checkpoints and denoised
outputs remain outside Git.

## DOES_SUPPORT

- The rung-4 non-reversal replicates across two seeds, agreeing in sign on all
  four metrics with every round-9 interval excluding zero.
- The rung-3 metric reversal replicates on 20 fresh seeds with directions
  agreeing, permitting a labelled post-hoc pooled aggregate.
- The reversal survives one fixed estimated-covariance condition with a
  measured, very small degradation.
- Refits at the archived configuration reproduce the archived held-out
  chi-square to ~1e-15, so the Round 9 semantic analysis is computed on the
  archived fits.
- In the separable condition NFPA attains substantially lower planted-factor
  projector errors than Iso-MPCA on every reported axis, across all 10 seeds.
- The whitening dynamic range refutes the collinearity account
  (`cv_inverse_J` = 0.69 excluding DC, `collinearity_supported: false`).

## DOES_NOT_SUPPORT

- Any binary confirmation of H3. The protocol set no quantitative recovery
  threshold; only continuous errors and paired intervals exist.
- Any claim that the archived-condition (α=0.1) decoding result is
  predeclared. Block H was dropped; that analysis is post-hoc exploratory.
- Any claim that NFPA recovers the planted temporal subspace as well as the
  channel subspace — the separable-condition errors are 0.7677 versus 0.0349.
- Any general covariance-robustness statement. E10 is one fixed condition.
- Any seed-level uncertainty statement for rung 4. Two seeds, with the
  bootstrap resampling evaluations along one pair of trajectories.
- Any statement about `angle_to_cw_pca_*` for Iso-MPCA until the desk resolves
  the `basis_whitened` defect.
- A 30-seed confirmatory Block A. The pooled set is `post_hoc_pooled`.
