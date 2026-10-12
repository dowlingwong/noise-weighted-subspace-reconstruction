# Y19 preparation (stage B0) — execution report

**Date:** 12 October 2026. **Role:** execution agent, stage B0 of
`plan/onthefly/RUNBOOK_MLST_MULTICHANNEL_UPGRADE_2026-10-12.md` §4 (author decisions D1–D8 adopted,
`reviews/z_decisions_20261012/AUTHOR_DECISIONS.md`). **Machine:** the author's Cowork Linux VM (4 cores,
3 GB RAM, Python 3.10.12, numpy 2.2.6; scipy 1.15.3, pytest 9.1.1 and h5py 3.16.0 installed into
`/tmp/pylib`, outside both repositories).

**Outcome.** A registrable kit for Y19 exists in EXP `experiments/mlst2026/y19_20261012/` (kit identity
`bbf4e8c64bfb1126bcccb8645a18e78196d9e03c5a2ea1f8117e2464eaa2bfe8`), with the human-readable protocol
`plan/Y19_PROTOCOL_2026-10-12.md` and the host prompt `plan/onthefly/AGENT_PROMPT_Y19_HOST_2026-10-12.md`.
Validation gate: PASS. Tests: 14 passed. Nothing is committed (read-only git only); the author commits the
kit, which then is the freeze.

**Hard-rule compliance.** No G2 record was drawn, no EMPCA/NFPA fit touched G2 data, and no G2 risk contrast
was computed. G2 was used only through population quantities (K(φ), eigen-structure, B_cert, the population
product optimum, c_U, c_F, a, N\*). Runner smoke tests use the T25 fixture geometry (C = 3, T = 4); the
timing benchmark uses a non-G2 T20-S/T25-family generator of the same d = 384 and records no risk. Seeds come
from a fresh SHA-256 namespace. No MS file outside the four listed deliverables was edited; in EXP only the
new kit directory was written. No `__pycache__`/`.pytest_cache` was left in either repository.

## 1. Commands run (EXP root; `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/pylib:.:src`)

```bash
# validation gate, core level (first computation of the session, before any G2 quantity)
python3 /tmp/y19dev/val.py          # core.population_product_optimum/first_order/summary on SparseLatentModel(theta, p=1)
# development scans of G2 population quantities only (scratch scripts in /tmp/y19dev, not in the kit)
python3 /tmp/y19dev/scan.py 0 0.5 1 2 3 ; python3 /tmp/y19dev/scan.py 0.05 ... 0.3 ; ... 0.11 ... 0.22
# kit
python3 -m experiments.mlst2026.y19_20261012.predictions write        # predictions.json + protocol.json (68 s, 4 threads)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m experiments.mlst2026.y19_20261012.benchmark --out /tmp/y19dev/bench.json --seeds 2
python3 -m experiments.mlst2026.y19_20261012.validation_gate
python3 -m experiments.mlst2026.y19_20261012.runner smoke --out /tmp/y19dev/smoke2 --workers 3   # T25 fixture, pool path
python3 -m experiments.mlst2026.y19_20261012.runner write-kit-hashes
python3 -m experiments.mlst2026.y19_20261012.runner verify-kit        # kit_identity bbf4e8c6…, 15 files, dependencies ok
python3 -m experiments.mlst2026.y19_20261012.predictions check        # ok, grids equal, max diff 0.0
python3 -m pytest experiments/mlst2026/y19_20261012/tests -v -p no:cacheprovider
```

## 2. Validation gate (T25 generator; `validation_gate.py`, `tests/test_validation_gate.py`)

Reference values are copied verbatim into `tests/t25_reference.json` from
`reviews/p1nl_preparation_20261006/tr09/weight_table.json` (SHA-256 `b26e3dd4…`) and
`reviews/upgrade_y15_y24_2026-10-09/y15/RECOMPUTATION.json` (SHA-256 `4e2074d1…`). The Y19 constants path
(`predictions.constants`, wrapping `core.py` unchanged, SHA-256 `3112bd3f…`, identical to the hash recorded in
the weight table's provenance) was run with the weight-table restart stream (n_random = 6), and again with the
Y19 budget of 64 restarts.

| θ | quantity | recomputed | saved | rel. diff | tolerance |
|---|---|---|---|---|---|
| all | c_U | 3010.3570787525 | 3010.3570787525 | 0 | 1e−10 |
| 0 | a | 2.8e−14 | 2.8e−14 | (abs) 2.8e−14 | abs 1e−10 |
| 0.12 / 0.25 / 0.40 | a | 0.538362 / 2.307372 / 5.757193 | same | 5.3e−14 / 0 / 4.9e−15 | 1e−8 |
| 0.12 / 0.25 / 0.40 | a (64 restarts) | same | same | ≤ 6.6e−14 | 1e−8 |
| 0 / 0.12 / 0.25 / 0.40 | c_F | 187.665 / 188.285 / 190.374 / 194.793 | same | ≤ 7.5e−16 | 1e−6 |
| 0.12 / 0.25 / 0.40 | N\* | 5240.96 / 1221.16 / 488.05 | same | ≤ 5.3e−14 | 1e−6 |
| 0.12 / 0.25 / 0.40 | N\* rounded | 5241 / 1221 / 488 | tracker §4c: 5241 / 1221 / 488 | exact | — |
| 0.12 / 0.25 / 0.40 | B_cert | 0.226806 / 0.972107 / 2.425384 | Y15 recomputation | ≤ 1.9e−13 | 1e−8 |
| 0.12 / 0.25 / 0.40 | interval upper end | 12441.65 / 2899.90 / 1159.87 | Y15 recomputation | ≤ 1.4e−9 | 1e−6 |

c_F lies in the brief's 187.7–194.8 range at every θ. **Result: PASS.** Order: the core-level reproduction
(`/tmp/y19dev/val.py`, same functions, same numbers) was the first computation of the session, before any G2
quantity; the kit-level gate script was written and run after `predictions.json`, through the same code path,
and is also a test that must pass on the host.

## 3. G2 design choices (every choice and why)

| Choice | Value | Reason |
|---|---|---|
| Format | C = 6, T = 64, d = 384, k = 6, (k_c, k_t) = (2, 3) | brief; differs from T25 (8 × 32); larger T (review M8) |
| Noise | Σ = S_t ⊗ S_c, AR(1), ρ_c = 0.4, ρ_t = 0.7, unit diagonal, via `t25.science.covariance` | "as in T25, re-dimensioned": same correlation per lag, so the temporal correlation length (≈ 2.8 samples) is unchanged while T doubles |
| Spectrum | λ_i = 24/i (i ≤ 6), gap 2, tail 14/j (j ≥ 7) | 1/i law (brief); top-k energy 58.8 ≈ T25's 57; gap/λ_k = 0.5 (T25 0.57); one law throughout, power-law tail instead of geometric. tr K = 115.9 (T25 206.6): same order, stated |
| Nonseparability | channel-dependent pulse delay φ·(c − 2.5) samples applied to a common 3-pulse basis | "channel-dependent pulse-shape mixing" (brief), physically a timing skew across channels; φ = 0 exactly separable |
| Pulses | double exponentials, onset 16, (τ_r, τ_d) = (1, 6), (2.5, 15), (5, 35) | fast/medium/slow components typical of detector pulses; onsets stay inside the window for the delays used, but the sampled templates are truncated at t = 63, where the slow pulse is still at 42% of its peak and the medium pulse at 7.5% (corrected after B1, F8; part of the G2 definition, not an approximation) |
| Channel patterns | Gaussian (width 2) and linear gradient | two smooth, linearly independent spatial modes |
| Eigenvectors | eigenvectors of the whitened template Gram K₀ with weights (1, .7, .45) ⊗ (1, .6), eigenvalues imposed | keeps the spectrum identical at every φ (φ moves geometry only, like θ in T20-S/T25), while the physical top-k subspace is exactly the template span |
| Tail eigenvectors | seeded Gaussian completion off the top-k span | the T20-S/T25 device |
| Latents | Gaussian along K's eigenbasis | Supplement S4.3 (Proposition S4.1) assumptions hold (TR09 caveat on non-eigenbasis latents does not arise) |

## 4. φ selection (population quantities only; `predictions.json` → `phi_selection`)

Rule fixed in `predictions.py` before the final computation: on φ ∈ {0.01, …, 0.50} (8 restarts each),
φ_A = smallest φ with N\*_pred ≤ 3000, φ_B = largest φ with N\*_pred ≥ 1200, ratio ≥ 2. Selected scan points:

| φ | 0.05 | 0.10 | 0.11 | **0.12** | 0.13 | 0.15 | 0.19 | **0.20** | 0.21 | 0.25 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_best | 0.188 | 0.699 | 0.832 | 0.973 | 1.122 | 1.442 | 2.155 | 2.347 | 2.543 | 3.380 | 4.538 | 7.247 |
| N\*_pred | 15140 | 4063 | 3414 | 2917 | 2529 | 1967 | 1315 | 1208 | 1114 | 837 | 623 | 389 |

The knob is steep (φ = 0.5 already gives a ≈ 8, N\* ≈ 350), because a fraction of a sample of relative delay
moves the fast pulse appreciably in the whitened metric. The lower limit 1200 keeps both formal zeros above
N ≈ 1000 (the local domain stated in Supplement S4.3, inherited from T20-S/T25; see B1 fix F3); the upper limit 3000 is the brief's window. Result: φ_A = 0.12, φ_B = 0.20,
N\*_A/N\*_B = 2.42. During development I also looked at φ ∈ {0.5, 1, 2, 3} (a = 8.1–25.9, N\* 105–349) to find
the scale; nothing fitted entered.

## 5. Predictions (final, 64 Newton restarts; all reach the same maximum; all stationary)

| Geometry | φ | B_cert | a_best | c_U | c_F | B | N\*_pred | interval | grid N |
|---|---|---|---|---|---|---|---|---|---|
| control | 0 | 0 | 1.4e−14 | 3097.356 | 256.245 | −2841.11 | none | — | 302, 604, 729, 1208, 1458, 2415, 2917, 4830, 5834, 11668 |
| A | 0.12 | 0.26033 | 0.97299 | 3097.356 | 258.216 | −2838.17 | **2916.96** | [2916.96, 10904.80] | 729, 1458, 2917, 5834, 11668 |
| B | 0.20 | 0.62010 | 2.34666 | 3097.356 | 261.200 | −2833.81 | **1207.59** | [1207.59, 4572.71] | 302, 604, 1208, 2415, 4830 |

Other population checks: H min eigenvalue 16.14 / 15.42 / 13.85 (≻ 0); gradient norm at P\*_F ≤ 2.4e−9;
k − cap = 0 / 0.130 / 0.310; Δ_k = 2; spectrum error ≤ 4e−14; φ = 0 separability residual 2.2e−14; template-span
residual ≤ 1.6e−14. Family size 20 (5 + 5 + 10), Bonferroni t_{1−0.05/40, 23} = 3.393. First-order means a + B/N
and a power heuristic (N·sd Δ_U = 96.5, N·sd Δ_F ≈ 26.5, both population quantities) are in the protocol §3;
they suggest every deciding cell is decided if the first-order picture holds.

## 6. Runtime estimate (non-G2 timing; `benchmark.py`)

Generator: T20-S/T25 family at C = 6, T = 64, (2, 3), T25 spectrum, θ = 0.25, benchmark geometry seed and
namespace `y19-benchmark-nonG2`; the confirmation fit `runner.fit_record` (4 NFPA restarts, max_iter 6000,
tol 1e-8), single thread; no risk kept.

| N | 302 | 1208 | 2917 | 11668 |
|---|---|---|---|---|
| seconds per record (2 seeds) | 2.90, 2.91 | 3.08, 2.98 | 2.97, 3.04 | 2.99, 2.87 |
| NFPA iterations per restart | 362–489 | 409–520 | 413–492 | 372–483 |

All restarts stationary. Pool generation ≤ 0.33 s per (geometry, seed). Estimate for 480 records: ≈ 25 CPU-min,
≈ 7–10 min wall with 4 workers plus ≈ 30 s preflight. Worst case (every restart hits 6000 iterations,
≈ 13× more) ≈ 5–6 CPU-h, ≈ 1.5 h wall. Peak memory measured for one worker at N = 11668: 217 MB max RSS (largest pool 11668 × 384 doubles = 36 MB).
Output ≈ 25–30 MB (records + compressed arrays).

## 7. Tests (`python -m pytest experiments/mlst2026/y19_20261012/tests -v -p no:cacheprovider`)

```
test_g2_population.py::test_separable_at_phi0_to_1e12                  PASSED  (‖P*_U − P_t⊗P_c‖ ≤ 1e−12; templates = kron(P0, Q); B_cert ≤ 1e−12)
test_g2_population.py::test_spectrum_invariant_and_nonseparable        PASSED
test_g2_population.py::test_predictions_and_protocol_consistent        PASSED  (grids, family 20, df 23, ≥ 50 restarts, 300 ≤ N* ≤ 3000, ratio ≥ 2)
test_g2_population.py::test_population_recomputation_matches_predictions PASSED
test_runner_smoke.py::test_smoke_never_uses_g2                         PASSED
test_runner_smoke.py::test_smoke_records_and_summary                   PASSED  (T25 fixture, 3 seeds, 60 records, summary + results.h5)
test_runner_smoke.py::test_resume_and_determinism                      PASSED  (other records untouched; removed record recomputed identically)
test_runner_smoke.py::test_reading_rule_logic                          PASSED  (MET / NOT MET / control / optimization-limited flag / N*_fit)
test_runner_smoke.py::test_kit_identity_and_dependencies_subprocess    PASSED
test_runner_smoke.py::test_confirmation_preflight_subprocess           PASSED  (and fails closed with OPENBLAS_NUM_THREADS=4)
test_seed_namespace.py::test_seed_derivation                           PASSED
test_seed_namespace.py::test_disjoint_from_prior_studies               PASSED  (T20-S, T25, TR09, E-NL seed sets)
test_seed_namespace.py::test_namespace_unused_elsewhere                PASSED
test_validation_gate.py::test_t25_constants_reproduced                 PASSED
14 passed in 42.14s
```

## 8. Files

EXP `experiments/mlst2026/y19_20261012/` (SHA-256 of every file in `KIT_SHA256.json`; kit identity
`bbf4e8c64bfb1126bcccb8645a18e78196d9e03c5a2ea1f8117e2464eaa2bfe8`; `KIT_SHA256.json` itself
`8d8ac8441feaee4b28cfc46637bfbd06813c0131a1f7578967a21c59274909d9`):
`__init__.py`, `generator.py`, `predictions.py`, `runner.py`, `validation_gate.py`, `benchmark.py`,
`protocol.json` (`a4ecaea1…`), `predictions.json` (`329a0cac…`), `README.md`, `tests/__init__.py`,
`tests/t25_reference.json`, `tests/test_validation_gate.py`, `tests/test_g2_population.py`,
`tests/test_runner_smoke.py`, `tests/test_seed_namespace.py`, `KIT_SHA256.json`.

Pinned external dependencies (`protocol.json` → `dependencies`, 35 in-repo files): the 30 accepted T25 closure
members actually loaded (checked against closure `01f2f82f…` by the X3 guard) plus five files outside it,
`p1nl_dev_20261006/__init__.py`, `p1nl_dev_20261006/tr09_first_order/{__init__,core}.py` and
`x3_nonlinear_metric_20261005/{__init__,guard}.py` — pinned by SHA-256 and re-checked by the preflight.

MS: `plan/Y19_PROTOCOL_2026-10-12.md`, `plan/onthefly/AGENT_PROMPT_Y19_HOST_2026-10-12.md`, this report,
`reviews/y19_preparation_20261012/ZENODO_PACKAGE.md`.

## 9. Deviations from the brief and open issues

1. **Evaluation size.** The T25 oracle arm scores the fitted affine map by the closed-form expected clean risk
   on an independent record (`t25.science.risks`), not by a finite evaluation sample. I kept that (the brief
   asks for the identical T25 recipe), so the "evaluation size" is infinite and stated as such.
2. **Gate ordering.** The core-level reproduction ran first; the kit-level `validation_gate.py` was written
   after `predictions.json`. Same functions, same numbers; it is also a host test.
3. **tr K.** 115.9 versus T25's 206.6 (top-k energy matched: 58.8 vs 57). Choosing one 1/i law throughout
   seemed cleaner than inflating the tail; stated in the protocol.
4. **Below the local domain and below d.** N\*/4 and N\*/2 of B (302, 604) and N\*/4 of A (729) are below
   N ≈ 1000; N = 302 < d = 384. The reading uses only their signs; flagged in the protocol §11.
5. **φ is steep.** Small φ changes move N\* a lot (dN\*/dφ ≈ −44000 per unit φ at A). The predictions are
   population quantities at exactly φ = 0.12 and 0.20, so this is not a source of error, but a reviewer may
   want to know it.
6. **Kit status.** The kit is complete and self-verifying but not yet frozen: the author's commit is the freeze.
   If the B1 review requests a change, rerun `predictions write` (only if a prediction-relevant parameter
   changes) and `runner write-kit-hashes`, then update the hashes in the protocol, host prompt and
   ZENODO_PACKAGE.md.
7. **Not done (out of scope or forbidden by the brief).** No CHANGELOG or tracker entry (Z2/Y19 status) and no
   commit; the planning agent/author should add them. EXP shows an unrelated untracked `paper_rps2026/`.
8. **Host environment.** The kit needs scipy (T25 imports `scipy.stats.t`, T20-S `scipy.linalg.expm`). The
   runner uses the `fork` multiprocessing context (Linux hosts). Thread variables must be 1 or the preflight refuses.

## 10. B1 fixes (12 October 2026)

Review: `reviews/y19_preparation_review_20261012_01/INDEPENDENT_REVIEW.md`, verdict ACCEPTED WITH MINOR CHANGES.
The reviewer required no kit change. **The kit is unchanged**: identity
`bbf4e8c64bfb1126bcccb8645a18e78196d9e03c5a2ea1f8117e2464eaa2bfe8`, `KIT_SHA256.json` `8d8ac844…09d9`,
`protocol.json` `a4ecaea1…771e`, `predictions.json` `329a0cac…6996`. All fixes are in the MS documents.

| Finding | Fix | Location |
|---|---|---|
| F1 (required) | The host commit uses `git add -f` (EXP `.gitignore` ignores `*.h5`, `*.npz`). Before committing, `git status --porcelain --ignored` must show nothing ignored or unstaged. After committing, `git ls-files` must show 480 record JSON, 480 NPZ and the 9 named run files (969 in total, plus any listed temp file). The counts are required in the host report; nothing is pushed if a count differs. | host prompt §6 |
| F2 (required) | Author decision (runbook §4): frozen interpretation layered on the unchanged machine outputs. Reported overall = **MET** only if A, B and the control are MET; **MET WITH FAILED CONTROL** if A and B are MET but the control is NOT MET (no confirmation claimed; the paper must state it); **NOT MET** if A or B is NOT MET. `summary.json` → `overall` stays the A∧B machine component. The host prompt prints both. | protocol §7; host prompt §5 |
| F3 | The N ≈ 1000 local-domain threshold is inherited from the T20-S/T25 setting (Δ_k = 4, d = 256) and not calibrated for G2 (Δ_k = 2, d = 384). B's formal zero (1208) sits at its edge. Y19's rule is the fixed-grid analogue of Y15's decided-cells reading. | protocol §4, §11 (caveat "Inherited threshold"); this report §4 |
| F4 | Steep, piecewise-smooth φ dependence disclosed: a ∝ φ² near 0, N\* ∝ φ⁻² roughly, dN\*/dφ ≈ −4.4·10⁴ at A, onset-kink breakpoint at φ = 0.40. Predictions hold only at exactly φ = 0.12 and 0.20; no robustness claim. | protocol §11 (caveat and "Not claimed"); this report §9.5 |
| F5 | Order of events disclosed: the φ-rule limits and grid were set after exploratory population-only scans; no sampled or fitted quantity existed. | protocol §4 ("Order of events") |
| F6 | (a) Protocol and host-prompt hashes recomputed after the edits. The deposited protocol keeps its `<FREEZE_COMMIT>` placeholders; the commits go into the deposit metadata. (b) The 35 pinned dependency files are listed by path and SHA-256, and their `git archive` at the freeze commit is added as a required item. (c) `ZENODO_PACKAGE.md` states that the MS Y19 files must be committed at freeze and the MS commit recorded; the host prompt has a `<MS_FREEZE_COMMIT>` placeholder. | `ZENODO_PACKAGE.md`; protocol status; host prompt header/§0 |
| F7 | Host-prompt hygiene: allowed writes include the clone, venv, logs and pytest temp dir; explicit gate-4 commands (`git ls-remote origin 'refs/heads/y19-return*'`, `git log --all --oneline -- …_return`, absence test); `summarize --partial` and reading records during the run forbidden (progress by counting only); stderr and `START.json` checked about 90 s after the `nohup` start; pytest/stdout/stderr logs copied into the return dir before the commit; `verify-kit` after the run; numpy 2.2.6 / scipy 1.15.3 pinned. | host prompt §§0–6 |
| F8 | Wording corrected: pulse truncation at t = 63 (slow pulse at 42% of its peak, medium at 7.5%) stated; template-span residual is ≤ 1.6e−14 (control 1.6e−14, A/B 3.3e−15), not 3e−15. | protocol §2; this report §3 |
| F9 | Observation, no change. B3 should re-run `verify-kit` before reading; added to the host prompt §5/§6. | host prompt |
| Renumbering | The supplement was renumbered today. The MS documents now cite "Supplement S4.3 (label `supp:first-order-crossing`)", Proposition S4.1 (`prop:supp-first-order-crossing`) and Table S5 (`tab:supp-y15-retrospective`), and the protocol states that labels are authoritative. Not changed: two code comments in the kit (`predictions.py` lines 4 and 98) still say "S14.3". These are non-functional, and editing them would change the kit identity, so they are left; this sentence records it. Sections 1–9 of this report keep their original text except the corrections marked above. | protocol header, §1, §4, §5, §11; this report §3, §4 |

New MS hashes (after these edits): `plan/Y19_PROTOCOL_2026-10-12.md` and
`plan/onthefly/AGENT_PROMPT_Y19_HOST_2026-10-12.md` are listed in `ZENODO_PACKAGE.md` (computed after this report
was final; this report's own hash is there too).
