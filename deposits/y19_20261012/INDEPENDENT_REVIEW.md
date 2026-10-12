# Y19 stage B1 — independent protocol review

**Date:** 12 October 2026. **Role:** independent reviewer (fresh context; did not build the kit).
**Machine:** author's Cowork Linux VM (aarch64, Python 3.10.12, numpy 2.2.6, scipy 1.15.3 / pytest 9.1.1 /
h5py 3.16.0 from `/tmp/pylib`). Read-only git (`GIT_OPTIONAL_LOCKS=0`); no file in either repository was
changed except this report. Scratch: `/tmp/y19rev/` (own scripts `indep.py`, `rule_test.py`; outputs).

**Material reviewed.** EXP `experiments/mlst2026/y19_20261012/` (all 16 files), MS
`plan/Y19_PROTOCOL_2026-10-12.md`, `plan/onthefly/AGENT_PROMPT_Y19_HOST_2026-10-12.md`,
`reviews/y19_preparation_20261012/REPORT.md`, `ZENODO_PACKAGE.md`; context: runbook §2/§4,
`supp_successor.tex` S14.3 (lines 112–175), `03_restriction.tex` Eq. `mlst-zero-bound`, TR09 `core.py`,
`t25_production_20260930/science.py`, T25 weight table and Y15 `RECOMPUTATION.json`.

## Verdict: **ACCEPTED WITH MINOR CHANGES**

The science of the kit is sound and I reproduced every registered number independently to ≤ 2e−13
relative. Prospective integrity is intact. The runner is the T25 oracle arm, and the reading rule is
implemented exactly as written. **No kit change is required, so the kit identity
`bbf4e8c6…2bfe8` can stand.** Two required changes are in the MS documents only: F1 (host commit
silently drops `results.h5` and all NPZ arrays) and F2 (role of the separable control in the overall
reading). Once they and the recommended wording changes are made, recompute the protocol-MD and
host-prompt hashes in `ZENODO_PACKAGE.md` (F6) before freezing and uploading.

## Findings

| # | Severity | Location | Finding | Required / recommended fix |
|---|---|---|---|---|
| F1 | **Moderate (required)** | MS `plan/onthefly/AGENT_PROMPT_Y19_HOST_2026-10-12.md:107`; EXP `.gitignore:4` (`*.h5`), `.gitignore:56` (`*.npz`) | `git add experiments/mlst2026/y19_20261012_return` silently skips ignored files. `git check-ignore --no-index` confirms that `results.h5` and every `arrays/<label>/seed*_N*.npz` (480 files: fitted bases, restart factors, mean) are ignored. Only the JSON records, `summary.json`, logs and the host report would reach the remote. That contradicts runbook §2.4 (results committed on host; no rsync), and §5 of the prompt asks for the SHA-256 of a file that would not be committed. The frozen reading is still reproducible from the 480 JSON records, so no science is lost, but evidence would be. | Use `git add -f experiments/mlst2026/y19_20261012_return`. Before committing, check `git status --porcelain --ignored experiments/mlst2026/y19_20261012_return` shows nothing ignored. After committing, require `git ls-files experiments/mlst2026/y19_20261012_return \| wc -l` = 480 JSON + 480 NPZ + `START.json`, `ATTEMPTS.jsonl`, `RUN_LOG.txt`, `summary.json`, `results.h5`, `HOST_REPORT.md` (+ copied logs), and record the count in the host report. |
| F2 | **Moderate (required, wording only)** | MS protocol §7 lines 133–135; `protocol.json` → `reading_rule.overall`/`control`; runbook §4 lines 134–136 | The runbook's met condition includes "The separable control must show no positive contrast". The protocol instead makes the overall reading depend only on A and B, with the control "reported alongside". The code implements the protocol (`runner.py:332`). A decided-positive control (a = 0, so first order predicts D < 0) would point to a pipeline or optimization defect, yet an overall MET would still be reported. | Add one sentence to protocol §7 saying how a NOT MET control qualifies the overall reading (e.g. "if the control is NOT MET the overall reading is reported as '… with failed separable control' and no confirmation is claimed"). Alternatively, record the author's decision that the control is descriptive only and why. Keep the machine rule unchanged (no kit change). |
| F3 | Minor | MS protocol §11 lines 178–180 (and §4 line 91) | The "N ≲ 1000" local-domain limit comes from S14.3, which was written for T25/T20-S (Δ_k = 4, d = 256). G2 has half the gap (Δ_k = 2) and d = 384, so the limit is not calibrated for G2 and is plausibly higher. N\*_B = 1208 was chosen just above 1000 by the selection rule (§4), so B's N\* cell itself sits at the inherited edge. For reference, T25 θ = 0.25 (N\* = 1221) was where the Y15 strict reading failed (S14.3 lines 217–222). | State in §11 that the 1000 threshold is inherited, not G2-specific, and that B's formal zero lies at its edge. Optionally, note that the Y19 rule is the fixed-grid analogue of Y15's "decided-cells" reading (factor-2 bracket [N\*/2, 2N\*]). |
| F4 | Minor | MS protocol §11 (missing); disclosed only in REPORT §4 line 91 and §9.5 line 176 | φ is steep. a(φ) is quadratic near 0 (a = 0.0078, 0.031 at φ = 0.01, 0.02), so N\* scales roughly as φ⁻². N\* moves 3414 → 2529 over φ = 0.11 → 0.13 (dN\*/dφ ≈ −4.4e4). K(φ) is also only piecewise smooth: the double-exponential onset kink crosses a sample when φ·δ_c reaches an integer. The first breakpoint is φ = 0.40, visible in the scan as a 20× drop in da/dφ between 0.40 and 0.41. Both are harmless for the registered φ = 0.12 and 0.20, which lie in the smooth region, but the protocol itself should carry the disclosure. | Add a §11 caveat bullet: steep φ dependence; predictions hold only at exactly φ = 0.12 and 0.20; the result says nothing about robustness to mis-specified geometry. |
| F5 | Minor | MS protocol §4 lines 88–94 | "Rule … fixed before any fit" is true. But REPORT §1 (line 28) and §4 (lines 83, 94) show the rule was written after exploratory population-only scans (φ ∈ {0, 0.5, 1, 2, 3}, 0.05–0.3, 0.11–0.22). No fitted quantity entered (verified, see C1), so integrity is not affected; the order of events just isn't stated in the protocol. | Add one sentence to §4: "the rule's limits were set after exploratory population-only scans (REPORT §4); no sampled or fitted quantity was available." |
| F6 | Minor | MS `ZENODO_PACKAGE.md:15,19`; protocol lines 5 and 190 (placeholders); MS `git status` | (a) Item 1's hash `1d9088f9…` covers the protocol MD with `<FREEZE_COMMIT>` placeholders and without the F2–F5 edits. Say whether the placeholder version is uploaded (commit given in the deposit metadata) or recompute after filling it in. (b) The deposit is not self-contained: the 35 pinned dependencies (`protocol.json` → `dependencies`, e.g. `t25_production_20260930/science.py`, `tr09_first_order/core.py`, X3 `guard.py`) are only referenced by hash. (c) The protocol, host prompt, runbook and prep report are untracked in MS (`??`), so "the matching MS commit" in the host prompt (line 11) does not exist yet. | Recompute the MD and prompt hashes after edits. Extend item 5 with `git archive <FREEZE_COMMIT>` of the dependency paths (or the whole EXP tree). Commit the MS Y19 files at freeze (runbook §2.2). |
| F7 | Minor | MS host prompt lines 12–13, 30–31, 49, 69–72 | Hygiene. (a) Allowed writes omit `~/y19_host`, `~/y19_venv`, `~/y19_run_std*.txt` and the pytest temp dir. (b) Gate 4 has no command; suggest `git ls-remote origin 'refs/heads/y19-return*'` and `git log --all --oneline -- experiments/mlst2026/y19_20261012_return` (both must be empty). (c) `runner summarize --partial` lets an agent read interim readings mid-run; it has no decision role, so the prompt should forbid it during B2. (d) After `nohup … &`, check stderr and `START.json` within ~1 min for `KitMismatch` (exit 3). (e) Copy the stdout/stderr logs into the return dir before committing. (f) Unpinned `pip install` is guarded by the 1e−6 preflight. Grid rounding margins are ≥ 0.021 samples (A N\*/2 = 1458.479), far above numerical noise, but pinning numpy 2.2.6 / scipy 1.15.3 is still preferable. | Edit the prompt accordingly. |
| F8 | Nit | REPORT line 75; protocol line 50 | REPORT says the pulses "all stay inside the 64-sample window". In fact the slow pulse (5, 35) is cut at t = 63 at 42% of its peak (0.261/0.620) and the medium pulse at 7.5%. The templates are defined as sampled and truncated, so this is not a validity issue. Protocol "checked to 3e−15": the control residual is 1.6e−14 (A/B 3.3e−15). | Correct the wording. |
| F9 | Observation (no change) | EXP `runner.py:424–450, 499–502` | Run-once is enforced procedurally, not in code. Repeating `run` after completion only re-summarizes (no new draw; deterministic). `summarize` does not call `verify_kit`. Acceptable given protocol §9 and the host prompt. | None (B3 should re-verify the kit before reading). |

## What I verified clean

### C1. Prospective integrity
- No G2 sampling or fitting path was exercised. `generator.draw` and `runner.fit_record` are called only from `runner.run_group` (confirmation), `benchmark.py` (generator `family: 't25'`, geometry seed 81328948, namespace `y19-benchmark-nonG2`; `/tmp/y19dev/bench*.json` confirm) and the smoke path (`SMOKE_GENERATOR` family `t25`, C = 3, T = 4; `/tmp/y19dev/smoke2` records have φ ∈ {0, 0.3, 0.6} and pool 160).
- The development scripts in `/tmp/y19dev` (`val.py`, `g2.py`, `scan.py`) compute population quantities only (`core.population_product_optimum/first_order/summary` on K(φ)).
- No result or cache files exist in the kit dir (no `__pycache__`, `.pyc` or `.pytest_cache`, before or after my test run).
- No `y19_20261012_return` exists anywhere on the VM. EXP `git status` shows only `?? experiments/mlst2026/y19_20261012/` (and unrelated `paper_rps2026/`). EXP HEAD is `6ac4fbc`, as recorded in `predictions.json` provenance.
- The namespace string `y19-confirm-20261012` occurs, outside the kit, only in `/tmp/y19dev/g2.py`, as the population geometry key. No registered seed integer appears in any JSON under `experiments/mlst2026`.
- φ was chosen by `PHI_SCAN` from population N\* only. `predictions check` re-derives φ_A = 0.12 and φ_B = 0.20 (see F5 for the ordering disclosure).

### C2. Independent recomputation (own code `/tmp/y19rev/indep.py`; generator used only to build K(φ), Σ, L)
Method:
- Own eigendecomposition of K.
- B_cert = Δ_k(k − min{k, s_c, s_t}) from own partial traces of P\*_U (C-order `[t,c,s,d]` reshape).
- c_U from the S14.3 sum.
- **a_best by my own alternating partial-trace eigen-maximisation** (not Newton): HOSVD + φ = 0 factors + **300 random starts** = 302 starts per geometry.
- c_F = ½ tr(H⁻¹Σ_g), with H and Σ_g = 2 tr(G_α M G_β M) from my own analytic second derivatives in an exponential chart with rotated bases. My complement basis differs from `core.complement`; c_F is chart-invariant at a stationary point.
- Finite-difference check of H through `scipy.linalg.expm` charts: 100 random entries, max abs error 3.7e−6 (|H| entries 14–73).

| Geometry | quantity | mine | `predictions.json` | rel. diff |
|---|---|---|---|---|
| control φ=0 | a / B_cert | −1.1e−13 / 0 (s_c = s_t = 6 to 2e−15) | 1.4e−14 / 0 | abs 1.3e−13 / 0 |
| | c_U / c_F / H_min | 3097.356277 / 256.2445431 / 16.1438359 | same | 1.5e−16 / 5.5e−15 / 3.6e−14 |
| A φ=0.12 | a_best | 0.9729888846 | 0.97298888456 | 1.7e−13 |
| | c_F / B_cert | 258.2164758 / 0.2603328962 | same | 1.6e−13 / 2.1e−15 |
| | **N\*** / interval upper | **2916.957077** / 10904.80500 | 2916.957077 / 10904.80500 | 1.5e−13 / 1.7e−14 |
| | H_min / H_max | 15.4197733 / 73.3474034 | same | 2.1e−12 / 8.7e−13 |
| B φ=0.20 | a_best | 2.346655910 | 2.346655910 | 4.5e−14 |
| | c_F / B_cert | 261.2000586 / 0.6200993182 | same | 4.5e−14 / 1.0e−14 |
| | **N\*** / interval upper | **1207.594837** / 4572.712848 | 1207.594837 / 4572.712848 | 5.0e−14 / 1.4e−14 |
| | H_min / H_max | 13.8541769 / 73.2034107 | same | 3.7e−12 / 1.2e−12 |

- All 302 of my starts reached the same maximum at every geometry (spread ≤ 2.9e−13), agreeing with the kit's 64/64. Gradient at the optimum ≤ 4e−10. B = c_F − c_U + a < 0 at A and B, and B_cert ≤ a.
- Grids follow from half-up rounding: A [729, 1458, 2917, 5834, 11668], B [302, 604, 1208, 2415, 4830], control = union of 10. N\*_A/N\*_B = 2.4155.
- The protocol §3 first-order means a + B/N were rechecked: A −2.92, −0.97, 0, +0.49, +0.73; B −7.04, −2.35, 0, +1.17, +1.76; control −9.41 … −0.24.
- **Validation gate re-verified with my own code on the T25 generator** (`SparseLatentModel`, p = 1; 22 ALS starts): a = 0.538362419969 / 2.30737180868 / 5.75719310540, c_F = 187.6646906 / 188.2852282 / 190.3739211 / 194.7932479, N\* = 5240.955504 / 1221.162439 / 488.051484, B_cert = 0.226806420 / 0.972106933 / 2.425383539. All agree with the weight table (SHA-256 `b26e3dd4…`, verified) and Y15 `RECOMPUTATION.json` (`4e2074d1…`, verified) to ≤ 1e−13. `tests/t25_reference.json` matches both sources value for value. The kit's own gate passes (inside the test run).

### C3. G2 correctness
- Spectrum equals my own law (24/i; 14/j) exactly (max dev 0). eig(K(φ)) matches it to ≤ 2.8e−14 at φ = 0, 0.12, 0.20, so it is invariant across φ.
- Σ equals my own kron(S_t(0.7), S_c(0.4)) exactly. It is SPD (λ_min = 0.0795); L² = Σ and L L⁻¹ = I to 4e−15.
- φ = 0 is exactly separable: B_cert = 0, P\*_U equals the product of its partial-trace projectors to 2.2e−14, and a ≈ 1e−13.
- Δ_k = 2 > 0, and H ≻ 0 at all three geometries.
- With templates rebuilt by my own code from the protocol text, the physical top-k subspace L·U_k equals the delayed-template span to ≤ 1.4e−14. The design therefore matches protocol §2 (onset 16; (τ_r, τ_d) = (1, 6), (2.5, 15), (5, 35); q₁ = exp(−(c−2.5)²/8), q₂ = (c−2.5)/2.5; δ_c = c − 2.5; Ω = (1, .7, .45) ⊗ (1, .6)).
- K₀ eigenvalues stay distinct (min ratio 1.33) on [0, 0.45], with no column reordering. The φ = 0.40 kink is the onset-sampling breakpoint (F4).

### C4. Runner
- `fit_record` is line-for-line the oracle branch of `t25.science.fit_cell`: `moments`, `W = inv(Σ)`, `starts` keyed (namespace, seed, φ, N), `structured(…, 6000, 1e−8)` (4 restarts), `unrestricted(W, S, 6)`, `map_for`, `b = (I − A)·mean`, `risks(A, b, L K L, Σ, Σ⁻¹)['clean']`, `representation`, restart clean risks. The optimization-limited rule (nonstationary selected restart or spread ≥ |D|) matches `reduce_records`. Data streams `t25.rng(ns, seed, φ, 'signal'|'noise')` match `make_data`'s key structure; the calibration stream is omitted, as stated.
- Sign: D = R_F − R_U (product − unrestricted) = T25 `class/` contrast.
- Seeds: 24 SHA-256-derived seeds reproduce exactly and are disjoint from T20-S, T25, TR09 (`tr09_20261007/{confirmation,fixture,benchmark}.json` present) and E-NL.
- Resumability: per-record atomic JSON/NPZ, `START.json` binding, prefix pools per (geometry, seed); determinism is covered by a test.
- Inference: t_{1−0.05/40, 23} = 3.39309 (scipy), family 20.
- **Reading rule tested on synthetic inputs against the real `protocol.json`** (`/tmp/y19rev/rule_test.py`), all as written:
  - '- - ? + +' → MET; '- - + + +' and '- - - + +' → MET (N\* cell unconstrained).
  - '- ? ? + +', '- - ? ? +', '- - ? + -', '+ - ? + +' → NOT MET, and overall NOT MET when either geometry fails.
  - Zero-SD cell → undecided → NOT MET.
  - Upper limit exactly 0 → undecided; just below 0 → negative.
  - Control with one decided positive → control NOT MET, overall unchanged (see F2); all-undecided control → MET.
  - Optimization-limited deciding cell → flag `[5834]`, reading unchanged.
  - Half-width = 3.3931·sd/√24, to 5e−17.

### C5. Protocol, host prompt, package
- Claims and non-claims (§11) are clear. They include the S14.3 sign caveat, Gaussian/oracle-only scope, the local-domain caveat for cells below N ≈ 1000, N = 302 < d = 384, the best-found optimum and NFPA optimization error.
- The deviations policy (§9) matches runbook §2.3.
- The host prompt has hash gates, run-once wording, resume-only semantics, return committed on the host branch and no reruns (subject to F1, F7).
- **SHA-256 recomputed and all match:** `KIT_SHA256.json` `8d8ac844…09d9`; `protocol.json` `a4ecaea1…771e`; `predictions.json` `329a0cac…6996`; all 15 per-file kit hashes; protocol MD `1d9088f9…0709`; host prompt `d6b66453…4e09`; REPORT `98759263…f6f1`.
- Kit identity recomputed independently as SHA-256 of `json.dumps(files, sort_keys=True)` = `bbf4e8c6…2bfe8`. `runner verify-kit` → identity `bbf4e8c6…`, 15 files, dependencies ok.

### C6. Tests and checks run by me
- `python -m pytest experiments/mlst2026/y19_20261012/tests -v -p no:cacheprovider` (threads = 1, `PYTHONDONTWRITEBYTECODE=1`, basetemp in /tmp): **14 passed in 41.94 s**.
- `python -m experiments.mlst2026.y19_20261012.predictions check` (full φ scan): ok = true, grids equal, max diff 0.0, φ selection re-derived.
- `runner verify-kit`: rc 0.
