# E2X+ protocol: plug-in certificate and plug-in crossing size (prespecified, version 2.1, 23 Oct 2026)

**Status.** Version 2.1. It was revised after the design review `E2XPLUS_PROTOCOL_REVIEW.md` (CHANGES REQUIRED; F1–F9, see §11) and the gate-1/3 review `E2XPLUS_KIT_REVIEW.md` (CHANGES REQUIRED; P1–P4, K1–K3 and minor items, see §12). It becomes binding at the freeze commit (§9). Before the freeze nothing in §3–§7 is run on the study namespace.

**Source.** The external critique of `paper1_main_20261021.pdf` puts this first: "Evaluate plug-in certificate and crossing predictions using estimated K, Σ". Design book §2; it extends E2-X of 15 Oct, which was never run.

**Character.**
- Prospective and descriptive.
- It fits no learner. It computes only:
  - sample second moments;
  - covariance estimates;
  - eigendecompositions;
  - partial traces;
  - population product optima of plug-in matrices;
  - first-order constants.
- It is prespecified on an internal, hash-stamped record, not registered on OSF.

## 0. Disclosure of what was seen before the freeze

1. **Population quantities.** All frozen population values (N⋆_pred, a, B_cert, c_U, c_F) are public in the manuscript evidence.
2. **The design reviewer's pilot.**
   - The reviewer ran 24 plug-in constants calls, oracle arm, G1 only (θ ∈ {0, 0.25}, N ∈ {400, 1250, 3200}, 4 seeds), in the separate namespace `e2xplus-REVIEWER-PILOT-20261023`. It also ran timing calls.
   - Reported values:
     - naive N⋆_plug: 364–385, 650–694 and 920–952 at θ = 0.25 for N = 400, 1250 and 3200;
     - debiased N⋆_db: 1,290–1,560 at N = 1250 and 3200, and 4,200–10,700 at N = 400;
     - at θ = 0, the debiased â_db ≤ 0 in 11 of 12 runs.
   - The reviewer also computed pilot certificate values but did not report them.
   - **The debiased arm (§4) was added after this pilot.** So the G1 θ = 0.25 / θ = 0 oracle results of both plug-in arms are partly anticipated. G2 and all estimated arms are not.
   - Naive values at θ = 0 in the same pilot: N⋆_plug ≈ 1.05–1.2 × N, and the naive rule chose F in 11 of 12 runs.
3. **Kit smoke run.** It used the fixture namespace `e2xplus-fixture` (3 seeds, 1 seed with constants, version-1 kit). Only its outcome letters were displayed (G1 θ = 0.25: A; G2 A: B, under the naive arm and version-1 rules); no cell values were read.
4. **Gate-1/3 reviewer activity.** It used only the fixture namespace (a 3-seed run, `summarize`, field names and match counts). It also ran synthetic `readings()` inputs and a determinism check by hash. Nothing was run on the study namespace.

## 1. Questions

From N signal-bearing records, and m noise records when Σ is not known:

- **Q1 (detection).** How far is the plug-in certificate B_cert(K̂) from B_cert(K)? From what N does it separate a nonseparable geometry from the separable control, against an **oracle-calibrated** null (§6, O2)?
- **Q2 (location).** How far is the plug-in crossing from the population N⋆_pred, and from what N is it within a factor of two? Two plug-in arms are co-primary:
  - **Naive:** N⋆_plug = (ĉ_U − ĉ_F)/â − 1. All three constants come from K̂ itself, where â = tr(P̂_U M̂) − tr(P̂_F M̂) is the **in-sample** gain on the same N records.
  - **Debiased:** â_db = â − (ĉ_U − ĉ_F)/N and N⋆_db = (ĉ_U − ĉ_F)/â_db − 1. This is a penalised in-sample comparison of the Takeuchi/AIC type: restrict iff N·â < 2(ĉ_U − ĉ_F), approximately. No novelty is claimed for it.
- **Q3 (decision).** Does the rule "restrict iff N < N⋆" agree with the decided sign of the fitted class contrast the paper already reports? Each arm's own N⋆ is applied to the same N records. The use is in-sample selection, which the paper does not claim (§3 l.76–80).
  - For the debiased arm the rule is: restrict iff â_db ≤ 0 or N < N⋆_db.

**Prediction for the naive arm, written before the freeze, from the paper's own first-order expansion (review F1):**
- E â = a(1 − 1/N) + (c_U − c_F)/N + o(1/N), so to first order N⋆_plug ≈ N·N⋆/(N + N⋆), that is, 1/N⋆_plug ≈ 1/N⋆ + 1/N.
- Consequences:
  - N⋆_plug < min(N, N⋆) to first order whenever a > 0, so the naive rule mostly chooses the unrestricted class.
  - The sampling inflation of ĉ_U (second order; about +10–25% in the pilot) can lift N⋆_plug above N where N ≪ N⋆, for example G1 θ = 0.12 at N = 400.
- **Mechanical criterion:** the prediction is *confirmed* iff, in at least 80% of the nonseparable oracle Q2 cells, the median of N⋆_plug / [N·N⋆_pred/(N + N⋆_pred)] lies in [2/3, 3/2]. Otherwise it is *refuted*.
- The decision consequence (how often the naive rule chooses F where the reference says restrict) is reported descriptively.
- If the naive arm fails Q3, that is the predicted in-sample optimism, not a finding against the certificate.

## 2. Generators (imported unchanged; SHA-256 pinned in `protocol.json`)

| Label | Generator | Geometries | Population reference |
|---|---|---|---|
| G1 | T25 / T20-S fixed spectrum. Population: `p1nl_dev_20261006.generator.SparseLatentModel(theta, p=1)`. Records: `t25_production_20260930.science.make_data`. C = 8, T = 32, d = 256, k = 6, (k_c, k_t) = (2, 3). Σ = Σ_t ⊗ Σ_c, ρ_c = 0.4, ρ_t = 0.7. | θ ∈ {0, 0.12, 0.25, 0.40} | N⋆_pred, a, B_cert from the pinned `y15_RESULT_20261012.json` (T20-S rows). c_U and c_F from the same constants path, pinned in `pinned/population_constants.json` and checked by T1. |
| G2 | Y19 pulse-template: `y19_20261012.generator.build(predictions.GENERATOR)`, `PopulationModel`, `draw`. C = 6, T = 64, d = 384, k = 6. | control (φ = 0), A (φ = 0.12), B (φ = 0.20) | `y19_20261012/predictions.json` |

**Records.**
- **G1:** `t25.make_data(c, seed, theta)`, with `c` equal to `production.json` except:
  - `namespace = "e2xplus-20261023/G1"`;
  - `N = [6400]`;
  - `m = [1024]`.
  
  These are nested prefixes of one pool per (seed, θ). The calibration pool is shared across θ for a seed, as in T25.
- **G2:**
  - Records: `gen.draw(g, "e2xplus-20261023/G2", seed, phi, n_max)` with n_max = max(6400, the largest Q2 N of the geometry): 11668 (A, control) and 6400 (B). Training rows are nested prefixes.
  - Calibration: `t25.rng("e2xplus-20261023/G2", seed, "calibration").normal(size=(1024, d)) @ g.L`. This is the only new data-generating line; it mirrors the T25 calibration stream and is shared across φ for a seed.
- **Seeds:** seed_i = `int.from_bytes(sha256(f"e2xplus-20261023/seed/{i}".encode()).digest()[:8], "little") % 10**12`.
  - i = 0…199 for Q1.
  - Q2 and Q3 use i < n_Q2.
  - **Independence of seed sets.** E2X+, T25 and ZC use three independent seed sets. The references are cell-level decided signs.

**Oracle information.** The ranks (k, k_c, k_t) are taken as known, as they are for every fitted learner in the paper, and the oracle arm uses the true Σ by design. No other population quantity enters the plug-in.

## 3. Metric arms and their estimands

| Arm | Whitening W | What the plug-in estimates |
|---|---|---|
| oracle | Σ^{-1/2} (`t25.powers`) | The paper's true-whitened quantities (same target as N⋆_pred) |
| kron256, kron1024 | Σ̂^{-1/2}, Σ̂ = `t25.estimate(pool[:m], C, T)["kron"]` | Σ̂ is separable, so the product class maps onto itself. The plug-in assumes W Σ W = I and scores in Σ̂⁻¹. Its target is therefore the crossing of a Σ̂-scored contrast, **not** the Σ⁻¹-scored T25 contrast. |
| full256, full1024 | Σ̂^{-1/2}, Σ̂ = `t25.estimate(...)["full"]` | Σ̂^{-1/2} is not separable, so the whitened product class is **a different class** from the physical product class T25 fitted. Descriptive only. |

The T25 estimators are used unchanged, including `regularize()`, the fixed 0.9·A + 0.1·(tr A/d)·I applied to the full estimate and to each Kronecker factor. Its bias is part of what is measured.

## 4. Estimators

- **Sample second moment.** Ŝ is the centred, divisor-N sample covariance of the first N records (`t25.moments`). One draw and one whitening per (generator, geometry, seed); all N and arms reuse them.
- **Plug-in moment.** Symmetrize A = W Ŝ W − I, eigendecompose, and set K̂ := V diag(max(μ̂, 0)) Vᵀ. Then M := K̂ + I, ℓ = max(μ̂, 0) + 1, and V and μ are in descending order. Record the clipped count and mass.
- **Duck-typed model** for TR09 `core.py` / Y19 `predictions.constants`. Fields: K, M, V, ℓ, μ, k, k_c, k_t, C, T, d, p = 1, κ = 3, n_sparse = k, `theta` (a label) and `g` (a start holder).
  - `g.U_c0` and `g.U_t0` are the top eigenvectors of the channel and time partial traces of the top-k projector of K̂: a feasible start. The oracle θ = 0 factors are never read.
- **Plug-in constants.** `predictions.constants(model, rng, n_random=6)` gives 8 restarts: HOSVD, the partial-trace start and 6 random. It returns â, ĉ_U, ĉ_F, B̂ = ĉ_F − ĉ_U + â, B_cert(K̂), the Hessian minimum, restart agreement, the naive N⋆_plug and its interval [(ĉ_U−ĉ_F)/â − 1, (ĉ_U−ĉ_F)/B_cert(K̂) − 1].
  - The rng is `t25.rng("e2xplus-20261023", generator, geometry, N, arm, seed, "restarts")`.
- **Naive arm decisions.**
  - available, N⋆_plug defined: F iff N < N⋆_plug, else U;
  - â ≤ 1e-9: F (ZC convention; expected never to fire);
  - B̂ ≥ 0 with â > 0, so the first-order contrast is positive at every N: U;
  - nonpositive gap, or Hessian not positive definite: no decision (counted against the rule).
- **Debiased arm.** â_db = â − (ĉ_U − ĉ_F)/N.
  - â_db ≤ 0: N⋆_db undefined and decision F.
  - Otherwise N⋆_db = (ĉ_U − ĉ_F)/â_db − 1, and the decision is F iff N < N⋆_db, else U.
  - If ĉ_U − ĉ_F ≤ 0: U.
  - Same no-decision causes as the naive arm.

## 5. Grids

| Output | G1 | G2 |
|---|---|---|
| Q1 (certificate only) | N ∈ {50, 100, 200, 400, 800, 1600, 3200, 6400}; 4 θ; 5 arms; 200 seeds | Same N grid; control, A, B; 5 arms; 200 seeds |
| Q2, Q3 (constants) | N ∈ {400, 800, 1000, 1250, 1600, 3200}; 4 θ; 5 arms; n_Q2 seeds | A: {729, 1458, 2917, 5834, 11668}; B: {302, 604, 1208, 2415, 4830}; control: their union; 5 arms; n_Q2 seeds |

## 6. Pre-declared outputs (descriptive)

| ID | Output | Definition |
|---|---|---|
| O1 | Certificate bias and spread | Per cell: mean, sd and 5/50/95% quantiles of B_cert(K̂) − B_cert(K) |
| O2 | **Oracle-calibrated** detection size | The null is θ = 0 (G1) or the control (G2), at the same N and arm; q95 is its 95% quantile over 200 seeds, and power(N) = P(B_cert(K̂) > q95). N_det is the smallest Q1-grid N at which power ≥ 0.9 **at that N and at every larger grid N**; ">6400" if none, with non-monotone power reported. This is an optimistic bound on a feasible test, because a practitioner has no oracle null. |
| O3 | Crossing accuracy, per arm (naive, debiased) | Per cell:<ul><li>median and IQR of N⋆/N⋆_pred (â_db ≤ 0 enters as +∞ and is also counted separately as `n_a_db_nonpositive`; +∞ is written as the string "+inf" in SUMMARY.json);</li><li>fraction of all seeds with N⋆/N⋆_pred ∈ [½, 2];</li><li>unavailable count by cause (for the debiased arm, "â_db ≤ 0" is reported as a cause);</li><li>median ĉ_U/c_U and ĉ_F/c_F;</li><li>median plug-in interval.</li></ul>For the estimated arms the ratio mixes Σ̂ error, shrinkage bias and a change of target (§3). |
| O4 | Decision agreement, per arm | Scored only where a fitted reference for the same contrast exists:<ul><li>G1 oracle, kron256 and kron1024, against the decided signs in `pinned/t25_contrasts.csv` rows `class/{oracle,kron_256,kron_1024}`. The `ci95` columns are the 864-contrast Bonferroni intervals of the T25 factorial (Table S8).</li><li>G2 oracle, against the Y19 decided signs (`pinned/y19_summary.json`), flagging the reference cells that contain optimization-limited records.</li></ul>For each scored cell report:<ul><li>agreement count out of n_Q2, with a Wilson 95% interval;</li><li>the agreement of the population N⋆_pred rule;</li><li>CV's agreement on G1 as a count out of 24 from ZC `descriptive_cells` (`cv_choice_F`). G2 has no CV reference.</li></ul>Separable cells (θ = 0, control) and nonseparable cells are reported separately. Undecided reference cells are listed and not scored. The full arms are not scored. |
| O5 | Diagnostics | Clipped eigenvalues, gap of K̂, restart agreement, Hessian minimum, unavailable counts, time per call |

## 7. Reading rules (fixed at the freeze; descriptive; per geometry and arm)

Notation: g ∈ {G1 θ = 0.12, 0.25, 0.40; G2 A, B}. N⋆_cell(g) = `round_half_up(N⋆_pred)`: 5241, 1221, 488, 2917 and 1208.

- **R-Q1(g).** N_det(g, oracle) ≤ N⋆_cell(g). All five N⋆_cell values lie inside the Q1 grid range (≤ 6400). For G1 θ = 0.12, N⋆_cell = 5241 is above the Q2 grid, so N_last = 3200 there.
- **R-Q2(g, arm, p)**, for plug-in arm p ∈ {naive, debiased}:
  - Let N_loc(g, arm, p) be the smallest Q2-grid N such that, at that N **and at every larger Q2-grid N**, at least 80% of seeds have N⋆/N⋆_pred ∈ [½, 2] and at most 10% are unavailable. "None" if there is no such N.
  - R-Q2 holds iff N_loc exists and N_loc ≤ N⋆_cell(g).
  - For the debiased arm, â_db ≤ 0 is treated as N⋆_db = +∞ (ratio +∞, outside [½, 2], not unavailable); its decision is F (§4).
- **R-Q3.** Descriptive only. No statement that a plug-in rule beats, matches or is worse than CV. The registered ZC R-C1 reading (MET) is unchanged.
- **Outcome per (g, p)** under the oracle arm. The order of strength is A > B > D > C.

  | Outcome | Condition |
  |---|---|
  | A | R-Q1 and R-Q2 |
  | B | R-Q1 only |
  | D | R-Q2 only |
  | C | Neither |

- **Main-text sentence** (the abstract uses the weakest outcome over the headline geometries G1 θ = 0.25 and G2 A).
  - The **debiased** arm gives the sentence.
  - The **naive** arm is reported as the confirmed or refuted first-order prediction of §1.
  - Templates (blanks filled by the export script):
    - **A:** "Computed from the training records against an oracle-calibrated null, the certificate flagged nonseparability from N = {N_det} records, before the crossing, and the bias-corrected plug-in N⋆ was within a factor of two from N = {N_loc}."
    - **B:** "The certificate flagged nonseparability from N = {N_det} records (oracle-calibrated), but the bias-corrected plug-in N⋆ did not locate the crossing reliably: at N = {N_last}, {P}% of seeds were within a factor of two (median factor {F}), so N⋆ is a population characterization, not yet a selector."
    - **D:** "The bias-corrected plug-in N⋆ was within a factor of two from N = {N_loc}, but the certificate did not separate the geometry from the separable control before the crossing."
    - **C:** "Neither plug-in quantity was reliable at the record counts studied."
  - Definitions of the blanks:
    - N_last is the largest Q2-grid N ≤ N⋆_cell.
    - {P} is the percentage of seeds at N_last with N⋆/N⋆_pred ∈ [½, 2], with â_db ≤ 0 counting as outside.
    - {F} is the median N⋆/N⋆_pred at N_last, or its reciprocal if < 1, with +∞ for â_db ≤ 0. It is printed with two decimals, or as "more than 10".
  - **Which geometry fills the blanks.** The headline geometry with the weaker outcome. On a tie, G1 θ = 0.25 fills the main-text sentence and G2 A goes to the supplement. The export fills the templates verbatim.
  - **Estimated arms.** One clause: "with estimated covariances the result {held | degraded}".
    - "Degraded" iff, for either headline geometry and either Kronecker arm (kron256, kron1024), N_det(arm) or the debiased N_loc(arm) is later than the oracle's.
    - "None" counts as later than any grid value. "None" for both the arm and the oracle counts as not later.
    - The full arms appear only in the supplement.

## 8. Compute, benchmark and seed fallback

- **Run host.** The Claude cloud workspace (2 cores), isolated from the HCP confirmation on Deepthought. The run uses the frozen kit tree.
- **Threads.** Results depend on the BLAS thread count (kit review K1). `run` refuses to start unless `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`, and records them in START.json. Recomputation in the receiving review uses the same setting.
- **Fail-closed run.** `run` requires `KIT_SHA256.json` and a passing `verify-kit`. Every module of the loaded import closure is pinned.
- **Errors.** An exception or a non-finite â in one constants call is recorded as cause `exception` or `nonfinite`, and both arms make no decision. The unit is still written.
- **Benchmark.**
  - Single-threaded per worker (`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`), 2 workers.
  - It times whole units, Q1 included, per generator. Review F6 estimated about 5.2 h at n_Q2 = 100.
- **Seed rule.**
  - n_Q2 = 100 if the projected total is ≤ 6 h.
  - Otherwise n_Q2 = 50.
  - If 50 projects above 10 h, STOP.
- **Run once.** The runner writes one atomic JSON per (generator, geometry, seed index) holding all N and arms. A completed record is never recomputed. A crash resumes. A completed run is never repeated.

## 9. Gates (in order; each is a STOP if it fails)

1. **Protocol.** This version 2.1 is accepted by the gate-1/3 reviewer (re-review of F1–F9 and P1–P4, K1–K3).
2. **Kit.** `experiments/mlst2026/e2xplus_plugin_20261023/` on EXP branch `claude/e2xplus-20261023`, with tests:
   - **T1.** With K̂ := K, the constants path reproduces the frozen N⋆_pred of G1 (3 θ) and G2 (A, B) to relative 1e-6, and B_cert(K) to 1e-9. The pinned population c_U and c_F are reproduced to 1e-9.
   - **T2.** θ = 0 and the control give B_cert(K) = 0.
   - **T3.** A hand-checked 2 × 2 Kronecker case of the partial-trace cap.
   - **T4.** The plug-in never reads oracle factors:
     - `PluginModel.g` is the start holder;
     - with `Geometry.U_c0`/`U_t0` and `G2Geometry` patched at class level to raise on access, a plug-in constants call succeeds.
   - **T5.** The G2 calibration stream is reproducible and distinct from the signal and noise streams.
   - **T6.** Seed and rng-key strings are distinct across namespaces, and no seed equals any T25, ZC or Y19 seed recorded in EXP.
   - **T7.** A smoke run on `e2xplus-fixture` produces every output field for both plug-in arms.
   - **T8.** Every imported and pinned file matches `protocol.json`.
   - **T9.** The debiased and naive decision functions follow §4 on hand-built inputs, covering each cause.
   - **T10.** `readings()` on synthetic boundary cases: grid-cell bounds, monotone N_det and N_loc, outcomes A/B/D/C, the template B blanks including +∞, the naive-prediction criterion, and the degraded rule.
   - **T7** also runs a G2 unit.
3. **Kit review.** A fresh-context reviewer is given the protocol and the kit.
4. **Freeze.**
   - A local commit with `KIT_SHA256.json`, `protocol.json` and this file's SHA-256.
   - An annotated tag `e2xplus-registration-20261023`.
   - The SHA-256 manifest is recorded in MS.
   - An OTS stamp is an author action, because no OpenTimestamps client is available here. Until then the time is the git commit time only, which is disclosed.
5. **Run** (§8).
6. **Receiving review.** A fresh-context reviewer checks:
   - the records against the freeze;
   - that `summarize` is reproducible;
   - five random records recomputed exactly;
   - that the outcome and text are selected mechanically.
7. **Integration.** Only through an export script into `evidence/data/e2xplus_20261023/`.

## 10. Limitations stated in advance

- Gaussian signal and noise. The plug-in uses Gaussian fourth moments (κ = 3) in c_F.
- Two synthetic generators.
- Oracle ranks.
- The certificate's detection is calibrated against an oracle null (O2).
- For the estimated arms the plug-in target differs from the fitted T25 contrast (§3).
- K̂ is clipped at zero.
- 8 restarts against 64 for the frozen truth.
- The debiased arm uses a first-order correction whose own constants are estimated: ĉ_U is inflated by sampling.
- Q3 is an in-sample selection use the paper does not claim.
- No real data.

## 11. Changes from version 1 (review F1–F9)

| Finding | Change |
|---|---|
| F1 | Debiased co-primary arm (§1, §4). Naive-arm first-order prediction stated (§1). TIC/AIC naming. Pilot disclosure (§0). |
| F2 | Estimands per arm (§3). O4 restricted to oracle and kron (G1) and oracle (G2). Full arms descriptive. |
| F3 | Oracle-calibrated detection size (O2, templates) |
| F4 | Per-geometry rules. Grid-cell bounds. Outcome D. Blanks defined. Monotone N_det and N_loc. Mapping for both arms. |
| F5 | Decision table (§4) matches the kit, including B̂ ≥ 0 → U and â_db ≤ 0 → F |
| F6 | Single-threaded whole-unit benchmark. One draw per unit (already so in the kit). |
| F7 | Duck-typed model text corrected (§4) |
| F8 | Reference verification stated. Counts with Wilson intervals. Separable and nonseparable reported separately. G2 scored for oracle only. Optimization-limited reference cells flagged. |
| F9 | Exact seed formula. T4 and T6 strengthened. Oracle ranks stated. |

## 12. Changes in version 2.1 (kit review P1–P4, K1–K3)

| Finding | Change |
|---|---|
| P1 | The naive prediction is restated with its second-order caveat. A mechanical confirmed/refuted criterion on N⋆_plug / [N·N⋆/(N+N⋆)] is added. |
| P2 | â_db ≤ 0 is treated as N⋆_db = +∞ for location. Template B uses {P} and {F}, both always defined. |
| P3 | Rule for which geometry fills the blanks. Aggregation for "degraded", including the none/none case. Templates filled verbatim by the export. |
| P4 | T10 tests `readings()`; T7 covers G2 |
| K1 | Single-threaded BLAS is enforced and recorded |
| K2 | `run` requires `KIT_SHA256.json` and `verify-kit`. A `write-manifest` command is added. The freeze happens in the EXP git repository on the author's Mac. |
| K3 | The full loaded import closure (experiments, src and package `__init__`) is pinned |
| Re-check R1/R2 | O5 tolerates None constants (nonfinite records) and T11 passes error records through summarize. O3 reports `n_a_db_nonpositive`. SUMMARY.json is strict JSON. T10b covers unequal headline outcomes and the G2 B boundary. |
| Minor | T1 c_F tolerance 1e-9 relative; G2 n_max stated; T6 uses the real data-key strings; §0 completed; per-call exception policy; non-finite â has no decision; κ = 3 stated |
