# Follow-up proposals — for review only; nothing launched

Each item follows from an audited exception or limitation (`audit/SCIENTIFIC_AUDIT.md`). None changes an existing
outcome. Any new compute requires a **new, reviewed protocol version with fresh seed namespaces**. Confirmation seeds
already used must not guide design (every protocol's no-retuning rule).

**Cost basis:** measured single-worker wall time, one BLAS thread, deepthought (`followup/MEASURED_CELL_TIMES.txt`).
Extrapolations are marked **(est.)** and should be re-measured on a fixture/pilot before any cap is set. Serial times
assume the current one-worker-per-ledger constraint on CephFS (see F8).

## Priority 0 — no new experiments

**F1. Complete T22's predeclared H5 reporting.**
* *Question:* how does the random-effects interval's coverage and length compare with the primary, as H5 requires?
* *Inputs:* reviewed reducer amendment adding the random-effects summary (`re_lo`/`re_hi` are already in every stored
  record); re-reduce a **copy** of the existing root.
* *Cost:* minutes; zero new cells.
* *Note:* a read-only computation already gives 24/65 in band, median 0.876. The amendment would make it part of the
  reproducible reduction rather than a supplementary post-hoc number.

**F2. Manuscript revision against the evidence table** (`evidence-table/`).
* *Cost:* author time only.

## Priority 1 — new protocol versions (simulation; data available)

**F3. T22-S v3: calibrate the equal-file interval at few files.**
* *Question:* does a pre-registered small-sample-corrected interval achieve nominal coverage across the frozen non-drift
  scenario grid? Candidates must be chosen before seeing new data, e.g. a studentised or degrees-of-freedom-adjusted
  hierarchical bootstrap.
* *Why:* H1 fails in 13/65 non-drift scenarios, and SE miscalibration tracks coverage (r = 0.79): too narrow at 4–8
  files, too wide at 16.
* *Inputs:* new protocol with the candidate interval(s) frozen; development on development seeds only; fresh
  confirmation seeds; same 65 scenarios and tolerances.
* *Cost:* n4 25×10×203 s + n8 15×10×322 s + n16 25×10×561 s ≈ **66.5 worker-hours**, plus pilot ≈ 5.6 h →
  **≈ 72 h (≈ 3 days serial)**; ≈ 5 h wall at 16 workers if F8 is resolved.

**F4. T22-S: how many independent files make a matched comparison feasible?**
* *Question:* at what file count do medium-effect power ≥ 0.5 **and** withholding ≤ 0.5 hold under the frozen
  feasibility rule?
* *Why:* the rule fails at 4, 8 and 16 files (medium power 0.02–0.04; withholding 0.94–0.96).
* *Inputs:* new protocol extending `n_files` (e.g. 32, 64) for the medium-effect scenarios (φ ∈ {0, 0.3, 0.6, 0.85},
  τ ∈ {0.01, 0.03}); best run after F3, since coverage must be in tolerance too.
* *Cost (est.):* cell time grows with files (203 → 322 → 561 s), extrapolating ≈ 1,000 s at 32 and ≈ 2,000 s at 64;
  8 scenarios × 10 cells × both sizes ≈ **67 worker-hours**.
* *Caveat:* simulation can only show what a real comparison would need. It remains blocked on data (F11).

**F5. T20-S: resolve the θ = 0.12 contrast-model exception and the unlocated θ₁ bracket.**
* *Question:* (a) is the systematic positive residual at θ = 0.12 a misspecified scaling family? (b) where does the
  predicted above-core-grid sign change for θ₁ occur?
* *Why:* the frozen model was selected on a 2.7% CV margin; no sign change was observed up to N = 3200.
* *Inputs:* new protocol; candidate models pre-registered from pilot-phase reasoning only; extended grid
  (e.g. N = 6400, 12800); fresh pilot and confirmation seeds.
* *Cost (est.):* cells ≈ 9–10 s at N ≤ 3200; assuming cost ∝ N, ≈ 20–40 s at 6400–12800. θ = 0.12 only
  (8 N × 48 seeds) ≈ **1.5–2 h**; all four θ ≈ **6–8 h**.

**F6. T21-S: finite-size origin of the orthogonal-arm failures.**
* *Question:* does orthogonal-arm agreement recover at larger d and N (fixed γ), and does a pre-registered, wider
  near-edge band remove the ratio-1 disagreements?
* *Why:* both primary failures and all 7 near-edge failures are at high γ, N = 128, at or below the BBP threshold.
  Sub-threshold spikes predicted at 0 recover 0.03–0.05.
* *Inputs:* new protocol with d scaled (e.g. 256 → 1024) at the same γ arms; the near-edge band pre-registered before
  new data.
* *Cost (est.):* cells now 0.1–0.2 s; PCA cost up to ~d³ gives ≈ 64× → ≈ 10 s/cell; full 2,499-cell matrix
  ≈ **7 worker-hours** (upper bound).

**F7. T19-S: reachability of the theorem domain for the full sample covariance.**
* *Question:* at what calibration size m does `S_full` reach η ≤ 0.1 on G1, so that H2 can be checked on that arm?
* *Why:* the domain is not reachable at the allocated m. Median η falls as m^(−1/2) (0.454 at 131,072; 0.226 at
  524,288), extrapolating to η = 0.1 at **m ≈ 2.7×10⁶**.
* *Inputs:* new protocol allocating m ≈ 3–4×10⁶ for `S_full` only. The protocol's rule forbids enlarging m after the
  fact in v2, so this must be v3, labelled as a reachability extension.
* *Cost (est.):* covariance formation ~m·d² ≈ 2×10¹¹ flops/fit → minutes per fit; 100 seeds ≈ **3–6 worker-hours**.
  Measure one pilot cell first.

**F8. Infrastructure: parallel-safe execution and a protected environment** (engineering; no science cost).
* *Why:* T22 took 80 worker-hours serially because the ledger's advisory `flock` is certified on local disk only.
  Separately, the pinned interpreter in `$HOME` was deleted mid-run by an unknown actor.
* *Inputs:* reviewed launcher amendment for a CephFS-safe ledger, or per-worker local roots with a reviewed merge;
  interpreter placed outside user-cleaned paths with a checksum-verified preflight.
* *Cost:* engineering review; turns F3 from ~3 days into hours.

## Blocked on data (not computable here)

**F9. CRESST full robustness** (excluded in this round).
* *Question:* is the archived channel effect robust across rank / shrinkage / time-cut settings?
* *Inputs:* raw pulse/noise/time arrays in `cresst-s-pulse-root/1` format with verified provenance and independent
  splits, plus a validated release-to-schema adapter.
* *Cost:* compute is trivial; the package's fixture-scaled estimate for all 350 cells is ≈ 7 CPU-minutes, unvalidated
  for real pulses. **The blocker is data and adapter validation.**

**F10. CRESST new-run or new-detector claims.**
* *Question:* can any interval cover a new-run effect?
* *Why:* calibration shows 0.44–0.61 coverage at 2 runs under between-run spread, only 0.58–0.82 at 8 runs; more
  channels per run does not help.
* *Inputs:* more independent runs — the package requires ≥ 4 runs to report any between-run component — or a different
  construction pre-registered and calibrated first.
* *Cost:* data acquisition; re-calibration compute is minutes.

**F11. T22 real-data matched comparison** (excluded in this round).
* *Inputs:* a frozen matched design; the number of **independent, unexposed** instrument files F4 shows to be feasible;
  raw data; the release file list (an unresolved input in the protocol).
* *Note:* the protocol states that a matched rerun on already-exposed data is not independent evidence.

## Summary

| ID | Kind | Question (short) | Blocker | Cost |
|---|---|---|---|---|
| F1 | reporting | T22 H5 random-effects summary | reviewed reducer amendment | minutes |
| F2 | writing | manuscript revision | — | author time |
| F3 | new protocol | T22 interval calibration, few files | review | ≈ 72 worker-h |
| F4 | new protocol | T22 file count for feasibility | review; after F3 | ≈ 67 worker-h (est.) |
| F5 | new protocol | T20 θ=0.12 model; θ₁ bracket | review | ≈ 2–8 worker-h (est.) |
| F6 | new protocol | T21 finite-size origin | review | ≤ 7 worker-h (est.) |
| F7 | new protocol | T19 S_full domain reachability | review | ≈ 3–6 worker-h (est.) |
| F8 | engineering | parallel-safe ledger; protected env | review | engineering |
| F9 | data | CRESST robustness | raw pulses + adapter | ≈ 7 CPU-min |
| F10 | data | CRESST new-run claims | ≥ 4 independent runs | data acquisition |
| F11 | data | T22 real-data comparison | matched design + unexposed files | data acquisition |
