# MLST 13 September successor programme — close-out

**Completed 2026-09-17, deepthought.** No production experiment was launched. No protocol, threshold, tolerance or
outcome was changed. The manuscript (`99040f8`) and the experiment repository (`1d29658`) were read only and remain
clean. Nothing was written to `/ceph` (spare entries unchanged at 1,785). Nothing was pushed or published.

## 1. Checksummed transfer — `TRANSFER_VERIFIED`

`MLST_SUCCESSOR_TRANSFER_20260917.tar` — sha256 `fac5498b81aa9150ce136228548fbb447c70d9c816692e3d5261ee229657c5d8`,
859,248,640 bytes, **4,758 members**. Every member was hashed against its source file (0 mismatches), and the 4,636
members also in the return inventory match it (`TRANSFER_VERIFICATION.json`, `TRANSFER_CONTENTS.sha256`).

Preserved inside, as required:
* **All seven run roots** (both CRESST roots labelled intentionally partial), launcher stdouts, ledgers, cells and
  invocation logs.
* **All inventoried preflights**: T22 original receipt, T19 amended (152/0/0/0), rebuilt environment (136/0/0/0).
* **Both validator runs and the correction**: run 1 `STOP_REVIEW_REQUIRED`, run 2 `VALIDATED` (35/35), and the
  unified diff between them (`bundle-staging/validator-runs/`).
* **Freeze artifacts**: all four, with a package-specific binding index. Every completed confirmation cell in every
  selection was drawn after its freeze. The superseded wrong-method index is kept and labelled.
* **Environment records**: rebuild logs; the full file hash list of the rebuilt venv; the base-interpreter hash; the
  damaged-venv skeleton with mtimes; every launch-time `environment.json` compared with the receipt.
* **Supplied (`3c22548d…`) and accepted (`4dc49d5c…`) manifests**; T19 amendment patch and transition inventories;
  the 13 Sep audit archive; the 16 Sep records.
* **A git bundle of the accepted T19 amendment commit `56de23e`**, which existed only in a local `/var/tmp` clone.
  Verified, and restore-tested: it recreates the commit and reproduces patch `4a5b968c…`.

Two provenance findings surfaced while assembling this:
* **The 16 Sep deletion was blocked, not interrupted.** `/home` is nfs4: files open by T22 production became hidden
  `.nfs` files and could not be removed until production exited at 23:44:45 (reproduced on this mount). This is also
  evidence that production held original-environment modules to the end.
* **365 completed cells** (T19 20, T20 14, T21 310, CRESST archive 21) ran on 13 Sep from a same-version interpreter
  at a `/ceph` path that no longer exists. Their reductions reproduce on copies.

## 2. Scientific audit (`audit/SCIENTIFIC_AUDIT.md`)

Each selection was re-scored against its own frozen hypotheses:

| Selection | Supported | Not supported / not reachable | Descriptive and other |
|---|---|---|---|
| T19-S | H1 model, H2 bound (0/788 violations), H3 in-domain, H4 orders | `S_full` domain **not reachable** at allocated m | registered T19 prediction stays NOT MET |
| T20-S | H1, H2, H4 x_U (4/4 θ), H4 contrast at 3 θ | **H4 contrast at θ = 0.12** (5/6; 3.370 vs 2.984) | H3 brackets: 2/3 as designed, 1 unconfirmable in grid |
| T21-S | H2 overlap confound, H0 control | **H1 overall** (296/298; orthogonal arm) → loadings step **untested** | H1′ near-edge 9/16 orthogonal, all at ratio 1 |
| T22-S | H2 type-I (0/30), controls 5/5 | **H1 coverage 13/65 non-drift**; matched real-data comparison **infeasible** at 4–16 files | H3 withholding as expected; **H5 random-effects interval omitted from reduction** |
| CRESST-S (**intentionally partial**) | H1 exact reproduction (10/10 bit-identical), H3 refusal, H4 calibration as predicted | — | 350 robustness cells unexecuted (excluded) |

**Corrections to earlier reports**, including my consolidated report:
* T22's H1 scope is non-drift, so the count is **13/65**, not 15/69.
* T19's `S_full/beyond` is outside H3's scope, so it is **not** a negative finding.
* T21's H1 fails overall, so the loadings explanation is **untested**, despite the reducer's per-geometry label.
* T20's H3 had not been scored before.
* T22's H5 random-effects summary is missing from the reduction.

## 3. Exceptions diagnosed from existing outputs only (`diagnostics/`)

* **T22 coverage.** Standard-error miscalibration explains the failures: bootstrap SD ÷ empirical SD correlates **0.79**
  with coverage. All 10 under-covering scenarios have ratio < 1, and all 3 over-covering have ratio > 1.3. The two-stage
  interval is 0.73× the t-interval at 4 files and 1.24× at 16. Neither withholding nor bias explains the failures. A
  read-only computation of the omitted random-effects interval puts it in band in only 24/65 scenarios.
* **T20, θ = 0.12.** The observed contrast lies above the frozen prediction at **every** N (other θ show mixed signs),
  and the model was selected on a 2.7% CV margin. That is systematic mild misspecification, not optimisation error.
* **T21.** Every failure is at N = 128, high γ, at or below the BBP threshold. That is finite-size smoothing of an
  asymptotic formula; spikes predicted at zero recover 0.03–0.05.
* **CRESST calibration.** Under run-to-run variance, adding channels per run **lowers** coverage (0.44 vs 0.61 at 2 runs).

## 4. Follow-up (`followup/FOLLOWUP_PROPOSALS.md`) — proposals only

* **F1** add T22's H5 summary to the reducer (minutes, no new cells).
* **F3** T22 interval recalibration at few files (≈ 72 worker-h).
* **F4** file count needed for feasibility (≈ 67 worker-h, est.).
* **F5** T20 θ = 0.12 model and θ₁ bracket (≈ 2–8 h).
* **F6** T21 finite-size origin (≤ 7 h).
* **F7** T19 `S_full` reachability, m ≈ 2.7×10⁶ (≈ 3–6 h).
* **F8** parallel-safe ledger and a protected environment.
* **F9–F11** blocked on data: CRESST raw pulses, ≥ 4 independent runs, unexposed TIDMAD files.

Costs are from measured cell times (`followup/MEASURED_CELL_TIMES.txt`). Every new-compute item needs a new reviewed
protocol version with fresh seeds.

## 5. Manuscript evidence table (`evidence-table/MANUSCRIPT_EVIDENCE_TABLE.md` and `.csv`)

28 claims across the canonical build, each tied to a manuscript location, a successor result, an artifact hash, a
classification and a revision recommendation. All 45 cited line ranges were verified mechanically.

* **VERIFIED:** every CRESST main-text number — 21.6%, 1.6–62.0%, [10.3, 38.8], [17.4, 27.1], [12.1, 39.8], +3.5/−8.7.
* **NOT SUPPORTED:** the spike-and-loadings reading (C3). The manuscript says it "awaits an independent test"; T21-S
  was that test, and it did not pass. Its supplement line "the explanation that does fit is the cited spiked-covariance
  one" should be softened.
* **QUALIFIED:**
  * A2 — observable orders now prospectively confirmed.
  * A3 — an in-domain remainder-bound check now exists, and `S_full` is unreachable.
  * D3 — the 4-file two-stage interval needs a calibration caveat.
* **ADDRESSED:** the discussion's "what would settle these" items for covariance, crossover, excitation and T22
  calibration each correspond to an executed successor study.
* **Non-transferable:** T20-S results do not revise the pulse-proxy claims, by protocol.

## Self-corrections during this close-out (all fixed before delivery)

* The first freeze-binding index applied T19/T22's method to T20/T21 and wrongly reported them "not bound". It is
  superseded, and kept.
* A draft T20 diagnosis subtracted `a_ref` from a prediction that already included it. It was discarded, with a note in
  the corrected file.
* A file-hash listing broke on a filename containing a space; it was regenerated with null-delimited paths.
* Count errors fixed: "7 of 25" became 8 of 25 at 4 files, and "10 of 13 alternatives" became 9 of 10 under-covering.
* Two introduction line citations were off by several lines; they were fixed and all citations re-verified.
