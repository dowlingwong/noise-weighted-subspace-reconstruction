# TR03: X3 representation endpoints S1/S2 (descriptive)

**Status: implemented (v2, 6 Oct, after review round 1 findings R2/R3 and the author's overlay
decisions); independent re-review pending.** Not yet run on any confirmation data.

Interpretive overlay: MS `plan/onthefly/X3_ADDENDUM_OVERLAY_v1_2026-10-06.md`. The overlay is
versioned separately; the registered addendum and the frozen receiving package keep their bytes.

Registered definition: MS `plan/onthefly/X3_ADDENDUM_REPRESENTATION_ENDPOINTS.md`, committed in MS
`8bad592` (6 Oct 2026, 12:23 UTC). The script checks that the addendum bytes are unchanged
(SHA-256 `37bad503…5491`). Label for any later use: *registered before the run, outside the
reviewed package, descriptive.*

- `representation_endpoints.py` computes S1 (subspace distance d and the largest principal angle to
  P_Σ and to P_iso) and S2 (leakage beyond the top k = 6 whitened output directions) for every
  selected X3 model. It reduces them to medians and IQRs over the 24 seeds per (class, loss, N), plus
  the seed-paired difference d_Σ(Σ⁻¹ loss) − d_Σ(isotropic loss) per (class, N).
- It imports the frozen X3 package (`x3_nonlinear_metric_20261005`, code identity `83360c2a…`)
  read-only. It does not change the package, its 54-contrast family or the P1/P2/SURE verdicts.
- It adds no intervals, tests or decision rules. `expectations.json` gives the descriptive status of
  the registered expectations E1–E3. E1/E2 are evaluated only on complete summaries. The addendum's
  section-5 qualitative sentence ("approximately the covariance-optimal subspace") is **withdrawn**
  by the overlay; no sentence gate is computed. It never supports a superposition claim from leakage
  alone.

## Order of use (phase 4, after the X3 return)

1. Receive the X3 return: verify the outer `SHA256SUMS`, extract `x3_return.tgz` into a fresh
   directory, and run the frozen portable audit (`reviews/production_20261005/x3/INTEGRATION_KIT.md`).
2. **Record the primary verdicts first.** Copy the P1, P2 and SURE-agreement verdicts from the
   export's `reduction.json` into the receiving record. Write them as JSON, for example
   `reviews/x3_receiving_<date>/primary_verdicts.json`:
   `{"P1": "...", "P2": "...", "SURE_agreement": "..."}`.
3. Only then compute S1/S2:

```sh
cd ~/Documents/noise-weighted-subspace-reconstruction
export PYTHONPATH=$PWD/src:$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=~/x3_env/bin/python        # the pinned torch 2.9.1 environment used for X3
R=~/Documents/arXiv-v1-polished-extended-preprint/reviews/x3_receiving_<date>
$PY -m experiments.mlst2026.x3_representation_addendum.representation_endpoints confirmation \
    --export <extracted x3 export dir> --primary-verdicts $R/primary_verdicts.json --out $R/tr03_endpoints
```

The output directory must not exist yet. It receives `per_seed.csv`, `summary.csv`, `paired.csv`,
`expectations.json`, `checks.json` and `RUN.json`. On refusal it receives only `REFUSED.json`, and
the command exits with an error.

Runtime: on the fixture export (3 seeds × 2 N × 6 models, D = 12) the whole run takes 1.4 s. The
confirmation export has 72 cells × 6 models with D = 256 and 2048 records. Each model costs one
forward pass and one 256 × 256 eigendecomposition, and each seed's data are regenerated twice. The
expected runtime is a few minutes on one thread; this has not been measured at confirmation size.

## What is checked before any endpoint is computed (refusal otherwise)

- The export's `SHA256SUMS`, `EXPORT.json` and `tables/` pass the frozen `workflow.check_export`,
  and the run ledgers and cell inventory pass `workflow.check_ledgers`.
- The `EXPORT.json` code identity, the freeze sources' identity and the identity of the frozen
  source on disk all equal the registered `83360c2a…`.
- The configuration is exactly the registered one: `BASE` (k = 6, 2048 evaluation records) in
  confirmation mode, or `FIXTURE` in fixture mode. The export status is `CONFIRMATION_EXPORT` or
  `DEVELOPMENT_EXPORT` accordingly.
- Each cell's `COMPLETE.json` matches its files, and the record's (seed, N) matches the enumeration.
- Every seed's training and evaluation data are regenerated with the frozen `science.make_data`.
  Both digests must equal the record. The stored closed-form tied-linear maps must equal their
  recomputation (rtol 1e-9, atol 1e-11, as in the frozen verifier). Every model record and every set
  of deep-model weights must be present.
- The registered addendum bytes are unchanged.
- No in-repository module outside this package, the X3 package and the byte-identical accepted T25
  closure is loaded.
- Confirmation mode only: `--primary-verdicts` is required and must equal the reduction's verdicts.

## Interpretations and declared choices (overlay v1; none changes the registered addendum bytes)

1. **"The two largest principal angles" (addendum §3).** Read as the largest principal angle to
   each of the two references, consistent with §2. Every per-seed row also carries all k angles to
   both references, so the other reading can be summarised from the same file.
2. **When U_f is undefined.** Only where the registered top-6 eigenspace is not unique, up to
   floating-point resolution τ = 64·ε·D·ν₁ (≈ 3.6e-12·ν₁ at D = 256; a numerical-resolution
   tolerance, not a substantive threshold). Each case keeps an explicit reason:
   - `zero_output_variance`: tr C_f ≤ 0; every endpoint is undefined.
   - `rank_deficient`: fewer than k eigenvalues exceed τ, so the zero eigenspace makes the top-k choice
     arbitrary. Leakage stays defined.
   - `unresolved_tie`: ν_k − ν_{k+1} ≤ τ. Leakage stays defined.

   A small but resolved gap is **not** excluded. d and the angles are reported, with the relative gap
   (ν_k − ν_{k+1})/ν₁ and the numerical rank as diagnostics in `per_seed.csv`. This corrects review
   finding R2: the earlier 1e-8·ν₁ cutoff treated ill-conditioning as non-uniqueness.
3. **Incomplete summaries.** Undefined values are never imputed. A summary over fewer than all
   registered seeds is labelled `INCOMPLETE available-case summary (n/24 …)`, carries `n_defined` and
   the `undefined_reasons` counts, and is never presented as the complete 24-seed result. E1/E2 are
   evaluated only when both compared groups are complete; otherwise they are
   `not_evaluable_incomplete` (review finding R3).
4. **Section-5 wording.** The qualitative sentence is withdrawn (author decision, 6 Oct). Report the
   registered distances, angles and leakage measures and the relative comparisons E1/E2. No absolute
   threshold for "approximately" is introduced.
5. **Strict data digests.** The frozen portable audit only *notes* a data-digest mismatch, which can
   occur when data are regenerated under a different BLAS. This script refuses with a `BLOCKED`
   reason.
   - Preferred route: run it on the original X3 host with its pinned environment, after the frozen
     audit and the primary verdicts are recorded.
   - If parity still fails, report TR03 as **blocked**. Never weaken the check.
6. **IQR.** Q3 − Q1 with numpy's default linear percentile interpolation; Q1 and Q3 are also reported.
7. **Linear-class check.** |ℓ| < 1e-8 for the tied-linear and deep-linear models is reported in
   `checks.json`. As registered, it is a check and not an endpoint; a failure is reported, not hidden.

## Tests

```sh
python -m pytest -q -p no:cacheprovider experiments/mlst2026/x3_representation_addendum/tests
```

See the test file for the current count (v2 adds the review-R2/R3 cases). They cover:

- hand-built distance and angle references;
- descending eigenvalue order;
- the reference subspaces against an independent construction (the addendum's Σ^{1/2}(K + I)Σ^{1/2}
  formula, orthonormalised by SVD rather than QR);
- zero leakage for affine maps of rank 1, 3 and 4 (= k);
- positive leakage for a nonlinear map;
- rows as records, and centring invariance;
- invariance under an orthogonal change of whitening;
- refusal of a non-whitening matrix;
- the zero-variance and non-identifiable cases;
- the fitted Σ⁻¹ tied-linear map's output subspace against the fitted CW-PCA projector (and the
  isotropic map against fitted PCA);
- the reducer's medians, IQRs, seed pairing, duplicate and missing seeds, and undefined values;
- a complete X3 fixture-namespace lifecycle, built in a child process with the frozen workflow;
- refusals on fabricated corruptions: a flipped byte, missing checksums, a changed export identity
  (re-checksummed), an unexported run, a changed addendum hash, an existing output directory, a data
  digest mismatch, a changed closed form, missing weights, a missing model record, mismatching recorded
  verdicts, a development export in confirmation mode, the confirmation verdict gate, and a module
  outside the closure.

The tests use only the X3 fixture namespace (`paper1-x3-v1/fixture`) and synthetic inputs.
