# TIDMAD Stage Runbook

One command runs the whole stage: `scripts/run_tidmad_stage.py`. This page is
what to type, in what order, and what to do when something fails.

Predeclaration: [`../results/TIDMAD_PREDECLARED_GATES.md`](../results/TIDMAD_PREDECLARED_GATES.md) — read it before running anything on real data.
Roadmap: [`../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`](../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md).

## Quick reference

```bash
# Everything that needs no data: phases 1, 2, 3, 4, 6 in synthetic mode (~15 s)
uv run python scripts/run_tidmad_stage.py --mode synthetic

# See what the real run would do, touch nothing
uv run python scripts/run_tidmad_stage.py --mode file --dry-run \
    --data-root /ceph/dwong/paper1_dataset

# The real run, from an empty data root
git clone https://github.com/jessicafry/TIDMAD ~/src/TIDMAD
uv run python scripts/run_tidmad_stage.py --mode file --download \
    --data-root /ceph/dwong/paper1_dataset \
    --tidmad-checkout ~/src/TIDMAD

# Re-run one phase after changing something
uv run python scripts/run_experiment.py \
    --config configs/tidmad/phase2_metric_reversal.yaml
```

## What the driver does

| Step | Runs | Failure is |
| --- | --- | --- |
| download | `scripts/download/download_tidmad.py` → upstream `download_data.py` | hard |
| phase0_contract | `scripts/tidmad_phase0_contract.py` | hard |
| phase0_G0_baseline | upstream `inference.py` + `benchmark.py` for `none` and `savgol` | **hard** |
| phase1…6 | `scripts/run_experiment.py` per config | soft (a finding) |
| archive | copies run records into `evidence/tidmad/<run-id>/` with SHA-256 | — |

**Hard vs soft is the important distinction.** A hard failure is an
infrastructure problem: the data is wrong, the channels are ambiguous, or the
published baseline cannot be reproduced. The driver stops, because a stage built
on an unreproducible baseline manufactures a result rather than measuring one.
A soft failure is a gate that returned a negative *finding* — that gets recorded
and the run continues, because a negative finding is evidence.

The driver is idempotent: every step skips if its output exists. Use `--force`
to re-run. A 2 GB download should never have to restart because a later step
failed.

## Order of operations on real data

1. **Download a small subset.** Two validation files (they carry the injected
   reference channel — the ground truth) and two science files (signal-free
   noise for PSD estimation). That is everything Phases 0–4 and 6 need; the full
   release is ~TB and is only required for Phase 5's full scoring.

2. **Establish the data contract.** `tidmad_phase0_contract.py` classifies the
   SQUID and reference channels from spectral evidence — the reference channel
   is line-dominated, the SQUID channel broadband — and **refuses to guess**
   when they are not separable. Resolve any ambiguity by hand against the
   release documentation before continuing. A wrong channel assumption would not
   crash; it would silently exchange data and truth in Phase 2.

3. **Gate G0.** The driver captures the upstream `none` and `savgol` scores
   verbatim into `evidence/tidmad/phase0/G0_baseline_reproduction.json`.
   **Compare them against the published values by hand**, record the verdict in
   the predeclaration, and only then set `verified: true` in
   `docs/tidmad_data_contract.json`. The loader refuses to run on an unverified
   contract, which is deliberate.

4. **Fill in the remaining predeclaration fields** — injection frequencies and
   amplitudes, confirmed sampling rate — and re-freeze the document. Nothing
   below may be adjusted after this point.

5. **Run the phases.** Phases 1–4 and 6 need only the contract; Phase 5 also
   needs `--tidmad-checkout`.

## What to expect, and what it means

| Outcome | Reading |
| --- | --- |
| Phase 1 G1b degrades on real noise | Expected. 8-bit quantization and glitches are not in the Gaussian model. Record how much — it previews Phase 4. |
| Phase 2 G2 fails | A real scope limitation on the claim's reach. Report it; do not change the operating point until it passes. |
| Phase 3 knee is not at 2 | Suspect a convention error first (frequency grid, phase handling) and re-verify Phase 1 before treating it as a finding. |
| Phase 4 synthetic control fails | **Halt.** This is a bug in our code, not a property of the data. |
| Phase 4 global/local models fail on real data | Expected, and it *is* the result — the in-domain replacement for GWOSC's boundary lesson. Report `cost_of_mismatch` alongside. |
| A Phase 6 control does not degrade | A finding to publish. CRESST had exactly one such case and reported it. Do not tune it away. |

## Outputs

- Run records: `results/metrics/TIDMAD_P*.json` (gitignored).
- Archived evidence: `evidence/tidmad/<run-id>/` with `manifest.json` carrying
  SHA-256 per file, the git commit, and the per-step status.
- Write-ups: [`../results/TIDMAD_PHASE1_RESULT.md`](../results/TIDMAD_PHASE1_RESULT.md),
  [`../results/TIDMAD_PHASES_2_TO_6_RESULT.md`](../results/TIDMAD_PHASES_2_TO_6_RESULT.md).

## Troubleshooting

**`h5py is required`** — install the extra: `uv sync --extra tidmad`, or
`pip install h5py`.

**`TIDMAD data contract is not marked verified`** — working as designed. Finish
step 3 above.

**`no files matched 'tidmad/raw/abra_validation_*.h5'`** — the download either
did not run or wrote elsewhere. Check `--data-root`; the expected layout is
`<data-root>/tidmad/raw/`.

**`upstream benchmark not found`** — pass `--tidmad-checkout`. There is no
fallback scorer, deliberately: an internally reimplemented score would look
comparable to the published scoreboard without being comparable to it.

**Download stalls** — the release's downloader takes `-c NY|NorCal|SoCal|Director`;
the default Director cache is documented upstream as sometimes unstable. Pass a
specific cache to `scripts/download/download_tidmad.py` and re-run.
