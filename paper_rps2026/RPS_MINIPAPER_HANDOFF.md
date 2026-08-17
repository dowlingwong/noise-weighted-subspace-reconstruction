# R/PS mini-paper handoff

**Last rewritten:** 2026-08-08, after the architecture-ladder restructure
(governing plan: `plan/PLAN_00_MASTER_RESTRUCTURE_2026-08-08.md`).

## Deliverable

The anonymous R/PS 2026 short-paper draft is `paper.pdf`, built from `paper.tex`
with the official 2026 workshop style. The document currently has 9 PDF pages:

- pages 1–4: main content, sections 1–5 (Introduction; Covariance-weighted
  representation geometry; The architecture ladder; Evidence; Discussion and
  limitations), plus the rung-4 table float;
- page 5: remainder of §4/§5 text and the rung-4 table float (see "Page budget");
- pages 6–7: references;
- pages 8–9: optional technical appendix.

**Page budget status.** Main content currently renders at 5 pages. Text alone is
~4.3 pages; the rung-4 baselines table plus the tail of §4/§5 spill onto page 5.
This is expected to resolve when the `\PENDING{...}` placeholders (which occupy
~6 lines of main text and 2 table cells) are replaced by compact numbers, and
when the final rung-4 figure is budgeted. See `plan/MANUSCRIPT_IMPL_LOG.md` §7.

## Build

```sh
make
```

The submission artifact is the single generated `paper.pdf`. R/PS does not
permit additional submission files. Any code or data link added later must be
anonymized (see repository hygiene below).

## Structure

The paper is organized around one claim, already present as Eq. (1): a
reconstruction method is a pair `(Σ⁻¹, F)`. The restructured paper holds `Σ⁻¹`
fixed and walks `F` up four rungs:

| Rung | Architecture class `F` | Status |
|---|---|---|
| 1 | rank-one fixed template — optimal filtering | classical, §3 |
| 2 | unrestricted rank-`k` linear — CW-PCA / tied linear AE | Proposition 1 |
| 3 | Kronecker-separable band × frame — NFPA | Proposition 2 |
| 4 | factored attention (intra-band, then cross-band) | empirical, TIDMAD |

All four rungs live on one public dataset (TIDMAD, CC-BY-4.0) in one
representation (the STFT band × frame plane). **No DELight data, simulation,
physics, figure, number, or checkpoint appears anywhere in this paper.**

## Main claims and their status

1. Inverse covariance defines the reconstruction inner product under a linear
   Gaussian model (archived theory).
2. OF, CW-PCA, and the tied noise-aware linear autoencoder are projections in
   this metric (Proposition 1).
3. Raw MSE is likelihood-correct only for isotropic noise and can induce metric
   reversal (archived 30-seed synthetic demonstration in §4).
4. NFPA restricts the objective to Kronecker-separable band–frame subspaces and
   equals CW-PCA only when a CW-PCA minimizer is separable (Proposition 2).
5. Rungs 3 and 4 share a factorization: `range(U_f) ⊗ range(U_m)` vs intra-band
   temporal then cross-band attention — the paper's most valuable sentence.
6. **Rung 4 (TIDMAD) is NOT yet run.** The `T-mse` vs `T-chi2` paired contrast,
   its coarse scores, and the `T-chi2+OF` secondary arm are `\PENDING{...}`.
   Only the published baselines (FCNet 6.55, Transformer 4.18, PUNet 3.84, none
   1.1026, moving average 0.64), the archived linear Wiener (−1.8025), and the
   TIDMAD calibration diagnostics are stated, all archived.
7. NFPA's non-degenerate `C = 8 > k_c = 2` result is `\PENDING{Block A, 10
   seeds}`; the single-split `C = k_c = 2` run is an appendix implementation
   check only.
8. Covariance adequacy is instrument-dependent: a stationary-PSD model is
   adequate on the TIDMAD readout (`χ²/dof = 1.0002`) but fails its calibration
   check on public GWOSC strain (archived).

The authoritative claim list with per-claim controls is `notes/claim_audit.md`.

## Attribution (rung 4)

All three conditions from `PLAN_03_RUNG4_TIDMAD_2026-08-08.md` §5 are satisfied
in the text: (1) the architecture and OF-injection design are not claimed as
contributions; (2) the prior work is cited where the design is described
(`priorThesis`); (3) the `T-chi2+OF` arm is explicitly attributed where
introduced. **The `priorThesis` bib entry is a placeholder — its metadata must be
completed by the authors and reconciled with double-blind anonymity before
submission.**

## Repository hygiene

If an anonymized code repository is published, `src/scripts/TraceSimulator.py`
and the `artifacts/` tree (DELight property) must never appear in it — build the
anonymized repo from a clean subtree, not by pruning this one. `.gitignore` is
not sufficient once files are tracked; check `git log --all --name-only` before
publishing.

## Open items

- Block A (NFPA, 10 seeds) and Block B (white-noise control): **not run** —
  results are `\PENDING{EXPERIMENT_PLAN}`.
- Rung 4 (TIDMAD `T-mse`/`T-chi2`): **not run** — results are
  `\PENDING{PLAN_03_RUNG4_TIDMAD}`.
- Anonymized repository link (D5 fix 1): author action; verify the link resolves.
- The `\PENDING` placeholders must be replaced only by archived numbers.
- Rung-4 figure: slot reserved in §4; produced by Track C from its results.

Author names and affiliations should be added only to a camera-ready version.
Any preprint and concurrent-submission plan should be checked against the other
venue's policy; R/PS itself is non-archival.
