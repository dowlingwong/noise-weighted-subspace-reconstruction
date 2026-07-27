# Open Issues — require author verification before arXiv submission

1. **Main-text length is ~31 pages (incl. references), below the 35–45 target.**
   All required content is present; the shortfall is density, not missing material.
   Knobs if you want to land inside the band: raise `\linespread` to 1.25,
   widen margins to 1.25in, or promote Appendix E's symmetry theorem statement
   back into §5. I recommend leaving it compact rather than padding.

2. **Legacy Appendix F provenance (release blocking).** The duplicated
   `1029.2 / 1028.9 / 1028.6` tables and the other untraceable quantitative
   panels were removed from the revision candidate. Restore them only from a
   checksum-backed record containing the exact configuration, seeds, metrics,
   and code revision.

3. **Amplitude correlation r = 0.9942 vs. old pass threshold r > 0.999.**
   The old text asserted all metrics were "well within pass thresholds", but the
   stated threshold for r was not met. The threshold claim was removed in the
   rewrite; verify whether 0.9942 has a known explanation (e.g., real-data event
   selection) worth a sentence in Experiment A.

4. **Detector naming: DELight (intro) vs. LAMCAL (Appendix G) vs. "AL" spectra
   (Appendix G).** Confirm the experiment/detector names and that you are allowed
   to publish the noise spectra and Study A tables.

5. **Real K-α dataset provenance.** Experiment A says "K-α calibration pulses from
   a cryogenic microcalorimeter"; add instrument, run, and selection details (or a
   citation) if available.

6. ~~**TIDMAD Phase 2 primary result is not in the evidence bundle
   (stop-ship).**~~ **RESOLVED 2026-07-27.** Root cause confirmed exactly as
   diagnosed here: `configs/tidmad/phase2_metric_reversal_real_noise.yaml`
   and `phase3_rank_manifold_real_noise.yaml` were both mid-investigation
   repointed at `abra_validation_0010.h5` for a diagnostic detour, while
   still sharing the file-`0000` primary run's output path
   (`results/metrics/TIDMAD_P{2,3}_real_seed20260727.json`), silently
   overwriting it before the bundle was synced. Fixed:
   - the true file-`0000` result was restored from checksummed archive
     (`evidence/tidmad/20260727T063119Z/`, verified against its own
     `manifest.json` SHA-256 before restoring) into
     `results/metrics/TIDMAD_P{2,3}_real_seed20260727.json`;
   - the file-`0010` diagnostic run was preserved (not discarded) under its
     own distinct filenames,
     `TIDMAD_P{2,3}_diagnostic_0010_seed20260727.json`, with a new config
     pair (`phase{2,3}_..._diagnostic_0010.yaml`) so it can be regenerated
     without touching the primary run again;
   - both `*_real_noise.yaml` configs were pointed back at
     `abra_validation_0000.h5`, matching their own docstrings;
   - `transfer_paper/scripts/refresh_bundle.py` now has a
     `_check_tidmad_file_identity` guard that raises loudly if any of the
     six TIDMAD Phase 2/3 result files (real / diagnostic-0010 /
     confirmatory) ever again fails to match its expected source file, so
     this cannot silently recur.

   `data/tidmad/current/TIDMAD_P2_real_seed20260727.json` now correctly
   shows the file-`0000` headline run (`+0.0190`, CI `[0.0045, 0.0321]`
   window-level, `[0.0027, 0.0344]` block-level, both excluding zero) —
   matching what `TIDMAD_REVISION_INSTRUCTION_2026-07-27.md` and
   `MANUSCRIPT_EVIDENCE_MAP.md` always cited. **The §7.4 placeholder can be
   replaced and the abstract's ground-truth metric-reversal sentence can be
   added** — re-sync this bundle into the manuscript build and re-check.

7. **Three appendix figures are missing (pre-existing, unrelated to TIDMAD).**
   `appG_simulation.tex` includes `transfer_paper/figures/studyA_v2_{quadrature_validation,
   overview,fractional_contributions}.pdf`; only the non-`_v2` variants exist,
   so pdflatex emits three draft boxes. Confirm whether the `_v2` files should
   be synced or the includes repointed to the existing files.

8. **Main text is ~12,300 words** against the July 12 instruction's 7,000–8,000
   target. TIDMAD added ~800; the bulk predates it, in the theory sections
   §2–§6 that the July 12 instruction asked to compress into the supplement.
   That compression has not been done.

9. **Corrective evidence is partial.** S5 and Study A now map to the
   `paper1-revision-20260726-partial-v1` archive. Unsupported legacy Appendix F
   values and the old Study B table were removed. The manuscript must not be
   called an arXiv revision candidate until the manifest's missing-evidence
   list is empty or the author formally removes those studies from scope.

7. **Fisher-information endpoint convention.** Appendix G now states the
   one-sided synthesis powers at DC, interior bins, and Nyquist. Independently
   verify Appendix A's information formula against the canonical OF weights,
   including the repository's legacy Nyquist factor, before closing this issue.

8. **`algorithm`/`algpseudocode` packages removed** because the build environment
   here lacked them; the two pseudocode blocks are now plain enumerated lists
   (Appendix A.6 and C.2). arXiv's TeX Live has these packages, so you may restore
   floating algorithm environments if you prefer.

9. **Uncited bib entries.** `wurgaft2026manifold`, `johnstone2001distribution`,
   `paul2007asymptotics`, `kolda2009`, `dempster1977maximum`, `roweis1998algorithms`,
   `fixsen1993pulse` (still cited), `oja1982simplified`, `cappe2009line`,
   `kingma2013auto` (still cited) — with `unsrt` style uncited entries are simply
   dropped; no action needed unless you want them cited.
   Also: `wurgaft2026manifold` is dated 2026 — verify it is a real, citable entry
   if you re-introduce it.

10. **Unused image files** remain in `img/` (all) and
    `figures/paper2_nfpa_demo_results.png`, `figures/figure_placement.md`.
    Decide whether to delete before uploading the source to arXiv.

11. **Author metadata**: affiliation and email are placeholders from the draft
    (`KIT`, `dowling.wong@kit.edu`); `\date{\today}`. Confirm before submission.

12. **MMC reference spectrum**: Appendix G retains the draft's note that its
    physical normalization "has not been independently verified" — resolve or keep
    the caveat deliberately.

13. **ε_ph ≃ 1 meV** and other DELightSignalFormation constants are described as
    "code conventions"; confirm the values quoted in Appendix G match the released
    simulation version.

14. **Assumption wording.** Equivalence claims now consistently use "under matched
    preprocessing, whitening, weighting, and normalization." If there are
    additional matched conventions in the notebooks (e.g., DC-bin handling,
    detrending), add them to the theorem assumptions.

15. **Study B metadata (release blocking).** `signal.dat` is magnitude-only and
    its calibration, sidedness, and physical interpretation are not independently
    documented in the current archive. The analogue acquisition/anti-alias
    transfer function is also unavailable. Freeze these inputs and run both
    `inband_no_alias` and `unfiltered_alias_fold` branches before restoring a
    numerical sensitivity table.
