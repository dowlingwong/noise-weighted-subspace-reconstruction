# MEG R1 kit — R2-A implementation

Additive EXP branch `codex/megz-r1-kit-20261014` from
`bfe1139d3923f3b38915c3637696d7afbbf65641`. The DESIGN rev 2.3 (RD-12) CONFIG is
encoded verbatim in `protocol_r1.json`; CONFIG SHA-256 is
`35a892276b298cc349434b0907bc3e79ab962a3038f9b0ea178c42bde56d1c47`.
This development kit is not a frozen confirmation protocol. R2-A never runs the
controls or real-data preprocessing/gates/prediction/fits.

## Scope, inputs and interpretation

The author explicitly authorized development events.tsv and recording-length
metadata reads on 14 October 2026 after the initial input-geometry STOP. The
metadata exporter reads FIF headers with `preload=False` only for n_times/sfreq,
never signal samples, STI101, filtering or epochs. `DEV_GEOMETRY.json` retains
exact rational sample origins from `(TSV sample - 37)/1100 + 0.0345`, band lengths, response-excluded contiguous
stretches and final gaps. BAD annotations are not applied to synthetic controls,
matching the R0 upper-bound census convention. R3 applies the actual BAD mask.

`partition.py` is byte-identical to the reference script (SHA-256
809672165ee76d009f478eef0180478fe281834273aa01ef56f265d8e80fb8d7).
Its historical census excludes final gaps; runtime plans include every onset's
gap. Adding a final gap changes no existing role. Train, cal and eval windows sit
on the fixed onset-locked grid of band offsets 0, T, 2T, ... from the gap's first
band sample (DESIGN rev 2.3, RD-12a; R2-B F1): a grid window that intersects a
response exclusion or a BAD sample is dropped and the grid is not restarted, so
position k is a fixed latency after the onset, as in the registered kit. The
reference keeps stride 1 inside allowed stretches. Backward position (rposition)
mirrors position within the full, untruncated band (8 slots at T = 10; R2-B F6),
so a final gap cut short by the recording end keeps the mirror. The two
alignments are computed and counted once.

Channel selection and spatial templates use evoked_T=20 from runs 1–2; only
noise windows and the temporal signal basis use T=10. R3 asserts the newly
selected channels against each sealed D1 list, before any G1 retest drop.
R2 cannot recompute real evoked fields under its data-read restriction; its
channel tests validate that assertion and the selection path on fabricated data.

Control law and cell indices are 1–7 and 0–12; subject and replicate indices are
0–3 and 0–199. All five SeedSequence entropy integers are recorded. Laws L6/L7
scale variance using lognormal factors (multiplying amplitudes by their square
roots); factors are shared by every role, reference included. Their top group
uses generator population eigenvectors. C2–C4 use the replicate's own S_ref
vectors; C5/C6 use its spatial partial trace. C12 uses the loudest generator
population direction, as review 03's probe did.

C7/C8 use the review probe's random physical CxT profile, normalized to 4x/2x
the energy threshold under S_ref, at eval position 0 (onset-locked under the
fixed grid). C9 (descriptive; RD-12e, R2-B F5) adds that same profile to the
continuous eval band stretch at band offset 0 + Uniform[0, 200 ms] (one delay per
eval gap, linear interpolation on the 91.667-Hz clock, zero outside its support)
before the windows are cut, so delays spill into positions 1-2; each fixed-grid
window receives its slice of that stretch.

Run-blocked descriptives compare each run with all other runs, pooling train +
cal + eval windows only (no reference, no guard), identically in the controls and
in R3 (RD-12d, R2-B F4), using G2-S alone; they are not endpoints. Every role is
centred at its own mean.
D3-S draws exactly 50 reference estimates; callers can pass multiple models to
reuse each estimate across theta rows. Its nominal a_hat is from M=K+I;
D_W uses learners under K+N and risks under K-N. The general-noise crossing is
the disclosed S4.3 analogue, never a gate. Preflight conservatively charges
seven full 64-start rows per estimate (four certificate grid rows plus three
fit geometries), including the nominal and actual general paths. Actual R4
signal templates and theta roots are unavailable in R2; timing uses fabricated
signal factors with the sealed generator shapes and real window geometry.

## Reused old-kit code (imported unchanged)

- `data_manifest.run_stem`: canonical snapshot paths (metadata export and R3).
- `preprocess`: preprocess_config, load_run, epochs, select_channels, cut,
  subject_payload, write_h5, load_subject and TriggerCheckError. R2 uses only
  fabricated-data assembly/serialization tests; load_run is reserved for R3.
- `common`: vec, symmetric, sha_file, atomic_json, sha_json.
- `estimators`: moment_stats, Kronecker flip-flop from S, whitened_error,
  nearest_kronecker, MomentStats and ledoit_wolf_from_stats (retired
  descriptives only); no old shrunk reference is used as the R1 metric.
- `gates`: g1_spd, noisiest_channel, g2_stationarity (retired LW
  descriptives), g5_mardia and lag_one_dependence (unchanged limits/once-only policy).
- `signal`: orth, spatial_templates, base_variances and SignalModel. Temporal
  basis generation is new and independent of evoked_T.
- `population`: general_row, spd_powers, GeneralProductClass, b_dagger and its
  imported core. All transitive old dependencies retain their pinned bytes.
- `runner`: kit_hashes and verify_kit for the old sealed identity. The new runner
  checks the old dependency file hashes without treating additive modules as
  members of the old kit's import closure.

No old file is edited. `verify-kit` checks old identity
f121776c80c3cd424f1bd14dc1360e8c7930de6f840b34140190f1108e8acdfb plus its
pinned dependency hashes, then the complete additive kit manifest.

## Host commands

```bash
E='cd /ceph/dwong/megz_r1_host_20261014 && . /ceph/dwong/meg_env/bin/activate && export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src && unset PYTHONPYCACHEPREFIX'
eval "$E"
K=experiments.mlst2026.meg_z_r1_20261014
python -m pytest experiments/mlst2026/meg_z_r1_20261014/tests -q -p no:cacheprovider
python -m $K.runner verify-kit
python -m $K.runner census-check
python -m $K.validate_host --out experiments/mlst2026/meg_z_r1_20261014_r2bfix/VALIDATION.json
python -m $K.runner preflight --derived /ceph/dwong/meg_z/derived_timingfix_20261013 --gates experiments/mlst2026/meg_z_20261013_dev_timingfix/gates.json --out experiments/mlst2026/meg_z_r1_20261014_r2bfix/PREFLIGHT.json
```

`preflight` and `validate_host` refuse an existing output path, so returned
evidence (`meg_z_r1_20261014_preflight/`, `_optimization/`, `_rd11/`) is never
overwritten; each new identity writes to a new folder (R2-B F8).

R3 CLIs are `python -m $K.preprocess` and `python -m $K.gate_cli` (tested but
not run on real data in R2). The campaign (`runner controls`) requires an R2-B
ACCEPTED review, the exact reviewed HEAD and identity, an explicit `--preflight`
path to a controls-only preflight carrying that identity (R2-B F8), and a new
output directory; it runs four workers. Start only when separately authorized, inside private
`tmux -L paper1megzr1`, session `r1-controls`; inspect with capture-pane only.
`summarize-controls` refuses incomplete/duplicate returns. Any failed endpoint
returns STOP; never tune, rerun under changed seeds, or alter a gate/law/cell.

## Design-to-code map

The table below maps every CONFIG dictionary key to a concrete code line and
unit test. Text-only provenance/disclosure fields are exact snapshot contracts,
not executable rules. `DESIGN_TO_CODE.json` provides the same mapping for tools.

The historical R0 census uses rounded TSV onset values; production timing uses
exact stable stimulus samples. `metadata_census_audit.py` preserves all count
differences. No historical census or budget is rewritten. Exact metadata
geometry and every response/band boundary are tested.

Preflight projects the control campaign against its own 6 h x 4-worker cap and
D3-S against its separate R4 budget (16 workers, 8 h wall; RD-11). Each D3-S
timing row runs under a deadline derived from the R4 D3-S budget and the stated
conservative seven-row plan (8 h x 16 workers / (4 x 50 x 7) = 329 s); a censored
row records a cost lower bound and STOP, without changing statistical functions
or starts (R2-B F10).

Each key maps to the top-level function that implements it and the test that
exercises it (R2-B F9). Snapshot-contract keys are text fields checked byte for
byte against MS CONFIG.

| CONFIG key | Code (function) | Test | Scope |
|---|---|---|---|
| `candidate` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `status` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `base` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `base.exp_source_commit` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `base.kit_identity` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `base.r0_evidence` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `base.decisions` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `unchanged` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `noise_windows` | `windows.py:12` `gap_geometry` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` | executable |
| `noise_windows.band_s` | `windows.py:12` `gap_geometry` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` | executable |
| `noise_windows.band_rule` | `windows.py:12` `gap_geometry` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` | executable |
| `noise_windows.before_next_s` | `windows.py:12` `gap_geometry` | `tests/test_windows_signal.py::test_next_onset_end_margin_and_final_gap` | executable |
| `noise_windows.end_margin_s` | `windows.py:12` `gap_geometry` | `tests/test_windows_signal.py::test_next_onset_end_margin_and_final_gap` | executable |
| `noise_windows.response_exclusion_s` | `windows.py:12` `gap_geometry` | `tests/test_windows_signal.py::test_response_exclusion_closed_endpoints_and_bad_mask` | executable |
| `noise_windows.tiling` | `windows.py:45` `window_plan` | `tests/test_windows_signal.py::test_fixed_grid_drops_and_never_restarts` | executable |
| `noise_windows.replaces` | `windows.py:12` `gap_geometry` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` | executable |
| `noise_windows.replaces.after_onset_s` | `windows.py:12` `gap_geometry` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` | executable |
| `T` | `windows.py:45` `window_plan` | `tests/test_windows_signal.py::test_stride_contiguous_and_mirror` | executable |
| `T.value` | `windows.py:45` `window_plan` | `tests/test_windows_signal.py::test_stride_contiguous_and_mirror` | executable |
| `T.rule` | `partition.py:32` `census` | `tests/test_contract.py::test_census_exact_and_budget` | executable |
| `T.d` | `windows.py:45` `window_plan` | `tests/test_windows_signal.py::test_stride_contiguous_and_mirror` | executable |
| `T.note` | `partition.py:32` `census` | `tests/test_contract.py::test_census_exact_and_budget` | executable |
| `signal` | `signal.py:29` `build` | `tests/test_windows_signal.py::test_signal_T10_from_T20_templates_fixed_trace_and_latents` | executable |
| `signal.bump_centers_s` | `signal.py:22` `temporal_bases` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` | executable |
| `signal.bump_width_s` | `signal.py:22` `temporal_bases` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` | executable |
| `signal.scaling_rule` | `signal.py:22` `temporal_bases` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` | executable |
| `signal.filter_basis` | `signal.py:10` `kernel` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` | executable |
| `signal.spatial_templates` | `signal.py:29` `build` | `tests/test_windows_signal.py::test_signal_T10_from_T20_templates_fixed_trace_and_latents` | executable |
| `signal.channel_selection` | `preprocess.py:16` `assemble` | `tests/test_R3.py::test_evoked_T20_and_R3_synthetic_assembly_roundtrip` | executable |
| `signal.kc` | `signal.py:29` `build` | `tests/test_windows_signal.py::test_signal_T10_from_T20_templates_fixed_trace_and_latents` | executable |
| `signal.kt` | `signal.py:29` `build` | `tests/test_windows_signal.py::test_signal_T10_from_T20_templates_fixed_trace_and_latents` | executable |
| `signal.latent_variances_l` | `signal.py:29` `build` | `tests/test_windows_signal.py::test_signal_T10_from_T20_templates_fixed_trace_and_latents` | executable |
| `signal.source_weights_j` | `signal.py:29` `build` | `tests/test_windows_signal.py::test_signal_T10_from_T20_templates_fixed_trace_and_latents` | executable |
| `signal.filter_rationale` | `signal.py:10` `kernel` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` | executable |
| `signal.channels_assert` | `preprocess.py:13` `assert_channels` | `tests/test_R3.py::test_channel_assertion_sealed_lists_and_single_drop` | executable |
| `reference_metric` | `gates.py:8` `reference` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` | executable |
| `reference_metric.kind` | `gates.py:8` `reference` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` | executable |
| `reference_metric.definition` | `gates.py:8` `reference` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` | executable |
| `reference_metric.role` | `gates.py:8` `reference` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` | executable |
| `reference_metric.replaces` | `gates.py:8` `reference` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` | executable |
| `gates` | `gates.py:74` `e1` | `tests/test_gates.py::test_e1_dev_and_budget_rules` | executable |
| `gates.G1` | `gates.py:66` `g1_with_retest` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` | executable |
| `gates.G1.unchanged_limits` | `gates.py:66` `g1_with_retest` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` | executable |
| `gates.G1.unchanged_limits.max_condition_spatial` | `gates.py:66` `g1_with_retest` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` | executable |
| `gates.G1.unchanged_limits.max_condition_full` | `gates.py:66` `g1_with_retest` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` | executable |
| `gates.G1.applied_to` | `gates.py:66` `g1_with_retest` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` | executable |
| `gates.G1.retest` | `gate_cli.py:41` `with_retest` | `tests/test_r2b_fixes.py::test_G1_retest_records_first_files` | executable |
| `gates.G1.failure` | `gates.py:66` `g1_with_retest` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` | executable |
| `gates.G2` | `gates.py:19` `role_g2` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` | executable |
| `gates.G2.band` | `gates.py:25` `g2_diagonal` | `tests/test_gates.py::test_g2_physical_ratio_separate_and_no_inflation_correction` | executable |
| `gates.G2.metric` | `gates.py:14` `coordinates` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` | executable |
| `gates.G2.G2R_reference_transport` | `gates.py:25` `g2_diagonal` | `tests/test_gates.py::test_g2_physical_ratio_separate_and_no_inflation_correction` | executable |
| `gates.G2.G2S_stationarity` | `gates.py:25` `g2_diagonal` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` | executable |
| `gates.G2.group_definition` | `gates.py:10` `groups` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` | executable |
| `gates.G2.pass` | `gates.py:25` `g2_diagonal` | `tests/test_gates.py::test_g2_physical_ratio_separate_and_no_inflation_correction` | executable |
| `gates.G2.replaces` | `gates.py:123` `retired_descriptives` | `tests/test_gates.py::test_retired_LW_descriptive_does_not_gate` | executable |
| `gates.G2.descriptive` | `gates.py:25` `g2_diagonal` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` | executable |
| `gates.G2.not_gated` | `gates.py:123` `retired_descriptives` | `tests/test_gates.py::test_retired_LW_descriptive_does_not_gate` | executable |
| `gates.G3` | `gates.py:49` `g3` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` | executable |
| `gates.G3.estimand_and_threshold` | `gates.py:49` `g3` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` | executable |
| `gates.G3.statistic` | `gates.py:40` `cluster_energy` | `tests/test_gates.py::test_cluster_formula_exact` | executable |
| `gates.G3.min_clusters` | `gates.py:40` `cluster_energy` | `tests/test_gates.py::test_cluster_minima` | executable |
| `gates.G3.slots` | `windows.py:45` `window_plan` | `tests/test_r2b_fixes.py::test_truncated_final_gap_keeps_full_band_mirror` | executable |
| `gates.G3.naive_debias` | `gates.py:40` `cluster_energy` | `tests/test_gates.py::test_cluster_formula_exact` | executable |
| `gates.G3.dependence_screen` | `gates.py:49` `g3` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` | executable |
| `gates.G3.calibrated_range` | `gates.py:40` `cluster_energy` | `tests/test_gates.py::test_cluster_independent_and_ar05_MC` | executable |
| `gates.E1` | `gates.py:74` `e1` | `tests/test_gates.py::test_e1_dev_and_budget_rules` | executable |
| `gates.E1.dev` | `gates.py:74` `e1` | `tests/test_gates.py::test_e1_dev_and_budget_rules` | executable |
| `gates.E1.confirmation_min` | `gates.py:91` `n_min` | `tests/test_gates.py::test_n_min_per_cell_upper80_SD_rule_and_no_pass_branch` | executable |
| `gates.E1.power_rule_evaluated_at` | `gates.py:91` `n_min` | `tests/test_gates.py::test_n_min_per_cell_upper80_SD_rule_and_no_pass_branch` | executable |
| `gates.E1.budget_rule` | `gates.py:78` `choose_m_hi` | `tests/test_gates.py::test_e1_dev_and_budget_rules` | executable |
| `gates.E1.n_min_sd` | `gates.py:91` `n_min` | `tests/test_gates.py::test_n_min_per_cell_upper80_SD_rule_and_no_pass_branch` | executable |
| `gates.E1.motivation` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `gates.E1.n_min_no_pass` | `gates.py:91` `n_min` | `tests/test_gates.py::test_n_min_per_cell_upper80_SD_rule_and_no_pass_branch` | executable |
| `d3` | `population.py:7` `primary` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` | executable |
| `d3.noise_model` | `population.py:7` `primary` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` | executable |
| `d3.prestated_reasons_for_a_PZ2_miss_added` | `population.py:7` `primary` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` | executable |
| `d3.D3S_descriptive` | `population.py:37` `d3s` | `tests/test_controls_population.py::test_D3S_50_estimates_descriptive` | executable |
| `d3.other` | `gates.py:118` `g4` | `tests/test_gates.py::test_g4_compares_unshrunk_reference` | executable |
| `d3.N_star_bracket` | `population.py:7` `primary` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` | executable |
| `d3.D3S_budget` | `runner.py:47` `budget_projection` | `tests/test_rd11.py::test_RD11_independent_budgets_and_exact_boundaries` | executable |
| `d3.D3S_law` | `runner.py:60` `preflight` | `tests/test_optimization.py::test_full_preflight_measures_all_D3S_even_when_controls_over_cap` | executable |
| `d3.R4_extension` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | recorded R4 requirement (README, REPORT); not implemented in R2 |
| `controls` | `controls.py:199` `replicate` | `tests/test_controls_population.py::test_exact_prescribed_seed_and_distinct_cells` | executable |
| `controls.generator` | `controls.py:41` `ar_shape` | `tests/test_controls_population.py::test_AR10_Yule_Walker_lags_and_stationary_covariance` | executable |
| `controls.laws` | `controls.py:99` `generate` | `tests/test_controls_population.py::test_laws_generation_finite_guard_reset_and_windows` | executable |
| `controls.laws.L1` | `controls.py:99` `generate` | `tests/test_controls_population.py::test_laws_generation_finite_guard_reset_and_windows` | executable |
| `controls.laws.L2` | `controls.py:99` `generate` | `tests/test_controls_population.py::test_laws_generation_finite_guard_reset_and_windows` | executable |
| `controls.laws.L3` | `controls.py:99` `generate` | `tests/test_controls_population.py::test_laws_generation_finite_guard_reset_and_windows` | executable |
| `controls.laws.L4` | `controls.py:99` `generate` | `tests/test_controls_population.py::test_laws_generation_finite_guard_reset_and_windows` | executable |
| `controls.laws.L5` | `controls.py:99` `generate` | `tests/test_controls_population.py::test_laws_generation_finite_guard_reset_and_windows` | executable |
| `controls.laws.L6` | `controls.py:99` `generate` | `tests/test_controls_population.py::test_laws_generation_finite_guard_reset_and_windows` | executable |
| `controls.laws.L7` | `controls.py:99` `generate` | `tests/test_controls_population.py::test_laws_generation_finite_guard_reset_and_windows` | executable |
| `controls.cells` | `controls.py:122` `apply_cell` | `tests/test_controls_population.py::test_cells_only_modify_eval_with_fixed_metric_and_group_definition` | executable |
| `controls.cells.C0` | `controls.py:122` `apply_cell` | `tests/test_controls_population.py::test_cells_only_modify_eval_with_fixed_metric_and_group_definition` | executable |
| `controls.cells.G2 detection (endpoints)` | `controls.py:122` `apply_cell` | `tests/test_controls_population.py::test_cells_only_modify_eval_with_fixed_metric_and_group_definition` | executable |
| `controls.cells.G3 detection (endpoint)` | `controls.py:122` `apply_cell` | `tests/test_controls_population.py::test_cells_only_modify_eval_with_fixed_metric_and_group_definition` | executable |
| `controls.cells.descriptive` | `controls.py:122` `apply_cell` | `tests/test_r2b_fixes.py::test_C9_jitter_on_continuous_stretch_spills_into_positions_1_2` | executable |
| `controls.replicates` | `controls.py:224` `summarize` | `tests/test_controls_population.py::test_incomplete_campaign_summary_refused` | executable |
| `controls.subjects` | `controls.py:224` `summarize` | `tests/test_controls_population.py::test_incomplete_campaign_summary_refused` | executable |
| `controls.endpoint_unit` | `controls.py:224` `summarize` | `tests/test_controls_population.py::test_incomplete_campaign_summary_refused` | executable |
| `controls.endpoints` | `controls.py:206` `claims` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` | executable |
| `controls.endpoints.G2 null false exclusion` | `controls.py:206` `claims` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` | executable |
| `controls.endpoints.G3 dependence-screen false flag` | `controls.py:206` `claims` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` | executable |
| `controls.endpoints.G2 detection C1-C6` | `controls.py:206` `claims` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` | executable |
| `controls.endpoints.G3 detection C7` | `controls.py:206` `claims` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` | executable |
| `controls.endpoints.G3 slot-statistic null exceedance (max E_cr over evaluated slots >= 2.06604; the screen is not counted)` | `controls.py:206` `claims` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` | executable |
| `controls.K` | `controls.py:206` `claims` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` | executable |
| `controls.alpha_per_claim` | `controls.py:219` `claim_bound` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` | executable |
| `controls.rules` | `controls.py:219` `claim_bound` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` | executable |
| `controls.descriptive_rates` | `controls.py:224` `summarize` | `tests/test_controls_population.py::test_incomplete_campaign_summary_refused` | executable |
| `controls.budget_hours` | `runner.py:47` `budget_projection` | `tests/test_rd11.py::test_RD11_independent_budgets_and_exact_boundaries` | executable |
| `controls.workers` | `runner.py:47` `budget_projection` | `tests/test_rd11.py::test_RD11_independent_budgets_and_exact_boundaries` | executable |
| `controls.estimated_cpu_hours` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `controls.preflight` | `runner.py:60` `preflight` | `tests/test_r2b_fixes.py::test_preflight_refuses_existing_output_before_any_input` | executable |
| `controls.failure_rule` | `controls.py:224` `summarize` | `tests/test_controls_population.py::test_incomplete_campaign_summary_refused` | executable |
| `controls.budget_scope` | `runner.py:127` `campaign` | `tests/test_rd11.py::test_campaign_guard_uses_control_budget_only` | executable |
| `controls.geometry_input` | `export_geometry.py:14` `export` | `tests/test_contract.py::test_geometry_census_and_real_final_gaps` | executable |
| `controls.run_blocked_descriptive` | `controls.py:161` `assess` | `tests/test_r2b_fixes.py::test_run_blocked_pools_exclude_reference` | executable |
| `pre_R2_readonly_check_R0e` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `pre_R2_readonly_check_R0e.what` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `pre_R2_readonly_check_R0e.G3_proxy` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `pre_R2_readonly_check_R0e.G2S_proxy` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `pre_R2_readonly_check_R0e.readings` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `pre_R2_readonly_check_R0e.readings.G3_any_subject_ge_threshold` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `pre_R2_readonly_check_R0e.readings.fewer_than_3_dev_subjects_pass_G3_and_G2S_proxy` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `pre_R2_readonly_check_R0e.readings.otherwise` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `pre_R2_readonly_check_R0e.order` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `revision` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `revision.r1` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `revision.r1a` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `revision.r2` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `revision.r2_1` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `revision.r2_2` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `revision.r2_3` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |
| `partition` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.kind` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_source_hash` | executable |
| `partition.reference_implementation` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_source_hash` | executable |
| `partition.block_gaps` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.cycle` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.run_offset` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.remainder` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.guard` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.inputs` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.runs` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.replaces` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.replaces.reference` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.replaces.training` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.replaces.calibration` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.replaces.evaluation` | `partition.py:19` `assign_roles` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` | executable |
| `partition.role_covariances` | `gates.py:8` `reference` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` | executable |
| `partition.evoked_selection_and_templates` | `preprocess.py:16` `assemble` | `tests/test_R3.py::test_evoked_T20_and_R3_synthetic_assembly_roundtrip` | executable |
| `partition.descriptive` | `controls.py:161` `assess` | `tests/test_r2b_fixes.py::test_run_blocked_pools_exclude_reference` | executable |
| `partition.census` | `partition.py:32` `census` | `tests/test_contract.py::test_census_exact_and_budget` | executable |
| `changed_vs_registered_kit` | `protocol.py:8` `load` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` | snapshot contract |

## Optimization pass (14 October 2026)

AR(10) draws retain the dense implementation's stationary first ten samples, then
apply the identical innovation recursion with scipy.signal.lfilter. Gap AR uses
lfilter along gaps, preserving the unattenuated independent run-start draw.
Window extraction gathers from stride views, preserving role/run/gap/window order.
Run descriptives use identical divisor-n formulas from per-run sufficient sums.
PCG64DXSM draw shape, draw order and final RNG state remain unchanged; numerical
equivalence is tested against the dffa807 implementation for all seven laws and
all thirteen cells. Floating-point reassociation is allowed by those tests;
bitwise equality of numerical samples is not claimed. Baseline source is preserved
in tests/BASELINE_CONTROLS.txt. No statistical law, cell, threshold or scope changed.
After RD-12 the equivalence test excludes C9 and the run-blocked pools, which were
redefined (R2-B F4, F5) and are tested in tests/test_r2b_fixes.py.

For the author-requested complete timing pass use runner preflight --full-timing:
this measures all four full 64-start D3-S timing rows. RD-11 applies the 6 h /
4-worker cap to controls only; D3-S retains all content and has a separate R4
budget of 16 workers and 8 h wall. Each budget must pass independently. Historical preflight evidence
is retained; optimization evidence uses a separate output directory.

## R2-B fixes (DESIGN rev 2.3, RD-12; 14 October 2026)

Review: MS `reviews/megz_r2_implementation_review_20261014_01/` (CHANGES REQUIRED).
CONFIG snapshot updated to MS `bfeaab7` (`35a89227...`). No threshold, law, cell,
statistic, seed or endpoint changed.

- F1 (RD-12a): `windows.window_plan` puts train/cal/eval windows on the fixed
  onset grid (drop, never restart). Tests: `test_fixed_grid_drops_and_never_restarts`
  (replaces the restart test), `test_onset_locked_mean_unbiased_on_sub03_geometry`,
  `test_fixed_grid_budget_audit` (`metadata_census_audit.budget_audit`). Fixed-grid
  pools: train 2204/2115/1972/1954, cal 1088/1084/999/987, eval 1080/1069/991/1000
  (sub-01..04); every budget x 1.1 passes; T = 10, m_hi = 768.
- F3 (RD-12c): `gates.n_min` applies the registered power rule per ref-arm cell
  with each cell's 80 % upper-limit SD; if no n <= 12 passes it asks for the
  registered remedies once (`after_remedy=True` on re-evaluation) and then returns
  STOP with route A for the confirmation stage and no confirmation access.
- F4 (RD-12d): run-blocked pools are train + cal + eval only in `controls.assess`.
- F5 (RD-12e): C9 on the continuous eval stretch (above).
- F6: rposition mirrors position within the full band (`windows.band_slots`).
- F7: `gate_cli.with_retest` records the first derived file's SHA-256 and summary
  before the once-only retest rewrites them.
- F8: `runner controls --preflight PATH` is required; `preflight` and
  `validate_host --out` refuse existing paths. New host evidence goes to
  `meg_z_r1_20261014_r2bfix/`.
- F9: finer design-to-code map (above, and `DESIGN_TO_CODE.json`).
- F10: README budget text updated (above).

## Required R4 extension (R2-B F2; CONFIG `d3.R4_extension`, RD-12b)

This kit does not run R4. Before any R4 computation, an additive kit extension
with its own independent implementation review must provide: predictions with
`general_row(ref_cov=S_ref)` (theta rule over d = 240, the N* bracket in
`predictions.json`); `b_dagger` with S_ref and its flip-flop factors; the D3-S
runner (50 reference estimates per subject drawn under L2 from the control
generator with the real stride-1 geometry, 16 workers, 8 h wall); and the ref-arm
fit and `clean_ref` scoring in the S_ref metric. The old runner refuses the
`sample_full_unshrunk` metric kind, so no shrunk metric can be used silently.
