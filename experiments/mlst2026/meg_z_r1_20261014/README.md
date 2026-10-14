# MEG R1 kit — R2-A implementation

Additive EXP branch `codex/megz-r1-kit-20261014` from
`bfe1139d3923f3b38915c3637696d7afbbf65641`. DESIGN rev 2.1 is encoded verbatim
in `protocol_r1.json`; CONFIG SHA-256 is
`da2e4de3777589ef5c204355461dce2cb1bb613823d5ce08c6035efa3c640c1b`.
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
gap. Adding a final gap changes no existing role. Forward position is the
band-offset index `offset // T`; backward position mirrors it within the full
band. Response exclusions restart tiling at the first allowed sample of each
stretch. The two alignments are computed and counted once.

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
the energy threshold under S_ref, at eval position 0. C9 shifts that same profile
by independent Uniform[0,200 ms] delays with linear interpolation on the
91.667-Hz clock and zero extension; it is descriptive. This fixes the otherwise
unstated interpolation/profile convention before any campaign outcome.

Run-blocked descriptives compare each run with the pooled other runs, using
G2-S alone; they are not endpoints. Every role is centred at its own mean.
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
python -m $K.runner preflight --derived /ceph/dwong/meg_z/derived_timingfix_20261013 --gates experiments/mlst2026/meg_z_20261013_dev_timingfix/gates.json --out experiments/mlst2026/meg_z_r1_20261014_preflight/PREFLIGHT.json
```

R3 CLIs are `python -m $K.preprocess` and `python -m $K.gate_cli` (tested but
not run on real data in R2). The campaign requires an R2-B ACCEPTED review,
exact reviewed HEAD and identity, and a new output directory; it runs four
workers. Start only when separately authorized, inside private
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

| CONFIG key | Code | Test |
|---|---|---|
| `candidate` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `status` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `base` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `base.exp_source_commit` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `base.kit_identity` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `base.r0_evidence` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `base.decisions` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `unchanged` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `noise_windows` | `windows.py:12` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` |
| `noise_windows.band_s` | `windows.py:12` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` |
| `noise_windows.band_rule` | `windows.py:12` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` |
| `noise_windows.before_next_s` | `windows.py:12` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` |
| `noise_windows.end_margin_s` | `windows.py:12` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` |
| `noise_windows.response_exclusion_s` | `windows.py:12` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` |
| `noise_windows.tiling` | `windows.py:12` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` |
| `noise_windows.replaces` | `windows.py:12` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` |
| `noise_windows.replaces.after_onset_s` | `windows.py:12` | `tests/test_windows_signal.py::test_exact_band_sample_boundaries` |
| `T` | `windows.py:34` | `tests/test_windows_signal.py::test_stride_contiguous_and_mirror` |
| `T.value` | `windows.py:34` | `tests/test_windows_signal.py::test_stride_contiguous_and_mirror` |
| `T.rule` | `windows.py:34` | `tests/test_windows_signal.py::test_stride_contiguous_and_mirror` |
| `T.d` | `windows.py:34` | `tests/test_windows_signal.py::test_stride_contiguous_and_mirror` |
| `T.note` | `windows.py:34` | `tests/test_windows_signal.py::test_stride_contiguous_and_mirror` |
| `signal` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.bump_centers_s` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.bump_width_s` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.scaling_rule` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.filter_basis` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.spatial_templates` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.channel_selection` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.kc` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.kt` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.latent_variances_l` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.source_weights_j` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.filter_rationale` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `signal.channels_assert` | `signal.py:29` | `tests/test_windows_signal.py::test_fir_equals_raw_filter_and_geometry` |
| `reference_metric` | `gates.py:8` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` |
| `reference_metric.kind` | `gates.py:8` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` |
| `reference_metric.definition` | `gates.py:8` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` |
| `reference_metric.role` | `gates.py:8` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` |
| `reference_metric.replaces` | `gates.py:8` | `tests/test_gates.py::test_role_covariance_unshrunk_divisor_n_and_shift_invariant` |
| `gates` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `gates.G1` | `gates.py:66` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` |
| `gates.G1.unchanged_limits` | `gates.py:66` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` |
| `gates.G1.unchanged_limits.max_condition_spatial` | `gates.py:66` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` |
| `gates.G1.unchanged_limits.max_condition_full` | `gates.py:66` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` |
| `gates.G1.applied_to` | `gates.py:66` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` |
| `gates.G1.retest` | `gates.py:66` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` |
| `gates.G1.failure` | `gates.py:66` | `tests/test_gates.py::test_once_only_retest_and_failure_exclusion` |
| `gates.G2` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G2.band` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G2.metric` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G2.G2R_reference_transport` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G2.G2S_stationarity` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G2.group_definition` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G2.pass` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G2.replaces` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G2.descriptive` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G2.not_gated` | `gates.py:19` | `tests/test_gates.py::test_g2_own_eigenvectors_and_five_statistics` |
| `gates.G3` | `gates.py:49` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` |
| `gates.G3.estimand_and_threshold` | `gates.py:49` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` |
| `gates.G3.statistic` | `gates.py:49` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` |
| `gates.G3.min_clusters` | `gates.py:49` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` |
| `gates.G3.slots` | `gates.py:49` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` |
| `gates.G3.naive_debias` | `gates.py:49` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` |
| `gates.G3.dependence_screen` | `gates.py:49` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` |
| `gates.G3.calibrated_range` | `gates.py:49` | `tests/test_gates.py::test_g3_strict_threshold_screen_and_dedup` |
| `gates.E1` | `gates.py:87` | `tests/test_gates.py::test_n_min_upper80_SD_rule_and_stop` |
| `gates.E1.dev` | `gates.py:87` | `tests/test_gates.py::test_n_min_upper80_SD_rule_and_stop` |
| `gates.E1.confirmation_min` | `gates.py:87` | `tests/test_gates.py::test_n_min_upper80_SD_rule_and_stop` |
| `gates.E1.power_rule_evaluated_at` | `gates.py:87` | `tests/test_gates.py::test_n_min_upper80_SD_rule_and_stop` |
| `gates.E1.budget_rule` | `gates.py:87` | `tests/test_gates.py::test_n_min_upper80_SD_rule_and_stop` |
| `gates.E1.n_min_sd` | `gates.py:87` | `tests/test_gates.py::test_n_min_upper80_SD_rule_and_stop` |
| `gates.E1.motivation` | `gates.py:87` | `tests/test_gates.py::test_n_min_upper80_SD_rule_and_stop` |
| `d3` | `population.py:13` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` |
| `d3.noise_model` | `population.py:13` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` |
| `d3.prestated_reasons_for_a_PZ2_miss_added` | `population.py:13` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` |
| `d3.D3S_descriptive` | `population.py:13` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` |
| `d3.other` | `population.py:13` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` |
| `d3.N_star_bracket` | `population.py:13` | `tests/test_controls_population.py::test_primary_general_noise_identity_and_bracket` |
| `controls` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.generator` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.laws` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.laws.L1` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.laws.L2` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.laws.L3` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.laws.L4` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.laws.L5` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.laws.L6` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.laws.L7` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.cells` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.cells.C0` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.cells.G2 detection (endpoints)` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.cells.G3 detection (endpoint)` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.cells.descriptive` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.replicates` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.subjects` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.endpoint_unit` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.endpoints` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.endpoints.G2 null false exclusion` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.endpoints.G3 dependence-screen false flag` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.endpoints.G2 detection C1-C6` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.endpoints.G3 detection C7` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.endpoints.G3 slot-statistic null exceedance (max E_cr over evaluated slots >= 2.06604; the screen is not counted)` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.K` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.alpha_per_claim` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.rules` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.descriptive_rates` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.budget_hours` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.workers` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.estimated_cpu_hours` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.preflight` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `controls.failure_rule` | `controls.py:155` | `tests/test_controls_population.py::test_35_claims_exact_and_CP_thresholds` |
| `pre_R2_readonly_check_R0e` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `pre_R2_readonly_check_R0e.what` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `pre_R2_readonly_check_R0e.G3_proxy` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `pre_R2_readonly_check_R0e.G2S_proxy` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `pre_R2_readonly_check_R0e.readings` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `pre_R2_readonly_check_R0e.readings.G3_any_subject_ge_threshold` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `pre_R2_readonly_check_R0e.readings.fewer_than_3_dev_subjects_pass_G3_and_G2S_proxy` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `pre_R2_readonly_check_R0e.readings.otherwise` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `pre_R2_readonly_check_R0e.order` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `revision` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `revision.r1` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `revision.r1a` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `revision.r2` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `revision.r2_1` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
| `partition` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.kind` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.reference_implementation` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.block_gaps` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.cycle` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.run_offset` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.remainder` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.guard` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.inputs` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.runs` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.replaces` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.replaces.reference` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.replaces.training` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.replaces.calibration` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.replaces.evaluation` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.role_covariances` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.evoked_selection_and_templates` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.descriptive` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `partition.census` | `partition.py:19` | `tests/test_contract.py::test_partition_guards_final_gap_and_no_cross_role_adjacency` |
| `changed_vs_registered_kit` | `protocol.py:8` | `tests/test_contract.py::test_config_equals_sha_pinned_MS` |
