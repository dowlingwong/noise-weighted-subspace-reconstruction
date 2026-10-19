# HCP Rnoise candidate kit — H0–H3, 19 October 2026

Separate revision-stage study, **not registered or ready for real-data execution**. This kit is additive to EXP f8d7545. MS design accepted at bfa3d33, review round02; the seven author decisions remain pending. No Rnoise samples were decoded in preparation. Confirmation headers, bytes and noise statistics were not accessed.

Machine contract: protocol.json is byte-identical to MS reviews/hcp_rnoise_design_20261019/CONFIG.json, SHA256 8b90257f72f765aa006ee3d927329018284d4c04a69d983d5f8589a8a0bf35fa. SPLIT and sensor geometry are pinned metadata fixtures. Full case-sensitive channel names and device coordinates come from a dev header, not noise statistics. T is **6 samples**, C12, d72, fs101.7255066Hz. Retained metadata counts per date: ref3630 (overlapping stride1), train1111, cal606, eval606. The last incomplete block is discarded. These are window counts, not independent observations.

## Design-to-code map

| MS DESIGN | Additive implementation | Key checks |
|---|---|---|
| 0 split and access seal | protocol.load/split_check/unit/guard; preprocess.open_header/read_scan | every confirmation date and every reserved real CLI refuses before reader; mocked dev header never loads samples; wrong representative rejected |
| 1.1 fixed geometry/FIR | preprocess.selected_geometry/kernel/preprocess_array | pinned12 names/loc; FIR6715 taps/hash; fabricated array agrees with RawArray.filter |
| 1.2 guarded grid/sampling | windows.plan/cut/sample_blocks | full blocks, fixed starts, c+C*t layout; exhaustive equal inclusion under block order/rotation; unequal blocks rejected |
| 1.3 budgets | windows.census/grids; runner.census_check; gates.choose_m_hi | all72 date×T rows and grids, N ceiling1010, m512; no inference from stride counts |
| 1.4 signal | signal.sarvas/spatial_templates/temporal_bases/build | compare point-Sarvas formula with MNE; fixed rank/cosine; filtered rank3+3; one theta0 scale |
| 2 D3 / D3-S | population.primary/SubjectPopulation/predictions/d3s_summary | general_row and metric-corrected Bdagger reused; synthetic zero/nonzero certificate and generalized noise trace; no inherited numeric inflation bracket |
| 2 learning/inference | learning.fit_cell; readings.date_interval/pz1–5 | whole-block orders/rotations; R1 EMPCA/NFPA and shared clean_ref; date means and reading signs |
| 3 G1/G2/G4/G5 | gates.assess, descriptives.lag_covariance/block_bootstrap, reused role_g2/g4/g5_mardia | direct G1 exclusion, no retest; fabricated scale drift, finite ill-conditioned and singular cases; cal/scan-third descriptives |
| 4 artifacts/E1/power | preprocess.artifact_screen; gates.budget/e1/n_min | native amplitudes/steps/nonfinite; nine-of-twelve, no replacements; unchanged R1 n_min/pilot SD upper rule |
| 5 controls | controls.ref_moments/fit_shape/generate/apply_cell/replicate/event/claim_bound/summarize | within-ref-block lags only; stationary first-block AR; eval-only cells; asymmetric G1 classification, exact CP boundaries, exact24 tuples, completeness |
| 5 timing | runner.preflight | synthetic shape only; shared replicate/generator and R1 general_noise; no controls campaign |
| 6 identities | runner.verify_kit | current file manifest plus frozen R1 a1825db… and original f121776c… verification |

R1 imports preserve its unshrunk covariance, G2 eigen groups/band, general population paths, D3-S, spatial-mode cell transform, AR recursion, optimizers, scores, t intervals, G4/G5 and power rule. This package changes no inherited file. The learning adapter normalizes each fit metric by its positive mean eigenvalue before calling the unchanged optimizer. This leaves the fitted map and normalized gradients invariant, avoids its absolute QR cutoff at Tesla units, records the scalar, and scores in the original physical reference metric. IMPLEMENTATION_DEFAULTS.md records PD7 qualification dependence diagnostics and PD8 numerical scaling. Invalid/incomplete power and prediction evidence fails closed; date intervals require explicit dates and resample identities, and PZ3/PZ5 require explicit expected registered cells. The new primitive contracts and exact human choices are documented in MS DESIGN; no hidden data selection or threshold adaptation.

The homogeneous sphere is a **point-magnetometer** mathematical forward model, not MRI-derived source localization or Magnes coil integration. Native preprocessing is 1–40Hz zero FIR then phase0 slicing q20. Fixed 10s full blocks lose4s guards each and2s scan ends. Uniform block permutation plus uniform cyclic rotation gives equal window inclusion. A wrapped partial draw contains two observed segments; it is not a physically contiguous interval. Six sparse-role blocks and date-unit inference do not prove independence. D3-S sizes reference estimation, not the serial-dependence correction to IID S4.3.

Controls: seven laws, cells0–6,10–12; no G3/locked-mean cells. Fit AR10 only from future ref-active lag0–19 moments. L5/C0 is a null endpoint; L5 detection cells and all L7 are descriptive. Mean-one lognormal *variance* factors act per scan third/block; transformed overlapping windows need not reconstruct one continuous process. Null reference failures are exclusions; detection reference failures are non-successes. CP n2400, alpha.05/24: null≤90 failures, detection≥2202 successes. Never drop failed replicates. Controls retain cal/third G2-S rates and C0/L2–L5 generalized-extreme quantiles, with explicit unavailable denominators; missing required descriptions stops the report without changing endpoint denominators. The test that exercises the full summary uses fabricated outcome dictionaries only, not a simulation campaign.

## Safe commands

Run from EXP root with /ceph/dwong/meg_env/bin/python, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1; unset PYTHONPYCACHEPREFIX. Long work belongs inside tmux -L paper1hcp.

    python -m experiments.mlst2026.hcp_rnoise_20261019.runner verify-kit
    python -m experiments.mlst2026.hcp_rnoise_20261019.runner split-check
    python -m experiments.mlst2026.hcp_rnoise_20261019.runner census-check
    python -m pytest -p no:cacheprovider experiments/mlst2026/hcp_rnoise_20261019/tests
    python -m experiments.mlst2026.hcp_rnoise_20261019.runner preflight --out /new/evidence/PREFLIGHT.json

Optional split-check --inventory accepts only the original hash-bound MS inventory JSON; it never opens recordings. Preflight is H3, after H2 independent review: five synthetic control timing probes and two complete D3-S general rows at64 starts, with128s per-row timeout. It projects168000 controls and up to3600 D3-S rows at4/16 workers, reports ideal and illustrative25% overhead scenarios. It neither qualifies controls nor measures multiworker throughput. Output paths must be new.

Reserved CLI names decode-dev/gates/predict/controls/fit/confirmation always fail closed, even for dev, with no data-root filesystem access. There is no flag, alternate protocol, environment variable or wrapper that unlocks the current candidate. Array kernels exist and are tested. A future author-adopted, independently reviewed launch extension must serialize/seal derived moments, connect orchestration, freeze family/seeds and enable dev execution; a later separately registered frozen protocol is additionally required for confirmation. Preparation does not supply an unreviewed real-data launcher.

HCP original and derived metadata remain subject to WU-Minn HCP Open Access Data Use Terms; no attempt at identifying participants. Data were provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators David Van Essen and Kamil Ugurbil;1U54MH091657), funded by the16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.
