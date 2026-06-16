# Experiment Registry

| ID | Dataset | Purpose | Entry Point | Config | Expected Outputs | Status | Notes |
|---|---|---|---|---|---|---|---|
| SYN-A | synthetic | OF = rank-1 weighted ML / EMPCA | `experiments/synthetic/of_empca_equivalence/run.py` | `configs/synthetic/of_empca.yaml` | summary JSON; later Fig. 1 equivalence plot | implemented | Fast smoke script is runnable. |
| SYN-B | synthetic | Bridge theorem: weighted tied linear AE = EMPCA | `experiments/synthetic/ae_empca_bridge/run.py` | TBD | principal-angle table and subspace plot | planned | Core functions are scaffolded through closed-form tied AE helpers. |
| SYN-C | synthetic | MSE-vs-Mahalanobis metric reversal | `experiments/synthetic/metric_reversal/run.py` | `configs/synthetic/metric_reversal.yaml` | summary JSON; later triptych plot | implemented | Compact smoke script; figure generation still planned. |
| SYN-D | synthetic | Template mismatch and timing jitter | `experiments/synthetic/template_mismatch/run.py` | TBD | bias/error curves versus jitter | planned | Should reuse pulse and OF utilities. |
| GWOSC-A | GWOSC | Known-event matched-filter validation | `experiments/gwosc/smoke.py` then `scripts/download/download_gwosc.py` | `configs/gwosc/gw150914_smoke.yaml` | PSD, whitening, matched-filter score, residual comparison | scaffolded | Smoke check does not download data. |
| GWOSC-B | GWOSC | Injection recovery in off-source real noise | TBD | TBD | recovery curve versus SNR; false-positive table | planned | Requires gwosc/gwpy dependency and data cache. |
| CRESST-A | CRESST | Pulse-shape reconstruction under estimated detector noise | `scripts/download/download_cresst.md`, `scripts/preprocess/preprocess_cresst.py` | `configs/cresst/pulse_shape_smoke.yaml` | traces, PSD/covariance, basis comparison, residual table | scaffolded | Manual download expected. |
| TIDMAD-A | TIDMAD | Optional denoising validation | TBD | `configs/tidmad/optional_smoke.yaml` | denoising score and frequency residual figure | planned | Optional; should not block Paper 1. |
