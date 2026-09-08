# Bootstrap-seed sensitivity audit (P1-E2, P1-E9, P1-E11)

Archived blocks recomputed: 30.  Exact reproduction of the archived interval at seed 20260819: 30/30.
Verdict (`excludes_zero`) changes at the predeclared seed 20260808: 0.
Largest absolute shift of an interval end between the two seeds: 1.383e-03; largest shift relative to the interval half-width: 3.8 %.

| experiment | block | archived CI (20260819) | CI at 20260808 | max rel. shift | verdict change |
|---|---|---|---|---|---|
| P1-E2 | `per_alpha/0.0/approximation_term_chi2_nfpa_minus_cwpca` | [-0.0028003, -0.0027095] | [-0.0028013, -0.0027093] | 2.2 % | no |
| P1-E2 | `per_alpha/0.0/metric_term_chi2_nfpa_minus_isompca` | [-0.02795, -0.027467] | [-0.027953, -0.027465] | 1.3 % | no |
| P1-E2 | `per_alpha/0.0/metric_term_mse_nfpa_minus_isompca` | [+0.0032839, +0.0033309] | [+0.0032845, +0.0033309] | 2.3 % | no |
| P1-E2 | `per_alpha/0.05/approximation_term_chi2_nfpa_minus_cwpca` | [-0.0027509, -0.0026676] | [-0.0027513, -0.0026676] | 1.1 % | no |
| P1-E2 | `per_alpha/0.05/metric_term_chi2_nfpa_minus_isompca` | [-0.025163, -0.024699] | [-0.025168, -0.024698] | 2.0 % | no |
| P1-E2 | `per_alpha/0.05/metric_term_mse_nfpa_minus_isompca` | [+0.0033217, +0.0033657] | [+0.0033223, +0.0033658] | 2.7 % | no |
| P1-E2 | `per_alpha/0.1/approximation_term_chi2_nfpa_minus_cwpca` | [-0.0025917, -0.0025205] | [-0.0025916, -0.00252] | 1.6 % | no |
| P1-E2 | `per_alpha/0.1/metric_term_chi2_nfpa_minus_isompca` | [-0.019411, -0.019003] | [-0.019414, -0.019] | 1.3 % | no |
| P1-E2 | `per_alpha/0.1/metric_term_mse_nfpa_minus_isompca` | [+0.0033941, +0.0034342] | [+0.0033946, +0.0034345] | 2.3 % | no |
| P1-E2 | `per_alpha/0.2/approximation_term_chi2_nfpa_minus_cwpca` | [-0.0021511, -0.0020738] | [-0.002152, -0.0020741] | 2.2 % | no |
| P1-E2 | `per_alpha/0.2/metric_term_chi2_nfpa_minus_isompca` | [-0.010238, -0.0099669] | [-0.010238, -0.0099652] | 1.3 % | no |
| P1-E2 | `per_alpha/0.2/metric_term_mse_nfpa_minus_isompca` | [+0.0034968, +0.0035376] | [+0.0034972, +0.0035377] | 2.1 % | no |
| P1-E2 | `per_alpha/0.4/approximation_term_chi2_nfpa_minus_cwpca` | [-0.0014397, -0.0013299] | [-0.0014394, -0.00133] | 0.5 % | no |
| P1-E2 | `per_alpha/0.4/metric_term_chi2_nfpa_minus_isompca` | [-0.0036141, -0.0034813] | [-0.0036137, -0.0034818] | 0.9 % | no |
| P1-E2 | `per_alpha/0.4/metric_term_mse_nfpa_minus_isompca` | [+0.0035717, +0.0036121] | [+0.003572, +0.0036122] | 1.5 % | no |
| P1-E2 | `per_alpha/0.8/approximation_term_chi2_nfpa_minus_cwpca` | [-0.00038051, -0.0002427] | [-0.00037952, -0.00024231] | 1.4 % | no |
| P1-E2 | `per_alpha/0.8/metric_term_chi2_nfpa_minus_isompca` | [-0.0010781, -0.00095278] | [-0.001077, -0.0009538] | 1.7 % | no |
| P1-E2 | `per_alpha/0.8/metric_term_mse_nfpa_minus_isompca` | [+0.0036243, +0.0036696] | [+0.0036246, +0.0036699] | 1.7 % | no |
| P1-E9 | `contrasts/NFPA-Iso-MPCA|amplitude` | [+0.45691, +0.48643] | [+0.45658, +0.48693] | 3.4 % | no |
| P1-E9 | `contrasts/NFPA-Iso-MPCA|rise_time` | [+0.87282, +0.88361] | [+0.87278, +0.88356] | 1.0 % | no |
| P1-E9 | `contrasts/NFPA-Iso-MPCA|decay_time` | [+0.23073, +0.28859] | [+0.23042, +0.28857] | 1.1 % | no |
| P1-E9 | `contrasts/NFPA-Iso-MPCA|secondary` | [+0.93878, +0.9508] | [+0.93877, +0.95103] | 3.8 % | no |
| P1-E9 | `contrasts/CW-PCA-IsoPCA|amplitude` | [+0.10332, +0.13001] | [+0.10337, +0.13014] | 0.9 % | no |
| P1-E9 | `contrasts/CW-PCA-IsoPCA|rise_time` | [+0.82299, +0.83289] | [+0.82298, +0.83279] | 1.9 % | no |
| P1-E9 | `contrasts/CW-PCA-IsoPCA|decay_time` | [+0.12972, +0.16288] | [+0.12946, +0.16319] | 1.8 % | no |
| P1-E9 | `contrasts/CW-PCA-IsoPCA|secondary` | [+0.83454, +0.91054] | [+0.83564, +0.91193] | 3.6 % | no |
| P1-E11 | `p_c_held_out_chi2_gap/CW-PCA|n_train_1x` | [+0.017361, +0.018023] | [+0.017352, +0.01802] | 2.8 % | no |
| P1-E11 | `p_c_held_out_chi2_gap/CW-PCA|n_train_4x` | [+0.017769, +0.01832] | [+0.017765, +0.018317] | 1.6 % | no |
| P1-E11 | `p_c_held_out_chi2_gap/NFPA|n_train_1x` | [+0.0059099, +0.0084133] | [+0.0058898, +0.0084601] | 3.7 % | no |
| P1-E11 | `p_c_held_out_chi2_gap/NFPA|n_train_4x` | [+0.0016379, +0.0019924] | [+0.0016368, +0.0019898] | 1.5 % | no |
