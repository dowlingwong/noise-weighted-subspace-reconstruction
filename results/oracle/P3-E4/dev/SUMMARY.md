# P3-E4 (dev) — NOT citable (dev)

Rows: 15120, seeds: [0, 1, 2]

## Mechanism signature (S vs N)

| statistic | S mean | N mean | S−N bootstrap 95 % |
|---|---|---|---|
| mean_disp_excess_z | 11.024 | 4.444 | 6.580 [6.096, 6.856] |
| mean_disp_excess_z_T_S | 7.830 | 2.500 | 5.330 [4.073, 7.252] |
| mean_disp_excess_z_T_U | 89.324 | 5.316 | 84.007 [78.893, 94.132] |
| mean_disp_excess_frac_in_T_S | 0.579 | 0.223 | 0.352 [0.306, 0.419] |
| mean_disp_outside_T_S | 0.860 | 0.941 | -0.081 [-0.090, -0.071] |
| resid_var_T_S | 0.998 | 1.110 | -0.112 [-0.132, -0.082] |
| event_disp_in_T_U_excess | 0.003 | 0.000 | 0.002 [0.002, 0.002] |

## C2 attribution (leave-one-seed-out)

| subject/stage | layerwise full F1 | deployable F1 | generic F1 | ΔF1 full−generic | ΔF1 deployable−generic | unknown AUROC |
|---|---|---|---|---|---|---|
| lin_cw|code | 0.823 | 0.769 | 0.701 | 0.123 | 0.069 | 0.254 (margin) / 0.708 (Mahalanobis) |
| mlp_ae|code | 0.811 | 0.726 | 0.602 | 0.209 | 0.124 | 0.266 (margin) / 0.709 (Mahalanobis) |
| mlp_ae|h1 | 0.826 | 0.731 | 0.612 | 0.215 | 0.120 | 0.295 (margin) / 0.732 (Mahalanobis) |
| mlp_ae|h2 | 0.812 | 0.711 | 0.585 | 0.227 | 0.126 | 0.251 (margin) / 0.710 (Mahalanobis) |

## C4 alarm–consequence (pooled over strata)

| subject/stage/monitor | within-stratum AUROC | pooled AUROC | missed-consequence | harmless-alarm |
|---|---|---|---|---|
| lin_cw|code|euclidean | 0.520 | 0.591 | 0.722 | 0.216 |
| lin_cw|code|pullback_whitened | 0.520 | 0.591 | 0.722 | 0.216 |
| lin_cw|code|jacobian_projected | 0.520 | 0.591 | 0.722 | 0.216 |
| lin_cw|code|pullback_output | 0.515 | 0.586 | 0.537 | 0.315 |
| lin_cw|code|output_residual | 0.432 | 0.520 | 0.185 | 0.821 |
| lin_cw|code|layer_alignment_k4 | 0.500 | 0.500 | 1.000 | 0.000 |
| lin_cw|code|input_ks_bh | 0.511 | 0.560 | 0.630 | 0.253 |
| lin_cw|code|mmd_rbf | 0.571 | 0.675 | 0.481 | 0.275 |
| lin_cw|code|c2st_logistic_auc | 0.411 | 0.534 | 0.796 | 0.154 |
| lin_cw|code|embedding_moments | 0.612 | 0.713 | 0.481 | 0.170 |
| mlp_ae|h1|euclidean | 0.452 | 0.427 | 0.859 | 0.236 |
| mlp_ae|h1|pullback_whitened | 0.446 | 0.443 | 0.938 | 0.083 |
| mlp_ae|h1|jacobian_projected | 0.485 | 0.466 | 0.828 | 0.242 |
| mlp_ae|h1|pullback_output | 0.456 | 0.488 | 0.797 | 0.201 |
| mlp_ae|h1|output_residual | 0.453 | 0.365 | 0.172 | 0.828 |
| mlp_ae|h1|layer_alignment_k4 | 0.540 | 0.488 | 0.875 | 0.156 |
| mlp_ae|h1|input_ks_bh | 0.547 | 0.495 | 0.719 | 0.268 |
| mlp_ae|h1|mmd_rbf | 0.481 | 0.497 | 0.797 | 0.283 |
| mlp_ae|h1|c2st_logistic_auc | 0.373 | 0.451 | 0.938 | 0.134 |
| mlp_ae|h1|embedding_moments | 0.567 | 0.602 | 0.484 | 0.376 |
| mlp_ae|h2|euclidean | 0.519 | 0.465 | 0.750 | 0.315 |
| mlp_ae|h2|pullback_whitened | 0.444 | 0.461 | 0.891 | 0.150 |
| mlp_ae|h2|jacobian_projected | 0.498 | 0.470 | 0.750 | 0.315 |
| mlp_ae|h2|pullback_output | 0.469 | 0.498 | 0.812 | 0.201 |
| mlp_ae|h2|output_residual | 0.453 | 0.365 | 0.172 | 0.828 |
| mlp_ae|h2|layer_alignment_k4 | 0.531 | 0.500 | 0.906 | 0.143 |
| mlp_ae|h2|input_ks_bh | 0.547 | 0.495 | 0.719 | 0.268 |
| mlp_ae|h2|mmd_rbf | 0.544 | 0.514 | 0.719 | 0.296 |
| mlp_ae|h2|c2st_logistic_auc | 0.506 | 0.508 | 0.906 | 0.127 |
| mlp_ae|h2|embedding_moments | 0.518 | 0.572 | 0.578 | 0.350 |
| mlp_ae|code|euclidean | 0.505 | 0.469 | 0.750 | 0.322 |
| mlp_ae|code|pullback_whitened | 0.444 | 0.461 | 0.891 | 0.150 |
| mlp_ae|code|jacobian_projected | 0.505 | 0.469 | 0.750 | 0.322 |
| mlp_ae|code|pullback_output | 0.469 | 0.498 | 0.812 | 0.201 |
| mlp_ae|code|output_residual | 0.453 | 0.365 | 0.172 | 0.828 |
| mlp_ae|code|layer_alignment_k4 | 0.584 | 0.509 | 0.922 | 0.111 |
| mlp_ae|code|input_ks_bh | 0.547 | 0.495 | 0.719 | 0.268 |
| mlp_ae|code|mmd_rbf | 0.544 | 0.509 | 0.656 | 0.354 |
| mlp_ae|code|c2st_logistic_auc | 0.500 | 0.511 | 0.812 | 0.175 |
| mlp_ae|code|embedding_moments | 0.520 | 0.582 | 0.594 | 0.280 |

## C4 with the reconstruction consequence K_recon (whitened reconstruction error vs clean signal)

| subject/stage/monitor | within-stratum AUROC | pooled AUROC | missed-consequence | harmless-alarm | frac. high |
|---|---|---|---|---|---|
| lin_cw|code|euclidean | 0.576 | 0.695 | 0.659 | 0.133 | 0.442 |
| lin_cw|code|pullback_whitened | 0.576 | 0.695 | 0.659 | 0.133 | 0.442 |
| lin_cw|code|jacobian_projected | 0.576 | 0.695 | 0.659 | 0.133 | 0.442 |
| lin_cw|code|pullback_output | 0.499 | 0.721 | 0.479 | 0.190 | 0.442 |
| lin_cw|code|output_residual | 0.459 | 0.498 | 0.102 | 0.758 | 0.442 |
| lin_cw|code|layer_alignment_k4 | 0.500 | 0.500 | 1.000 | 0.000 | 0.442 |
| lin_cw|code|input_ks_bh | 0.569 | 0.563 | 0.671 | 0.223 | 0.442 |
| lin_cw|code|mmd_rbf | 0.393 | 0.759 | 0.455 | 0.123 | 0.442 |
| lin_cw|code|c2st_logistic_auc | 0.443 | 0.619 | 0.731 | 0.076 | 0.442 |
| lin_cw|code|embedding_moments | 0.415 | 0.760 | 0.557 | 0.043 | 0.442 |
| mlp_ae|h1|euclidean | 0.487 | 0.654 | 0.609 | 0.144 | 0.304 |
| mlp_ae|h1|pullback_whitened | 0.452 | 0.539 | 0.878 | 0.061 | 0.304 |
| mlp_ae|h1|jacobian_projected | 0.470 | 0.615 | 0.635 | 0.171 | 0.304 |
| mlp_ae|h1|pullback_output | 0.498 | 0.621 | 0.687 | 0.152 | 0.304 |
| mlp_ae|h1|output_residual | 0.456 | 0.542 | 0.035 | 0.768 | 0.304 |
| mlp_ae|h1|layer_alignment_k4 | 0.545 | 0.556 | 0.800 | 0.129 | 0.304 |
| mlp_ae|h1|input_ks_bh | 0.574 | 0.625 | 0.600 | 0.213 | 0.304 |
| mlp_ae|h1|mmd_rbf | 0.404 | 0.712 | 0.513 | 0.175 | 0.304 |
| mlp_ae|h1|c2st_logistic_auc | 0.472 | 0.644 | 0.757 | 0.068 | 0.304 |
| mlp_ae|h1|embedding_moments | 0.365 | 0.775 | 0.287 | 0.262 | 0.304 |
| mlp_ae|h2|euclidean | 0.403 | 0.636 | 0.539 | 0.236 | 0.304 |
| mlp_ae|h2|pullback_whitened | 0.310 | 0.566 | 0.765 | 0.103 | 0.304 |
| mlp_ae|h2|jacobian_projected | 0.365 | 0.614 | 0.565 | 0.247 | 0.304 |
| mlp_ae|h2|pullback_output | 0.451 | 0.622 | 0.678 | 0.144 | 0.304 |
| mlp_ae|h2|output_residual | 0.456 | 0.542 | 0.035 | 0.768 | 0.304 |
| mlp_ae|h2|layer_alignment_k4 | 0.401 | 0.524 | 0.809 | 0.110 | 0.304 |
| mlp_ae|h2|input_ks_bh | 0.574 | 0.625 | 0.600 | 0.213 | 0.304 |
| mlp_ae|h2|mmd_rbf | 0.365 | 0.693 | 0.504 | 0.205 | 0.304 |
| mlp_ae|h2|c2st_logistic_auc | 0.416 | 0.644 | 0.791 | 0.084 | 0.304 |
| mlp_ae|h2|embedding_moments | 0.307 | 0.758 | 0.357 | 0.240 | 0.304 |
| mlp_ae|code|euclidean | 0.356 | 0.612 | 0.548 | 0.247 | 0.304 |
| mlp_ae|code|pullback_whitened | 0.310 | 0.566 | 0.765 | 0.103 | 0.304 |
| mlp_ae|code|jacobian_projected | 0.356 | 0.612 | 0.548 | 0.247 | 0.304 |
| mlp_ae|code|pullback_output | 0.451 | 0.622 | 0.678 | 0.144 | 0.304 |
| mlp_ae|code|output_residual | 0.456 | 0.542 | 0.035 | 0.768 | 0.304 |
| mlp_ae|code|layer_alignment_k4 | 0.429 | 0.508 | 0.817 | 0.072 | 0.304 |
| mlp_ae|code|input_ks_bh | 0.574 | 0.625 | 0.600 | 0.213 | 0.304 |
| mlp_ae|code|mmd_rbf | 0.358 | 0.694 | 0.435 | 0.259 | 0.304 |
| mlp_ae|code|c2st_logistic_auc | 0.383 | 0.566 | 0.739 | 0.141 | 0.304 |
| mlp_ae|code|embedding_moments | 0.339 | 0.752 | 0.409 | 0.175 | 0.304 |

## C1 detection power (fraction of windows above the 1 % FAR threshold)

| subject/stage/monitor/family | power |
|---|---|
| lin_cw|code|c2st_logistic_auc|N1_cov_slope | 0.02 |
| lin_cw|code|c2st_logistic_auc|N2_cov_channel | 0.00 |
| lin_cw|code|c2st_logistic_auc|N3_jitter | 0.17 |
| lin_cw|code|c2st_logistic_auc|N4_gain_drift | 0.00 |
| lin_cw|code|c2st_logistic_auc|N5_glitch | 0.00 |
| lin_cw|code|c2st_logistic_auc|N6_channel_loss | 0.44 |
| lin_cw|code|c2st_logistic_auc|S1_amplitude | 0.56 |
| lin_cw|code|c2st_logistic_auc|S2_decay | 0.30 |
| lin_cw|code|c2st_logistic_auc|S3_t0 | 0.53 |
| lin_cw|code|c2st_logistic_auc|S4_alpha | 0.00 |
| lin_cw|code|c2st_logistic_auc|U1_tails | 0.00 |
| lin_cw|code|c2st_logistic_auc|U2_line | 0.00 |
| lin_cw|code|embedding_moments|N1_cov_slope | 0.00 |
| lin_cw|code|embedding_moments|N2_cov_channel | 0.00 |
| lin_cw|code|embedding_moments|N3_jitter | 0.31 |
| lin_cw|code|embedding_moments|N4_gain_drift | 0.08 |
| lin_cw|code|embedding_moments|N5_glitch | 0.03 |
| lin_cw|code|embedding_moments|N6_channel_loss | 0.33 |
| lin_cw|code|embedding_moments|S1_amplitude | 0.89 |
| lin_cw|code|embedding_moments|S2_decay | 0.04 |
| lin_cw|code|embedding_moments|S3_t0 | 0.42 |
| lin_cw|code|embedding_moments|S4_alpha | 0.04 |
| lin_cw|code|embedding_moments|U1_tails | 0.00 |
| lin_cw|code|embedding_moments|U2_line | 1.00 |
| lin_cw|code|euclidean|N1_cov_slope | 0.04 |
| lin_cw|code|euclidean|N2_cov_channel | 0.03 |
| lin_cw|code|euclidean|N3_jitter | 0.36 |
| lin_cw|code|euclidean|N4_gain_drift | 0.03 |
| lin_cw|code|euclidean|N5_glitch | 0.03 |
| lin_cw|code|euclidean|N6_channel_loss | 0.52 |
| lin_cw|code|euclidean|S1_amplitude | 0.37 |
| lin_cw|code|euclidean|S2_decay | 0.67 |
| lin_cw|code|euclidean|S3_t0 | 0.39 |
| lin_cw|code|euclidean|S4_alpha | 0.07 |
| lin_cw|code|euclidean|U1_tails | 0.00 |
| lin_cw|code|euclidean|U2_line | 0.50 |
| lin_cw|code|input_ks_bh|N1_cov_slope | 0.35 |
| lin_cw|code|input_ks_bh|N2_cov_channel | 0.08 |
| lin_cw|code|input_ks_bh|N3_jitter | 0.03 |
| lin_cw|code|input_ks_bh|N4_gain_drift | 0.14 |
| lin_cw|code|input_ks_bh|N5_glitch | 0.06 |
| lin_cw|code|input_ks_bh|N6_channel_loss | 1.00 |
| lin_cw|code|input_ks_bh|S1_amplitude | 0.19 |
| lin_cw|code|input_ks_bh|S2_decay | 0.19 |
| lin_cw|code|input_ks_bh|S3_t0 | 0.31 |
| lin_cw|code|input_ks_bh|S4_alpha | 0.04 |
| lin_cw|code|input_ks_bh|U1_tails | 0.28 |
| lin_cw|code|input_ks_bh|U2_line | 1.00 |
| lin_cw|code|jacobian_projected|N1_cov_slope | 0.04 |
| lin_cw|code|jacobian_projected|N2_cov_channel | 0.03 |
| lin_cw|code|jacobian_projected|N3_jitter | 0.36 |
| lin_cw|code|jacobian_projected|N4_gain_drift | 0.03 |
| lin_cw|code|jacobian_projected|N5_glitch | 0.03 |
| lin_cw|code|jacobian_projected|N6_channel_loss | 0.52 |
| lin_cw|code|jacobian_projected|S1_amplitude | 0.37 |
| lin_cw|code|jacobian_projected|S2_decay | 0.67 |
| lin_cw|code|jacobian_projected|S3_t0 | 0.39 |
| lin_cw|code|jacobian_projected|S4_alpha | 0.07 |
| lin_cw|code|jacobian_projected|U1_tails | 0.00 |
| lin_cw|code|jacobian_projected|U2_line | 0.50 |
| lin_cw|code|layer_alignment_k4|N1_cov_slope | 0.00 |
| lin_cw|code|layer_alignment_k4|N2_cov_channel | 0.00 |
| lin_cw|code|layer_alignment_k4|N3_jitter | 0.00 |
| lin_cw|code|layer_alignment_k4|N4_gain_drift | 0.00 |
| lin_cw|code|layer_alignment_k4|N5_glitch | 0.00 |
| lin_cw|code|layer_alignment_k4|N6_channel_loss | 0.00 |
| lin_cw|code|layer_alignment_k4|S1_amplitude | 0.00 |
| lin_cw|code|layer_alignment_k4|S2_decay | 0.00 |
| lin_cw|code|layer_alignment_k4|S3_t0 | 0.00 |
| lin_cw|code|layer_alignment_k4|S4_alpha | 0.00 |
| lin_cw|code|layer_alignment_k4|U1_tails | 0.00 |
| lin_cw|code|layer_alignment_k4|U2_line | 0.00 |
| lin_cw|code|mmd_rbf|N1_cov_slope | 0.06 |
| lin_cw|code|mmd_rbf|N2_cov_channel | 0.00 |
| lin_cw|code|mmd_rbf|N3_jitter | 0.44 |
| lin_cw|code|mmd_rbf|N4_gain_drift | 0.06 |
| lin_cw|code|mmd_rbf|N5_glitch | 0.08 |
| lin_cw|code|mmd_rbf|N6_channel_loss | 0.56 |
| lin_cw|code|mmd_rbf|S1_amplitude | 0.85 |
| lin_cw|code|mmd_rbf|S2_decay | 0.56 |
| lin_cw|code|mmd_rbf|S3_t0 | 0.61 |
| lin_cw|code|mmd_rbf|S4_alpha | 0.00 |
| lin_cw|code|mmd_rbf|U1_tails | 0.00 |
| lin_cw|code|mmd_rbf|U2_line | 1.00 |
| lin_cw|code|output_residual|N1_cov_slope | 1.00 |
| lin_cw|code|output_residual|N2_cov_channel | 1.00 |
| lin_cw|code|output_residual|N3_jitter | 0.56 |
| lin_cw|code|output_residual|N4_gain_drift | 0.78 |
| lin_cw|code|output_residual|N5_glitch | 1.00 |
| lin_cw|code|output_residual|N6_channel_loss | 1.00 |
| lin_cw|code|output_residual|S1_amplitude | 0.96 |
| lin_cw|code|output_residual|S2_decay | 0.07 |
| lin_cw|code|output_residual|S3_t0 | 0.53 |
| lin_cw|code|output_residual|S4_alpha | 1.00 |
| lin_cw|code|output_residual|U1_tails | 0.94 |
| lin_cw|code|output_residual|U2_line | 1.00 |
| lin_cw|code|pullback_output|N1_cov_slope | 0.00 |
| lin_cw|code|pullback_output|N2_cov_channel | 0.03 |
| lin_cw|code|pullback_output|N3_jitter | 0.53 |
| lin_cw|code|pullback_output|N4_gain_drift | 0.11 |
| lin_cw|code|pullback_output|N5_glitch | 0.06 |
| lin_cw|code|pullback_output|N6_channel_loss | 0.67 |
| lin_cw|code|pullback_output|S1_amplitude | 1.00 |
| lin_cw|code|pullback_output|S2_decay | 1.00 |
| lin_cw|code|pullback_output|S3_t0 | 0.58 |
| lin_cw|code|pullback_output|S4_alpha | 0.11 |
| lin_cw|code|pullback_output|U1_tails | 0.00 |
| lin_cw|code|pullback_output|U2_line | 0.28 |
| lin_cw|code|pullback_whitened|N1_cov_slope | 0.04 |
| lin_cw|code|pullback_whitened|N2_cov_channel | 0.03 |
| lin_cw|code|pullback_whitened|N3_jitter | 0.36 |
| lin_cw|code|pullback_whitened|N4_gain_drift | 0.03 |
| lin_cw|code|pullback_whitened|N5_glitch | 0.03 |
| lin_cw|code|pullback_whitened|N6_channel_loss | 0.52 |
| lin_cw|code|pullback_whitened|S1_amplitude | 0.37 |
| lin_cw|code|pullback_whitened|S2_decay | 0.67 |
| lin_cw|code|pullback_whitened|S3_t0 | 0.39 |
| lin_cw|code|pullback_whitened|S4_alpha | 0.07 |
| lin_cw|code|pullback_whitened|U1_tails | 0.00 |
| lin_cw|code|pullback_whitened|U2_line | 0.50 |
| mlp_ae|code|c2st_logistic_auc|N1_cov_slope | 0.15 |
| mlp_ae|code|c2st_logistic_auc|N2_cov_channel | 0.06 |
| mlp_ae|code|c2st_logistic_auc|N3_jitter | 0.14 |
| mlp_ae|code|c2st_logistic_auc|N4_gain_drift | 0.06 |
| mlp_ae|code|c2st_logistic_auc|N5_glitch | 0.11 |
| mlp_ae|code|c2st_logistic_auc|N6_channel_loss | 0.33 |
| mlp_ae|code|c2st_logistic_auc|S1_amplitude | 0.19 |
| mlp_ae|code|c2st_logistic_auc|S2_decay | 0.30 |
| mlp_ae|code|c2st_logistic_auc|S3_t0 | 0.39 |
| mlp_ae|code|c2st_logistic_auc|S4_alpha | 0.04 |
| mlp_ae|code|c2st_logistic_auc|U1_tails | 0.06 |
| mlp_ae|code|c2st_logistic_auc|U2_line | 0.44 |
| mlp_ae|code|embedding_moments|N1_cov_slope | 0.46 |
| mlp_ae|code|embedding_moments|N2_cov_channel | 0.08 |
| mlp_ae|code|embedding_moments|N3_jitter | 0.25 |
| mlp_ae|code|embedding_moments|N4_gain_drift | 0.00 |
| mlp_ae|code|embedding_moments|N5_glitch | 0.17 |
| mlp_ae|code|embedding_moments|N6_channel_loss | 0.41 |
| mlp_ae|code|embedding_moments|S1_amplitude | 0.52 |
| mlp_ae|code|embedding_moments|S2_decay | 0.11 |
| mlp_ae|code|embedding_moments|S3_t0 | 0.44 |
| mlp_ae|code|embedding_moments|S4_alpha | 0.04 |
| mlp_ae|code|embedding_moments|U1_tails | 0.44 |
| mlp_ae|code|embedding_moments|U2_line | 1.00 |
| mlp_ae|code|euclidean|N1_cov_slope | 0.22 |
| mlp_ae|code|euclidean|N2_cov_channel | 0.08 |
| mlp_ae|code|euclidean|N3_jitter | 0.31 |
| mlp_ae|code|euclidean|N4_gain_drift | 0.11 |
| mlp_ae|code|euclidean|N5_glitch | 0.14 |
| mlp_ae|code|euclidean|N6_channel_loss | 0.59 |
| mlp_ae|code|euclidean|S1_amplitude | 0.22 |
| mlp_ae|code|euclidean|S2_decay | 0.74 |
| mlp_ae|code|euclidean|S3_t0 | 0.44 |
| mlp_ae|code|euclidean|S4_alpha | 0.19 |
| mlp_ae|code|euclidean|U1_tails | 0.06 |
| mlp_ae|code|euclidean|U2_line | 1.00 |
| mlp_ae|code|input_ks_bh|N1_cov_slope | 0.35 |
| mlp_ae|code|input_ks_bh|N2_cov_channel | 0.08 |
| mlp_ae|code|input_ks_bh|N3_jitter | 0.03 |
| mlp_ae|code|input_ks_bh|N4_gain_drift | 0.14 |
| mlp_ae|code|input_ks_bh|N5_glitch | 0.06 |
| mlp_ae|code|input_ks_bh|N6_channel_loss | 1.00 |
| mlp_ae|code|input_ks_bh|S1_amplitude | 0.19 |
| mlp_ae|code|input_ks_bh|S2_decay | 0.19 |
| mlp_ae|code|input_ks_bh|S3_t0 | 0.31 |
| mlp_ae|code|input_ks_bh|S4_alpha | 0.04 |
| mlp_ae|code|input_ks_bh|U1_tails | 0.28 |
| mlp_ae|code|input_ks_bh|U2_line | 1.00 |
| mlp_ae|code|jacobian_projected|N1_cov_slope | 0.22 |
| mlp_ae|code|jacobian_projected|N2_cov_channel | 0.08 |
| mlp_ae|code|jacobian_projected|N3_jitter | 0.31 |
| mlp_ae|code|jacobian_projected|N4_gain_drift | 0.11 |
| mlp_ae|code|jacobian_projected|N5_glitch | 0.14 |
| mlp_ae|code|jacobian_projected|N6_channel_loss | 0.59 |
| mlp_ae|code|jacobian_projected|S1_amplitude | 0.22 |
| mlp_ae|code|jacobian_projected|S2_decay | 0.74 |
| mlp_ae|code|jacobian_projected|S3_t0 | 0.44 |
| mlp_ae|code|jacobian_projected|S4_alpha | 0.19 |
| mlp_ae|code|jacobian_projected|U1_tails | 0.06 |
| mlp_ae|code|jacobian_projected|U2_line | 1.00 |
| mlp_ae|code|layer_alignment_k4|N1_cov_slope | 0.04 |
| mlp_ae|code|layer_alignment_k4|N2_cov_channel | 0.00 |
| mlp_ae|code|layer_alignment_k4|N3_jitter | 0.11 |
| mlp_ae|code|layer_alignment_k4|N4_gain_drift | 0.03 |
| mlp_ae|code|layer_alignment_k4|N5_glitch | 0.08 |
| mlp_ae|code|layer_alignment_k4|N6_channel_loss | 0.26 |
| mlp_ae|code|layer_alignment_k4|S1_amplitude | 0.00 |
| mlp_ae|code|layer_alignment_k4|S2_decay | 0.26 |
| mlp_ae|code|layer_alignment_k4|S3_t0 | 0.06 |
| mlp_ae|code|layer_alignment_k4|S4_alpha | 0.04 |
| mlp_ae|code|layer_alignment_k4|U1_tails | 0.00 |
| mlp_ae|code|layer_alignment_k4|U2_line | 0.72 |
| mlp_ae|code|mmd_rbf|N1_cov_slope | 0.37 |
| mlp_ae|code|mmd_rbf|N2_cov_channel | 0.11 |
| mlp_ae|code|mmd_rbf|N3_jitter | 0.33 |
| mlp_ae|code|mmd_rbf|N4_gain_drift | 0.08 |
| mlp_ae|code|mmd_rbf|N5_glitch | 0.11 |
| mlp_ae|code|mmd_rbf|N6_channel_loss | 0.59 |
| mlp_ae|code|mmd_rbf|S1_amplitude | 0.33 |
| mlp_ae|code|mmd_rbf|S2_decay | 0.63 |
| mlp_ae|code|mmd_rbf|S3_t0 | 0.64 |
| mlp_ae|code|mmd_rbf|S4_alpha | 0.11 |
| mlp_ae|code|mmd_rbf|U1_tails | 0.22 |
| mlp_ae|code|mmd_rbf|U2_line | 1.00 |
| mlp_ae|code|output_residual|N1_cov_slope | 1.00 |
| mlp_ae|code|output_residual|N2_cov_channel | 1.00 |
| mlp_ae|code|output_residual|N3_jitter | 0.50 |
| mlp_ae|code|output_residual|N4_gain_drift | 0.78 |
| mlp_ae|code|output_residual|N5_glitch | 1.00 |
| mlp_ae|code|output_residual|N6_channel_loss | 1.00 |
| mlp_ae|code|output_residual|S1_amplitude | 0.96 |
| mlp_ae|code|output_residual|S2_decay | 0.26 |
| mlp_ae|code|output_residual|S3_t0 | 0.53 |
| mlp_ae|code|output_residual|S4_alpha | 1.00 |
| mlp_ae|code|output_residual|U1_tails | 0.94 |
| mlp_ae|code|output_residual|U2_line | 1.00 |
| mlp_ae|code|pullback_output|N1_cov_slope | 0.09 |
| mlp_ae|code|pullback_output|N2_cov_channel | 0.06 |
| mlp_ae|code|pullback_output|N3_jitter | 0.28 |
| mlp_ae|code|pullback_output|N4_gain_drift | 0.03 |
| mlp_ae|code|pullback_output|N5_glitch | 0.11 |
| mlp_ae|code|pullback_output|N6_channel_loss | 0.11 |
| mlp_ae|code|pullback_output|S1_amplitude | 0.15 |
| mlp_ae|code|pullback_output|S2_decay | 0.74 |
| mlp_ae|code|pullback_output|S3_t0 | 0.22 |
| mlp_ae|code|pullback_output|S4_alpha | 0.04 |
| mlp_ae|code|pullback_output|U1_tails | 0.00 |
| mlp_ae|code|pullback_output|U2_line | 0.94 |
| mlp_ae|code|pullback_whitened|N1_cov_slope | 0.15 |
| mlp_ae|code|pullback_whitened|N2_cov_channel | 0.03 |
| mlp_ae|code|pullback_whitened|N3_jitter | 0.06 |
| mlp_ae|code|pullback_whitened|N4_gain_drift | 0.03 |
| mlp_ae|code|pullback_whitened|N5_glitch | 0.03 |
| mlp_ae|code|pullback_whitened|N6_channel_loss | 0.52 |
| mlp_ae|code|pullback_whitened|S1_amplitude | 0.11 |
| mlp_ae|code|pullback_whitened|S2_decay | 0.22 |
| mlp_ae|code|pullback_whitened|S3_t0 | 0.22 |
| mlp_ae|code|pullback_whitened|S4_alpha | 0.11 |
| mlp_ae|code|pullback_whitened|U1_tails | 0.00 |
| mlp_ae|code|pullback_whitened|U2_line | 0.39 |
| mlp_ae|h1|c2st_logistic_auc|N1_cov_slope | 0.11 |
| mlp_ae|h1|c2st_logistic_auc|N2_cov_channel | 0.06 |
| mlp_ae|h1|c2st_logistic_auc|N3_jitter | 0.00 |
| mlp_ae|h1|c2st_logistic_auc|N4_gain_drift | 0.06 |
| mlp_ae|h1|c2st_logistic_auc|N5_glitch | 0.03 |
| mlp_ae|h1|c2st_logistic_auc|N6_channel_loss | 0.37 |
| mlp_ae|h1|c2st_logistic_auc|S1_amplitude | 0.07 |
| mlp_ae|h1|c2st_logistic_auc|S2_decay | 0.00 |
| mlp_ae|h1|c2st_logistic_auc|S3_t0 | 0.33 |
| mlp_ae|h1|c2st_logistic_auc|S4_alpha | 0.04 |
| mlp_ae|h1|c2st_logistic_auc|U1_tails | 0.06 |
| mlp_ae|h1|c2st_logistic_auc|U2_line | 0.50 |
| mlp_ae|h1|embedding_moments|N1_cov_slope | 0.48 |
| mlp_ae|h1|embedding_moments|N2_cov_channel | 0.08 |
| mlp_ae|h1|embedding_moments|N3_jitter | 0.39 |
| mlp_ae|h1|embedding_moments|N4_gain_drift | 0.08 |
| mlp_ae|h1|embedding_moments|N5_glitch | 0.39 |
| mlp_ae|h1|embedding_moments|N6_channel_loss | 0.56 |
| mlp_ae|h1|embedding_moments|S1_amplitude | 0.67 |
| mlp_ae|h1|embedding_moments|S2_decay | 0.33 |
| mlp_ae|h1|embedding_moments|S3_t0 | 0.47 |
| mlp_ae|h1|embedding_moments|S4_alpha | 0.19 |
| mlp_ae|h1|embedding_moments|U1_tails | 0.50 |
| mlp_ae|h1|embedding_moments|U2_line | 1.00 |
| mlp_ae|h1|euclidean|N1_cov_slope | 0.07 |
| mlp_ae|h1|euclidean|N2_cov_channel | 0.00 |
| mlp_ae|h1|euclidean|N3_jitter | 0.31 |
| mlp_ae|h1|euclidean|N4_gain_drift | 0.06 |
| mlp_ae|h1|euclidean|N5_glitch | 0.08 |
| mlp_ae|h1|euclidean|N6_channel_loss | 0.52 |
| mlp_ae|h1|euclidean|S1_amplitude | 0.04 |
| mlp_ae|h1|euclidean|S2_decay | 0.63 |
| mlp_ae|h1|euclidean|S3_t0 | 0.33 |
| mlp_ae|h1|euclidean|S4_alpha | 0.04 |
| mlp_ae|h1|euclidean|U1_tails | 0.00 |
| mlp_ae|h1|euclidean|U2_line | 1.00 |
| mlp_ae|h1|input_ks_bh|N1_cov_slope | 0.35 |
| mlp_ae|h1|input_ks_bh|N2_cov_channel | 0.08 |
| mlp_ae|h1|input_ks_bh|N3_jitter | 0.03 |
| mlp_ae|h1|input_ks_bh|N4_gain_drift | 0.14 |
| mlp_ae|h1|input_ks_bh|N5_glitch | 0.06 |
| mlp_ae|h1|input_ks_bh|N6_channel_loss | 1.00 |
| mlp_ae|h1|input_ks_bh|S1_amplitude | 0.19 |
| mlp_ae|h1|input_ks_bh|S2_decay | 0.19 |
| mlp_ae|h1|input_ks_bh|S3_t0 | 0.31 |
| mlp_ae|h1|input_ks_bh|S4_alpha | 0.04 |
| mlp_ae|h1|input_ks_bh|U1_tails | 0.28 |
| mlp_ae|h1|input_ks_bh|U2_line | 1.00 |
| mlp_ae|h1|jacobian_projected|N1_cov_slope | 0.13 |
| mlp_ae|h1|jacobian_projected|N2_cov_channel | 0.03 |
| mlp_ae|h1|jacobian_projected|N3_jitter | 0.31 |
| mlp_ae|h1|jacobian_projected|N4_gain_drift | 0.03 |
| mlp_ae|h1|jacobian_projected|N5_glitch | 0.06 |
| mlp_ae|h1|jacobian_projected|N6_channel_loss | 0.52 |
| mlp_ae|h1|jacobian_projected|S1_amplitude | 0.11 |
| mlp_ae|h1|jacobian_projected|S2_decay | 0.63 |
| mlp_ae|h1|jacobian_projected|S3_t0 | 0.33 |
| mlp_ae|h1|jacobian_projected|S4_alpha | 0.07 |
| mlp_ae|h1|jacobian_projected|U1_tails | 0.00 |
| mlp_ae|h1|jacobian_projected|U2_line | 0.94 |
| mlp_ae|h1|layer_alignment_k4|N1_cov_slope | 0.02 |
| mlp_ae|h1|layer_alignment_k4|N2_cov_channel | 0.06 |
| mlp_ae|h1|layer_alignment_k4|N3_jitter | 0.22 |
| mlp_ae|h1|layer_alignment_k4|N4_gain_drift | 0.08 |
| mlp_ae|h1|layer_alignment_k4|N5_glitch | 0.03 |
| mlp_ae|h1|layer_alignment_k4|N6_channel_loss | 0.52 |
| mlp_ae|h1|layer_alignment_k4|S1_amplitude | 0.07 |
| mlp_ae|h1|layer_alignment_k4|S2_decay | 0.44 |
| mlp_ae|h1|layer_alignment_k4|S3_t0 | 0.06 |
| mlp_ae|h1|layer_alignment_k4|S4_alpha | 0.00 |
| mlp_ae|h1|layer_alignment_k4|U1_tails | 0.00 |
| mlp_ae|h1|layer_alignment_k4|U2_line | 0.67 |
| mlp_ae|h1|mmd_rbf|N1_cov_slope | 0.31 |
| mlp_ae|h1|mmd_rbf|N2_cov_channel | 0.03 |
| mlp_ae|h1|mmd_rbf|N3_jitter | 0.28 |
| mlp_ae|h1|mmd_rbf|N4_gain_drift | 0.03 |
| mlp_ae|h1|mmd_rbf|N5_glitch | 0.11 |
| mlp_ae|h1|mmd_rbf|N6_channel_loss | 0.56 |
| mlp_ae|h1|mmd_rbf|S1_amplitude | 0.15 |
| mlp_ae|h1|mmd_rbf|S2_decay | 0.48 |
| mlp_ae|h1|mmd_rbf|S3_t0 | 0.44 |
| mlp_ae|h1|mmd_rbf|S4_alpha | 0.04 |
| mlp_ae|h1|mmd_rbf|U1_tails | 0.11 |
| mlp_ae|h1|mmd_rbf|U2_line | 1.00 |
| mlp_ae|h1|output_residual|N1_cov_slope | 1.00 |
| mlp_ae|h1|output_residual|N2_cov_channel | 1.00 |
| mlp_ae|h1|output_residual|N3_jitter | 0.50 |
| mlp_ae|h1|output_residual|N4_gain_drift | 0.78 |
| mlp_ae|h1|output_residual|N5_glitch | 1.00 |
| mlp_ae|h1|output_residual|N6_channel_loss | 1.00 |
| mlp_ae|h1|output_residual|S1_amplitude | 0.96 |
| mlp_ae|h1|output_residual|S2_decay | 0.26 |
| mlp_ae|h1|output_residual|S3_t0 | 0.53 |
| mlp_ae|h1|output_residual|S4_alpha | 1.00 |
| mlp_ae|h1|output_residual|U1_tails | 0.94 |
| mlp_ae|h1|output_residual|U2_line | 1.00 |
| mlp_ae|h1|pullback_output|N1_cov_slope | 0.13 |
| mlp_ae|h1|pullback_output|N2_cov_channel | 0.03 |
| mlp_ae|h1|pullback_output|N3_jitter | 0.28 |
| mlp_ae|h1|pullback_output|N4_gain_drift | 0.03 |
| mlp_ae|h1|pullback_output|N5_glitch | 0.14 |
| mlp_ae|h1|pullback_output|N6_channel_loss | 0.11 |
| mlp_ae|h1|pullback_output|S1_amplitude | 0.19 |
| mlp_ae|h1|pullback_output|S2_decay | 0.70 |
| mlp_ae|h1|pullback_output|S3_t0 | 0.17 |
| mlp_ae|h1|pullback_output|S4_alpha | 0.04 |
| mlp_ae|h1|pullback_output|U1_tails | 0.06 |
| mlp_ae|h1|pullback_output|U2_line | 0.94 |
| mlp_ae|h1|pullback_whitened|N1_cov_slope | 0.06 |
| mlp_ae|h1|pullback_whitened|N2_cov_channel | 0.06 |
| mlp_ae|h1|pullback_whitened|N3_jitter | 0.03 |
| mlp_ae|h1|pullback_whitened|N4_gain_drift | 0.06 |
| mlp_ae|h1|pullback_whitened|N5_glitch | 0.00 |
| mlp_ae|h1|pullback_whitened|N6_channel_loss | 0.52 |
| mlp_ae|h1|pullback_whitened|S1_amplitude | 0.07 |
| mlp_ae|h1|pullback_whitened|S2_decay | 0.00 |
| mlp_ae|h1|pullback_whitened|S3_t0 | 0.11 |
| mlp_ae|h1|pullback_whitened|S4_alpha | 0.04 |
| mlp_ae|h1|pullback_whitened|U1_tails | 0.00 |
| mlp_ae|h1|pullback_whitened|U2_line | 0.06 |
| mlp_ae|h2|c2st_logistic_auc|N1_cov_slope | 0.11 |
| mlp_ae|h2|c2st_logistic_auc|N2_cov_channel | 0.00 |
| mlp_ae|h2|c2st_logistic_auc|N3_jitter | 0.06 |
| mlp_ae|h2|c2st_logistic_auc|N4_gain_drift | 0.03 |
| mlp_ae|h2|c2st_logistic_auc|N5_glitch | 0.03 |
| mlp_ae|h2|c2st_logistic_auc|N6_channel_loss | 0.41 |
| mlp_ae|h2|c2st_logistic_auc|S1_amplitude | 0.04 |
| mlp_ae|h2|c2st_logistic_auc|S2_decay | 0.07 |
| mlp_ae|h2|c2st_logistic_auc|S3_t0 | 0.36 |
| mlp_ae|h2|c2st_logistic_auc|S4_alpha | 0.00 |
| mlp_ae|h2|c2st_logistic_auc|U1_tails | 0.06 |
| mlp_ae|h2|c2st_logistic_auc|U2_line | 0.44 |
| mlp_ae|h2|embedding_moments|N1_cov_slope | 0.46 |
| mlp_ae|h2|embedding_moments|N2_cov_channel | 0.08 |
| mlp_ae|h2|embedding_moments|N3_jitter | 0.33 |
| mlp_ae|h2|embedding_moments|N4_gain_drift | 0.06 |
| mlp_ae|h2|embedding_moments|N5_glitch | 0.25 |
| mlp_ae|h2|embedding_moments|N6_channel_loss | 0.56 |
| mlp_ae|h2|embedding_moments|S1_amplitude | 0.48 |
| mlp_ae|h2|embedding_moments|S2_decay | 0.44 |
| mlp_ae|h2|embedding_moments|S3_t0 | 0.50 |
| mlp_ae|h2|embedding_moments|S4_alpha | 0.07 |
| mlp_ae|h2|embedding_moments|U1_tails | 0.44 |
| mlp_ae|h2|embedding_moments|U2_line | 1.00 |
| mlp_ae|h2|euclidean|N1_cov_slope | 0.26 |
| mlp_ae|h2|euclidean|N2_cov_channel | 0.03 |
| mlp_ae|h2|euclidean|N3_jitter | 0.33 |
| mlp_ae|h2|euclidean|N4_gain_drift | 0.08 |
| mlp_ae|h2|euclidean|N5_glitch | 0.17 |
| mlp_ae|h2|euclidean|N6_channel_loss | 0.56 |
| mlp_ae|h2|euclidean|S1_amplitude | 0.19 |
| mlp_ae|h2|euclidean|S2_decay | 0.74 |
| mlp_ae|h2|euclidean|S3_t0 | 0.44 |
| mlp_ae|h2|euclidean|S4_alpha | 0.15 |
| mlp_ae|h2|euclidean|U1_tails | 0.06 |
| mlp_ae|h2|euclidean|U2_line | 1.00 |
| mlp_ae|h2|input_ks_bh|N1_cov_slope | 0.35 |
| mlp_ae|h2|input_ks_bh|N2_cov_channel | 0.08 |
| mlp_ae|h2|input_ks_bh|N3_jitter | 0.03 |
| mlp_ae|h2|input_ks_bh|N4_gain_drift | 0.14 |
| mlp_ae|h2|input_ks_bh|N5_glitch | 0.06 |
| mlp_ae|h2|input_ks_bh|N6_channel_loss | 1.00 |
| mlp_ae|h2|input_ks_bh|S1_amplitude | 0.19 |
| mlp_ae|h2|input_ks_bh|S2_decay | 0.19 |
| mlp_ae|h2|input_ks_bh|S3_t0 | 0.31 |
| mlp_ae|h2|input_ks_bh|S4_alpha | 0.04 |
| mlp_ae|h2|input_ks_bh|U1_tails | 0.28 |
| mlp_ae|h2|input_ks_bh|U2_line | 1.00 |
| mlp_ae|h2|jacobian_projected|N1_cov_slope | 0.20 |
| mlp_ae|h2|jacobian_projected|N2_cov_channel | 0.06 |
| mlp_ae|h2|jacobian_projected|N3_jitter | 0.31 |
| mlp_ae|h2|jacobian_projected|N4_gain_drift | 0.08 |
| mlp_ae|h2|jacobian_projected|N5_glitch | 0.14 |
| mlp_ae|h2|jacobian_projected|N6_channel_loss | 0.59 |
| mlp_ae|h2|jacobian_projected|S1_amplitude | 0.22 |
| mlp_ae|h2|jacobian_projected|S2_decay | 0.74 |
| mlp_ae|h2|jacobian_projected|S3_t0 | 0.47 |
| mlp_ae|h2|jacobian_projected|S4_alpha | 0.19 |
| mlp_ae|h2|jacobian_projected|U1_tails | 0.06 |
| mlp_ae|h2|jacobian_projected|U2_line | 1.00 |
| mlp_ae|h2|layer_alignment_k4|N1_cov_slope | 0.07 |
| mlp_ae|h2|layer_alignment_k4|N2_cov_channel | 0.03 |
| mlp_ae|h2|layer_alignment_k4|N3_jitter | 0.17 |
| mlp_ae|h2|layer_alignment_k4|N4_gain_drift | 0.08 |
| mlp_ae|h2|layer_alignment_k4|N5_glitch | 0.08 |
| mlp_ae|h2|layer_alignment_k4|N6_channel_loss | 0.26 |
| mlp_ae|h2|layer_alignment_k4|S1_amplitude | 0.04 |
| mlp_ae|h2|layer_alignment_k4|S2_decay | 0.41 |
| mlp_ae|h2|layer_alignment_k4|S3_t0 | 0.11 |
| mlp_ae|h2|layer_alignment_k4|S4_alpha | 0.00 |
| mlp_ae|h2|layer_alignment_k4|U1_tails | 0.00 |
| mlp_ae|h2|layer_alignment_k4|U2_line | 0.61 |
| mlp_ae|h2|mmd_rbf|N1_cov_slope | 0.33 |
| mlp_ae|h2|mmd_rbf|N2_cov_channel | 0.03 |
| mlp_ae|h2|mmd_rbf|N3_jitter | 0.28 |
| mlp_ae|h2|mmd_rbf|N4_gain_drift | 0.03 |
| mlp_ae|h2|mmd_rbf|N5_glitch | 0.11 |
| mlp_ae|h2|mmd_rbf|N6_channel_loss | 0.56 |
| mlp_ae|h2|mmd_rbf|S1_amplitude | 0.26 |
| mlp_ae|h2|mmd_rbf|S2_decay | 0.56 |
| mlp_ae|h2|mmd_rbf|S3_t0 | 0.53 |
| mlp_ae|h2|mmd_rbf|S4_alpha | 0.04 |
| mlp_ae|h2|mmd_rbf|U1_tails | 0.11 |
| mlp_ae|h2|mmd_rbf|U2_line | 1.00 |
| mlp_ae|h2|output_residual|N1_cov_slope | 1.00 |
| mlp_ae|h2|output_residual|N2_cov_channel | 1.00 |
| mlp_ae|h2|output_residual|N3_jitter | 0.50 |
| mlp_ae|h2|output_residual|N4_gain_drift | 0.78 |
| mlp_ae|h2|output_residual|N5_glitch | 1.00 |
| mlp_ae|h2|output_residual|N6_channel_loss | 1.00 |
| mlp_ae|h2|output_residual|S1_amplitude | 0.96 |
| mlp_ae|h2|output_residual|S2_decay | 0.26 |
| mlp_ae|h2|output_residual|S3_t0 | 0.53 |
| mlp_ae|h2|output_residual|S4_alpha | 1.00 |
| mlp_ae|h2|output_residual|U1_tails | 0.94 |
| mlp_ae|h2|output_residual|U2_line | 1.00 |
| mlp_ae|h2|pullback_output|N1_cov_slope | 0.09 |
| mlp_ae|h2|pullback_output|N2_cov_channel | 0.06 |
| mlp_ae|h2|pullback_output|N3_jitter | 0.28 |
| mlp_ae|h2|pullback_output|N4_gain_drift | 0.03 |
| mlp_ae|h2|pullback_output|N5_glitch | 0.11 |
| mlp_ae|h2|pullback_output|N6_channel_loss | 0.11 |
| mlp_ae|h2|pullback_output|S1_amplitude | 0.15 |
| mlp_ae|h2|pullback_output|S2_decay | 0.74 |
| mlp_ae|h2|pullback_output|S3_t0 | 0.22 |
| mlp_ae|h2|pullback_output|S4_alpha | 0.04 |
| mlp_ae|h2|pullback_output|U1_tails | 0.00 |
| mlp_ae|h2|pullback_output|U2_line | 0.94 |
| mlp_ae|h2|pullback_whitened|N1_cov_slope | 0.15 |
| mlp_ae|h2|pullback_whitened|N2_cov_channel | 0.03 |
| mlp_ae|h2|pullback_whitened|N3_jitter | 0.06 |
| mlp_ae|h2|pullback_whitened|N4_gain_drift | 0.03 |
| mlp_ae|h2|pullback_whitened|N5_glitch | 0.03 |
| mlp_ae|h2|pullback_whitened|N6_channel_loss | 0.52 |
| mlp_ae|h2|pullback_whitened|S1_amplitude | 0.11 |
| mlp_ae|h2|pullback_whitened|S2_decay | 0.22 |
| mlp_ae|h2|pullback_whitened|S3_t0 | 0.22 |
| mlp_ae|h2|pullback_whitened|S4_alpha | 0.11 |
| mlp_ae|h2|pullback_whitened|U1_tails | 0.00 |
| mlp_ae|h2|pullback_whitened|U2_line | 0.39 |

## Layerwise alignment-drop score (mlp_ae)

| contract | stage | mean score |
|---|---|---|
| N | h1 | 0.07 |
| N | h2 | 0.30 |
| N | code | 0.25 |
| S | h1 | -0.01 |
| S | h2 | 0.50 |
| S | code | 0.40 |
