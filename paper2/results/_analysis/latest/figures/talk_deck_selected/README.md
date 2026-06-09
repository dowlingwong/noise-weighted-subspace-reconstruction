# NPML Talk Figure Map

Curated figures for `NPML_dwong_2026 (1).pdf`, revised after external plot review.

Main framing rule: the QP figures are controlled stress tests, and the Transformer figures are conceptual/diagnostic motivation. Do not present either as final performance claims.

## Recommended Slide Placement

| Slide | Topic | Use this figure | Why it belongs there |
| --- | --- | --- | --- |
| 2 | Physics and signal formation | Optional preview: `s03_qp_clean_shape_family.png` | If there is room, it gives the audience an immediate visual for pulse-shape variation before the math. |
| 3 | Physical signal model | `s03_qp_clean_shape_family.png` | Shows the controlled signal manifold: arrival time and decay change the trace, so the model is `x = s(z) + n`, not one immutable pulse. |
| 4 | Likelihood determines the loss | `s04_likelihood_geometry_cartoon.png` | Makes the theorem visual: white-noise geometry and detector-noise geometry rank residuals differently. |
| 5 | OF as constrained maximum likelihood | `s05_of_fixed_template_limitation.png` | Replaces the old mismatched waveform overlay. It directly shows that an amplitude-only template fit leaves structured residuals when timing/decay vary. |
| 6 | PCA: learning a signal subspace | Optional backup: `backup_metric_ablation_triptych.png` | Use only if you want data evidence here. It supports the transition from fixed template to learned subspace. |
| 7 | EMPCA under detector geometry | Prefer verbal callback to slide 4, or backup: `backup_metric_ablation_triptych.png` | Avoid repeating the same geometry cartoon full-size. If using a figure, use the metric-ablation backup to show that changing the metric changes the learned subspace. |
| 8 | Linear AEs inherit likelihood geometry | `s08_ae_metric_decision_panel.png` | Main AE figure. Same models, three scoring rules, different winners: native MSE, detector-weighted residual, and downstream amplitude RMSE do not tell the same story. |
| 8 | AE downstream consequence | Optional: `s08_amplitude_calibration_panels.png` | Use as backup or second panel if there is room. It connects reconstruction choices to amplitude residuals. |
| 9 | Correct loss is necessary, not sufficient | `s09_qp_noise_pipeline_examples.png` | Best stress-test opener. It cleanly separates ideal stationary Gaussian noise from non-stationarity and artifacts. |
| 10 | NFPA structured extension | Backup only: `backup_nfpa_vs_empca_triptych.png` | Use carefully. Frame it as a structured/separable restriction related to EMPCA, not necessarily a dramatically different subspace. |
| 11 | Architecture/noise model motivation | `s11_qp_multichannel_correlation.png` | Shows why channel correlations matter: shared readout/cryostat-like noise can create geometry that per-channel models miss. |
| 12 | From sequence to channel-time tokens | `s15_transformer_tokenization_schematic.png` | Use as conceptual motivation. It shows trace tokenization and latent activation diagnostics without claiming performance superiority. |
| 13 | Wrong loss = wrong geometry | `s13_ae_residual_spectrum.png` | Smoothed and annotated after review. It shows that residuals that look acceptable in raw frequency space can become costly after detector-PSD weighting. |
| 14 | Correct loss is necessary, but not sufficient | `s14_qp_noise_psd_shift.png` and `s14_qp_artifact_before_after.png` | Shows two assumption failures: one global PSD can miss non-stationary segments, and artifacts need flag/refit or robust modeling. |
| 15 | Transformer direction | `s15_transformer_tokenization_schematic.png`; optional backup `s15_transformer_latent_activation_diagnostic.png` | Say explicitly: this is a diagnostic/motivation plot, not an attention map and not a win claim. The separate activation map is optional because it repeats the bottom panel. |
| 16 | Conclusion | `backup_final_2x2_matrix.png` only if reframed | This figure is about metric correctness and coverage, not architecture hierarchy. If slide 16 keeps the architecture-hierarchy text, use this only as backup or revise the slide text. |
| 17 | Key result | Prefer `s08_amplitude_calibration_panels.png`; backup `s17_metric_tradeoff_scatter_backup.png` | End with amplitude if possible. The scatter is now cleaner, but it remains a backup performance view because Transformer points are diagnostic and not competitive. |

## Strongest Talk Spine

1. `s03_qp_clean_shape_family.png`: signal is a manifold, not one template.
2. `s04_likelihood_geometry_cartoon.png`: detector PSD defines the geometry.
3. `s05_of_fixed_template_limitation.png`: OF is the fixed-template constrained case.
4. `s08_ae_metric_decision_panel.png`: model choice changes with the metric.
5. `s13_ae_residual_spectrum.png`: detector weighting explains why.
6. `s09_qp_noise_pipeline_examples.png`: theorem assumptions versus structured-noise violations.
7. `s14_qp_noise_psd_shift.png` and `s14_qp_artifact_before_after.png`: why correct loss is necessary but not sufficient.
8. `s11_qp_multichannel_correlation.png` and `s15_transformer_tokenization_schematic.png`: why architecture/noise models need channel-time geometry.

## Speaker Notes

Slide 4: "Circles are what small residual means under white-noise assumptions; tilted ellipses are what small residual means under detector noise. A residual can be small in the wrong geometry and large in the right one."

Slide 5: "OF is maximum likelihood only after I restrict the signal to one template direction. Once timing or decay changes, the best amplitude fit leaves structured residuals."

Slide 8: "Same four AE variants, three scoring rules, different winners. The left panel is the usual ML objective; the middle and right panels are closer to the detector and physics goals."

Slide 8 nuance: "Raw plus Mahalanobis and prewhite plus Mahalanobis are close in detector residual. Mahalanobis loss already encodes the covariance; prewhitening becomes especially useful when the architecture cannot apply the full covariance directly."

Slide 12/15: "This is not a result claiming the Transformer wins. It is a diagnostic/motivation figure: if we use attention, the natural units are channel-time tokens after aligning the input with detector geometry."

Slide 13: "Look at the highlighted low-frequency band. A residual that seems smooth in native space can be exactly the residual that matters after PSD weighting."

Slide 14 PSD: "The black curve is an idealized stationary target shape. The colored curves are segment estimates from a non-stationary process drifting away from one global PSD."

Slide 15 activation map: "This is latent activation magnitude, not an attention weight. It is a sanity-check diagnostic, not an interpretability claim."

## Files Kept As Backup

| File | Backup use |
| --- | --- |
| `backup_ae_waveform_reconstruction_overlay.png` | AE trace-level sanity check only. Do not use for slide 5. |
| `backup_metric_ablation_triptych.png` | Optional evidence that metric choice changes the learned geometry. |
| `backup_nfpa_vs_empca_triptych.png` | Use carefully; frame NFPA as structured/separable and related to EMPCA. |
| `backup_coverage_ablation_heatmap.png` | Optional coverage/latent-factor backup. |
| `backup_final_2x2_matrix.png` | Use only if discussing metric correctness versus coverage, not architecture hierarchy. |
| `s17_metric_tradeoff_scatter_backup.png` | Backup performance view; not the main closing figure. |
