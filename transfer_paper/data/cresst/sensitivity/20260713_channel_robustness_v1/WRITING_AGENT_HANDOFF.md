# Writing-agent handoff: CRESST post-freeze sensitivity

## Claim recommendation

The channel-balanced sensitivity analysis supports lower EMPCA residual than fixed-template OF. The matched-rank PCA contrast also favors EMPCA. State that only seven clusters make the hierarchical interval coarse and potentially unstable. Keep the archived pooled 300-replicate event bootstrap as the predeclared primary result.

The conclusion is limited to held-out within-channel linear reconstruction on the seven released channels. Do not claim cross-channel/detector transfer, detector sensitivity or threshold improvement, 19,335 independent detector-level replications, nonlinear-autoencoder evidence, a universal covariance-weighting advantage when the PCA contrast is heterogeneous, or replacement of the frozen primary gate.

## Pooled-primary points, channel-balanced points, and hierarchical intervals

| estimand | pooled event-weighted mean | equal-channel mean | hierarchical 95% interval |
|---|---:|---:|---:|
| d_OF | 1012.266448 | 731.874857 | [110.6883176, 1537.475104] |
| d_comp | 352.1707059 | 279.2600917 | [36.51588735, 569.014246] |
| d_multi_raw | 318.6584144 | 252.6588896 | [32.70533847, 514.9194549] |
| d_PCA | 3.571959551 | 3.782055086 | [1.220081216, 6.69084711] |
| d_control_mismatched_of_template | 7924.090897 | 7027.363609 | [1317.601661, 14127.19174] |
| d_control_noise_only_train | 489.4012569 | 418.6693091 | [38.69159153, 905.0049062] |
| d_control_permuted_time_bins | 59812.71729 | 47165.19306 | [13228.83587, 87875.34527] |
| d_control_random_weighted_basis | 4013.472613 | 3179.235333 | [690.7085764, 6021.773954] |
| d_control_wrong_psd | 1.567418692 | 1.884531913 | [0.4564353374, 3.715111148] |

## Normalized channel effects

Values are residual-reduction fractions, not percentages.

| channel | OF − EMPCA | multi-template OF − EMPCA | PCA − EMPCA |
|---|---:|---:|---:|
| 33_36.0 | 0.9878352008 | 0.9578148913 | 0.1427545633 |
| 33_37.0 | 0.8972825052 | 0.801020502 | 0.1149211256 |
| 35_26.0 | 0.9811710082 | 0.9464500655 | 0.6203613355 |
| 35_27.0 | 0.9571598647 | 0.8107403287 | 0.1306372333 |
| 35_56.0 | 0.9718541635 | 0.8892783488 | 0.2527792547 |
| 35_57.0 | 0.9805295275 | 0.960428855 | 0.2323252065 |
| 35_58.0 | 0.9571118874 | 0.9031746143 | 0.01604533779 |

## Reproduction status

The event export reproduced the frozen archive within the configured tolerance; no reproduction discrepancy remained when this handoff was written.

## Key output checksums

- `channel_summary.csv` — `8549866cb2183a557a4600ee7e4255eeb220ce908009e0a54ca1bba6d98d04d6`
- `estimand_summary.csv` — `2eba7315c23af56ab24271031b5ec4c3ee85477212b303c73e991d32ed38cd00`
- `event_residuals_lossless.npz` — `0f2cd048e029d7beb04bf754de50a50ee232eec957e11d1cebba7f7fd733ebd5`
- `paper_facing_table.csv` — `1d724d5a3f3229db18501edb98273250048fbb6a51b2de83a454cdd29e81322b`
- `paper_figure.pdf` — `41398295a53737d4b7086ce0eb260f9ab0480b1f0f4e1d68925b96094ed9e1d4`
- `reproduction_and_leakage_audit.md` — `43fd3489b6824c809057ca1522951aff2cbc2b67ba5b8096590efa034bc2ae1b`
- `reproduction_audit.json` — `45ce222a7acb7d397ced6737e0f56464eccea826fe2d4d5e2bc0a67076d8f261`
- `summary.json` — `18a78c8a0a46e977075045c7615a605d1ff4d07468d040650a0c31f8005a0729`

See `summary.json`, `channel_summary.csv`, and `estimand_summary.csv` for exact numerical values; see `SHA256SUMS` for the complete output manifest.
