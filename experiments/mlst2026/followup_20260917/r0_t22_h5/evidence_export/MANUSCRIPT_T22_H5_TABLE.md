# T22 H5 random-effects interval summary (reporting amendment R0-v1)

Computed read-only from the stored T22-S Monte-Carlo records.  No new cells were executed; the frozen
reduction is unchanged.  Coverage uses the declared population equal-file estimand (``true_mu``).

| scenario scope | n | RE in band | RE coverage median (range) | two-stage median | t median | median RE/two-stage length | exploratory displacement scenarios |
|---|---|---|---|---|---|---|---|
| non-drift (H1/H5 scope) | 65 | 24 | 0.876 (0.427–0.998) | 0.959 | 0.932 | 0.636 | 29 |
| drift stress (separate) | 4 | 0 | 0.822 (0.817–0.842) | 0.878 | 0.944 | 0.702 | 0 |
| all scenarios (descriptive) | 69 | 24 | 0.871 (0.427–0.998) | 0.958 | 0.933 | 0.636 | 29 |

The random-effects interval is **not** a replacement for the primary interval.  It is a DerSimonian-Laird
inverse-variance mean of the per-file effects and targets the precision-weighted file-effect mean, while the
two-stage interval targets the equal-file mean.  The last column counts only *exploratory finite-sample
estimator displacement*: where the RE estimate sits materially away from the equal-file point on these
replicates.  It does **not** establish a population estimand mismatch and does **not** prove the cause of the
coverage shortfall; those need the two population targets and the generator's file-sampling law, stated
separately in the report's ``target.population_targets``/``target.assumptions``.
H5 is a *secondary* reporting hypothesis.  Its intervals are reported here beside the primary and the t interval;
the frozen H1 verdict (13 of 65 non-drift scenarios outside the coverage band) is unchanged.

