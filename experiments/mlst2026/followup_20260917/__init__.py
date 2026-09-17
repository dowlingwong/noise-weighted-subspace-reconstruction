"""MLST follow-up round, 17 September 2026.

This package holds the next-round experiment implementation defined in
``arXiv-v1-polished-extended-preprint/output/agent-instructions-20260917/02_IMPLEMENT_NEXT_EXPERIMENTS.md``.

Packages (all new protocols are labelled as designed *after* the September
results were observed):

* ``r0_t22_h5``          reporting overlay: the omitted T22 H5 random-effects
                         summary, computed from copies of the stored records.
                         Zero new cells.
* ``r1_t22_calibration`` new protocol: small-file interval calibration.
* ``r2_t20_theta012``    new protocol: T20 theta=0.12 scaling and crossover.
* ``r3_t21_finite``      new protocol: T21 finite-size recovery.
* ``r4_t22_feasibility`` new protocol: evaluation-file power and development-data
                         adequacy for the T22 feasibility gate.
* ``r5_t19_reach``       optional: T19 full-covariance theorem-domain reachability.
* ``r6_instrument``      data-blocked: CRESST robustness / TIDMAD matched design.

The historical protocols, manifests, freezes, ledgers, cells and reductions
under ``successor_20260912`` and ``successor_20260913`` are immutable and are
never edited by this package.
"""
