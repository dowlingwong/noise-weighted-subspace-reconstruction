# HCP stage F additive extension

Frozen dependencies are imported without edits: dev kit 4c6d2e69, HCP 79dcb968,
R1 a1825db and original f121776c. `prediction.py` checks the committed dev seal,
uses `population.predictions` (four fixed plus two adaptive rows), and records
point/certificate crossings without moving them into the training grid. Final
population optimizations use HOSVD + theta0 + 62 random starts; the accepted root
search uses six random starts. A missing root stays unavailable.

The metadata-fixed grids are A={56,111,222,444,888} and
B={28,56,111,222,444}; theta0 control uses the A absolute grid. The fitted signal
geometries carried from R1 are theta0, adaptive A and adaptive B. The four fixed
rows remain the PZ1 certificate family and have descriptive PZ2 range labels.
All final population restart and Hessian diagnostics are retained.

`confirmation_guard.py` has no recording reader. The future confirmation
launcher must fetch GitHub main/tags read-only and call its registration gate
before any recording access. `fit_confirmation` additionally checks that exact
per-unit prediction bytes have been committed and pushed into origin/main before
entering the learning kernel. It cannot be unlocked by a local protocol flag.
The public registration bundle must match the named git tag and current main;
current OTS receipt and signed authorization identities must match. Timestamp
maturation/verification remains an author action described in the registration
guide. This kit does not tag, stamp, register or access confirmation recordings.

Run fabricated tests with meg_env, PYTHONPATH=src, PYTHONDONTWRITEBYTECODE=1,
and OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1. Long jobs run under tmux -L paper1hcp.

`pilot.py` reads only the nine E1-passing dates' previously saved decimated dev
arrays after exact source hashes and sealed Sref equality checks. Every pilot
fit calls the inherited `learning.fit_cell`, including original sorted12 date
indices and learning law8/cells100–102. `summary.py` requires the complete exact
date/geometry/N/arm/replicate key set before any reading. It reports date means,
paired interactions, restart flags, population Hessian diagnostics and all SDs.
The prospective family34 and once-only reference R50 pilot extension are recorded
in MS PILOT_PLAN.md before learning. `power.py` calls unchanged HCP/R1 n_min and
also tabulates every required sign-claim cell at every n8…12 (signed AD8, commit4fa00e7; all15 reference cells stay in family34), with strict inequalities and one-sided
80% chi-square SD upper limits. Failure after R50 stops before F5. No optimiser
iteration-limit adaptation is performed: the inherited T25 recipe stays fixed.
