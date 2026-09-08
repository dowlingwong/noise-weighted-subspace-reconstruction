Single-seed pilot (S3, seed 0) run to check the plumbing, with the "converged"
arm capped at 5000 L-BFGS iterations. The untied weighted arm hit that cap, so
the cap was raised to 20000 (declared in PREDECLARED.md) before the reported
run, to make "converged" mean gtol-limited wherever the optimiser can get
there. Superseded; not citable.
