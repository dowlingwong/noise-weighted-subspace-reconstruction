One cell (alpha=0, seed=0) computed before the restart-chi2 diagnostic was
moved into whitened coordinates. The quantity is mathematically identical
(reconstruct/normalized_chi2 is an exact whiten-unwhiten-whiten round trip);
the change is a ~40x reduction in the number of whitening calls. Archived so
every cell of the reported run comes from one code path.
