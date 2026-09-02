# Cut list

> ## ⚠ SUPERSEDED — 2026-08-08
>
> This cut order predates the architecture-ladder restructure and is **no longer
> authoritative**. The governing cut order is §2 of
> `plan/PLAN_00_MASTER_RESTRUCTURE_2026-08-08.md` (Corollary 1 → appendix, the
> parameter-count sentence → appendix, the `C = k_c = 2` proof-of-concept →
> appendix as an implementation check, TIDMAD single-file result → one sentence,
> then §2's centering/covariance-estimation contract → one sentence). In
> particular, **do not cut Eq. (1)** — the `(Σ⁻¹, F)` tuple is now the paper's
> organizing device. The old order is retained below for provenance only.

The four-page draft intentionally excludes the full optimal-filter and Fisher-information derivations, the detailed CRESST/GWOSC/TIDMAD protocols, detector sensitivity and trigger claims, covariance-sensitivity experiments, the external U-Net case study, the coverage theorem and Fisher-rank diagnostic, the proposed NFPA approximation bound, VAE and Transformer extensions, and all Paper 3 monitoring terminology. The curved likelihood-contour figure is also excluded because its learned-manifold language is inconsistent with the present linear analysis.

If the main text exceeds four pages after the planned empirical figure is inserted, cut in this order: extra experimental setup detail, the rank-one diagnostic paragraph, the tuple in Eq. (1), then the parameter-count sentence. Do not cut the self-supervision motivation, covariance objective, projection bridge, metric reversal, NFPA definition, restricted bridge, or the `C=k_c` proof-of-concept limitation until that result is replaced.
