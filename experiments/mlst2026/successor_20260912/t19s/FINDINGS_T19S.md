# T19-S findings — 12 September 2026

Deterministic mathematics only. **Zero estimator fits and zero inference
replicates** were spent. Self-checks, not independent approval.

## Status

| part | state |
|---|---|
| Part 1 — observable/order derivation and checks | **complete, corrected after local review** |
| Part 2 — in-domain/stress grid (120 fits) | **NOT EXECUTED** — mission elapsed ceiling expired first |

## 1. What the archived T19 actually measured

Established by reading `experiments/paper1/t19_coupling.py`, not inferred:

| aspect | value | source |
|---|---|---|
| operator | `A (I + E)^{-1}` | `perturbed_spectrum`, line 143 |
| observable | `T = C_disc C_ret^{-1}` — the **graph matrix** | `displacement`, lines 122-126 |
| residual reported | `||T_measured - T_first_order||_F` | line 398 |
| first-order prediction | `T_ji = -lam_j E_ji / (lam_i - lam_j)` | `first_order`, line 133 |
| family C | pure coupling, `E_11 = E_22 = 0` exactly | `make_E_family`, line 195 |
| family F | full random symmetric | line 200 |
| measured slopes | **C 2.9227, F 2.0530** (sample basis) | `summary_v2.json` |

**Correction to my earlier report.** I previously wrote that "the archive
measured a norm" and explained the 2.9 slope by a Frobenius-orthogonality
argument about the projector. That was wrong. The archive measured the **graph
matrix residual**. The correct explanation is a parity argument about the graph,
and it has to be made for the covariance model.

## 2. Corrected derivation (covariance model)

In the eigenbasis of `A`, with `Ehat = Q^T E Q`:

    M(eps) = Lam (I + eps Ehat)^{-1}
           = Lam - eps Lam Ehat + eps^2 Lam Ehat^2 - eps^3 Lam Ehat^3 + ...
    G1 = -Lam Ehat,   G2 = +Lam Ehat^2,   G3 = -Lam Ehat^3

Riccati expansion of the graph `X = eps X1 + eps^2 X2 + ...`:

    order 1:  Lam2 X1 - X1 Lam1 = -(G1)21
    order 2:  Lam2 X2 - X2 Lam1 = -(G2)21 - (G1)22 X1 + X1 (G1)11

Order 1 reproduces the registered prediction exactly.

**Pure coupling** (`Ehat = [[0, B^T], [B, 0]]`):

- `(G1)11 = -Lam1 Ehat_11 = 0` and `(G1)22 = -Lam2 Ehat_22 = 0`;
- `Ehat^2 = [[B^T B, 0], [0, B B^T]]` is **block diagonal**, so `(Ehat^2)_21 = 0`
  and therefore `(G2)21 = 0`.

The whole order-2 right-hand side vanishes, so **X2 = 0 exactly** and the graph
residual is `O(eps^3)`.

The second bullet is the step the additive model never had to face: there is no
`G2` term in `A + eps G`, so its absence was assumed rather than shown. In the
covariance model it must be proved, and it holds.

At order 3, `(Ehat^3)_21 = B^T B B^T` is generally nonzero, so `X3 != 0`.

**Generic E**: `(G1)11`, `(G1)22` are nonzero, so `X2 != 0` and the residual is
`O(eps^2)`.

## 3. Observable/order table (measured, both models separately labelled)

Predicted orders declared before measurement; all 16 slope assertions pass.

| observable | generic | pure coupling | covariance model | additive control |
|---|---|---|---|---|
| **graph matrix** (the archived observable) | 2 | **3** | 2.01 / 3.00 | 2.00 / 3.00 |
| projector matrix | 2 | 2 | 2.01 / 2.00 | 2.00 / 2.00 |
| projector Frobenius norm | 2 | 3 | 2.05 / 3.00 | 2.00 / 3.00 |
| principal angles (sin) | 2 | 3 | 2.05 / 3.00 | 2.00 / 3.00 |

The covariance model reproduces the archived **orders** (2.92 C, 2.05 F). It
does not reproduce the archived *numbers*: the archive used a different
generator, dimension and rank (rank-2 CW-PCA on G1, d = 256). The parity
argument is rank- and dimension-independent.

Different observables genuinely have different leading orders: the projector
*matrix* stays quadratic for pure coupling because its diagonal blocks
`-S(I+S)^{-1}` and `X(I+S)^{-1}X^T` survive, while the graph, the norm and the
angles all go cubic. **Imposing one residual order on every observable — as the
registered T19 prediction did — is a category error.**

## 4. Corrected projector identity

    ||P(X) - P(0)||_F^2 = 2 tr[X^T X (I + X^T X)^{-1}]
                        = 2||X||_F^2 - 2 tr[(X^T X)^2] + O(||X||^6)

Verified numerically to 3.5e-18 against the exact projector, and the truncated
series agrees with its own leading omitted term `2 tr[S^3]` (2.17e-07 against
2.18e-07). My earlier docstring had the wrong sign and omitted the
normalization correction.

## 5. Checker defect fixed

The local reviewer injected a faulty block-diagonal fixture and observed the
script print `FAIL` and then return **0**: the edge checks did not update the
aggregate status.

Fixed: a `Checks` collector now separates **assertions** (which gate the exit
code) from **diagnostics** (which never do), and every asserted check reaches
the exit status. A mutation regression `--self-test` injects the same faulty
fixture and requires a non-zero status; it reports
`FAILED 1 of 22 assertions: block-diagonal E leaves the invariant subspace fixed`.

The former "shrinking cutoff gap ~ 1/delta" line was **demoted to a
diagnostic** and its label corrected: moving `lam_k` moves every gap involving
it, so that check never isolated a `1/delta` law.

## 6. What is NOT established

- **Part 2 was not run.** No in-domain/stress grid, no independently estimated
  covariance arm, no training-sample arm. The mission's elapsed ceiling expired
  before it started. It remains a frozen executable design.
- **No finite-perturbation theorem.** Proposition 5.2's `44 eta^2` bound stays
  restricted to `eta <= 0.1`; the archived T19 runs at `eta = 0.22-1.3` remain
  outside it, and matching orders outside that range do not extend it.
- The registered T19 coupling-family slope prediction remains **not met**.
- This is a synthetic, deterministic study of an operator identity. It says
  nothing about any real instrument.
