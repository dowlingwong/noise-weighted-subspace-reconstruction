#!/usr/bin/env python3
"""T19-S: which perturbation observable has which leading residual order.

Deterministic linear algebra only.  No data is fitted, so this script spends
ZERO estimator fits from the mission budget.

=============================================================================
TWO MODELS, DELIBERATELY KEPT SEPARATE
=============================================================================

  covariance   M(eps) = A (I + eps E)^{-1}     <-- the model the paper and the
                                                   ARCHIVED T19 actually use
  additive     M(eps) = A + eps G              <-- a mathematical control only;
                                                   NOT the paper's model

An earlier revision of this script studied only the additive model and then
described the archived mechanism as settled.  That was wrong: the additive
recurrence omits the higher-order operator terms -eps A E, +eps^2 A E^2, ...
that the covariance model carries.  The covariance derivation is given below
and is what the archived result must be judged against.

=============================================================================
WHAT THE ARCHIVED T19 ACTUALLY MEASURED
=============================================================================

From experiments/paper1/t19_coupling.py:

  * operator    : A (I + E)^{-1}, with A = Sigma^{-1/2} R_x Sigma^{-1/2} and
                  E = Sigma^{-1/2}(Sigma_hat - Sigma) Sigma^{-1/2}   (line 143)
  * observable  : T = C_disc C_ret^{-1}, i.e. the GRAPH MATRIX of the fitted
                  invariant subspace in the eigenbasis Q           (lines 122-126)
  * residual    : ||T_measured - T_first_order||_F                  (line 398)
  * first order : T_ji = -lam_j E_ji / (lam_i - lam_j)              (line 133)
  * families    : C = pure coupling, E_11 = E_22 = 0 exactly        (line 195)
                  F = full random symmetric                         (line 200)
  * measured    : slope 2.9227 (C), 2.0530 (F), sample basis

So the archived observable is the GRAPH MATRIX, not a projector and not a
norm of a projector displacement.  Any explanation of the 2.92 slope must be
an explanation about the graph.

=============================================================================
DERIVATION FOR THE COVARIANCE MODEL (the one that matters)
=============================================================================

Work in the eigenbasis Q of A, so A = Lam = diag(lam) and E is Ehat = Q^T E Q.
Expanding the resolvent,

    M(eps) = Lam (I + eps Ehat)^{-1}
           = Lam - eps Lam Ehat + eps^2 Lam Ehat^2 - eps^3 Lam Ehat^3 + ...

so the operator perturbation has a whole series of terms,

    G1 = -Lam Ehat,    G2 = +Lam Ehat^2,    G3 = -Lam Ehat^3.

The perturbed top-k invariant subspace is the graph of X solving the Riccati
equation M21 + M22 X - X M11 - X M12 X = 0.  With X = eps X1 + eps^2 X2 + ...:

    order 1:  Lam2 X1 - X1 Lam1 = -(G1)21
    order 2:  Lam2 X2 - X2 Lam1 = -(G2)21 - (G1)22 X1 + X1 (G1)11

Order 1 reproduces the registered prediction exactly:
    (X1)_ji (lam_j - lam_i) = lam_j Ehat_ji
    => (X1)_ji = -lam_j Ehat_ji / (lam_i - lam_j)                   [matches]

PURE COUPLING (Ehat_11 = Ehat_22 = 0, i.e. Ehat = [[0, B^T], [B, 0]]):

    (G1)11 = -Lam1 Ehat_11 = 0
    (G1)22 = -Lam2 Ehat_22 = 0
    Ehat^2 = [[B^T B, 0], [0, B B^T]]   -- BLOCK DIAGONAL
      => (Ehat^2)_21 = 0  =>  (G2)21 = Lam2 (Ehat^2)_21 = 0

    Hence the entire order-2 right-hand side vanishes and X2 = 0 EXACTLY.

The step (Ehat^2)_21 = 0 is new relative to the additive model and is exactly
what the earlier additive-only argument failed to establish: in the additive
model there simply is no G2 term, so its vanishing was assumed rather than
shown.  Here it has to be proved, and it holds.

At order 3, (Ehat^3)_21 = B^T B B^T is generally nonzero, and (G2)11, (G2)22
are nonzero, so X3 != 0.  The graph residual is therefore O(eps^3).

GENERIC E: (G1)11 and (G1)22 are nonzero, so X2 != 0 and the graph residual
is O(eps^2).

This is the mechanism behind the archived 2.92 (C) against 2.05 (F).

=============================================================================
THE PROJECTOR DISTANCE (corrected)
=============================================================================

For P(X) = [I;X](I + X^T X)^{-1}[I;X]^T, writing S = X^T X:

    ||P(X) - P(0)||_F^2 = 2 tr[S (I + S)^{-1}]
                        = 2||X||_F^2 - 2 tr[(X^T X)^2] + O(||X||^6)

An earlier revision of this docstring added the squared diagonal-block
contributions with the WRONG SIGN and omitted the off-diagonal normalization
correction of the same order.  The corrected expansion is above; the cubic
conclusion for the norm survives it, because the correction is relatively
O(eps^2).

Note that the projector MATRIX and the projector NORM behave differently:
P has diagonal blocks -S(I+S)^{-1} and X(I+S)^{-1}X^T which are O(eps^2)
whether or not X2 vanishes, while the norm above depends on X only through
tr[S(I+S)^{-1}] and so returns to O(eps^3) when X2 = 0.
"""
import argparse
import sys

import numpy as np

K = 3  # retained rank, matching the block split used throughout


# ---------------------------------------------------------------------------
# assertion bookkeeping: every asserted check must reach the exit status
# ---------------------------------------------------------------------------
class Checks:
    """Assertions decide the exit code; diagnostics are printed but never gate."""

    def __init__(self):
        self.failures = []
        self.n_assert = 0

    def assert_(self, name, passed, detail=""):
        self.n_assert += 1
        if not passed:
            self.failures.append(name)
        print(f"  [{'ok  ' if passed else 'FAIL'}] {name:<52s} {detail}")
        return passed

    def diagnostic(self, name, detail=""):
        print(f"  [diag] {name:<52s} {detail}")

    def report(self):
        print()
        if self.failures:
            print(f"FAILED {len(self.failures)} of {self.n_assert} assertions: "
                  + ", ".join(self.failures))
            return 1
        print(f"ALL {self.n_assert} ASSERTIONS PASSED")
        return 0


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------
def sym(M):
    return 0.5 * (M + M.T)


def orth(M):
    q, _ = np.linalg.qr(M)
    return q


def graph_of(V, Q):
    """T = C_disc C_ret^{-1}: the archived T19 observable (t19_coupling.py:122)."""
    C = Q.T @ V
    return C[K:, :] @ np.linalg.inv(C[:K, :])


def first_order_graph(lam, Eq, eps, model="covariance"):
    """First-order graph displacement, which differs between the two models.

    covariance  M = A(I + eps E)^{-1}, so G1 = -Lam E and
                (X1)_ji = -lam_j E_ji / (lam_i - lam_j)   <- registered prediction
    additive    M = A + eps G, so G1 = G and
                (X1)_ji =  G_ji / (lam_i - lam_j)
    """
    denom = lam[:K][None, :] - lam[K:][:, None]      # lam_i - lam_j
    blk = (eps * Eq)[K:, :K]
    if model == "covariance":
        return (-lam[K:][:, None] * blk) / denom
    if model == "additive":
        return blk / denom
    raise ValueError(model)


def subspace_covariance_model(A, Eq, Q, eps):
    """Top-k invariant subspace of A (I + eps E)^{-1}, via the similar symmetric
    operator -- the same construction as t19_coupling.perturbed_spectrum."""
    E = Q @ (eps * Eq) @ Q.T
    IE = np.eye(A.shape[0]) + E
    vals, vecs = np.linalg.eigh(sym(IE))
    if vals.min() <= 0:
        raise FloatingPointError("I + eps E is not positive definite")
    S_isqrt = (vecs / np.sqrt(vals)[None, :]) @ vecs.T
    S_sqrt = (vecs * np.sqrt(vals)[None, :]) @ vecs.T
    B = S_isqrt @ A @ S_isqrt
    mu, U = np.linalg.eigh(sym(B))
    idx = np.argsort(mu)[::-1][:K]
    return orth(S_sqrt @ U[:, idx])


def subspace_additive_model(A, Eq, Q, eps):
    """Top-k invariant subspace of A + eps G (control model, NOT the paper's)."""
    G = Q @ (eps * Eq) @ Q.T
    w, V = np.linalg.eigh(sym(A + G))
    idx = np.argsort(w)[::-1][:K]
    return V[:, idx]


MODELS = {"covariance": subspace_covariance_model, "additive": subspace_additive_model}


def make_Eq(d, kind, rng):
    """Perturbation in the eigenbasis, normalised to unit operator norm.

    Mirrors t19_coupling.make_E_family: 'pure_coupling' is family C (off-diagonal
    BLOCKS only, E_11 = E_22 = 0), 'generic' is family F (full random symmetric).
    """
    Eq = np.zeros((d, d))
    if kind == "pure_coupling":
        B = rng.standard_normal((d - K, K))
        B /= np.linalg.norm(B, 2)
        Eq[K:, :K] = B
        Eq[:K, K:] = B.T
    elif kind == "generic":
        M = sym(rng.standard_normal((d, d)))
        Eq = M / np.linalg.norm(M, 2)
    elif kind == "block_diagonal":
        M1 = sym(rng.standard_normal((K, K)))
        M2 = sym(rng.standard_normal((d - K, d - K)))
        Eq[:K, :K] = M1
        Eq[K:, K:] = M2
        Eq /= np.linalg.norm(Eq, 2)
    elif kind == "zero":
        pass
    else:
        raise ValueError(kind)
    return Eq


def observables(A, Q, lam, Eq, eps, model):
    """Residuals of four observables against their first-order values."""
    V = MODELS[model](A, Eq, Q, eps)
    X = graph_of(V, Q)
    X1 = first_order_graph(lam, Eq, eps, model)

    d = A.shape[0]
    P0 = np.zeros((d, d))
    P0[:K, :K] = np.eye(K)
    C = Q.T @ V                       # orthonormal: V is, and Q is orthogonal
    dP = C @ C.T - P0                 # projector displacement, exactly
    dP1 = np.zeros_like(dP)
    dP1[K:, :K] = X1
    dP1[:K, K:] = X1.T

    # sin of the principal angles: singular values of the discarded-block component
    st = np.sort(np.linalg.svd(C[K:, :], compute_uv=False))[::-1]
    st1 = np.sort(np.linalg.svd(X1, compute_uv=False))[::-1]

    return {
        "graph": np.linalg.norm(X - X1, "fro"),
        "projector_matrix": np.linalg.norm(dP - dP1, "fro"),
        "projector_fro_norm": abs(np.linalg.norm(dP, "fro")
                                  - np.sqrt(2) * np.linalg.norm(X1, "fro")),
        "sin_theta": np.linalg.norm(st - st1),
    }


def slope(eps_grid, y):
    eps_grid = np.asarray(eps_grid, float)
    y = np.asarray(y, float)
    m = y > 1e-13
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(eps_grid[m]), np.log(y[m]), 1)[0])


OBS = ["graph", "projector_matrix", "projector_fro_norm", "sin_theta"]
PREDICTED = {
    "generic":       {"graph": 2, "projector_matrix": 2, "projector_fro_norm": 2, "sin_theta": 2},
    "pure_coupling": {"graph": 3, "projector_matrix": 2, "projector_fro_norm": 3, "sin_theta": 3},
}


def run(checks, *, break_block_diagonal=False):
    d = 24
    rng = np.random.default_rng(20260912)
    lam = np.concatenate([np.array([10.0, 9.4, 8.8]), np.linspace(4.0, 1.0, d - K)])
    Q = orth(rng.standard_normal((d, d)))
    A = Q @ np.diag(lam) @ Q.T
    eps_grid = np.logspace(-1, -3, 9)

    print(f"d={d} k={K} cutoff gap delta={lam[K-1]-lam[K]:.3f} lambda1={lam[0]:.3f}")
    print(f"eps grid {eps_grid[0]:.1e} .. {eps_grid[-1]:.1e}")

    for model in ("covariance", "additive"):
        label = ("THE PAPER'S MODEL  A (I + eps E)^{-1}" if model == "covariance"
                 else "CONTROL ONLY      A + eps G  (not the paper's model)")
        print(f"\n=== model: {model}   [{label}]")
        for kind in ("generic", "pure_coupling"):
            Eq = make_Eq(d, kind, np.random.default_rng(7))
            rows = [observables(A, Q, lam, Eq, e, model) for e in eps_grid]
            for obs in OBS:
                s = slope(eps_grid, [r[obs] for r in rows])
                pred = PREDICTED[kind][obs]
                checks.assert_(f"{model}/{kind}/{obs} slope",
                               abs(s - pred) < 0.25, f"slope {s:5.2f}  predicted {pred}")

    # ---------------- deterministic edge checks (all are ASSERTIONS) --------
    print("\n=== deterministic edge checks (covariance model)")
    P0 = np.zeros((d, d))
    P0[:K, :K] = np.eye(K)

    Eq0 = make_Eq(d, "zero", rng)
    V = subspace_covariance_model(A, Eq0, Q, 0.0)
    z = np.linalg.norm(Q.T @ V @ (Q.T @ V).T - P0, "fro")
    checks.assert_("zero perturbation leaves the projector unchanged", z < 1e-12, f"{z:.2e}")

    Eqb = make_Eq(d, "block_diagonal", np.random.default_rng(11))
    if break_block_diagonal:                       # mutation fixture, self-test only
        Eqb[K:, :K] = 0.05
        Eqb[:K, K:] = 0.05
    V = subspace_covariance_model(A, Eqb, Q, 0.05)
    X = graph_of(V, Q)
    zb = np.linalg.norm(X, "fro")
    checks.assert_("block-diagonal E leaves the invariant subspace fixed",
                   zb < 1e-10, f"||X||_F = {zb:.2e}")

    lam_rep = np.concatenate([np.array([10.0, 10.0, 8.8]), np.linspace(4.0, 1.0, d - K)])
    A_rep = Q @ np.diag(lam_rep) @ Q.T
    Eqc = make_Eq(d, "pure_coupling", np.random.default_rng(7))
    rows = [observables(A_rep, Q, lam_rep, Eqc, e, "covariance") for e in eps_grid]
    s = slope(eps_grid, [r["graph"] for r in rows])
    checks.assert_("repeated retained eigenvalues keep the cubic graph order",
                   abs(s - 3) < 0.3, f"slope {s:.2f}")

    lam_tie = np.concatenate([np.array([10.0, 9.4, 4.0]), np.linspace(4.0, 1.0, d - K)])
    with np.errstate(divide="ignore", invalid="ignore"):
        X1t = first_order_graph(lam_tie, Eqc, 0.1, 'covariance')
    nf = int((~np.isfinite(X1t)).sum())
    checks.assert_("exact tie at the cutoff is non-identifiable, not divided through",
                   nf > 0, f"{nf} non-finite first-order entries")

    # exact projector identity, corrected expansion
    Xr = np.random.default_rng(3).standard_normal((d - K, K)) * 0.01
    S = Xr.T @ Xr
    lhs = 2.0 * np.trace(S @ np.linalg.inv(np.eye(K) + S))
    Vg = orth(np.vstack([np.eye(K), Xr]))
    rhs = np.linalg.norm(Vg @ Vg.T - P0, "fro") ** 2
    checks.assert_("||P(X)-P(0)||_F^2 == 2 tr[S (I+S)^{-1}]",
                   abs(lhs - rhs) < 1e-12, f"|diff| = {abs(lhs-rhs):.2e}")
    series = 2 * np.trace(S) - 2 * np.trace(S @ S)
    omitted = 2 * abs(np.trace(S @ S @ S))          # leading O(||X||^6) term
    checks.assert_("corrected series 2||X||^2 - 2 tr[(X^T X)^2] matches to O(||X||^6)",
                   abs(series - lhs) < 10 * omitted + 1e-15,
                   f"|diff| = {abs(series-lhs):.2e}, leading omitted term {omitted:.2e}")

    # ---------------- informational diagnostics (never gate the exit code) --
    print("\n=== diagnostics (informational only, not assertions)")
    for gap in (2.0, 1.0, 0.5, 0.25):
        lam_g = np.concatenate([np.array([10.0, 9.4, 4.0 + gap]), np.linspace(4.0, 1.0, d - K)])
        X1g = first_order_graph(lam_g, Eqc, 0.1, 'covariance')
        checks.diagnostic(f"cutoff gap {gap:<5} -> ||X1||_F",
                          f"{np.linalg.norm(X1g,'fro'):.4f}  (all gaps move with lam_k, "
                          f"so this does NOT isolate a 1/delta law)")
    w, Vq = np.linalg.eigh(sym(A))
    checks.diagnostic("eigensolver residual floor",
                      f"{np.linalg.norm(A @ Vq - Vq @ np.diag(w), 'fro'):.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="mutation regression: inject a broken block-diagonal fixture "
                         "and require the exit status to become non-zero")
    args = ap.parse_args()

    if args.self_test:
        print("### mutation regression: injecting a faulty block-diagonal fixture\n")
        checks = Checks()
        run(checks, break_block_diagonal=True)
        status = checks.report()
        if status == 0:
            print("\nSELF-TEST FAILED: a broken fixture still returned success")
            return 1
        print("\nSELF-TEST PASSED: the broken fixture is detected and fails the exit status")
        return 0

    checks = Checks()
    run(checks)
    return checks.report()


if __name__ == "__main__":
    raise SystemExit(main())
