#!/usr/bin/env python3
"""T19-S: which perturbation observable has which leading residual order.

Deterministic linear algebra only.  No data is fitted, so this script spends
zero estimator fits from the mission budget.

Setting.  A = diag(lambda) symmetric, retained block = top k, discarded block =
the rest, cutoff gap delta = lambda_k - lambda_{k+1}.  Perturbation A + eps*G
with G symmetric.  Write G in 2x2 block form (G11 retained-retained, G21
discarded-retained, G22 discarded-discarded).

The perturbed top-k invariant subspace is the graph of X: span([I; X]).  X
solves the Riccati equation, and expanding X = eps*X1 + eps^2*X2 + eps^3*X3:

    order 1:  Lam2 X1 - X1 Lam1 = -G21
    order 2:  Lam2 X2 - X2 Lam1 = -G22 X1 + X1 G11
    order 3:  Lam2 X3 - X3 Lam1 = -G22 X2 + X2 G11 + X1 G12 X1

PREDICTION (derived, then tested below).  For a *pure coupling* perturbation
(G11 = 0 and G22 = 0) the order-2 right-hand side vanishes identically, so
X2 = 0 and the graph's first correction is O(eps^3).  For a generic G it is
O(eps^2).  But the orthogonal projector

    P = [I;X] (I + X^T X)^{-1} [I;X]^T

has diagonal blocks -X^T X + O(X^4) and X X^T + O(X^4), which are O(eps^2)
whether or not X2 vanishes.  So the projector *matrix* stays quadratic for
pure coupling while the graph goes cubic.

The Frobenius *norm* behaves differently again:
    ||P - P0||_F^2 = 2||X||_F^2 + ||X^T X||_F^2 + ||X X^T||_F^2 + ...
so ||P - P0||_F = sqrt(2) ||X||_F (1 + O(eps^2)), and its residual against the
first-order value is O(eps^3) for pure coupling -- the O(eps^2) diagonal part
is Frobenius-orthogonal to the O(eps) off-diagonal part and does not enter the
norm at second order.  A norm can therefore have a higher-order residual than
the matrix whose norm it is.  This is the mechanism behind the archived T19
observation of eps^2.9 for pure coupling against eps^2.05 for generic E: the
archive measured a norm.
"""
import numpy as np

np.random.seed(0)


def blocks(M, k):
    return M[:k, :k], M[:k, k:], M[k:, :k], M[k:, k:]


def first_order_graph(lam, G, k):
    """X1[i,j] = G21[i,j] / (lam1[j] - lam2[i]) from Lam2 X1 - X1 Lam1 = -G21."""
    lam1, lam2 = lam[:k], lam[k:]
    G21 = G[k:, :k]
    return G21 / (lam1[None, :] - lam2[:, None])


def top_k_basis(M, k):
    """Orthonormal basis of the top-k invariant subspace via symmetric eigh."""
    w, V = np.linalg.eigh(M)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx], w[np.argsort(w)[::-1]]


def graph_of(V, k):
    """X such that span(V) = span([I; X]); requires the top block invertible."""
    V1, V2 = V[:k, :], V[k:, :]
    return V2 @ np.linalg.inv(V1)


def projector(V):
    return V @ V.T


def sin_theta(V, k):
    """Singular values of the discarded-block component: sin of principal angles."""
    return np.linalg.svd(V[k:, :], compute_uv=False)


def make_perturbation(d, k, kind, rng):
    G = np.zeros((d, d))
    if kind == "generic":
        M = rng.standard_normal((d, d))
        G = (M + M.T) / 2
    elif kind == "pure_coupling":
        C = rng.standard_normal((d - k, k))
        G[k:, :k] = C
        G[:k, k:] = C.T
    elif kind == "block_diagonal":
        M1 = rng.standard_normal((k, k)); M2 = rng.standard_normal((d - k, d - k))
        G[:k, :k] = (M1 + M1.T) / 2
        G[k:, k:] = (M2 + M2.T) / 2
    elif kind == "zero":
        pass
    else:
        raise ValueError(kind)
    n = np.linalg.norm(G, 2)
    return G / n if n > 0 else G


def residual_orders(d, k, lam, kind, eps_grid, rng):
    """Residuals of four observables against their first-order values."""
    A = np.diag(lam)
    G = make_perturbation(d, k, kind, rng)
    X1 = first_order_graph(lam, G, k)
    P0 = np.zeros((d, d)); P0[:k, :k] = np.eye(k)
    rows = []
    for eps in eps_grid:
        V, _ = top_k_basis(A + eps * G, k)
        # fix column gauge: make the retained block close to identity
        V = V @ np.linalg.inv(np.linalg.qr(V[:k, :])[0]) if k > 1 else V
        X = graph_of(V, k)
        P = projector(V)
        st = np.sort(sin_theta(V, k))[::-1]
        st1 = np.sort(np.linalg.svd(eps * X1, compute_uv=False))[::-1]
        dP = P - P0
        dP_first = np.zeros_like(dP)
        dP_first[k:, :k] = eps * X1
        dP_first[:k, k:] = eps * X1.T
        rows.append(dict(
            eps=eps,
            graph=np.linalg.norm(X - eps * X1, "fro"),
            projector_matrix=np.linalg.norm(dP - dP_first, "fro"),
            projector_diag=np.linalg.norm(dP[:k, :k], "fro") + np.linalg.norm(dP[k:, k:], "fro"),
            projector_fro_norm=abs(np.linalg.norm(dP, "fro") - np.sqrt(2) * eps * np.linalg.norm(X1, "fro")),
            sin_theta=np.linalg.norm(st - st1),
        ))
    return rows, X1, G


def slope(eps, y):
    """Log-log slope, ignoring points at the numerical floor."""
    eps = np.asarray(eps); y = np.asarray(y)
    m = y > 1e-13
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(eps[m]), np.log(y[m]), 1)[0])


def main():
    d, k = 24, 3
    rng = np.random.default_rng(20260912)
    # retained eigenvalues well separated from the discarded ones
    lam = np.concatenate([np.array([10.0, 9.4, 8.8]), np.linspace(4.0, 1.0, d - k)])
    delta = lam[k - 1] - lam[k]
    eps_grid = np.logspace(-1, -3, 9)

    print(f"d={d} k={k} cutoff gap delta={delta:.3f} lambda1={lam[0]:.3f}")
    print(f"eps grid {eps_grid[0]:.1e} .. {eps_grid[-1]:.1e}\n")

    OBS = ["graph", "projector_matrix", "projector_diag", "projector_fro_norm", "sin_theta"]
    PREDICTED = {
        "generic":       dict(graph=2, projector_matrix=2, projector_diag=2, projector_fro_norm=2, sin_theta=2),
        "pure_coupling": dict(graph=3, projector_matrix=2, projector_diag=2, projector_fro_norm=3, sin_theta=3),
    }

    print(f"{'family':<15}{'observable':<22}{'slope':>8}{'predicted':>11}{'verdict':>10}")
    print("-" * 66)
    ok = True
    for kind in ("generic", "pure_coupling"):
        rows, X1, G = residual_orders(d, k, lam, kind, eps_grid, rng)
        for obs in OBS:
            s = slope([r["eps"] for r in rows], [r[obs] for r in rows])
            pred = PREDICTED[kind][obs]
            good = abs(s - pred) < 0.25
            ok &= good
            print(f"{kind:<15}{obs:<22}{s:>8.2f}{pred:>11}{'ok' if good else 'MISMATCH':>10}")
        print()

    # ---- deterministic edge checks -------------------------------------
    print("deterministic checks")
    print("-" * 66)
    A = np.diag(lam)

    # 1. zero perturbation moves nothing
    V, _ = top_k_basis(A, k)
    z = np.linalg.norm(projector(V) - np.diag([1.0] * k + [0.0] * (d - k)), "fro")
    print(f"  zero perturbation, ||dP||_F                     {z:.2e}  {'ok' if z < 1e-12 else 'FAIL'}")

    # 2. block-diagonal (commuting with the block structure) leaves the subspace fixed
    G = make_perturbation(d, k, "block_diagonal", rng)
    V, _ = top_k_basis(A + 0.05 * G, k)
    P0 = np.zeros((d, d)); P0[:k, :k] = np.eye(k)
    z = np.linalg.norm(projector(V) - P0, "fro")
    print(f"  block-diagonal E, ||dP||_F                      {z:.2e}  {'ok' if z < 1e-10 else 'FAIL'}")

    # 3. repeated eigenvalues *within* the retained block: first order still valid
    lam_rep = np.concatenate([np.array([10.0, 10.0, 8.8]), np.linspace(4.0, 1.0, d - k)])
    rows, X1, _ = residual_orders(d, k, lam_rep, "pure_coupling", eps_grid, rng)
    s = slope([r["eps"] for r in rows], [r["graph"] for r in rows])
    print(f"  repeated retained eigenvalues, graph slope      {s:.2f}      {'ok' if abs(s-3) < 0.3 else 'FAIL'}")

    # 4. shrinking cutoff gap: first-order graph grows like 1/delta
    print("  shrinking cutoff gap (first-order graph norm ~ 1/delta):")
    for gap in (2.0, 1.0, 0.5, 0.25):
        lam_g = np.concatenate([np.array([10.0, 9.4, 4.0 + gap]), np.linspace(4.0, 1.0, d - k)])
        Gp = make_perturbation(d, k, "pure_coupling", np.random.default_rng(7))
        X1g = first_order_graph(lam_g, Gp, k)
        print(f"      delta={gap:<5} ||X1||_F={np.linalg.norm(X1g,'fro'):8.3f}"
              f"   delta*||X1||={gap*np.linalg.norm(X1g,'fro'):8.3f}")

    # 5. exact tie at the cutoff: first order is undefined, not merely large
    lam_tie = np.concatenate([np.array([10.0, 9.4, 4.0]), np.linspace(4.0, 1.0, d - k)])
    Gp = make_perturbation(d, k, "pure_coupling", np.random.default_rng(7))
    with np.errstate(divide="ignore", invalid="ignore"):
        X1t = first_order_graph(lam_tie, Gp, k)
    nonfinite = int((~np.isfinite(X1t)).sum())
    print(f"  exact tie at the cutoff, non-finite X1 entries  {nonfinite:>8}  "
          f"{'ok (non-identifiable, reported not divided)' if nonfinite > 0 else 'FAIL'}")

    # 6. numerical floor: how small a residual this precision can resolve
    V, _ = top_k_basis(A, k)
    floor = np.linalg.norm(A @ V - V @ (V.T @ A @ V), "fro")
    print(f"  eigensolver residual floor                      {floor:.2e}")

    print("\nALL ORDER PREDICTIONS MATCHED" if ok else "\nSOME PREDICTIONS DID NOT MATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
