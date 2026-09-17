"""A3: bounded simulator-truth audit for the generator R2 inherits.

R2 reuses the archived t20s analytic Gaussian family.  The audit records, as an
executable receipt rather than protocol prose:

* whether the fixed-spectrum invariants hold at every claimed theta (via the
  archived ``generator.invariants``);
* whether the generator applies any clipping or other nonlinear post-processing
  (a static source scan plus a finite-value check);
* that the family has no stochastic components at truth time (it is analytic:
  no heavy-tail innovations, no cross-file dependence, no homogeneity knob), so
  the R1 truth-audit failure modes do not arise here.

Any generator correction discovered would be a new version with disclosure,
never a silent change to old truth.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np

from experiments.mlst2026.successor_20260913.t20s import generator as GEN

#: tokens that would indicate hidden clipping / winsorisation in truth construction
CLIP_TOKENS = ("np.clip(", ".clip(", "clamp(", "minimum(", "maximum(", "winsor")

#: the R1 truth-audit failure modes, explicitly declared absent for this family
STOCHASTIC_AUDIT = {
    "heavy_tail_initialisation": "absent: the family is analytic (no innovations are drawn at truth time)",
    "heterogeneity_dependent_generation": "absent: the file/geometry is a single fixed K(theta); no per-file effect is drawn",
    "clipping": "checked by source scan and by finite-value + spectrum invariance",
    "dependence_assumptions": "absent: truth is an exact second moment, not a sampled process",
}


def audit_generator(P: dict, *, fixture: bool = True) -> dict:
    from . import runner as RN  # local import to avoid a cycle

    D = RN.design(P, fixture)
    thetas = [float(t) for t in D["thetas"]]
    g = GEN.make_geometry(C=D["C"], T=D["T"], k_c=D["k_c"], k_t=D["k_t"], spec=D["spectrum"],
                          seed=D["geometry_seed"])
    inv = GEN.invariants(g, thetas, R_pop=int(D["R_pop"]), n_iter=int(D["pop_als_n_iter"]),
                         tol=float(D["pop_als_tol"]), spec=D["spectrum"])

    src = inspect.getsource(GEN)
    found = sorted({t for t in CLIP_TOKENS if t in src})

    # finite-value and spectrum checks on K(theta)
    finite_ok = True
    max_eig_dev = 0.0
    for th in thetas:
        K = g.K(th)
        if not np.all(np.isfinite(K)):
            finite_ok = False
        mu = np.linalg.eigvalsh(0.5 * (K + K.T))
        max_eig_dev = max(max_eig_dev, float(np.max(np.abs(np.sort(mu)[::-1] - g.mu))))

    checks = {
        "invariants_hold": {"passed": not inv["excluded_thetas"], "excluded": inv["excluded_thetas"],
                            "problems": inv["problems"]},
        "spectrum_fixed": {"passed": max_eig_dev <= 1e-8, "max_abs_eig_dev": max_eig_dev},
        "finite_values": {"passed": bool(finite_ok)},
        "no_clipping_in_source": {"passed": not found, "tokens_found": found},
        "nonseparability_moves": {"passed": bool(inv["nonseparability_moves"])},
    }
    return {"schema": "mlst-followup-truth-audit/1", "package": P["package"], "fixture": bool(fixture),
            "generator": "successor_20260913.t20s.generator.Geometry",
            "thetas": thetas, "checks": checks,
            "stochastic_audit": STOCHASTIC_AUDIT,
            "all_passed": all(c["passed"] for c in checks.values()),
            "note": "a fixture audit is a wiring check; it certifies nothing about the pulse-proxy family"}


def write_receipt(P: dict, out: Path, *, fixture: bool = True) -> dict:
    rec = audit_generator(P, fixture=fixture)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec
