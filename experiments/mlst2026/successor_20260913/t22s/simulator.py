"""T22-S part A: the synthetic simulator.

What is simulated
-----------------
One simulated *file* ``i`` is the injection-bin slice of a T22 stretch: complex
unitary-DFT coefficients at the injection bin for ``n_cal = 64`` transfer
calibration windows and ``n_eval = 300`` evaluation windows,

* reference ``rho(t)``: a tone of amplitude ``A`` with a random phase per window
  (an off-bin tone advances its phase every window, which is why the archived
  calibration-mean baseline is indistinguishable from the zero baseline) plus
  reference noise at bin power ``A^2 / ref_snr``;
* readout ``c(t) = g_i rho(t) + n_i(t)`` with complex gain ``g_i`` and readout
  noise ``n_i(t)`` of bin power ``N_i`` (log-normal across files, mean 1);
* method estimates at the bin: ``zero``, ``calibration-mean`` (the archived
  ``fit_transfer`` baseline, refitted inside every bootstrap replicate),
  ``raw-bin = c(t)``, and two learned competitors ``pca`` and ``empca`` whose
  residuals are correlated copies of the readout noise with reduced power,

      r_M(t) = sqrt(N_i (1 - lam_M,i) (1 + d W_t)) * (kappa n_hat(t) + sqrt(1 - kappa^2) m_M(t)),

  ``n_hat`` the unit-power readout noise process, ``m_M`` an independent unit
  process, ``kappa = 0.99`` (a learned reconstruction at the injection bin is
  highly correlated with the bin read), ``W_t`` an optional random walk (drift),
  and ``lam_M,i`` the fraction of noise power the method removes.

The paired squared-error contrasts the archived pipeline scores are then exact
functions of the declared parameters (bin-power units, times ``4/w``):

    E Delta(raw-bin, empca) = lam_e,i N_i          (denoising benefit)
    E Delta(pca,     empca) = (lam_e,i - lam_p,i) N_i   (metric contribution)
    E Delta(zero,    empca) = lam_z N_i            (limited recovery; the signal
                              power is S_i = N_i (1 - lam_e,i + lam_z))

so the equal-file estimand of the primary contrast is ``mu = lam_e * E[N_i] *
4/w = lam_e * 4/w`` under the declared file-sampling model
``lam_e,i = lam_e + tau b_i`` (``b_i`` standard normal, or the two-point
"dominant file" mixture), ``N_i`` log-normal with mean one.

Within-file dependence: the readout noise and the competitor processes are
complex AR(1) with coefficient ``phi`` (the squared-error contrast then has
lag-one autocorrelation about ``phi^2``); the optional drift ``d W_t`` makes the
block-variance curve keep rising at every block length.  Heavy tails: ``t_3``
innovations scaled to unit variance.  Transfer-calibration uncertainty: the
truth labels are ``g_hat rho_eval`` with ``g_hat`` from ``te.fit_transfer`` on
the 64 calibration windows, exactly as in the archived runner, and the joint
bootstrap refits it.

Nothing here is fitted: every learned-method estimate is *generated*, so a
simulation cell charges zero fits and reports its bootstrap replicates.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

KAPPA = 0.99            # correlation between a learned residual and the bin read
W_NOMINAL = 1024        # window length used only through the 4/w physical-amplitude factor
N_CAL = 64
N_EVAL = 300
REF_SNR = 1.0e4
NOISE_SCALE_SPREAD = 0.5   # log-normal sd of N_i across files (mean fixed at 1)
LAM_ZERO_ALT = 29.0        # signal power S = N (1 - lam_e + lam_z): readout SNR ~ 30 like the archive
METHODS = ("zero", "calibration-mean", "raw-bin", "pca", "empca")
CONTRASTS = (("zero", "empca"), ("calibration-mean", "empca"), ("raw-bin", "empca"), ("pca", "empca"))
EXACT_MU_CONTRASTS = ("raw-bin|empca", "pca|empca", "zero|empca")


@dataclass(frozen=True)
class Scenario:
    name: str
    n_files: int
    lam_emp: float                 # denoising effect (fraction of readout noise power removed by empca)
    lam_pca: float                 # same for pca
    lam_zero: float                # limited-recovery effect: S - sigma_emp^2 = lam_zero N
    phi: float = 0.0               # AR(1) coefficient of the within-file processes
    tau: float = 0.0               # sd of lam_emp across files
    heterogeneity: str = "gaussian"  # "gaussian" | "dominant"
    drift_sd: float = 0.0          # random-walk modulation of the competitor residual power
    heavy_tails: bool = False
    n_dev_files: int = 4           # development files whose curves decide the dependence verdict
    n_cal: int = N_CAL
    n_eval: int = N_EVAL
    w: int = W_NOMINAL

    def as_dict(self) -> dict:
        return asdict(self)

    def true_mu(self) -> dict:
        """Exact equal-file estimands (readout physical-amplitude units) of the
        three contrasts whose expectation is a declared function of the
        parameters.  ``E[N_i] = 1`` and ``E[b_i] = 0`` by construction."""
        f = 4.0 / self.w
        return {"raw-bin|empca": self.lam_emp * f,
                "pca|empca": (self.lam_emp - self.lam_pca) * f,
                "zero|empca": self.lam_zero * f}

    def within_file_sd_per_window(self) -> float:
        """Per-window sd of Delta(raw-bin, empca) in units of N (kappa fixed),
        at the file-mean lam_emp: sqrt(1 + (1-lam)^2 - 2 kappa^2 (1-lam))."""
        lam = self.lam_emp
        return float(np.sqrt(max(1.0 + (1.0 - lam) ** 2 - 2.0 * KAPPA ** 2 * (1.0 - lam), 1e-12)))

    def effect_sizes(self) -> dict:
        sd = self.within_file_sd_per_window()
        e_w = self.lam_emp / sd
        return {"per_window_sd_over_N": sd, "effect_per_window_sd_units": float(e_w),
                "effect_per_file_z": float(e_w * np.sqrt(self.n_eval))}


@dataclass
class SimFile:
    name: str
    N: float
    g: complex
    lam_emp: float
    lam_pca: float
    cal_read: np.ndarray
    cal_ref: np.ndarray
    eval_read: np.ndarray
    eval_ref: np.ndarray
    truth_eval: np.ndarray            # g rho_eval: the true label (never used by the pipeline)
    generated_estimates: dict = field(default_factory=dict)   # pca / empca at the bin

    def as_summary(self) -> dict:
        return {"name": self.name, "N": self.N, "g_abs": abs(self.g), "lam_emp": self.lam_emp, "lam_pca": self.lam_pca}


# --- random processes -------------------------------------------------------------
def _innovations(rng, n, heavy_tails):
    if heavy_tails:
        x = rng.standard_t(3, size=n) / np.sqrt(3.0)
        y = rng.standard_t(3, size=n) / np.sqrt(3.0)
    else:
        x = rng.normal(size=n)
        y = rng.normal(size=n)
    return (x + 1j * y) / np.sqrt(2.0)


def complex_process(rng, n: int, phi: float, heavy_tails: bool = False) -> np.ndarray:
    """Unit-power complex AR(1): z_t = phi z_{t-1} + sqrt(1-phi^2) eps_t, stationary start."""
    eps = _innovations(rng, n, heavy_tails)
    if phi <= 0.0:
        return eps
    z = np.empty(n, dtype=np.complex128)
    z[0] = eps[0]
    s = np.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + s * eps[t]
    return z


def random_walk(rng, n: int) -> np.ndarray:
    """Standardised random walk: increments N(0, 1/n) so |W_t| is O(1)."""
    return np.cumsum(rng.normal(size=n) / np.sqrt(n))


def file_effects(rng, sc: Scenario, n: int) -> np.ndarray:
    """Per-file lam_emp under the declared file-sampling model (mean lam_emp, sd tau)."""
    if sc.tau <= 0.0:
        return np.full(n, sc.lam_emp)
    if sc.heterogeneity == "gaussian":
        b = rng.normal(size=n)
    elif sc.heterogeneity == "dominant":
        # two-point mixture with mean 0, variance 1: one file carries the effect
        b = np.full(n, -1.0 / np.sqrt(max(n - 1, 1)))
        b[rng.integers(0, n)] = np.sqrt(max(n - 1, 1))
    else:
        raise ValueError(f"unknown heterogeneity model {sc.heterogeneity!r}")
    return sc.lam_emp + sc.tau * b


# --- one file -------------------------------------------------------------------
def simulate_file(rng, sc: Scenario, name: str, lam_emp_i: float) -> SimFile:
    n_c, n_e = sc.n_cal, sc.n_eval
    N = float(np.exp(NOISE_SCALE_SPREAD * rng.normal() - 0.5 * NOISE_SCALE_SPREAD ** 2))
    lam_e = float(np.clip(lam_emp_i, -0.5, 0.95))
    lam_p = float(np.clip(sc.lam_pca, -0.5, 0.95))
    S = N * (1.0 - lam_e + sc.lam_zero)             # readout signal power at the bin
    A = 1.0                                          # reference tone amplitude (arbitrary units)
    g = np.sqrt(S) / A * np.exp(1j * rng.uniform(0, 2 * np.pi))
    ref_noise = np.sqrt(A ** 2 / REF_SNR)

    def reference(n):
        return A * np.exp(1j * rng.uniform(0, 2 * np.pi, size=n)) + ref_noise * _innovations(rng, n, False)

    rho_c, rho_e = reference(n_c), reference(n_e)
    n_hat_c = complex_process(rng, n_c, sc.phi, sc.heavy_tails)
    n_hat_e = complex_process(rng, n_e, sc.phi, sc.heavy_tails)
    c_c = g * rho_c + np.sqrt(N) * n_hat_c
    c_e = g * rho_e + np.sqrt(N) * n_hat_e
    truth_e = g * rho_e

    drift = 1.0 + sc.drift_sd * random_walk(rng, n_e) if sc.drift_sd > 0 else np.ones(n_e)
    drift = np.clip(drift, 0.05, None)
    est = {}
    for method, lam in (("pca", lam_p), ("empca", lam_e)):
        m = complex_process(rng, n_e, sc.phi, sc.heavy_tails)
        r = np.sqrt(N * (1.0 - lam) * drift) * (KAPPA * n_hat_e + np.sqrt(1.0 - KAPPA ** 2) * m)
        est[method] = truth_e + r
    return SimFile(name=name, N=N, g=complex(g), lam_emp=lam_e, lam_pca=lam_p,
                   cal_read=c_c, cal_ref=rho_c, eval_read=c_e, eval_ref=rho_e,
                   truth_eval=truth_e, generated_estimates=est)


def simulate_scenario(seed: int, sc: Scenario) -> tuple[list[SimFile], list[SimFile], dict]:
    """Development files (curves only) and evaluation files for one Monte-Carlo
    replicate; returns ``(dev_files, eval_files, info)``."""
    rng = np.random.default_rng(seed)
    lam_dev = file_effects(rng, sc, sc.n_dev_files)
    lam_eval = file_effects(rng, sc, sc.n_files)
    dev = [simulate_file(rng, sc, f"dev{i:02d}", lam_dev[i]) for i in range(sc.n_dev_files)]
    ev = [simulate_file(rng, sc, f"file{i:02d}", lam_eval[i]) for i in range(sc.n_files)]
    info = {"lam_emp_files": [float(x) for x in lam_eval], "N_files": [f.N for f in ev],
            "realised_equal_file_mu_raw-bin|empca": float(np.mean([f.lam_emp * f.N for f in ev]) * 4.0 / sc.w)}
    return dev, ev, info


# --- scenario catalogue -----------------------------------------------------------
EFFECTS = {
    # name: (lam_emp, lam_pca, lam_zero)
    "null-complete": (0.0, 0.0, 0.0),        # every contrast null (readout SNR 1)
    "null-partial": (0.0, 0.0, LAM_ZERO_ALT),  # recovery real, denoising and metric null
    "alt-small": (0.01, 0.005, LAM_ZERO_ALT),
    "alt-medium": (0.03, 0.015, LAM_ZERO_ALT),
    "alt-large": (0.06, 0.03, LAM_ZERO_ALT),
}
NULL_EFFECTS = ("null-complete", "null-partial")
PHI_GRID = (0.0, 0.3, 0.6, 0.85)
TAU_GRID = (0.0, 0.01, 0.03)
N_FILES_GRID = (4, 8, 16)


def make_scenario(effect: str, n_files: int, phi: float, tau: float = 0.01, heterogeneity: str = "gaussian",
                  drift_sd: float = 0.0, heavy_tails: bool = False) -> Scenario:
    lam_e, lam_p, lam_z = EFFECTS[effect]
    tags = [effect, f"n{n_files}", f"phi{phi:g}", f"tau{tau:g}"]
    if heterogeneity != "gaussian":
        tags.append(heterogeneity)
    if drift_sd > 0:
        tags.append(f"drift{drift_sd:g}")
    if heavy_tails:
        tags.append("t3")
    return Scenario(name="-".join(tags), n_files=n_files, lam_emp=lam_e, lam_pca=lam_p, lam_zero=lam_z,
                    phi=phi, tau=tau, heterogeneity=heterogeneity, drift_sd=drift_sd, heavy_tails=heavy_tails)


def confirmation_scenarios() -> list[Scenario]:
    """The declared validation set (fixed before any pilot cell is drawn)."""
    out = []
    for eff in EFFECTS:
        for n in N_FILES_GRID:
            for phi in (0.0, 0.6, 0.85):
                out.append(make_scenario(eff, n, phi, tau=0.01))
    for eff in ("null-partial", "alt-medium"):
        for n in (4, 16):
            out.append(make_scenario(eff, n, 0.3, tau=0.01))
            out.append(make_scenario(eff, n, 0.6, tau=0.0))
            out.append(make_scenario(eff, n, 0.6, tau=0.03))
            out.append(make_scenario(eff, n, 0.6, tau=0.03, heterogeneity="dominant"))
            out.append(make_scenario(eff, n, 0.6, tau=0.01, drift_sd=0.15))
            out.append(make_scenario(eff, n, 0.6, tau=0.01, heavy_tails=True))
    names = [s.name for s in out]
    assert len(names) == len(set(names)), "duplicate confirmation scenario"
    return out


def pilot_scenarios() -> list[Scenario]:
    """Tuning scenarios: used only to choose the bootstrap replicate setting."""
    return [make_scenario("null-partial", 4, 0.0), make_scenario("null-partial", 16, 0.6),
            make_scenario("alt-medium", 4, 0.0), make_scenario("alt-medium", 16, 0.6),
            make_scenario("null-complete", 8, 0.85), make_scenario("alt-small", 8, 0.3),
            make_scenario("alt-medium", 8, 0.6, tau=0.03, heterogeneity="dominant"),
            make_scenario("alt-large", 4, 0.6, drift_sd=0.15)]


def development_scenarios() -> list[Scenario]:
    """Fixture scenarios (development namespace only; never citable)."""
    return [make_scenario("null-partial", 4, 0.0), make_scenario("alt-large", 4, 0.0),
            make_scenario("null-partial", 16, 0.0), make_scenario("alt-medium", 4, 0.6, drift_sd=0.3)]
