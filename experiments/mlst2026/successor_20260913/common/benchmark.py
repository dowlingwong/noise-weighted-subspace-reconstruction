"""Measurement harness for the local-package resource estimates.

Why this module exists
----------------------
The first version of ``cli._estimate`` multiplied one *fixture* per-cell wall
time by a constant declared in the protocol
(``budget.fixture_to_production_scale``).  That is a linear extrapolation of a
tiny fixture: it ignores the dimension the production cell actually runs at and
it ignores the algorithmic scaling of the kernel inside the cell.  Where the
fixture is overhead-dominated (a 24-dimensional PCA is mostly Python), the flat
factor is wrong by more than an order of magnitude, and nothing in the output
says so.

This module supplies the three things a defensible estimate needs:

``timed``
    run a callable at one problem size, discard a warm-up, report the median
    and the min/max over the repeats (so the estimate can carry a *range* that
    came from measurement rather than from an invented margin);

``peak_memory`` / :func:`probe_peak_memory_subprocess`
    ``tracemalloc`` peak (Python allocation only) *and*
    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (the whole process's resident
    high-water mark, which never goes down inside a process and is therefore
    only meaningful when the size in question is measured in its own process);

``fit_power_law``
    least squares of ``log t = log a + b log n``, reporting ``a``, the fitted
    exponent ``b`` and the residual, so the measurement can be printed next to
    the exponent the algorithm predicts.  A fit that disagrees with the theory
    is a fact about the benchmark (usually: the sizes are overhead-dominated),
    and this module reports the disagreement instead of hiding it.

Nothing here imports a package runner; the callables are supplied by the
caller so that every benchmark runs the package's own code path.
"""
from __future__ import annotations

import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Sequence

#: The machine every number in this repository's resource estimates was measured on.
#: Recorded verbatim; a projection made on other hardware is a different number.
HARDWARE_BLOCK = (
    "Intel(R) Xeon(R) Processor @ 2.10GHz, 2 vCPU (2 cores, 1 thread/core), 7 GiB RAM\n"
    "Linux container; python 3.11.15, numpy 2.4.4, scipy-openblas 0.3.31.188.0\n"
    "Benchmarks run single-threaded: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
)

#: The environment the benchmarks are run under.  BLAS threading changes wall
#: time by a factor the estimate cannot know, so it is pinned and recorded.
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def pin_single_thread(env: dict | None = None) -> dict:
    """Set (in ``env``, default ``os.environ``) the single-thread BLAS policy the
    benchmarks are measured under.  Must be called before numpy is imported for
    the setting to bind; the driver does that."""
    target = os.environ if env is None else env
    target.update(THREAD_ENV)
    return dict(THREAD_ENV)


def thread_settings_observed() -> dict:
    """What the thread environment actually is, as seen from this process."""
    return {k: os.environ.get(k) for k in THREAD_ENV}


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
@dataclass
class Timing:
    """Wall seconds for one callable at one problem size."""
    size: float
    repeats: int
    warmup: int
    median_s: float
    min_s: float
    max_s: float
    samples_s: list[float] = field(default_factory=list)

    @property
    def spread_fraction(self) -> float:
        """(max - min) / median: the measured, not invented, uncertainty."""
        return (self.max_s - self.min_s) / self.median_s if self.median_s > 0 else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["spread_fraction"] = self.spread_fraction
        return d


def timed(fn: Callable[[], object], *, size: float, repeats: int = 3, warmup: int = 1) -> Timing:
    """Call ``fn`` ``warmup + repeats`` times; the warm-up calls are discarded.

    The warm-up matters: the first call in a process pays import, BLAS buffer
    allocation and any per-process cache the package fills, and charging that to
    every production cell would overstate the total (it is instead reported
    separately, as a per-worker start-up cost, by the drivers that care)."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return Timing(size=float(size), repeats=int(repeats), warmup=int(max(0, warmup)),
                  median_s=float(statistics.median(samples)), min_s=float(min(samples)),
                  max_s=float(max(samples)), samples_s=[float(s) for s in samples])


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
def peak_memory(fn: Callable[[], object]) -> dict:
    """Peak memory of one call, by two independent measures.

    ``tracemalloc_peak_bytes``
        peak *Python* allocation attributable to this call.  numpy array data
        allocated through the Python allocator is included; memory that a
        library takes with plain ``malloc`` (some BLAS workspaces) is not.

    ``rss_high_water_bytes``
        ``ru_maxrss`` of the whole process after the call.  It is a high-water
        mark that never decreases, so inside a long-lived process it reports the
        largest thing that process ever did, not this call.  Use
        :func:`probe_peak_memory_subprocess` when the number has to be about one
        size only.
    """
    tracemalloc.start()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {"tracemalloc_peak_bytes": int(peak),
            "tracemalloc_is": "peak Python-allocator bytes for this call only",
            "rss_high_water_bytes": int(rss_after),
            "rss_before_bytes": int(rss_before),
            "rss_delta_bytes": int(rss_after - rss_before),
            "rss_is": ("ru_maxrss of the WHOLE process (high-water mark, never decreases); "
                       "meaningful per size only when measured in a fresh subprocess")}


def probe_peak_memory_subprocess(argv: Sequence[str], *, env: dict | None = None,
                                 cwd: str | None = None, timeout: float = 900.0) -> dict:
    """Run ``argv`` (which must print exactly one JSON object on its last stdout
    line) in a fresh process and return that object together with the child's own
    ``ru_maxrss``.  A fresh process is the only way an RSS high-water mark can be
    attributed to one problem size."""
    e = dict(os.environ if env is None else env)
    e.update(THREAD_ENV)
    proc = subprocess.run(list(argv), capture_output=True, text=True, env=e, cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"memory probe failed ({proc.returncode}): {proc.stderr[-2000:]}")
    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"memory probe printed no JSON: {proc.stdout[-2000:]}")
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# scaling fit
# ---------------------------------------------------------------------------
#: Theoretical exponents, kept next to the fits so a reader can compare.
THEORY = {
    "eigendecomposition": (3.0, "eigendecomposition of a d x d matrix is O(d^3)"),
    "svd": (3.0, "thin SVD of an N x d matrix is O(N d min(N, d)); at fixed aspect ratio that is O(d^3)"),
    "als_sweep": (1.0, "an ALS sweep is O(iter . N . d . k), so linear in N at fixed d, k, iter"),
    "bootstrap": (1.0, "a bootstrap is linear in the number of replicates"),
    "em_sweep": (1.0, "a weighted-EM sweep is O(niter . n_train . n_components . k), so linear in the rank k"),
}


@dataclass
class PowerLawFit:
    a: float
    b: float
    residual_log_rms: float
    max_abs_log_residual: float
    r_squared: float
    n_points: int
    sizes: list[float]
    times_s: list[float]

    def predict(self, size: float) -> float:
        return float(self.a * (float(size) ** self.b))

    def as_dict(self) -> dict:
        return asdict(self)


def fit_power_law(sizes: Sequence[float], times: Sequence[float]) -> PowerLawFit:
    """Least squares of ``log t = log a + b log n``; returns ``a``, ``b`` and the
    residual.  Two points give an exact fit (residual 0) and that is reported as
    such; three or more make the residual informative."""
    xs = [math.log(float(s)) for s in sizes]
    ys = [math.log(float(t)) for t in times]
    n = len(xs)
    if n < 2:
        raise ValueError("a scaling fit needs at least two sizes")
    if any(t <= 0 for t in times) or any(s <= 0 for s in sizes):
        raise ValueError("sizes and times must be positive to fit on the logs")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("all sizes are equal; no scaling can be fitted")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    loga = my - b * mx
    resid = [y - (loga + b * x) for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in resid)
    ss_tot = sum((y - my) ** 2 for y in ys)
    return PowerLawFit(a=float(math.exp(loga)), b=float(b),
                       residual_log_rms=float(math.sqrt(ss_res / n)),
                       max_abs_log_residual=float(max(abs(r) for r in resid)),
                       r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
                       n_points=int(n), sizes=[float(s) for s in sizes], times_s=[float(t) for t in times])


def exponent_verdict(fitted: float, theoretical: float, *, tol: float = 0.35) -> dict:
    """Compare a fitted exponent with the algorithm's, and say plainly when they
    disagree.  A large shortfall at small sizes means the benchmark is
    overhead-dominated; that is a statement about the benchmark, and it is made
    here rather than smoothed away."""
    diff = float(fitted) - float(theoretical)
    agrees = abs(diff) <= tol
    if agrees:
        note = "the measured exponent agrees with the algorithm within +/- %.2f" % tol
    elif diff < 0:
        note = ("the measured exponent is BELOW the algorithm's by %.2f: at these sizes the kernel is not "
                "yet flop-bound (Python/allocation overhead dominates), so a projection made with the "
                "fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical "
                "exponent overstates it at these ones" % abs(diff))
    else:
        note = ("the measured exponent is ABOVE the algorithm's by %.2f: something outside the counted "
                "flops grows with the size (memory traffic, cache misses, or a second kernel)" % diff)
    return {"fitted_exponent": float(fitted), "theoretical_exponent": float(theoretical),
            "difference": diff, "agrees_within": float(tol), "agrees": bool(agrees), "note": note}


def extrapolation(measured_max_size: float, target_size: float) -> dict:
    """Describe the gap between the largest size actually run and the size being
    projected to.  More than a decade in the dimension is labelled: it is a
    projection, not a measurement."""
    factor = float(target_size) / float(measured_max_size) if measured_max_size else float("inf")
    decades = math.log10(factor) if factor > 0 else float("inf")
    return {"largest_measured_size": float(measured_max_size), "target_size": float(target_size),
            "extrapolation_factor_in_size": factor, "decades_in_size": decades,
            "is_measurement": bool(abs(factor - 1.0) < 1e-9),
            "spans_more_than_one_decade": bool(decades > 1.0),
            "label": ("measured at the production size" if abs(factor - 1.0) < 1e-9 else
                      ("PROJECTED across %.2f decades in the size (%.3gx) - this is a projection, not a "
                       "measurement" % (decades, factor) if decades > 1.0 else
                       "projected %.3gx in the size from the largest measured point" % factor))}


# ---------------------------------------------------------------------------
# the whole benchmark of one kernel
# ---------------------------------------------------------------------------
def scaling_benchmark(name: str, make_call: Callable[[float], Callable[[], object]], sizes: Iterable[float], *,
                      theory_key: str, size_symbol: str, production_size: float, fixture_size: float | None = None,
                      repeats: int = 3, warmup: int = 1, memory_sizes: Sequence[float] | None = None,
                      memory_probe: Callable[[float], dict] | None = None, call_path: str = "",
                      notes: str = "") -> dict:
    """Time ``make_call(size)()`` at every size, fit the scaling, and project to
    ``production_size``.

    Every returned per-size row is a measurement.  The projection row says
    explicitly whether the production size was itself measured (then the
    projection *is* the measurement) or extrapolated, and with which exponent and
    across what factor in the dimension.
    """
    theory_b, theory_note = THEORY[theory_key]
    sizes = [float(s) for s in sizes]
    rows: list[dict] = []
    for s in sizes:
        t = timed(make_call(s), size=s, repeats=repeats, warmup=warmup)
        row = t.as_dict()
        if memory_sizes is None or s in memory_sizes:
            row["memory"] = (memory_probe(s) if memory_probe is not None else peak_memory(make_call(s)))
            row["memory_measured_in"] = "subprocess" if memory_probe is not None else "this process"
        rows.append(row)
    fit = fit_power_law([r["size"] for r in rows], [r["median_s"] for r in rows])
    largest = max(r["size"] for r in rows)
    at_production = [r for r in rows if abs(r["size"] - production_size) < 1e-9]
    # the extrapolation is measured from the production size itself when that size was
    # actually run (which may not be the largest one benchmarked), and from the largest
    # measured size otherwise
    extra = extrapolation(production_size if at_production else largest, production_size)
    extra["largest_size_benchmarked"] = largest
    if at_production:
        r = at_production[0]
        proj = {"basis": "measured", "seconds_per_call_median": r["median_s"],
                "seconds_per_call_low": r["min_s"], "seconds_per_call_high": r["max_s"],
                "how": "the production size was run directly; the range is the measured min/max over repeats"}
    else:
        centre = fit.predict(production_size)
        # the range widens by the fit residual (in the logs) and by the worst
        # measured repeat spread; both are measurements, neither is a margin.
        spread = max((r["max_s"] - r["min_s"]) / r["median_s"] for r in rows)
        lo = centre * math.exp(-fit.max_abs_log_residual) * (1.0 - min(0.9, spread / 2))
        hi = centre * math.exp(+fit.max_abs_log_residual) * (1.0 + spread / 2)
        proj = {"basis": "projected", "seconds_per_call_median": float(centre),
                "seconds_per_call_low": float(lo), "seconds_per_call_high": float(hi),
                "how": ("t = a * %s^b with a = %.6g and the FITTED exponent b = %.3f, evaluated at %s = %g; "
                        "the range is the fit's worst log residual (exp(+/-%.3f)) widened by the worst "
                        "measured repeat spread (%.1f%%)"
                        % (size_symbol, fit.a, fit.b, size_symbol, production_size,
                           fit.max_abs_log_residual, 100 * spread))}
    proj.update(extra)
    return {"kernel": name, "call_path": call_path, "size_symbol": size_symbol,
            "production_size": float(production_size), "fixture_size": (None if fixture_size is None else float(fixture_size)),
            "sizes_run": sizes, "repeats": int(repeats), "warmup": int(warmup),
            "points": rows, "fit": fit.as_dict(),
            "theory": {"key": theory_key, "exponent": theory_b, "note": theory_note},
            "exponent_check": exponent_verdict(fit.b, theory_b),
            "projection": proj, "notes": notes}


# ---------------------------------------------------------------------------
# small helpers used by the drivers and by cli._estimate
# ---------------------------------------------------------------------------
def wall_time_range(cpu_seconds_low: float, cpu_seconds_high: float, workers: Iterable[int],
                    *, efficiency: float = 1.0) -> dict:
    """Perfectly-parallel wall time at each worker count.  The assumption is
    stated in the output, not buried: the cells of every package are independent
    processes that share nothing but the ledger, so the only modelled loss is
    ``efficiency`` (1.0 = none) and the serial ledger append."""
    out = {}
    for w in workers:
        w = int(w)
        out[str(w)] = {"workers": w,
                       "hours_low": float(cpu_seconds_low / (w * efficiency) / 3600.0),
                       "hours_high": float(cpu_seconds_high / (w * efficiency) / 3600.0)}
    return out


def dir_bytes(path) -> int:
    from pathlib import Path
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def json_bytes(obj) -> int:
    """Bytes one result object costs on disk, in the encoding the runner uses."""
    return len(json.dumps(obj, indent=1, sort_keys=True, default=str).encode("utf-8"))


def main_probe() -> int:  # pragma: no cover - exercised through the driver
    """``python -m ...common.benchmark --self-test`` prints one JSON line; used to
    check the subprocess memory-probe plumbing."""
    if "--self-test" in sys.argv:
        import numpy as np
        n = int(sys.argv[sys.argv.index("--self-test") + 1]) if len(sys.argv) > 2 else 512

        def work():
            a = np.random.default_rng(0).standard_normal((n, n))
            return np.linalg.eigvalsh(a @ a.T)
        m = peak_memory(work)
        print(json.dumps({"size": n, **m}))
        return 0
    print("usage: python -m ...common.benchmark --self-test [n]", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_probe())
