"""
Minimal QP (Quasi-Particle) trace simulator — standalone.

Generates a clean (noise-free) QP signal trace from an explicit array of
QP arrival times. Each QP arrival contributes one single-QP response template
to the trace; the result is the sum.

Conventions match the existing DELight ``TraceSimulator`` phonon path:
    * Single-QP template = exp(+(t - trigger_time)/tau_rise) for t <= trigger_time,
                           exp(-(t - trigger_time)/tau_decay) for t >  trigger_time
      So a QP arriving at t=0 produces a peak at t=trigger_time, with a leading
      rise of timescale tau_rise and a trailing decay of timescale tau_decay.
    * Per-QP ADC amplitude = gain_QE * E_to_ADC * 1e-3 * meV_per_QP
      (the 1e-3 converts the assumed ~1 meV per phonon to eV).

This module has no dependency on the rest of the TraceSimulator package.
Add your own noise on top of the returned trace.

Example
-------
>>> sim = QPSimulator()
>>> rng = np.random.default_rng(0)
>>> arrival_times_ns = rng.exponential(scale=2e6, size=5000) + 6e6  # ns
>>> trace = sim.generate(arrival_times_ns)        # shape (16384,)
>>> trace_with_noise = trace + my_noise_module.generate(len(trace))
"""

from __future__ import annotations

import numpy as np


class QPSimulator:
    """Minimal clean-trace QP simulator.

    Parameters
    ----------
    sampling_frequency : float
        Digitizer sampling frequency in Hz. Default 2.5e5 (matches TraceSimulator).
    trace_samples : int
        Number of samples in the output trace. Default 16384.
    tau_rise : float
        Single-QP response rise time in ns. Default 50e3.
    tau_decay : float
        Single-QP response decay time in ns. Default 3e6.
    trigger_time : float or None
        Time within the trace (ns) at which a QP arriving at t=0 will peak.
        If None, defaults to 10% of the trace duration.
    gain_QE : float
        Quasi-particle energy-to-charge gain (matches TraceSimulator).
    E_to_ADC : float
        eV -> ADC conversion factor (matches TraceSimulator).
    meV_per_QP : float
        Mean energy per QP in meV. Default 1.0.
    """

    def __init__(
        self,
        sampling_frequency: float = 2.5e5,
        trace_samples: int = 16_384,
        tau_rise: float = 50e3,
        tau_decay: float = 3e6,
        trigger_time: float | None = None,
        gain_QE: float = 15.0,
        E_to_ADC: float = 2.0,
        meV_per_QP: float = 1.0,
    ):
        # Digitizer
        self.frequency = float(sampling_frequency)
        self.dt = 1.0 / self.frequency * 1e9              # ns per sample
        self.trace_samples = int(trace_samples)
        self.trace_duration = self.dt * self.trace_samples  # ns

        # Pulse shape
        self.tau_rise = float(tau_rise)
        self.tau_decay = float(tau_decay)
        self.trigger_time = (
            0.1 * self.trace_duration if trigger_time is None else float(trigger_time)
        )

        # Amplitude (ADC counts per single QP)
        self.gain_QE = float(gain_QE)
        self.E_to_ADC = float(E_to_ADC)
        self.meV_per_QP = float(meV_per_QP)
        self.qp_amplitude = self.gain_QE * self.E_to_ADC * 1e-3 * self.meV_per_QP

        # Histogram bin edges for arrival times: length trace_samples + 1
        self.t_edges = np.arange(0.0, self.trace_duration + self.dt / 2.0, self.dt)

        self._build_template()

    # ------------------------------------------------------------------ #
    # Template
    # ------------------------------------------------------------------ #
    def _build_template(self) -> None:
        """Build single-QP response template (length 2.5 * trace_samples)."""
        xs = np.arange(0.0, self.trace_duration * 2.5, self.dt)
        # Cap the exponent at 0 in each branch so the unused half never
        # overflows; the `np.where` selects the correct half regardless.
        rising = np.exp(np.minimum((xs - self.trigger_time) / self.tau_rise, 0.0))
        falling = np.exp(np.minimum(-(xs - self.trigger_time) / self.tau_decay, 0.0))
        self.template = np.where(xs <= self.trigger_time, rising, falling)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self, arrival_times) -> np.ndarray:
        """Return a clean QP trace (in ADC counts) for the given arrivals.

        Parameters
        ----------
        arrival_times : array_like of float
            QP arrival times in ns. Each entry is one QP. Times outside
            [0, trace_duration) are silently dropped.

        Returns
        -------
        np.ndarray of shape (trace_samples,)
            Clean QP trace, float ADC counts. Add noise yourself.
        """
        t = np.asarray(arrival_times, dtype=float).ravel()
        if t.size == 0:
            return np.zeros(self.trace_samples, dtype=float)
        counts, _ = np.histogram(t, bins=self.t_edges)
        return self._sum_template(counts) * self.qp_amplitude

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _sum_template(self, counts: np.ndarray) -> np.ndarray:
        """Sum-of-templates: place the template at every non-empty bin."""
        n = len(counts)
        out = np.zeros(n, dtype=float)
        for i in np.nonzero(counts)[0]:
            out[i:] += counts[i] * self.template[: n - i]
        return out


# ---------------------------------------------------------------------- #
# Self-test / quick demo
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = QPSimulator()
    print(
        f"dt = {sim.dt:g} ns | trace = {sim.trace_duration*1e-6:.2f} ms "
        f"| trigger_time = {sim.trigger_time*1e-6:.2f} ms "
        f"| QP amplitude = {sim.qp_amplitude:g} ADC"
    )

    rng = np.random.default_rng(0)
    n_qp = 5000

    # Same number of QPs, three different arrival-time distributions
    bursts = {
        "early (offset 0 ms, tau 2 ms)":   rng.exponential(2e6, n_qp) + 0.0,
        "mid   (offset 6 ms, tau 2 ms)":   rng.exponential(2e6, n_qp) + 6e6,
        "late  (offset 15 ms, tau 5 ms)":  rng.exponential(5e6, n_qp) + 15e6,
    }

    t_axis_ms = np.arange(sim.trace_samples) * sim.dt * 1e-6
    plt.figure(figsize=(9, 4))
    for label, t_arr in bursts.items():
        plt.plot(t_axis_ms, sim.generate(t_arr), lw=1.0, label=label)
    plt.xlabel("time [ms]")
    plt.ylabel("ADC counts")
    plt.title(f"QPSimulator — clean traces, {n_qp} QPs each")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png = "qp_simulator_demo.png"
    plt.savefig(out_png, dpi=120)
    print(f"saved {out_png}")
