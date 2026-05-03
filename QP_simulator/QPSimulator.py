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
    def generate(
        self,
        arrival_times,
        return_amplitude: bool = False,
    ) -> "np.ndarray | tuple[np.ndarray, float]":
        """Return a clean QP trace (in ADC counts) for the given arrivals.

        Parameters
        ----------
        arrival_times : array_like of float
            QP arrival times in ns. Each entry is one QP. Times outside
            [0, trace_duration) are silently dropped.
        return_amplitude : bool, optional
            When True, also return the true signal amplitude in ADC counts:
            ``amplitude_true = n_inside * self.qp_amplitude`` where
            ``n_inside`` is the number of arrival times that fall within
            ``[0, trace_duration)``. Default False (original behaviour).

        Returns
        -------
        trace : np.ndarray, shape (trace_samples,)
            Clean QP trace, float ADC counts. Add noise yourself.
        amplitude_true : float
            Only returned when ``return_amplitude=True``. True signal
            amplitude in ADC counts.
        """
        t = np.asarray(arrival_times, dtype=float).ravel()
        if t.size == 0:
            trace = np.zeros(self.trace_samples, dtype=float)
            return (trace, 0.0) if return_amplitude else trace
        counts, _ = np.histogram(t, bins=self.t_edges)
        trace = self._sum_template(counts) * self.qp_amplitude
        if return_amplitude:
            amplitude_true = float(counts.sum()) * self.qp_amplitude
            return trace, amplitude_true
        return trace

    def generate_family(
        self,
        n_events: int,
        tau_decay_range: "tuple[float, float]" = (3e6, 3e6),
        t0_jitter_range: "tuple[float, float]" = (0.0, 0.0),
        n_QP_range: "tuple[int, int]" = (5000, 5000),
        rng: "np.random.Generator | None" = None,
    ) -> "tuple[np.ndarray, dict]":
        """Generate ``n_events`` clean traces with independently drawn per-event
        parameters.

        For each event:

        * ``tau_decay`` is drawn uniformly from ``tau_decay_range``.
        * ``trigger_time`` = ``self.trigger_time`` + Uniform(``t0_jitter_range``).
        * ``n_QP`` is drawn uniformly (integer) from ``n_QP_range``.
        * Arrival times are drawn as Exponential(scale=2e6 ns) + trigger_time,
          matching the existing demo convention.
        * A temporary :class:`QPSimulator` is built with the per-event
          ``tau_decay`` and ``trigger_time`` to generate each trace; all
          other parameters are inherited from ``self``.

        No noise is added. Apply noise externally using the ``noise_module``
        classes before passing traces to an estimator.

        Parameters
        ----------
        n_events : int
            Number of events (traces) to generate.
        tau_decay_range : (float, float)
            Inclusive range for uniform draw of ``tau_decay`` in ns.
            Pass ``(x, x)`` for a fixed value.
        t0_jitter_range : (float, float)
            Inclusive range for uniform jitter (ns) added to
            ``self.trigger_time``. Pass ``(0.0, 0.0)`` for no jitter.
        n_QP_range : (int, int)
            Inclusive range for uniform integer draw of the number of QPs
            per event.
        rng : np.random.Generator or None
            Random generator for reproducibility. If None, uses
            ``np.random.default_rng()``.

        Returns
        -------
        traces : np.ndarray, shape (n_events, trace_samples)
            Clean traces (ADC counts). Add noise externally.
        params : dict
            Per-event ground-truth with keys:

            * ``'tau_decay'``    — np.ndarray, shape (n_events,)
            * ``'t0_shift'``    — np.ndarray, shape (n_events,), jitter in ns
            * ``'n_QP'``        — np.ndarray, shape (n_events,), dtype int
            * ``'amplitude_ADC'`` — np.ndarray, shape (n_events,), true peak ADC
        """
        if rng is None:
            rng = np.random.default_rng()

        tau_lo, tau_hi = float(tau_decay_range[0]), float(tau_decay_range[1])
        t0_lo, t0_hi = float(t0_jitter_range[0]), float(t0_jitter_range[1])
        nqp_lo, nqp_hi = int(n_QP_range[0]), int(n_QP_range[1])

        tau_decays = (
            np.full(n_events, tau_lo)
            if tau_lo == tau_hi
            else rng.uniform(tau_lo, tau_hi, size=n_events)
        )
        t0_shifts = (
            np.zeros(n_events)
            if t0_lo == t0_hi
            else rng.uniform(t0_lo, t0_hi, size=n_events)
        )
        n_qps = (
            np.full(n_events, nqp_lo, dtype=int)
            if nqp_lo == nqp_hi
            else rng.integers(nqp_lo, nqp_hi + 1, size=n_events)
        )

        traces = np.empty((n_events, self.trace_samples), dtype=float)
        amplitudes = np.empty(n_events, dtype=float)

        for i in range(n_events):
            ev_tau = tau_decays[i]
            ev_t0 = self.trigger_time + t0_shifts[i]
            ev_nqp = int(n_qps[i])

            ev_sim = QPSimulator(
                sampling_frequency=self.frequency,
                trace_samples=self.trace_samples,
                tau_rise=self.tau_rise,
                tau_decay=ev_tau,
                trigger_time=ev_t0,
                gain_QE=self.gain_QE,
                E_to_ADC=self.E_to_ADC,
                meV_per_QP=self.meV_per_QP,
            )

            arrivals = rng.exponential(scale=2e6, size=ev_nqp) + ev_t0
            trace, amp = ev_sim.generate(arrivals, return_amplitude=True)
            traces[i] = trace
            amplitudes[i] = amp

        params = {
            "tau_decay": tau_decays,
            "t0_shift": t0_shifts,
            "n_QP": n_qps,
            "amplitude_ADC": amplitudes,
        }
        return traces, params

    def get_template_at_shift(self, t0_shift_ns: float) -> np.ndarray:
        """Return the single-QP response template shifted by ``t0_shift_ns``
        nanoseconds relative to ``self.trigger_time``.

        Builds a temporary :class:`QPSimulator` with
        ``trigger_time = self.trigger_time + t0_shift_ns`` and returns its
        template (the ``self.template`` attribute) truncated to
        ``self.trace_samples``.

        Parameters
        ----------
        t0_shift_ns : float
            Shift in nanoseconds. Positive = later peak.

        Returns
        -------
        np.ndarray, shape (trace_samples,)
            Shifted single-QP template (normalised, not scaled by
            ``qp_amplitude``).
        """
        shifted_sim = QPSimulator(
            sampling_frequency=self.frequency,
            trace_samples=self.trace_samples,
            tau_rise=self.tau_rise,
            tau_decay=self.tau_decay,
            trigger_time=self.trigger_time + float(t0_shift_ns),
            gain_QE=self.gain_QE,
            E_to_ADC=self.E_to_ADC,
            meV_per_QP=self.meV_per_QP,
        )
        return shifted_sim.template[: self.trace_samples].copy()

    @staticmethod
    def estimate_psd(
        noise_traces: np.ndarray,
        sampling_frequency: float,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Estimate the one-sided noise PSD from an array of noise-only traces.

        For each trace, computes the real FFT, takes the squared magnitude,
        and averages across traces. The result is normalised so that
        ``sum(J_k) * (fs / N)`` equals the mean noise power, matching the
        convention used by ``NoiseGenerator``.

        Normalisation: ``J_k = mean(|FFT(x)|^2) * 2 / (N * fs)``

        where ``N = trace_samples`` and ``fs = sampling_frequency``.
        The factor of 2 accounts for folding negative frequencies into the
        one-sided spectrum (the DC and Nyquist bins are **not** doubled, in
        keeping with the standard ``numpy.fft.rfft`` convention, but for
        traces long enough for those bins to be negligible the difference
        is immaterial).

        Parameters
        ----------
        noise_traces : np.ndarray, shape (N_traces, trace_samples)
            Noise-only traces. Must be 2-D.
        sampling_frequency : float
            Digitizer sampling rate in Hz.

        Returns
        -------
        frequencies : np.ndarray, shape (trace_samples // 2 + 1,)
            One-sided frequency axis in Hz.
        J_k : np.ndarray, shape (trace_samples // 2 + 1,)
            One-sided PSD estimate (units: ADC² / Hz).
        """
        noise_traces = np.asarray(noise_traces, dtype=float)
        if noise_traces.ndim != 2:
            raise ValueError(
                "noise_traces must be 2-D with shape (N_traces, trace_samples)."
            )
        N = noise_traces.shape[1]
        fs = float(sampling_frequency)

        spectra = np.abs(np.fft.rfft(noise_traces, axis=1)) ** 2  # (N_traces, N//2+1)
        mean_spectrum = spectra.mean(axis=0)

        # One-sided PSD: multiply by 2 to fold negative frequencies, then
        # normalise by (N * fs) so the integral over positive freqs = power.
        J_k = mean_spectrum * 2.0 / (N * fs)

        frequencies = np.fft.rfftfreq(N, d=1.0 / fs)
        return frequencies, J_k

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
