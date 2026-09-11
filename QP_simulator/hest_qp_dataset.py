"""HeST -> QPSimulator -> noise_module dataset builder for the NFPA section.

The expensive stage (HeST quasiparticle transport) is separated from the cheap
one (rendering traces), because they answer different questions:

  * ``build_event_library`` runs HeST once per event and stores, for every
    detected quasiparticle, which CPD it hit and when. That is the whole of
    the physics; it costs ~1.5 s/event and is cached as a small .npz.
  * ``render_traces`` turns a library into (n, C, d) traces for a *chosen*
    trigger jitter, CPD decay time and thinning factor. It is exact -- not a
    surrogate -- and fast, so trigger-jitter sweeps cost nothing.
  * ``add_noise`` / ``noise_only`` draw Sigma_c (x) Sigma_t noise from
    ``noise_module`` at a chosen power.

One event = one ER deposit at (x, y, z) with energy E. Per-event truth:
E (amplitude), (x, y) vertex -> channel factor, t0 and tau -> time factor.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import HeST as H
from QPSimulator import QPSimulator
from noise_module.multichannel_noise import MultiChannelNoiseGenerator

# --------------------------------------------------------------------------
# Reference-cell configuration
# --------------------------------------------------------------------------
FS = 2.5e5            # Hz, DELight TraceSimulator convention
TRACE_SAMPLES = 2048  # 8.19 ms record (2.7 tau_decay)
TAU_RISE = 50e3       # ns
TAU_DECAY_NOM = 3e6   # ns
FILL_HEIGHT = 4.5     # cm
QP_FRACTION = 0.15    # thinning; traces are rescaled by 1/QP_FRACTION
NOISE_POWER = 16.0    # ADC^2 -> sigma = 4 ADC
NOISE_TYPE = "pink"
CORR_STRENGTH = 0.35

E_RANGE = (500.0, 2000.0)      # eV, log-uniform
R_MAX = 2.5                    # cm
Z_RANGE = (0.5, 4.0)           # cm
T0_JITTER = 25e3               # ns, uniform +- (0.5 tau_rise)
TAU_SPREAD = 0.40              # +-40 % uniform

ARRAY_MAP = np.array([[0, 0, 1, 1, 0, 0],
                      [0, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1, 0],
                      [0, 0, 1, 1, 0, 0]])
PITCH = 1.1
SENSOR_XY = np.array([((i - 2.5) * PITCH, (j - 2.5) * PITCH)
                      for i in range(6) for j in range(6) if ARRAY_MAP[i, j] > 0.5])
N_CHANNELS = len(SENSOR_XY)
TRIGGER_TIME = 0.1 * TRACE_SAMPLES / FS * 1e9      # ns, nominal peak position

# The CPD response is band-limited at f_sig = 1/(2 pi tau_rise) ~ 3 kHz, so a
# 250 kHz record spends 97% of its dimensions on noise-only bins. Decimating by
# DECIMATE (with the anti-alias filter a real DAQ would also apply) keeps every
# bit of signal and takes d from 2048 to 512, which is what moves the subspace
# estimate out of the high-dimensional regime where weak directions sit under
# the Marchenko-Pastur edge. Set DECIMATE = 1 to work at the native rate.
DECIMATE = 4
FS_EFF = FS / DECIMATE
D_EFF = TRACE_SAMPLES // DECIMATE


def downsample(x, factor=None):
    """Anti-alias filter + decimate along the last axis (no-op for factor 1)."""
    factor = DECIMATE if factor is None else factor
    if factor == 1:
        return np.asarray(x, float)
    from scipy.signal import decimate as _dec
    return _dec(np.asarray(x, float), int(factor), ftype="fir",
                zero_phase=True, axis=-1)


def qp_amplitude() -> float:
    return QPSimulator(sampling_frequency=FS, trace_samples=TRACE_SAMPLES).qp_amplitude


# --------------------------------------------------------------------------
# per-event truth
# --------------------------------------------------------------------------
def sample_truth(n, rng, cell="reference", t0_jitter=T0_JITTER,
                 tau_spread=TAU_SPREAD):
    """Draw per-event physical parameters; `cell` freezes all but one factor."""
    ref = dict(
        logE=rng.uniform(np.log(E_RANGE[0]), np.log(E_RANGE[1]), n),
        r=R_MAX * np.sqrt(rng.uniform(0, 1, n)),
        phi=rng.uniform(0, 2 * np.pi, n),
        z=rng.uniform(*Z_RANGE, size=n),
        t0=rng.uniform(-t0_jitter, t0_jitter, n),
        tau=TAU_DECAY_NOM * (1.0 + rng.uniform(-tau_spread, tau_spread, n)),
    )
    mid = dict(
        logE=np.full(n, np.log(np.sqrt(E_RANGE[0] * E_RANGE[1]))),
        r=np.zeros(n), phi=np.zeros(n), z=np.full(n, 2.0),
        t0=np.zeros(n), tau=np.full(n, TAU_DECAY_NOM),
    )
    if cell == "reference":
        d = ref
    elif cell == "energy_only":
        d = dict(mid); d["logE"] = ref["logE"]
    elif cell == "x_only":
        d = dict(mid)
        s = rng.uniform(-R_MAX, R_MAX, n)
        d["r"] = np.abs(s); d["phi"] = np.where(s < 0, np.pi, 0.0)
    elif cell == "y_only":
        d = dict(mid)
        s = rng.uniform(-R_MAX, R_MAX, n)
        d["r"] = np.abs(s); d["phi"] = np.where(s < 0, -np.pi / 2, np.pi / 2)
    elif cell == "t0_only":
        d = dict(mid); d["t0"] = ref["t0"]
    elif cell == "tau_only":
        d = dict(mid); d["tau"] = ref["tau"]
    else:
        raise ValueError(cell)
    out = {k: np.asarray(v, dtype=float) for k, v in d.items()}
    out["E"] = np.exp(out["logE"])
    out["x"] = out["r"] * np.cos(out["phi"])
    out["y"] = out["r"] * np.sin(out["phi"])
    return out


# --------------------------------------------------------------------------
# stage 1 (expensive): HeST quasiparticle transport
# --------------------------------------------------------------------------
def build_event_library(cell: str, n: int, seed: int, path,
                        qp_fraction: float = QP_FRACTION,
                        verbose_every: int = 200, overwrite: bool = False):
    """Run HeST once per event; cache (sensor id, arrival time) for every hit.

    Stored ragged: `hit_sensor` and `hit_time_ns` are concatenated over events
    with `hit_offset` giving the per-event boundaries.
    """
    path = str(path)
    if os.path.exists(path) and not overwrite:
        return np.load(path, allow_pickle=True)

    rng = np.random.default_rng(seed)
    truth = sample_truth(n, rng, cell=cell)
    np.random.seed(seed)                      # HeST uses the legacy global RNG
    det = H.HeRALD_v1(fill_height=FILL_HEIGHT)
    C = det.get_nsensors()

    sensors, times, offsets = [], [], [0]
    n_qp_full = np.zeros(n, dtype=np.int64)
    t_start = time.time()
    for i in range(n):
        nq = H.GetQuanta(float(truth["E"][i]), "ER").Quasiparticles
        n_qp_full[i] = nq
        out = H.GetEvaporationSignal(det, max(int(round(nq * qp_fraction)), 1),
                                     float(truth["x"][i]), float(truth["y"][i]),
                                     float(truth["z"][i]), useMap=False,
                                     plot_3d=False, debug=True, verbose=False,
                                     simulate_evap_probs=False)
        _, sid, _, tt, _, _, _ = out
        hit = sid > -0.5
        sensors.append(sid[hit].astype(np.int16))
        times.append(tt[hit].astype(np.float32))
        offsets.append(offsets[-1] + int(hit.sum()))
        if verbose_every and (i + 1) % verbose_every == 0:
            el = time.time() - t_start
            print(f"    [{cell} seed {seed}] {i+1}/{n}  {el:.0f}s "
                  f"({el/(i+1):.2f} s/event)", flush=True)

    np.savez_compressed(
        path,
        hit_sensor=np.concatenate(sensors) if sensors else np.zeros(0, np.int16),
        hit_time_ns=np.concatenate(times) if times else np.zeros(0, np.float32),
        hit_offset=np.array(offsets, dtype=np.int64),
        n_qp_full=n_qp_full, sensor_xy=SENSOR_XY, n_channels=C,
        **{f"truth_{k}": v for k, v in truth.items()},
        provenance=json.dumps(dict(
            cell=cell, seed=seed, n=n, fs=FS, trace_samples=TRACE_SAMPLES,
            tau_rise=TAU_RISE, tau_decay_nom=TAU_DECAY_NOM,
            qp_fraction=qp_fraction, fill_height=FILL_HEIGHT,
            hest="spice-herald/HeST 0.2.0 (source)")),
    )
    return np.load(path, allow_pickle=True)


def library_summary(lib):
    """Per-event channel share, detected count and median arrival time."""
    off = lib["hit_offset"]
    C = int(lib["n_channels"])
    n = len(off) - 1
    share = np.zeros((n, C))
    n_det = np.diff(off)
    t_med = np.zeros(n)
    sid, tt = lib["hit_sensor"], lib["hit_time_ns"]
    for i in range(n):
        s = sid[off[i]:off[i + 1]]
        if s.size:
            share[i] = np.bincount(s, minlength=C) / s.size
            t_med[i] = np.median(tt[off[i]:off[i + 1]])
    return share, n_det, t_med


def response_params(n, seed, cell="reference"):
    """Draw the CPD-response parameters (t0, tau) for a cell.

    These are *not* HeST quantities -- the event library knows nothing about
    the trigger or the sensor time constants -- so they are drawn at render
    time. That is what lets a trigger-jitter sweep reuse one library.
    """
    rng = np.random.default_rng(seed)
    t0 = rng.uniform(-T0_JITTER, T0_JITTER, n)
    tau = TAU_DECAY_NOM * (1.0 + rng.uniform(-TAU_SPREAD, TAU_SPREAD, n))
    if cell in ("energy_only", "x_only", "y_only"):
        t0, tau = np.zeros(n), np.full(n, TAU_DECAY_NOM)
    elif cell == "t0_only":
        tau = np.full(n, TAU_DECAY_NOM)
    elif cell == "tau_only":
        t0 = np.zeros(n)
    return t0, tau


# --------------------------------------------------------------------------
# stage 2 (cheap, exact): render traces
# --------------------------------------------------------------------------
def render_traces(lib, t0=None, tau=None, qp_fraction=QP_FRACTION,
                  fs=FS, n_time=TRACE_SAMPLES, tau_rise=TAU_RISE,
                  trigger_time=TRIGGER_TIME):
    """Clean (n, C, d) traces from a HeST event library.

    `t0` / `tau` default to the values stored with the library, but any array
    may be supplied -- the arrival times are physics and the CPD response is
    not, so re-rendering at a different trigger jitter is exact.
    """
    off = lib["hit_offset"]
    C = int(lib["n_channels"])
    n = len(off) - 1
    t0 = lib["truth_t0"] if t0 is None else np.broadcast_to(np.asarray(t0, float), (n,))
    tau = lib["truth_tau"] if tau is None else np.broadcast_to(np.asarray(tau, float), (n,))
    sid, tt = lib["hit_sensor"], lib["hit_time_ns"]
    out = np.zeros((n, C, n_time), dtype=np.float32)
    for i in range(n):
        sim = QPSimulator(sampling_frequency=fs, trace_samples=n_time,
                          tau_rise=tau_rise, tau_decay=float(tau[i]),
                          trigger_time=trigger_time + float(t0[i]))
        s = sid[off[i]:off[i + 1]]
        t = tt[off[i]:off[i + 1]].astype(float)
        for c in np.unique(s):
            sel = t[s == c]
            if sel.size:
                out[i, c] = (sim.generate(sel) / qp_fraction).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# stage 3: noise
# --------------------------------------------------------------------------
def _noise_gen(C, seed, noise_power=NOISE_POWER):
    return MultiChannelNoiseGenerator(
        dict(noise_type=NOISE_TYPE, noise_power=noise_power, sampling_frequency=FS),
        dict(n_channels=C, channel_gain_jitter=0.0,
             private_strength_range=[1.0, 1.0], normalize_channel_variance=False),
        seed=seed)


def noise_only(n, C=N_CHANNELS, T=TRACE_SAMPLES, seed=0, noise_power=NOISE_POWER):
    mc = _noise_gen(C, seed, noise_power)
    out = np.empty((n, C, T), dtype=np.float32)
    for i in range(n):
        out[i] = mc.generate_shared_private(C, T, corr_strength=CORR_STRENGTH)
    return out


def add_noise(clean, seed, noise_power=NOISE_POWER, return_meta=False):
    n, C, T = clean.shape
    mc = _noise_gen(C, seed, noise_power)
    noisy = np.empty_like(clean)
    meta = None
    for i in range(n):
        Xn, m = mc.generate_shared_private(C, T, corr_strength=CORR_STRENGTH,
                                           return_metadata=True)
        noisy[i] = clean[i] + Xn.astype(clean.dtype)
        if meta is None:
            meta = m
    return (noisy, meta) if return_meta else noisy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cell", default="reference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lib = build_event_library(a.cell, a.n, a.seed, a.out)
    share, n_det, t_med = library_summary(lib)
    print(f"wrote {a.out}: {a.n} events, median detected QP {np.median(n_det):.0f}, "
          f"median arrival {np.median(t_med)*1e-3:.2f} us")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# calibration grid: the channel response h(x, y, z) at high statistics
# --------------------------------------------------------------------------
def build_calibration_grid(path, n_qp: int = 120_000, seed: int = 4242,
                           z_values=(0.9, 1.8, 2.7, 3.6), step: float = 1.25,
                           r_max: float = R_MAX, overwrite: bool = False):
    """Measure the CPD collection pattern on a position grid.

    The channel tangent is a property of the detector, so it is measured once
    here at high QP statistics rather than estimated from the same noisy events
    the model is fitted to.
    """
    path = str(path)
    if os.path.exists(path) and not overwrite:
        return np.load(path)
    np.random.seed(seed)
    det = H.HeRALD_v1(fill_height=FILL_HEIGHT)
    C = det.get_nsensors()
    axis = np.arange(-2, 3) * step
    pts = [(x, y) for x in axis for y in axis if np.hypot(x, y) <= r_max + 1e-9]
    rows, meta = [], []
    for (x, y) in pts:
        for z in z_values:
            out = H.GetEvaporationSignal(det, int(n_qp), float(x), float(y),
                                         float(z), useMap=False, plot_3d=False,
                                         debug=True, verbose=False,
                                         simulate_evap_probs=False)
            _, sid, _, tt, _, _, _ = out
            hit = sid > -0.5
            s = sid[hit].astype(int)
            rows.append(np.bincount(s, minlength=C) / max(s.size, 1))
            meta.append((x, y, z, s.size,
                         float(np.median(tt[hit])) if s.size else 0.0))
    np.savez_compressed(path, share=np.array(rows), meta=np.array(meta),
                        sensor_xy=SENSOR_XY, n_qp=n_qp, seed=seed)
    return np.load(path)
