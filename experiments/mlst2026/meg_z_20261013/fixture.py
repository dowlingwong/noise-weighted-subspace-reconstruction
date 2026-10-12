"""Synthetic fixtures (never real data): derived-level subjects with known Kronecker or non-Kronecker noise,
and a tiny BIDS-like raw dataset built with ``mne.create_info`` / ``mne.io.RawArray`` for the preprocessing
tests.  Used by ``runner smoke`` and ``tests/``.  The smoke namespace and subject labels (sub-9x) are disjoint
from the analysed ds000117 subjects."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import common, estimators
from .common import vec

SMOKE_SUBJECTS = ['sub-91', 'sub-92', 'sub-93', 'sub-94', 'sub-95']


def ar1(rho: float, n: int) -> np.ndarray:
    return rho ** np.abs(np.arange(n)[:, None] - np.arange(n))


def fixture_covariance(C: int, T: int, kind: str = 'kron', mix: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kronecker S_t (x) S_c (kind 'kron') or the sum (1 - mix) S_t (x) S_c + mix S_t2 (x) S_c2 (kind 'nonkron')."""
    St = ar1(0.6, T)
    g = np.linspace(-1, 1, C)
    Sc = 0.5 * np.eye(C) + np.exp(-(g[:, None] - g[None]) ** 2 / 0.5)
    cov = np.kron(St, Sc)
    if kind == 'nonkron':
        St2 = ar1(-0.3, T) + 0.2 * np.eye(T)
        Sc2 = np.outer(g, g) + 0.3 * np.eye(C)
        cov = (1 - mix) * cov + mix * np.kron(St2, Sc2)
    elif kind != 'kron':
        raise ValueError(kind)
    return common.symmetric(cov), St, Sc


def evoked_pattern(C: int, T: int, sfreq: float, centers=(0.07, 0.13), width: float = 0.03) -> np.ndarray:
    t = np.arange(T) / sfreq
    g = np.linspace(-1, 1, C)
    a1 = np.exp(-(g - 0.3) ** 2 / 0.4)
    a2 = g - 0.2 * g ** 2
    b1 = np.exp(-0.5 * ((t - centers[0]) / width) ** 2)
    b2 = np.exp(-0.5 * ((t - centers[1]) / width) ** 2)
    return 3.0 * (np.outer(a1, b1) + 0.8 * np.outer(a2, b2))


def make_derived_subject(out_dir, subject: str, C: int = 4, T: int = 6, sfreq: float = 27.5, kind: str = 'kron',
                         n: dict | None = None, n_ref: int = 3000, seed: int = 0, locked_mean: float = 0.0,
                         heavy_tails: bool = False) -> dict:
    """Write OUT/<subject>.h5 in the preprocess layout and return its summary (with SHA-256)."""
    from .preprocess import subject_payload, write_h5
    n = n or {'training': 900, 'calibration': 600, 'evaluation': 300}
    cov, St, Sc = fixture_covariance(C, T, kind)
    L = np.linalg.cholesky(cov)
    gen = common.rng('megz-fixture', subject, seed)

    def draw(m):
        z = gen.normal(size=(m, C * T))
        if heavy_tails:
            z = z * np.sqrt(3.0 / gen.chisquare(5, size=(m, 1)))      # multivariate t_5, unit variance
        return z @ L.T

    wins, meta = {}, {}
    run_of = {'training': [3, 4], 'calibration': [5], 'evaluation': [6]}
    for part, m in n.items():
        X = draw(m)
        if locked_mean:
            X[:, :C] += locked_mean
        wins[part] = common.unvec(X, C, T)
        runs = np.array([run_of[part][i % len(run_of[part])] for i in range(m)])
        meta[part] = {'run': runs, 'start': np.arange(m) * T, 'gap': np.arange(m) // 3, 'position': np.arange(m) % 3,
                      'rposition': 2 - np.arange(m) % 3}
    ref = estimators.moment_stats(draw(n_ref))
    ref.n_eff = n_ref
    ev = evoked_pattern(C, T, sfreq)
    trials = {}
    for r in range(1, 7):
        face = np.arange(40) % 3 != 2
        Xt = common.unvec(draw(40), C, T) + np.where(face, 1.0, 0.5)[:, None, None] * ev
        trials[r] = {'X': Xt, 'face': face, 'value': np.where(face, 5, 17)}
    attrs = {'subject': subject, 'C': C, 'T': T, 'sfreq': sfreq, 'channels': [f'MEG{i:04d}' for i in range(C)],
             'bads_dropped': [], 'drop_selected': [], 'preprocess_config': {'fixture': kind}, 'config_sha256': 'fixture',
             'partition': {'reference': [1, 2], 'training': [3, 4], 'calibration': [5], 'evaluation': [6]},
             'created_by': 'meg_z_20261013.fixture', 'fixture_kind': kind, 'proc_history_empty': True}
    h = write_h5(Path(out_dir) / f'{subject}.h5', subject_payload(wins, meta, ref, ev, trials), attrs)
    return {'subject': subject, 'file': f'{subject}.h5', **h, 'C': C, 'T': T, 'config_sha256': 'fixture',
            'true_cov': cov, 'St': St, 'Sc': Sc}


def smoke_protocol(base: dict) -> dict:
    """Small-dimension protocol for the synthetic smoke run (C = 4, T = 6 at 27.5 Hz)."""
    p = json.loads(json.dumps(base))
    p.update(version='megz-smoke', status='SMOKE (synthetic fixture; not data)', frozen=False,
             namespace=common.NAMESPACE_SMOKE)
    p['subjects'] = {'analysed': SMOKE_SUBJECTS, 'dev': SMOKE_SUBJECTS[:4], 'confirmation': SMOKE_SUBJECTS[4:],
                     'rule': 'smoke fixture'}
    p['preprocess'] = dict(p['preprocess'], C=4, T=6)
    p['signal'] = dict(p['signal'], template_windows_s=[[0.03, 0.09], [0.1, 0.19]], bump_centers_s=[0.07, 0.13],
                       bump_width_s=0.03, trace_K_target=40.0)
    p['population'] = dict(p['population'], n_random_starts=6)
    p['theta_rule'] = dict(p['theta_rule'], n_min=20, search_random_starts=2, xtol=1e-3)
    p['grid'] = dict(p['grid'], control_absolute=[30, 60, 120, 240], n_min=20)
    p['calibration'] = dict(p['calibration'], m_lo=64, candidates_hi=[256, 128])
    p['budget_requirements'] = {'N_max': 240, 'm_max': 128, 'n_eval': 100}
    p['resamples'] = 2
    p['n_eval'] = 100
    p['learners'] = dict(p['learners'], max_iter=200)
    p['dependencies'] = None
    return p


# ----------------------------------------------------------------------------------------------- raw BIDS fixture
def make_bids_raw(root, subject: str = 'sub-01', n_mag: int = 8, duration_s: float = 60.0, sfreq: float = 1100.0,
                  isi_s: float = 3.2, seed: int = 0, bad: str | None = 'MEG0004', evoked_scale: float = 4e-13,
                  noise_scale: float = 1e-13, response_delay_s: float = 0.8, noisy_channel: str | None = None,
                  events_shift_s: float = 0.0) -> dict:
    """Six-run BIDS-like ds000117 layout with magnetometers, a gradiometer, STI101, events.tsv, channels.tsv.
    Noise: spatial mixing x AR(1) (approximately Kronecker after band-pass); evoked response at a known
    latency after trigger + 34.5 ms."""
    import mne
    from .data_manifest import run_stem
    root = Path(root)
    gen = common.rng('megz-raw-fixture', subject, seed)
    names = [f'MEG{i:04d}' for i in range(1, n_mag + 1)] + ['MEG9992', 'STI101']
    types = ['mag'] * n_mag + ['grad', 'stim']
    n = int(duration_s * sfreq)
    mix = np.linalg.cholesky(0.5 * np.eye(n_mag) + 0.5 * np.exp(-np.subtract.outer(np.arange(n_mag), np.arange(n_mag)) ** 2 / 4.0))
    spatial = np.linspace(1.0, 0.2, n_mag)
    codes = [5, 13, 17, 6, 14, 18]
    out = {'onsets': {}, 'resp': {}}
    for run in range(1, 7):
        info = mne.create_info(names, sfreq, types)
        e = gen.normal(size=(n_mag, n))
        from scipy.signal import lfilter
        e = lfilter([1.0], [1.0, -0.95], e, axis=1)
        data = np.zeros((len(names), n))
        data[:n_mag] = noise_scale * (mix @ e) / 3.0
        if noisy_channel is not None:
            ic = names.index(noisy_channel)
            data[ic] = data[ic] * 300.0 + 300.0 * noise_scale * gen.normal(size=n)
        data[n_mag] = noise_scale * gen.normal(size=n)
        onsets = np.arange(5.0, duration_s - 4.0, isi_s) + gen.uniform(0, 0.2, size=len(np.arange(5.0, duration_s - 4.0, isi_s)))
        values = np.array([codes[i % len(codes)] for i in range(len(onsets))])
        resp = onsets + response_delay_s
        t = np.arange(n) / sfreq
        for on, v in zip(onsets, values):
            lat = on + 0.0345 + 0.1
            amp = 1.0 if v in (5, 6, 13, 14) else 0.5
            data[:n_mag] += evoked_scale * amp * np.outer(spatial, np.exp(-0.5 * ((t - lat) / 0.02) ** 2))
            s = int(round(on * sfreq))
            data[-1, s:s + 5] = v
        for r in resp:
            s = int(round(r * sfreq))
            data[-1, s:s + 5] += 256
        raw = mne.io.RawArray(data, info, verbose='error')
        stem = root / run_stem(subject, run)
        stem.parent.mkdir(parents=True, exist_ok=True)
        raw.save(str(stem) + '_meg.fif', overwrite=True, verbose='error')
        rows = sorted([(on, int(v), 'stim') for on, v in zip(onsets, values)] + [(r, 256, 'btnL') for r in resp])
        with open(str(stem) + '_events.tsv', 'w') as f:
            f.write('onset\tduration\ttrial_type\tvalue\tsample\tstim_file\n')
            for on, v, tt in rows:
                on = on + events_shift_s                    # tests: an events.tsv that already contains a delay
                f.write(f'{on:.6f}\t0.0\t{tt}\t{v}\t{int(round(on * sfreq))}\tn/a\n')
        with open(str(stem) + '_channels.tsv', 'w') as f:
            f.write('name\ttype\tunits\tstatus\n')
            for nm, ty in zip(names, types):
                st = 'bad' if nm == bad else 'good'
                f.write(f'{nm}\t{"MEGMAG" if ty == "mag" else ty.upper()}\tT\t{st}\n')
        out['onsets'][run] = onsets
        out['resp'][run] = resp
    return out
