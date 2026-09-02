"""P3-E0 — analytic unit tests and positive controls (ORACLE IMPLEMENTATION_PLAN rule 3)."""

import numpy as np
import pytest

from experiments.oracle.oracle_cov import families as F
from experiments.oracle.oracle_cov.consequence import AmplitudeDegradation
from experiments.oracle.oracle_cov.monitors import EuclideanDisplacement, JacobianProjected, PullbackWhitened
from experiments.oracle.oracle_cov.protocol import RepresentationBatch
from experiments.oracle.oracle_cov.tangent import fisher, tangent_geometry
from experiments.oracle.oracle_cov.report import far_threshold, bh_fdr, macro_f1
from experiments.oracle.oracle_cov.abstain import conformal_threshold


def _batches(sub, ts, stage, n=3, size=40):
    out = []
    for w in range(n):
        wd = F.make_window(ts, F.family("clean"), size, seed=w)
        Z = sub.represent(wd.X, stage)
        out.append(RepresentationBatch(stage, Z, sub.outputs(wd.X), sub.recon_whitened(wd.X), {"chi2_bar": 1.0, "X_whitened": sub.whiten_hat(wd.X), "truth_amplitude": wd.latents["amplitude"]}))
    return out


def test_forward_model_reproduces_training_signal(ts):
    assert np.max(np.abs(ts.forward(ts.ds.latents_train) - ts.ds.clean_train)) < 1e-9


def test_excited_span_rank_equals_varied_latents(ts):
    tg = tangent_geometry(ts, ("amplitude", "rise_time", "decay_time", "secondary"), n_points=16)
    assert tg.T_S.shape[1] == 4             # the planted family spans exactly 4 directions
    assert np.abs(tg.T_S.T @ tg.T_U).max() < 1e-10
    lat = {k: v[:3] for k, v in ts.ds.latents_train.items()}
    _, r = fisher(ts, lat, names=("amplitude", "rise_time", "decay_time", "secondary"))
    assert (r == 4).all()


def test_fisher_rank_drops_when_a_latent_is_degenerate(ts):
    lat = {k: v[:2].copy() for k, v in ts.ds.latents_train.items()}
    lat["amplitude"][:] = 0.0                 # every direction is proportional to A: at A = 0 nothing is identifiable
    _, r = fisher(ts, lat, names=("rise_time", "decay_time", "secondary"))
    assert (r == 0).all()


def test_pullback_equals_mahalanobis_of_projected_displacement(lin, ts):
    stage = "code"
    b = _batches(lin, ts, stage)
    m = PullbackWhitened(lin, stage).fit(b)
    delta = b[0].Z.mean(0) - m.mu_stage
    expected = np.linalg.norm(lin.jac_recon(stage, m.mu_stage) @ delta)      # Sigma_w = I in matched-whitened coordinates
    assert abs(m.raw(b[0]) - expected) < 1e-9


def test_jacobian_projected_null_is_zero_and_aligned_is_norm(mlp, ts):
    stage = "h1"
    b = _batches(mlp, ts, stage)
    jp = JacobianProjected(mlp, stage).fit(b)
    pw = PullbackWhitened(mlp, stage).fit(b)
    rng = np.random.default_rng(0)
    Z0 = np.repeat(jp.mu_stage[None], 5, axis=0)          # sit exactly at the linearisation point
    Zn = F.designed_perturbation(mlp, stage, Z0, "null", 2.0, rng)
    Za = F.designed_perturbation(mlp, stage, Z0, "aligned", 2.0, rng)
    zero = RepresentationBatch(stage, Zn, None, None, {}); al = RepresentationBatch(stage, Za, None, None, {})
    assert jp.raw(zero) < 1e-8 and pw.raw(zero) < 1e-8
    assert abs(jp.raw(al) - 2.0) < 1e-8
    assert abs(pw.raw(al) - 2.0 * jp.singular_values[0]) < 1e-6


def test_white_regime_euclidean_equals_pullback_up_to_scale(lin, ts):
    """At the linear code stage with orthonormal P, |P delta| = |delta| exactly."""
    b = _batches(lin, ts, "code")
    e = EuclideanDisplacement().fit(b); p = PullbackWhitened(lin, "code").fit(b)
    for bb in b:
        assert abs(e.raw(bb) - p.raw(bb)) < 1e-9


def test_alarm_monotone_in_severity(lin, ts, cfg):
    from experiments.oracle.oracle_cov import pipeline as P
    from dataclasses import replace
    cfg = replace(cfg, window_size=100)
    ctx = P.build_seed_context(0, cfg, ts.cfg, subjects=("lin_cw",))
    scores = []
    for sev in (0.5, 1.0, 2.0, 4.0):
        rows = P.score_window(ctx, cfg, F.family("S3_t0", sev), 0)
        scores.append(rows[0]["mean_disp_excess_z_T_U"])      # positive control: unexcited-direction energy grows with the shift
    assert all(b > a for a, b in zip(scores, scores[1:])), scores


def test_consequence_and_threshold_helpers():
    c = AmplitudeDegradation(clean_rmse=0.2, kappa_m=0.1)
    assert abs(c.score(np.array([1.0, 1.0]), np.array([1.3, 0.7])) - 0.5) < 1e-12 and c.high(0.11) and not c.high(0.09)
    s = np.arange(100.0)
    assert far_threshold(s, 0.01) == 99.0
    assert conformal_threshold(np.arange(10.0), 0.1) == 9.0
    assert macro_f1(np.array(["N", "S"]), np.array(["N", "S"])) == 1.0
    assert bh_fdr(np.array([0.001, 0.5, 0.9])).tolist() == [True, False, False]


def test_families_record_parameters_and_shapes(ts):
    for name, sevs in F.SEVERITIES.items():
        w = F.make_window(ts, F.family(name, sevs[0]), 20, seed=3)
        assert w.X.shape == (20, ts.cfg.C, ts.cfg.T) and w.meta, name
