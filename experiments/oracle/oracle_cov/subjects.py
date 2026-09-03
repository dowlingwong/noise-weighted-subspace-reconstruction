"""Frozen subject models for ORACLE-Cov.

Every subject is trained once on the clean training support under an assumed
covariance ``Sigma_hat = kron(St_hat, Sc_hat)`` and then frozen.  It exposes

* ``stages()``            — named hook points;
* ``represent(X, stage)`` — the representation at a stage for raw events ``X``;
* ``outputs(X)``          — the amplitude estimate (linear readout on the code,
                             fitted on clean training latents);
* ``recon_whitened(X)``   — reconstruction in the *assumed*-whitened space;
* ``jac_recon(stage)``    — Jacobian of the whitened reconstruction w.r.t. the
                             stage representation, ``(C*T, d_stage)``, at a
                             stage vector (analytic for both subject classes);
* ``jac_output(stage)``   — gradient of the amplitude output w.r.t. the stage.

``lin_cw`` is CW-PCA / the tied linear autoencoder at rank ``k`` (analytic
everything); ``mlp_ae`` is a two-hidden-layer tanh autoencoder trained in
numpy with Adam on the assumed-whitened input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.synthetic.nfpa_confirmatory.covariance import _sqrt_inv, whiten_separable

from .generator import TrainingSet


def _flat(Z: np.ndarray) -> np.ndarray:
    return Z.reshape(Z.shape[0], -1, order="F")


@dataclass
class Subject:
    name: str
    ts: TrainingSet
    Sc_hat: np.ndarray
    St_hat: np.ndarray
    readout_w: np.ndarray = field(default=None)   # (d_code + 1,)
    frozen: bool = False

    # ---- interface -------------------------------------------------------
    def stages(self) -> list[str]:
        raise NotImplementedError

    def whiten_hat(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_Wc_hat"):
            self._Wc_hat, self._Wt_hat = _sqrt_inv(self.Sc_hat), _sqrt_inv(self.St_hat)   # cached; identical to whiten_separable
        return _flat(np.einsum("ab,...bd,de->...ae", self._Wc_hat, X - self.ts.ds.train_mean, self._Wt_hat))

    def represent(self, X: np.ndarray, stage: str) -> np.ndarray:
        raise NotImplementedError

    def code(self, X: np.ndarray) -> np.ndarray:
        return self.represent(X, "code")

    def recon_whitened(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac_recon(self, stage: str, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac_output(self, stage: str, z: np.ndarray) -> np.ndarray:
        """Gradient of the amplitude readout w.r.t. the stage vector (chain through the code)."""
        Jc = self.jac_code(stage, z)             # (d_code, d_stage)
        return self.readout_w[:-1] @ Jc          # (d_stage,)

    def jac_code(self, stage: str, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def outputs(self, X: np.ndarray) -> np.ndarray:
        c = self.code(X)
        return c @ self.readout_w[:-1] + self.readout_w[-1]

    def fit_readout(self) -> None:
        c = self.code(self.ts.ds.X_train)
        A = np.column_stack([c, np.ones(len(c))])
        y = self.ts.ds.latents_train["amplitude"]
        self.readout_w = np.linalg.lstsq(A, y, rcond=None)[0]


class LinearSubject(Subject):
    """CW-PCA / tied linear autoencoder under the assumed covariance."""

    def __init__(self, ts: TrainingSet, Sc_hat, St_hat, rank: int):
        super().__init__("lin_cw", ts, Sc_hat, St_hat)
        Z = self.whiten_hat(ts.ds.X_train)
        _, _, vh = np.linalg.svd(Z, full_matrices=False)
        self.P = vh[:rank].T                    # (CT, k) orthonormal in assumed-whitened space
        self.fit_readout(); self.frozen = True

    def stages(self):
        return ["code", "recon"]

    def represent(self, X, stage):
        Z = self.whiten_hat(X)
        c = Z @ self.P
        if stage == "code":
            return c
        if stage == "recon":
            return c @ self.P.T
        raise KeyError(stage)

    def recon_whitened(self, X):
        return self.represent(X, "recon")

    def jac_recon(self, stage, z):
        if stage == "code":
            return self.P                         # (CT, k)
        return np.eye(self.P.shape[0])

    def jac_code(self, stage, z):
        if stage == "code":
            return np.eye(self.P.shape[1])
        return self.P.T                           # code = recon @ P


class MLPSubject(Subject):
    """tanh autoencoder  Z -> h1 -> h2 -> code -> g1 -> g2 -> Z_hat, numpy + Adam."""

    def __init__(self, ts: TrainingSet, Sc_hat, St_hat, rank: int, hidden=(64, 32), epochs=60, lr=2e-3, seed=0):
        super().__init__("mlp_ae", ts, Sc_hat, St_hat)
        rng = np.random.default_rng(seed)
        d = ts.cfg.C * ts.cfg.T
        sizes = [d, hidden[0], hidden[1], rank, hidden[1], hidden[0], d]
        self.W = [rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / (sizes[i] + sizes[i + 1])) for i in range(6)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(6)]
        self._train(self.whiten_hat(ts.ds.X_train), self.whiten_hat(ts.ds.X_validation), epochs, lr, rng)
        self.fit_readout(); self.frozen = True

    # layer l output: a_l = tanh(a_{l-1} W_l + b_l) for l in {1,2,4,5}; linear for l in {3 (code), 6 (recon)}
    _linear = {2, 5}

    def _forward(self, Z):
        acts = [Z]
        for l in range(6):
            pre = acts[-1] @ self.W[l] + self.b[l]
            acts.append(pre if l in self._linear else np.tanh(pre))
        return acts

    def _train(self, Ztr, Zval, epochs, lr, rng):
        m = [np.zeros_like(w) for w in self.W]; v = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        t = 0; n = len(Ztr); bs = 64; hist = []
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                acts = self._forward(Ztr[idx])
                delta = (acts[-1] - Ztr[idx]) * (2.0 / (len(idx) * Ztr.shape[1]))
                gW = [None] * 6; gb = [None] * 6
                for l in range(5, -1, -1):
                    gW[l] = acts[l].T @ delta; gb[l] = delta.sum(axis=0)
                    if l > 0:
                        delta = delta @ self.W[l].T
                        if (l - 1) not in self._linear:
                            delta = delta * (1.0 - acts[l] ** 2)
                t += 1
                for l in range(6):
                    for p, g, mm, vv in ((self.W[l], gW[l], m[l], v[l]), (self.b[l], gb[l], mb[l], vb[l])):
                        mm *= 0.9; mm += 0.1 * g; vv *= 0.999; vv += 0.001 * g * g
                        p -= lr * (mm / (1 - 0.9 ** t)) / (np.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
            hist.append(float(np.mean((self._forward(Zval)[-1] - Zval) ** 2)))
        self.val_history = hist

    STAGES = {"h1": 1, "h2": 2, "code": 3, "g1": 4, "g2": 5, "recon": 6}

    def stages(self):
        return list(self.STAGES)

    def represent(self, X, stage):
        return self._forward(self.whiten_hat(X))[self.STAGES[stage]]

    def recon_whitened(self, X):
        return self.represent(X, "recon")

    def _forward_from(self, stage: str, z: np.ndarray) -> list[np.ndarray]:
        l0 = self.STAGES[stage]
        acts = [z]
        for l in range(l0, 6):
            pre = acts[-1] @ self.W[l] + self.b[l]
            acts.append(pre if l in self._linear else np.tanh(pre))
        return acts

    def _jac_from(self, stage: str, z: np.ndarray, upto: int) -> np.ndarray:
        """Jacobian of layer ``upto`` output w.r.t. the stage vector ``z`` (1-D)."""
        l0 = self.STAGES[stage]
        acts = self._forward_from(stage, z[None, :])
        J = np.eye(z.size)
        for i, l in enumerate(range(l0, upto)):
            J = J @ self.W[l]
            if l not in self._linear:
                J = J * (1.0 - acts[i + 1][0] ** 2)[None, :]
        return J.T                                  # (d_out, d_stage)

    def jac_recon(self, stage, z):
        return self._jac_from(stage, z, 6)

    def jac_code(self, stage, z):
        if stage == "code":
            return np.eye(z.size)
        if self.STAGES[stage] > 3:
            raise ValueError("code is upstream of this stage")
        return self._jac_from(stage, z, 3)


def make_subject(kind: str, ts: TrainingSet, Sc_hat=None, St_hat=None, *, rank=6, hidden=(64, 32), epochs=60, lr=2e-3, seed=0) -> Subject:
    Sc_hat = ts.Sigma_c if Sc_hat is None else Sc_hat
    St_hat = ts.Sigma_t if St_hat is None else St_hat
    if kind == "lin_cw":
        return LinearSubject(ts, Sc_hat, St_hat, rank)
    if kind == "mlp_ae":
        return MLPSubject(ts, Sc_hat, St_hat, rank, hidden, epochs, lr, seed)
    raise KeyError(kind)
