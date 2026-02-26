import numpy as np


def weighted_inner(a, b, w):
    return np.sum(np.conj(a) * b * w)


def weighted_cosine(a, b, w):
    num = np.abs(weighted_inner(a, b, w))
    den = np.sqrt(np.real(weighted_inner(a, a, w))) * np.sqrt(np.real(weighted_inner(b, b, w)))
    return float(num / den)


def weighted_residual_energy(x, xhat, w):
    x = np.asarray(x)
    xhat = np.asarray(xhat)
    w = np.asarray(w, dtype=np.float64)
    if x.shape != xhat.shape:
        raise ValueError("x and xhat must have same shape")
    return float(np.real(np.sum(np.abs(x - xhat) ** 2 * w)))
