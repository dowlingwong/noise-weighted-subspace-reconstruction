import numpy as np


def make_inverse_psd_weights(noise_psd, eps=1e-18):
    noise_psd = np.asarray(noise_psd, dtype=np.float64)
    if noise_psd.ndim != 1:
        raise ValueError(f"noise_psd must be 1D; got {noise_psd.shape}")
    return 1.0 / (noise_psd + eps)


def clip_psd_for_weights(psd, psd_floor=None, w_max=None):
    psd_eff = np.asarray(psd, dtype=np.float64).copy()
    if psd_floor is not None:
        psd_eff = np.maximum(psd_eff, float(psd_floor))
    if w_max is not None:
        psd_eff = np.maximum(psd_eff, 1.0 / float(w_max))
    return psd_eff


def build_of_one_sided_weights(psd_one_sided, trace_len):
    psd_one_sided = np.asarray(psd_one_sided, dtype=np.float64)
    n_rfft = trace_len // 2 + 1
    if psd_one_sided.shape[0] != n_rfft:
        raise ValueError(f"PSD length {psd_one_sided.shape[0]} does not match rfft bins {n_rfft}")

    w = np.zeros_like(psd_one_sided)
    w[0] = 0.0
    if trace_len % 2 == 0:
        w[1:-1] = 2.0 / psd_one_sided[1:-1]
        w[-1] = 1.0 / psd_one_sided[-1]
    else:
        w[1:] = 2.0 / psd_one_sided[1:]
    return w
