"""
NFPA Demo: Noise-Factored Projection Autoencoder
=================================================
Tests the OF → EMPCA → noise-aware linear AE hierarchy.
Pure numpy. Run: python3 nfpa_demo.py

Claims tested:
  1. NFPA (chi2 ALS) converges to EMPCA subspace   [Bridge Theorem]
  2. IsoAE (MSE ALS) learns a DIFFERENT subspace    [Metric Reversal]
  3. k_c=k_t=1 NFPA recovers Optimal Filtering      [Theorem 4.1]
  4. chi2 and MSE rank methods in opposite order     [Geometric misspecification]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
np.random.seed(42)

# ── 1. PARAMETERS ─────────────────────────────────────────────────────────────
C  = 2       # channels
T  = 256     # time samples
fs = 1000.0
dt = 1.0/fs
k_t, k_c = 3, 2        # NFPA modes; k_tot = 6
k_tot = k_t * k_c
N_tr, N_te = 800, 300
A_sigma = 0.30          # amplitude spread
print(f"C={C}, T={T}, k_c={k_c}, k_t={k_t}, k_tot={k_tot}")

# ── 2. SIGNAL TEMPLATE (asymmetric pulse) ─────────────────────────────────────
t   = np.arange(T) * dt
t0  = T * dt * 0.38
s_b = np.where(t < t0,
        np.exp(-(t-t0)**2/(2*(3*dt)**2)),
        np.exp(-(t-t0)**2/(2*(14*dt)**2)))
s_b /= np.abs(s_b).max()
v_true  = np.array([1.0, 0.65])
S_tmpl  = np.outer(v_true, s_b)     # (C, T)

# ── 3. NOISE MODEL: Brownian PSD ─────────────────────────────────────────────
freqs  = np.fft.rfftfreq(T, d=dt)
f_safe = np.where(freqs == 0, 1.0, np.abs(freqs))
S_n    = 1.0 / f_safe**2
S_n[0] = 0.0

def gen_noise(N):
    nf  = len(S_n)
    amp = np.sqrt(np.maximum(S_n, 0) / (2.0 * T))
    Xf  = (np.random.randn(N, nf) + 1j*np.random.randn(N, nf)) * amp
    Xf[:, 0] = 0
    return np.fft.irfft(Xf, n=T)

# ── 4. WHITENING LAYER ────────────────────────────────────────────────────────
# Calibrate so whitened noise has RMS ≈ 1 (empirical, T samples)
_wf_raw  = np.where(S_n > 0, 1.0 / np.sqrt(S_n + 1e-20), 0.0)
# Empirical calibration: generate reference noise, whiten, measure RMS
_ref     = gen_noise(2000)                    # (2000, T) reference noise
_ref_fft = np.fft.rfft(_ref, axis=-1) * _wf_raw
_ref_w   = np.fft.irfft(_ref_fft, n=T, axis=-1)
_scale   = np.std(_ref_w)
white_f  = _wf_raw / (_scale + 1e-20)        # calibrated whitening filter
_cf_raw  = np.where(white_f > 1e-10, 1.0/white_f, 0.0)
color_f  = _cf_raw                            # inverse whitening filter

def whiten(x):
    """x (...,T) → z (...,T), Sigma^{-1/2} in freq domain."""
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * white_f, n=T, axis=-1)

def unwhiten(z):
    """z (...,T) → x (...,T)."""
    return np.fft.irfft(np.fft.rfft(z, axis=-1) * color_f, n=T, axis=-1)

noise_check = np.std(whiten(gen_noise(500)))
print(f"Whitened noise RMS = {noise_check:.3f}  (target 1.0)")

# ── 5. DATASET ────────────────────────────────────────────────────────────────
# SNR in whitened space: SNR = ||s_white||^2 / sigma^2 per sample
s_white  = whiten(S_tmpl)               # (C, T) whitened template
snr_est  = np.linalg.norm(s_white) / (noise_check * np.sqrt(C * T))
print(f"Approx waveform SNR (whitened) = {snr_est:.2f}")

def make_dataset(N):
    A = np.random.normal(1.0, A_sigma, N)
    X = np.zeros((N, C, T))
    for c in range(C):
        X[:, c] = A[:, None] * S_tmpl[c] + gen_noise(N)
    return X, A

X_tr, A_tr = make_dataset(N_tr)
X_te, A_te = make_dataset(N_te)
Z_tr = whiten(X_tr)
Z_te = whiten(X_te)
Z_tr_f = Z_tr.reshape(N_tr, -1)
Z_te_f = Z_te.reshape(N_te, -1)
X_tr_f = X_tr.reshape(N_tr, -1)
X_te_f = X_te.reshape(N_te, -1)

# ── 6. BASELINES ──────────────────────────────────────────────────────────────
# EMPCA = PCA on whitened data (oracle: best possible linear reconstruction)
ev_w, Uw  = np.linalg.eigh(Z_tr_f.T @ Z_tr_f / N_tr)
U_empca   = Uw[:, np.argsort(ev_w)[::-1][:k_tot]]    # (CT, k_tot)
Z_te_emp  = Z_te_f @ U_empca @ U_empca.T              # recon in whitened space
X_te_emp  = unwhiten(Z_te_emp.reshape(N_te, C, T))

# IsoPCA = PCA on raw data (wrong geometry)
ev_r, Ur  = np.linalg.eigh(X_tr_f.T @ X_tr_f / N_tr)
U_isopca  = Ur[:, np.argsort(ev_r)[::-1][:k_tot]]    # (CT, k_tot)
X_te_iso  = X_te_f @ U_isopca @ U_isopca.T
Z_te_iso  = whiten(X_te_iso.reshape(N_te, C, T)).reshape(N_te, -1)

# Optimal Filter (joint over channels)
sw_flat   = s_white.flatten()
norm_sq   = sw_flat @ sw_flat
A_of_te   = Z_te_f @ sw_flat / norm_sq
of_corr   = np.corrcoef(A_te, A_of_te)[0, 1]
print(f"OF amplitude corr = {of_corr:.4f}")

# ── 7. ALS TRAINING ───────────────────────────────────────────────────────────
def als_Ut(Z_or_X, U_c, k_t):
    D   = np.einsum('nct,ck->nkt', Z_or_X, U_c)
    _, _, Vt = np.linalg.svd(D.reshape(len(Z_or_X)*U_c.shape[1], T),
                              full_matrices=False)
    return Vt[:k_t].T

def als_Uc(Z_or_X, U_t, k_c):
    E   = np.einsum('nct,tl->ncl', Z_or_X, U_t)
    _, _, Vt = np.linalg.svd(E.transpose(0,2,1).reshape(len(Z_or_X)*U_t.shape[1], C),
                              full_matrices=False)
    return Vt[:k_c].T

def recon_kron(D, U_c, U_t):
    c = np.einsum('nct,ck,tl->nkl', D, U_c, U_t)
    return np.einsum('nkl,ck,tl->nct', c, U_c, U_t)

def chi2_loss(Z, U_c, U_t):
    return float(np.mean((Z - recon_kron(Z, U_c, U_t))**2))

def mse_loss(X, U_c, U_t):
    return float(np.mean((X - recon_kron(X, U_c, U_t))**2))

def nfpa_als(Z, k_c, k_t, n_iter=150, seed=0):
    """ALS on whitened data = chi2 loss. Minimiser = EMPCA (Bridge Thm)."""
    rng = np.random.default_rng(seed)
    U_c = np.linalg.qr(rng.standard_normal((C, k_c)))[0]
    U_t = np.linalg.qr(rng.standard_normal((T, k_t)))[0]
    hist = []
    for _ in range(n_iter):
        U_t = als_Ut(Z, U_c, k_t)
        U_c = als_Uc(Z, U_t, k_c)
        hist.append(chi2_loss(Z, U_c, U_t))
    return U_c, U_t, hist

def iso_ae_als(X, Z, k_c, k_t, n_iter=150, seed=7):
    """ALS on raw data = MSE loss. Also tracks chi2 for comparison."""
    rng = np.random.default_rng(seed)
    U_c = np.linalg.qr(rng.standard_normal((C, k_c)))[0]
    U_t = np.linalg.qr(rng.standard_normal((T, k_t)))[0]
    mse_h, chi2_h = [], []
    for _ in range(n_iter):
        U_t = als_Ut(X, U_c, k_t)      # optimise MSE
        U_c = als_Uc(X, U_t, k_c)
        mse_h.append(mse_loss(X, U_c, U_t))
        chi2_h.append(chi2_loss(Z, U_c, U_t))
    return U_c, U_t, mse_h, chi2_h

print("Training NFPA  (chi² loss)...")
U_c_n, U_t_n, nfpa_h = nfpa_als(Z_tr, k_c, k_t)
print(f"  chi²: {nfpa_h[0]:.5f} → {nfpa_h[-1]:.5f}")

print("Training IsoAE (MSE loss)...")
U_c_i, U_t_i, iso_mh, iso_ch = iso_ae_als(X_tr, Z_tr, k_c, k_t)
print(f"  MSE:  {iso_mh[0]:.5f} → {iso_mh[-1]:.5f}")
print(f"  chi²: {iso_ch[0]:.5f} → {iso_ch[-1]:.5f}")

# ── 8. RECONSTRUCTION & METRICS ───────────────────────────────────────────────
Z_te_nfpa  = recon_kron(Z_te, U_c_n, U_t_n)
X_te_nfpa  = unwhiten(Z_te_nfpa)
X_te_isoae = recon_kron(X_te, U_c_i, U_t_i)
Z_te_isoae = whiten(X_te_isoae)

def chi2_m(Z_orig, Z_hat):
    return float(np.mean((Z_orig.reshape(len(Z_orig),-1) -
                           Z_hat.reshape(len(Z_hat),-1))**2))
def mse_m(X_orig, X_hat):
    return float(np.mean((X_orig.reshape(len(X_orig),-1) -
                           X_hat.reshape(len(X_hat),-1))**2))

R = {
    'EMPCA' : {'chi2': chi2_m(Z_te, Z_te_emp.reshape(N_te,C,T)),
               'mse':  mse_m(X_te, X_te_emp)},
    'NFPA'  : {'chi2': chi2_m(Z_te, Z_te_nfpa),
               'mse':  mse_m(X_te, X_te_nfpa)},
    'IsoAE' : {'chi2': chi2_m(Z_te, Z_te_isoae),
               'mse':  mse_m(X_te, X_te_isoae)},
    'IsoPCA': {'chi2': chi2_m(Z_te, Z_te_iso.reshape(N_te,C,T)),
               'mse':  mse_m(X_te, X_te_iso.reshape(N_te,C,T))},
}

print("\n── Reconstruction metrics ───────────────────────────────────")
print(f"{'Method':<10} {'chi²':>10} {'raw MSE':>12}")
for m, d in R.items():
    print(f"  {m:<8} {d['chi2']:>10.5f} {d['mse']:>12.5f}")
print(f"\n  Metric reversal:")
print(f"  NFPA chi² < IsoAE chi²? {R['NFPA']['chi2'] < R['IsoAE']['chi2']}  "
      f"({R['NFPA']['chi2']:.5f} vs {R['IsoAE']['chi2']:.5f})")
print(f"  NFPA MSE  > IsoAE MSE?  {R['NFPA']['mse'] > R['IsoAE']['mse']}   "
      f"({R['NFPA']['mse']:.5f} vs {R['IsoAE']['mse']:.5f})")

# ── 9. SUBSPACE ANGLES ────────────────────────────────────────────────────────
def kron_basis(U_c, U_t):
    return np.column_stack([np.kron(U_c[:,j], U_t[:,i])
                            for j in range(U_c.shape[1])
                            for i in range(U_t.shape[1])])  # (CT, k_tot)

def ang_deg(A, B):
    Qa = np.linalg.qr(A)[0]; Qb = np.linalg.qr(B)[0]
    s  = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(np.abs(s),0,1)) * 180/np.pi

B_nfpa  = kron_basis(U_c_n, U_t_n)     # whitened space
B_empca = U_empca

# IsoAE basis: convert to whitened coords
B_iso_r  = kron_basis(U_c_i, U_t_i)   # (CT, k_tot) raw coords
B_iso_w  = whiten(B_iso_r.reshape(k_tot, C, T)).reshape(k_tot, -1).T  # (CT, k_tot)
Q_iso, _ = np.linalg.qr(B_iso_w)

ang_NE = ang_deg(B_nfpa, B_empca)
ang_IE = ang_deg(Q_iso,  B_empca)
ang_NI = ang_deg(B_nfpa, Q_iso)

print("\n── Subspace principal angles (°) ────────────────────────────")
print(f"  NFPA  vs EMPCA: {np.round(ang_NE,1)}  ← Bridge Thm: SMALL")
print(f"  IsoAE vs EMPCA: {np.round(ang_IE,1)}  ← Metric reversal: LARGE")
print(f"  NFPA  vs IsoAE: {np.round(ang_NI,1)}")

# ── 10. AMPLITUDE RECOVERY ────────────────────────────────────────────────────
codes_te = np.einsum('nct,ck,tl->nkl', Z_te, U_c_n, U_t_n).reshape(N_te, -1)
corrs    = [np.corrcoef(A_te, codes_te[:,i])[0,1] for i in range(k_tot)]
best     = np.argmax(np.abs(corrs))
A_hat    = codes_te[:, best] * np.sign(corrs[best])
nfpa_rc  = np.corrcoef(A_te, A_hat)[0,1]
print(f"\n── Amplitude recovery ───────────────────────────────────────")
print(f"  OF   r = {of_corr:.4f}")
print(f"  NFPA r = {nfpa_rc:.4f}  (latent dim {best})")
print(f"  All dim corrs: {[f'{r:.3f}' for r in corrs]}")

# ── 11. THEOREM 4.1: k=1 NFPA → OF ──────────────────────────────────────────
U_c1, U_t1, h1 = nfpa_als(Z_tr, k_c=1, k_t=1, n_iter=80)
kv = np.kron(U_c1[:,0], U_t1[:,0])         # (CT,) learned basis
sw = sw_flat / np.linalg.norm(sw_flat)
kv_n = kv  / np.linalg.norm(kv)
align = np.abs(sw @ kv_n)
A_k1  = Z_te_f @ kv / (kv @ kv)
r_k1  = np.abs(np.corrcoef(A_te, A_k1)[0,1])
print(f"\n── k=1 NFPA vs OF ───────────────────────────────────────────")
print(f"  |cos θ| template alignment = {align:.4f}  (1.0 = identical)")
print(f"  k=1 NFPA ampl corr = {r_k1:.4f}")
print(f"  OF        ampl corr = {of_corr:.4f}")

# ── 12. FIGURE ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
fig.subplots_adjust(hspace=0.46, wspace=0.38)
cc = ['steelblue','darkorange']
n_ex = 5

# A: raw event + NFPA recon
ax = axes[0,0]
for c in range(C):
    ax.plot(t*1e3, X_te[n_ex,c], lw=0.9, alpha=0.45, color=cc[c])
    ax.plot(t*1e3, X_te_nfpa[n_ex,c], lw=1.5, color=cc[c],
            ls='--', label=f'ch{c+1}')
ax.set(xlabel='Time [ms]', ylabel='Ampl.', title='A  Event + NFPA recon')
ax.legend(fontsize=7)

# B: whitened event
ax = axes[0,1]
for c in range(C):
    ax.plot(t*1e3, Z_te[n_ex,c], lw=0.9, alpha=0.45, color=cc[c])
    ax.plot(t*1e3, Z_te_nfpa[n_ex,c], lw=1.5, color=cc[c], ls='--')
ax.set(xlabel='Time [ms]', ylabel='Whitened ampl.',
       title='B  Whitened space (chi² domain)')

# C: PSD + whitening
ax = axes[0,2]
f_pl = freqs[1:]
ax.loglog(f_pl, S_n[1:]/S_n[1:].max(), color='royalblue', lw=1.5,
          label='$S_n(f)$ Brownian')
ax.loglog(f_pl, white_f[1:]**2/white_f[1:].max()**2, color='tomato',
          lw=1.5, ls='--', label='$|H|^2$ whitening')
ax.set(xlabel='Freq [Hz]', ylabel='Norm. PSD', title='C  Noise + whitening filter')
ax.legend(fontsize=7)

# D: convergence
ax = axes[0,3]
it = np.arange(len(nfpa_h))
ax.semilogy(it, nfpa_h,  color='steelblue', lw=1.5, label='NFPA chi²')
ax.semilogy(it, iso_ch,  color='tomato',    lw=1.5, ls='--', label='IsoAE chi²')
ax.axhline(R['EMPCA']['chi2'], color='k', ls=':', lw=1, label='EMPCA (oracle)')
ax.set(xlabel='ALS iteration', ylabel='chi²', title='D  Convergence')
ax.legend(fontsize=7)

# E: subspace angles
ax = axes[1,0]
d = np.arange(1, k_tot+1); w = 0.26
ax.bar(d-w, ang_NE, width=w, color='steelblue', alpha=0.85, label='NFPA vs EMPCA')
ax.bar(d,   ang_IE, width=w, color='tomato',    alpha=0.85, label='IsoAE vs EMPCA')
ax.bar(d+w, ang_NI, width=w, color='seagreen',  alpha=0.85, label='NFPA vs IsoAE')
ax.axhline(90, color='gray', lw=0.8, ls=':')
ax.set(xlabel='Dim', ylabel='Angle [°]', title='E  Principal angles — Bridge Thm',
       ylim=(0, 96)); ax.set_xticks(d); ax.legend(fontsize=6.5)
ax.text(0.97,0.95, f'mean NFPA/EMP: {ang_NE.mean():.0f}°\nmean Iso/EMP:  {ang_IE.mean():.0f}°',
        ha='right', va='top', transform=ax.transAxes, fontsize=7,
        bbox=dict(boxstyle='round',fc='white',alpha=0.9))

# F: chi2 bar
ax = axes[1,1]
labels_b = ['EMPCA','NFPA','IsoAE','IsoPCA']
cv = [R[k]['chi2'] for k in labels_b]
brs = ax.bar(labels_b, cv, color=['k','steelblue','tomato','peru'], alpha=0.82, width=0.55)
ax.set(ylabel='chi² residual', title='F  chi² metric ↓ better')
for b,v in zip(brs,cv): ax.text(b.get_x()+b.get_width()/2, v+max(cv)*0.01,
                                 f'{v:.4f}', ha='center', fontsize=7)

# G: MSE bar
ax = axes[1,2]
mv = [R[k]['mse'] for k in labels_b]
brs2 = ax.bar(labels_b, mv, color=['k','steelblue','tomato','peru'], alpha=0.82, width=0.55)
ax.set(ylabel='Raw MSE', title='G  Raw MSE ↓ better')
for b,v in zip(brs2,mv): ax.text(b.get_x()+b.get_width()/2, v+max(mv)*0.01,
                                   f'{v:.4f}', ha='center', fontsize=7)

# H: metric reversal summary text
ax = axes[1,3]; ax.axis('off')
mr_ok  = R['NFPA']['chi2'] < R['IsoAE']['chi2']
mse_ok = R['NFPA']['mse']  > R['IsoAE']['mse']
txt = (
    "METRIC REVERSAL\n"
    "──────────────────────────\n"
    f"NFPA chi² < IsoAE chi²?\n"
    f"  {'✓ YES' if mr_ok  else '✗ NO'}\n"
    f"  {R['NFPA']['chi2']:.5f} < {R['IsoAE']['chi2']:.5f}\n\n"
    f"NFPA MSE  > IsoAE MSE?\n"
    f"  {'✓ YES' if mse_ok else '✗ NO'}\n"
    f"  {R['NFPA']['mse']:.5f} > {R['IsoAE']['mse']:.5f}\n\n"
    f"Geometric misspecification:\n"
    f"MSE optimiser moves to a\n"
    f"DIFFERENT Grassmannian\n"
    f"direction than chi² opt."
)
ax.text(0.05,0.97,txt,va='top',ha='left',transform=ax.transAxes,fontsize=8,
        family='monospace', bbox=dict(boxstyle='round',fc='#fff8e1',ec='#f9a825',lw=1.2))

# I: amplitude scatter
ax = axes[2,0]
ax.scatter(A_te, A_of_te, s=9, alpha=0.45, color='gray',  label=f'OF r={of_corr:.3f}')
ax.scatter(A_te, A_hat,   s=9, alpha=0.55, color='steelblue', label=f'NFPA r={nfpa_rc:.3f}')
lim=[A_te.min()-0.1,A_te.max()+0.1]; ax.plot(lim,lim,'k--',lw=0.8)
ax.set(xlabel='True amplitude', ylabel='Estimated', title='I  Amplitude recovery')
ax.legend(fontsize=7)

# J: temporal bases
ax = axes[2,1]
for i in range(k_t):
    ax.plot(t*1e3, U_t_n[:,i], lw=1.3, label=f'$u_{{t,{i+1}}}$')
sw_ref = s_white[0]; sw_ref /= np.abs(sw_ref).max()
ax.plot(t*1e3, sw_ref*0.75, 'k--', lw=1, alpha=0.5, label='s̃ (ref, ch0)')
ax.set(xlabel='Time [ms]', title='J  NFPA temporal bases $U_t$')
ax.legend(fontsize=6.5)

# K: Theorem 4.1 panel
ax = axes[2,2]
sw0  = s_white[0]; sw0 /= np.abs(sw0).max()
kv0  = U_c1[0,0] * U_t1[:,0]; kv0 /= np.abs(kv0).max()
ax.plot(t*1e3, sw0, 'k--', lw=1.4, label='OF template s̃')
ax.plot(t*1e3, kv0, 'steelblue', lw=1.4, label=f'k=1 NFPA (|cosθ|={align:.4f})')
ax.set(xlabel='Time [ms]', title='K  Thm 4.1: k=1 NFPA → OF')
ax.legend(fontsize=7)

# L: channel basis
ax = axes[2,3]
ax.bar(['ch1','ch2'], U_c_n[:,0], color='steelblue', alpha=0.7, label='$u_{c,1}$')
ax.bar(['ch1','ch2'], U_c_n[:,1], color='darkorange', alpha=0.7,
       bottom=U_c_n[:,0]*0, label='$u_{c,2}$')
ax.axhline(0, color='k', lw=0.5)
ax2 = ax.twinx()
ax2.plot(['ch1','ch2'], v_true/np.linalg.norm(v_true), 'k^--', ms=7,
         label='True v_true (norm.)')
ax2.set_ylabel('True channel weights', fontsize=7)
ax2.legend(fontsize=6.5, loc='lower right')
ax.set(title='L  NFPA channel basis $U_c$'); ax.legend(fontsize=7, loc='upper right')

fig.suptitle(
    'NFPA — Noise-Factored Projection AE: OF → EMPCA → noise-aware linear AE consistency tests',
    fontsize=10, fontweight='bold'
)

out_path = Path(__file__).with_name('nfpa_demo_results.png')
fig.savefig(out_path, bbox_inches='tight', dpi=120)
plt.close(fig)
print(f"\nFigure saved: {out_path}")

# ── 13. NUMERICAL SUMMARY ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("NUMERICAL SUMMARY")
print("="*55)
print(f"Bridge Theorem (NFPA→EMPCA):")
print(f"  mean principal angle: {ang_NE.mean():.1f}°  (IsoAE: {ang_IE.mean():.1f}°)")
print(f"Metric reversal:")
print(f"  chi²: NFPA={R['NFPA']['chi2']:.5f} < IsoAE={R['IsoAE']['chi2']:.5f}  {'✓' if R['NFPA']['chi2'] < R['IsoAE']['chi2'] else '✗'}")
print(f"  MSE:  NFPA={R['NFPA']['mse']:.5f} > IsoAE={R['IsoAE']['mse']:.5f}  {'✓' if R['NFPA']['mse'] > R['IsoAE']['mse'] else '✗'}")
print(f"Theorem 4.1 (k=1 → OF):  |cosθ| = {align:.4f}  r = {r_k1:.4f}")
print(f"Amplitude recovery:  OF r={of_corr:.4f}  NFPA r={nfpa_rc:.4f}")
print("="*55)
