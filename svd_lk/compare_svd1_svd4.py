"""
compare_svd1_svd4.py  —  Porównanie SVD-1 (najwyższa energia) vs SVD-4 (najwyższa korelacja)
==============================================================================================
Pokazuje, że wybór składowej EDR przez korelację Pearsona ma sens:
składowa o największej energii (SVD-1) NIE jest najlepszym estymatorem oddechu.

Wyniki zapisywane do results/:
    cmp_01_sygnaly.png      — SVD-1 vs SVD-4 vs RESP w czasie
    cmp_02_scatter.png      — scatter plot obu składowych względem RESP
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
IN_RESULTS = "results/svd_edr.npz"
IN_DATA    = "../preprocessing/preprocessed/cebsdb_b001.npz"
OUT_DIR    = "results"
FS_INTERP  = 4.0
RESP_LO    = 0.0666
RESP_HI    = 0.5
# ---------------------------------------------------------------------------

r = np.load(IN_RESULTS, allow_pickle=False)
d = np.load(IN_DATA,    allow_pickle=False)

U         = r["U"]              # (N_cykli × n_components)
best_idx  = int(r["best_idx"])  # powinno być 3 (SVD-4)
r_times_b = r["r_times_beats"]  # czasy cykli [s]
resp_norm = r["resp_norm"]      # znorm. referencja na siatce 4 Hz
r_times   = r["r_times"]        # siatka 4 Hz
correlations   = r["correlations"]
variance_ratio = r["variance_ratio"]

# Referencja beat-by-beat (taka sama jak w svd_edr.py)
r_peaks   = d["r_peaks"]
valid_idx = d["valid_idx"]
resp_ref  = d["resp_ref"]
fs        = float(d["fs"])

resp_per_cycle = np.array([
    resp_ref[r_peaks[valid_idx[i]]:r_peaks[valid_idx[i] + 1]].mean()
    if valid_idx[i] + 1 < len(r_peaks) else np.nan
    for i in range(len(valid_idx))
])
n      = min(U.shape[0], len(resp_per_cycle))
t_m    = r_times_b[:n]
resp_m = resp_per_cycle[:n]
mask   = ~np.isnan(resp_m)
t_m, resp_m = t_m[mask], resp_m[mask]
U_m = U[mask]

# Siatka regularna
t_reg = np.arange(t_m[0], t_m[-1], 1.0 / FS_INTERP)

def _interp(t_irr, y_irr, t_out):
    f = interp1d(t_irr, y_irr, kind="cubic", bounds_error=False,
                 fill_value=(float(y_irr[0]), float(y_irr[-1])))
    return f(t_out)

def _bandpass(sig):
    sos = butter(4, [RESP_LO, RESP_HI], btype="band", fs=FS_INTERP, output="sos")
    return sosfiltfilt(sos, sig)

def _norm(sig):
    return (sig - sig.mean()) / (sig.std() + 1e-9)

# Filtrowanie SVD-1 i SVD-4
svd1_filt = _bandpass(_interp(t_m, U_m[:, 0], t_reg))
svd4_filt = _bandpass(_interp(t_m, U_m[:, best_idx], t_reg))
resp_filt = _bandpass(_interp(t_m, resp_m, t_reg))

# Korelacje
r1, _ = pearsonr(svd1_filt, resp_filt)
r4, _ = pearsonr(svd4_filt, resp_filt)

# Korekcja znaku
svd1_norm = _norm(svd1_filt) * np.sign(r1)
svd4_norm = _norm(svd4_filt) * np.sign(r4)
resp_n    = _norm(resp_filt)

print(f"SVD-1:  energia = {variance_ratio[0]:.1f}%   |r| = {abs(r1):.3f}")
print(f"SVD-4:  energia = {variance_ratio[best_idx]:.1f}%   |r| = {abs(r4):.3f}")

# ---------------------------------------------------------------------------
# Wykres 1 — sygnały w czasie
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

axes[0].plot(t_reg, resp_n, color="tab:green", linewidth=1.4, label="Referencja RESP")
axes[0].set_ylabel("Amplituda (znorm.)", fontsize=10)
axes[0].set_title("Referencja oddechowa", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].plot(t_reg, resp_n,   color="tab:green",  linewidth=1.0, alpha=0.5, label="Referencja")
axes[1].plot(t_reg, svd1_norm, color="tab:orange", linewidth=1.2, linestyle="--",
             label=f"SVD-1  (energia {variance_ratio[0]:.1f}%,  |r| = {abs(r1):.3f})")
axes[1].set_ylabel("Amplituda (znorm.)", fontsize=10)
axes[1].set_title("SVD-1 — najwyższa energia", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

axes[2].plot(t_reg, resp_n,   color="tab:green", linewidth=1.0, alpha=0.5, label="Referencja")
axes[2].plot(t_reg, svd4_norm, color="tab:blue",  linewidth=1.2, linestyle="--",
             label=f"SVD-4  (energia {variance_ratio[best_idx]:.1f}%,  |r| = {abs(r4):.3f})")
axes[2].set_ylabel("Amplituda (znorm.)", fontsize=10)
axes[2].set_title("SVD-4 — najwyższa korelacja z RESP  ← wybrany EDR", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Czas [s]", fontsize=10)
axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

fig.suptitle("SVD-1 vs SVD-4 — porównanie z referencją oddechową  (CEBSDB b001)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUT_DIR, "cmp_01_sygnaly.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"\nZapisano: {path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Wykres 2 — scatter: składowa vs RESP
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, sig, label, color, r_val, e_val in [
    (ax1, svd1_norm, "SVD-1", "tab:orange", r1, variance_ratio[0]),
    (ax2, svd4_norm, "SVD-4", "tab:blue",   r4, variance_ratio[best_idx]),
]:
    ax.scatter(resp_n, sig, s=2, alpha=0.3, color=color)
    # linia trendu
    m, b = np.polyfit(resp_n, sig, 1)
    x_line = np.linspace(resp_n.min(), resp_n.max(), 200)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Referencja RESP (znorm.)", fontsize=11)
    ax.set_ylabel(f"{label} (znorm.)", fontsize=11)
    ax.set_title(f"{label}  |r| = {abs(r_val):.3f}   energia = {e_val:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

fig.suptitle("Scatter: składowe SVD vs referencja oddechowa  (CEBSDB b001)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUT_DIR, "cmp_02_scatter.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"Zapisano: {path}")
plt.close(fig)
