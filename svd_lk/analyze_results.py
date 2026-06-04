"""
analyze_results.py  -  ETAP 3: Analiza wyników EDR
====================================================
Wczytuje wyniki z etapu 2 (results/svd_edr.npz) oraz dane z etapu 1
(preprocessed/cebsdb_b001.npz) i generuje wykresy analizy.

Wykresy zapisywane do results/:
    01_edr_vs_referencja.png   - porównanie EDR z referencją oddechową
    02_psd_oddechowe.png       - widmo mocy (Welch) w paśmie oddechowym
    03_korelacje_skladowych.png - korelacja każdej składowej SVD z referencją
    04_morfologie_qrs.png      - kształty morfologiczne (wiersze Vᵀ)

Użycie:
    python analyze_results.py [--method svd]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Konfiguracja - możesz tu wpisać ścieżkę do wyników innej metody
# ---------------------------------------------------------------------------
METHOD     = sys.argv[1] if len(sys.argv) > 1 else "svd"
IN_RESULTS = f"results/{METHOD}_edr.npz"
IN_DATA    = "../preprocessing/preprocessed/cebsdb_b001.npz"
OUT_DIR    = f"results"

os.makedirs(OUT_DIR, exist_ok=True)

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Zapisano: {path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Wczytanie wyników
# ---------------------------------------------------------------------------
r = np.load(IN_RESULTS, allow_pickle=False)
d = np.load(IN_DATA,    allow_pickle=False)

edr_norm       = r["edr_norm"]
resp_norm      = r["resp_norm"]
r_times        = r["r_times"]
fs_edr         = float(r["fs_edr"])
best_idx       = int(r["best_idx"])
correlations   = r["correlations"]
variance_ratio = r["variance_ratio"]
S              = r["S"]
U              = r["U"]
Vt             = r["Vt"]
rr_edr         = float(r["rr_edr"])
rr_resp        = float(r["rr_resp"])
final_corr     = float(r["final_corr"])
edr_signal     = r["edr_signal"]
edr_filt       = r["edr_filt"]

print(f"Metoda:          {METHOD.upper()}")
print(f"Najlepsza skł.:  {METHOD.upper()}-{best_idx + 1}")
print(f"Korelacja r:     {final_corr:.3f}")
print(f"EDR:             {rr_edr:.1f} /min")
print(f"Referencja:      {rr_resp:.1f} /min")
print()

# ---------------------------------------------------------------------------
# Wykres 0 - heatmapa macierzy cykli X (i X po centrowaniu)
# ---------------------------------------------------------------------------
X = d["X"]
r_times_beats = r["r_times_beats"]
X_cent = X - X.mean(axis=0)

# Dla czytelności: redukujemy rozdzielczość kolumn do ~300 punktów
n_cols_display = 300
col_idx = np.linspace(0, X.shape[1] - 1, n_cols_display, dtype=int)
t_pct   = np.linspace(0, 100, n_cols_display)   # oś X w % cyklu RR

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im0 = axes[0].imshow(
    X[:, col_idx], aspect="auto", cmap="RdBu_r",
    extent=[0, 100, X.shape[0], 0],
    vmin=np.percentile(X, 2), vmax=np.percentile(X, 98)
)
axes[0].set_xlabel("Czas w cyklu RR [%]")
axes[0].set_ylabel("Numer cyklu")
axes[0].set_title("Macierz cykli X (surowa)", fontsize=11, fontweight="bold")
plt.colorbar(im0, ax=axes[0], label="Amplituda [mV]")

im1 = axes[1].imshow(
    X_cent[:, col_idx], aspect="auto", cmap="RdBu_r",
    extent=[0, 100, X.shape[0], 0],
    vmin=np.percentile(X_cent, 2), vmax=np.percentile(X_cent, 98)
)
axes[1].set_xlabel("Czas w cyklu RR [%]")
axes[1].set_ylabel("Numer cyklu")
axes[1].set_title("Macierz cykli (po centrowaniu)", fontsize=11, fontweight="bold")
plt.colorbar(im1, ax=axes[1], label="Amplituda [mV]")

plt.suptitle("Macierz cykli EKG - CEBSDB b001", fontsize=13, fontweight="bold")
plt.tight_layout()
_save(fig, "00_macierz_cykli.png")

# ---------------------------------------------------------------------------
# Wykres 1 - EDR vs referencja oddechowa
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(r_times, resp_norm, linewidth=1.4, color="tab:green", label="Referencja oddechowa")
ax.plot(r_times, edr_norm,  linewidth=1.2, color="tab:blue", linestyle="--",
        label=f"EDR  {METHOD.upper()}-{best_idx+1}  (r = {final_corr:.3f})")
ax.set_xlabel("Czas [s]", fontsize=12)
ax.set_ylabel("Amplituda (znorm.)", fontsize=12)
ax.set_title(f"EDR vs referencja oddechowa  -  CEBSDB b001  [{METHOD.upper()}]",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
_save(fig, f"01_edr_vs_referencja.png")

# ---------------------------------------------------------------------------
# Wykres 2 - widmo mocy (PSD) w paśmie oddechowym
# ---------------------------------------------------------------------------
nperseg = min(64, len(edr_filt))
f_edr,  psd_edr  = welch(edr_filt,  fs=fs_edr, nperseg=nperseg)
f_resp, psd_resp = welch(resp_norm,  fs=fs_edr, nperseg=nperseg)
mask_f = (f_edr >= 0.0666) & (f_edr <= 0.5)

fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogy(f_edr[mask_f] * 60, psd_edr[mask_f],
            label=f"EDR  {METHOD.upper()}-{best_idx+1}", linewidth=1.4)
ax.semilogy(f_resp[mask_f] * 60, psd_resp[mask_f],
            label="Referencja", linewidth=1.2, linestyle="--")
ax.axvline(rr_edr,  color="tab:blue",   linestyle=":", linewidth=1.2,
           label=f"EDR peak: {rr_edr:.1f} /min")
ax.axvline(rr_resp, color="tab:orange", linestyle=":", linewidth=1.2,
           label=f"Ref peak: {rr_resp:.1f} /min")
ax.set_xlabel("Częstość oddechów [/min]")
ax.set_ylabel("PSD")
ax.set_title(f"Widmo mocy (Welch) - zakres oddechowy  [{METHOD.upper()}]",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
_save(fig, f"02_psd_oddechowe.png")

# ---------------------------------------------------------------------------
# Wykres 3 - korelacja każdej składowej z referencją
# ---------------------------------------------------------------------------
n_show = len(correlations)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ["tab:red" if i == best_idx else "steelblue" for i in range(n_show)]
ax1.bar(range(1, n_show + 1), correlations, color=colors)
ax1.set_xlabel(f"Składowa {METHOD.upper()}")
ax1.set_ylabel("|r| z referencją oddechową")
ax1.set_title(f"Korelacja składowych z referencją  [{METHOD.upper()}]",
              fontsize=11, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")
ax1.bar(best_idx + 1, correlations[best_idx], color="tab:red",
        label=f"Najlepsza: {METHOD.upper()}-{best_idx+1}  (|r|={correlations[best_idx]:.3f})")
ax1.legend(fontsize=9)

n_var = min(15, len(variance_ratio))
ax2.bar(range(1, n_var + 1),
        variance_ratio[:n_var], color="steelblue")
ax2.bar(best_idx + 1, variance_ratio[best_idx], color="tab:red",
        label=f"{METHOD.upper()}-{best_idx+1}: {variance_ratio[best_idx]:.1f}%")
ax2.set_xlabel(f"Składowa {METHOD.upper()}")
ax2.set_ylabel("Udział w wariancji [%]")
ax2.set_title("Energia składowych (wartości osobliwe²)",
              fontsize=11, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")
ax2.legend(fontsize=9)

plt.tight_layout()
_save(fig, f"03_korelacje_skladowych.png")

# ---------------------------------------------------------------------------
# Wykres 4 - morfologie QRS (wiersze Vᵀ dla 6 pierwszych składowych)
# ---------------------------------------------------------------------------
n_morph = min(6, Vt.shape[0])
cycle_len = Vt.shape[1]
t_cycle   = np.linspace(0, 100, cycle_len)  # [% cyklu RR]

fig, axes = plt.subplots(n_morph, 1, figsize=(12, 2.5 * n_morph), sharex=True)
for k in range(n_morph):
    color = "tab:red" if k == best_idx else "steelblue"
    label = f"{METHOD.upper()}-{k+1}"
    axes[k].plot(t_cycle, Vt[k], linewidth=0.9, color=color)
    axes[k].set_ylabel(label, fontsize=9)
    axes[k].grid(True, alpha=0.3)
    axes[k].axhline(0, color="gray", linewidth=0.5)

axes[-1].set_xlabel("Faza cyklu RR [%]")
fig.suptitle(f"Morfologie składowych (wiersze Vᵀ)  -  CEBSDB b001  [{METHOD.upper()}]",
             fontsize=12, fontweight="bold")
plt.tight_layout()
_save(fig, f"04_morfologie_qrs.png")

# ---------------------------------------------------------------------------
# Wykres 5 - energia składowych (wartości osobliwe²)
# ---------------------------------------------------------------------------
n_show = min(15, len(variance_ratio))

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(1, n_show + 1), variance_ratio[:n_show], color="steelblue", edgecolor="white")
ax.set_xlabel(f"Składowa {METHOD.upper()}", fontsize=12)
ax.set_ylabel("Udział w wariancji [%]", fontsize=12)
ax.set_title(f"Energia składowych SVD  -  CEBSDB b001", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(variance_ratio[:n_show]):
    ax.text(i + 1, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
plt.tight_layout()
_save(fig, f"05_energia_skladowych.png")

# ---------------------------------------------------------------------------
# Wykres 6 - korelacja składowych z referencją (z U)
# ---------------------------------------------------------------------------
n_corr = len(correlations)
colors_c = ["tab:red" if i == best_idx else "steelblue" for i in range(n_corr)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, n_corr + 1), correlations, color=colors_c, edgecolor="white")
ax.bar(best_idx + 1, correlations[best_idx], color="tab:red", edgecolor="white",
       label=f"{METHOD.upper()}-{best_idx+1}:  |r| = {correlations[best_idx]:.3f}  (EDR)")
ax.set_xlabel(f"Składowa {METHOD.upper()}", fontsize=12)
ax.set_ylabel("|r|  z referencją oddechową", fontsize=12)
ax.set_title(f"Korelacja składowych z RESP  -  CEBSDB b001", fontsize=12, fontweight="bold")
ax.set_ylim(0, min(1.0, correlations.max() * 1.25))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(correlations):
    ax.text(i + 1, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
plt.tight_layout()
_save(fig, f"06_korelacje_z_resp.png")

print(f"\nWszystkie wykresy zapisane w: {OUT_DIR}/")
