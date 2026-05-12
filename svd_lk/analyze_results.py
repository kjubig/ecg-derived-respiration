"""
analyze_results.py  —  ETAP 3: Analiza wyników EDR
====================================================
Wczytuje wyniki z etapu 2 (results/svd_edr.npz) oraz dane z etapu 1
(preprocessed/cebsdb_b001.npz) i generuje wykresy analizy.

Wykresy zapisywane do results/:
    01_edr_vs_referencja.png   — porównanie EDR z referencją oddechową
    02_psd_oddechowe.png       — widmo mocy (Welch) w paśmie oddechowym
    03_korelacje_skladowych.png — korelacja każdej składowej SVD z referencją
    04_morfologie_qrs.png      — kształty morfologiczne (wiersze Vᵀ)

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
# Konfiguracja — możesz tu wpisać ścieżkę do wyników innej metody
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

print(f"Metoda:          {METHOD.upper()}")
print(f"Najlepsza skł.:  {METHOD.upper()}-{best_idx + 1}")
print(f"Korelacja r:     {final_corr:.3f}")
print(f"EDR:             {rr_edr:.1f} /min")
print(f"Referencja:      {rr_resp:.1f} /min")
print()

# ---------------------------------------------------------------------------
# Wykres 1 — EDR vs referencja oddechowa
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(r_times, resp_norm, linewidth=1.4, color="tab:green", label="Referencja oddechowa")
ax1.plot(r_times, edr_norm,  linewidth=1.2, color="tab:blue", linestyle="--",
         label=f"EDR  {METHOD.upper()}-{best_idx+1}  (r={final_corr:.3f})")
ax1.set_ylabel("Amplituda (znorm.)")
ax1.set_title(f"EDR vs referencja oddechowa  —  CEBSDB b001  [{METHOD.upper()}]",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Różnica (błąd)
diff = edr_norm - resp_norm
ax2.fill_between(r_times, diff, alpha=0.4, color="tab:red", label="EDR − referencja")
ax2.axhline(0, color="gray", linewidth=0.8)
ax2.set_xlabel("Czas [s]")
ax2.set_ylabel("Różnica")
ax2.set_title("Błąd (EDR − referencja)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
_save(fig, f"01_edr_vs_referencja.png")

# ---------------------------------------------------------------------------
# Wykres 2 — widmo mocy (PSD) w paśmie oddechowym
# ---------------------------------------------------------------------------
nperseg = min(64, len(edr_signal))
f_edr,  psd_edr  = welch(edr_signal, fs=fs_edr, nperseg=nperseg)
f_resp, psd_resp = welch(resp_norm,  fs=fs_edr, nperseg=nperseg)
mask_f = (f_edr >= 0.1) & (f_edr <= 0.5)

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
ax.set_title(f"Widmo mocy (Welch) — zakres oddechowy  [{METHOD.upper()}]",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
_save(fig, f"02_psd_oddechowe.png")

# ---------------------------------------------------------------------------
# Wykres 3 — korelacja każdej składowej z referencją
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
# Wykres 4 — morfologie QRS (wiersze Vᵀ dla 6 pierwszych składowych)
# ---------------------------------------------------------------------------
n_morph = min(6, Vt.shape[0])
cycle_len = Vt.shape[1]
t_cycle   = np.linspace(0, 100, cycle_len)  # [% cyklu RR]

fig, axes = plt.subplots(n_morph, 1, figsize=(12, 2.5 * n_morph), sharex=True)
for k in range(n_morph):
    color = "tab:red" if k == best_idx else "steelblue"
    label = f"{METHOD.upper()}-{k+1}" + (" ← EDR" if k == best_idx else "")
    axes[k].plot(t_cycle, Vt[k], linewidth=0.9, color=color)
    axes[k].set_ylabel(label, fontsize=9)
    axes[k].grid(True, alpha=0.3)
    axes[k].axhline(0, color="gray", linewidth=0.5)

axes[-1].set_xlabel("Faza cyklu RR [%]")
fig.suptitle(f"Morfologie składowych (wiersze Vᵀ)  —  CEBSDB b001  [{METHOD.upper()}]",
             fontsize=12, fontweight="bold")
plt.tight_layout()
_save(fig, f"04_morfologie_qrs.png")

print(f"\nWszystkie wykresy zapisane w: {OUT_DIR}/")
