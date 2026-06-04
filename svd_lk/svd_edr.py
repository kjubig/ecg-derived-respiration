"""
svd_edr.py  —  ETAP 2: Wyznaczenie składowej oddechowej metodą SVD
===================================================================
Wczytuje dane z etapu 1 (preprocessed/cebsdb_b001.npz),
wykonuje SVD na macierzy cykli i wybiera składową EDR.

Wynik:
    results/svd_edr.npz  — dane gotowe do analizy wyników

Plik .npz zawiera:
    edr_signal    — sygnał EDR (najlepsza kolumna U, surowa)
    edr_norm      — sygnał EDR (znormalizowany 0–1, faza skorygowana)
    resp_norm     — referencja oddechowa (znormalizowana 0–1)
    r_times       — czasy cykli [s]
    fs_edr        — częstotliwość sygnału EDR (≈ HR) [Hz]
    best_idx      — indeks wybranej składowej SVD (0-based)
    correlations  — korelacja każdej z 20 składowych z referencją
    variance_ratio — udział każdej składowej w wariancji [%]
    S             — wartości osobliwe
    U             — macierz U (N × n_components)
    Vt            — macierz Vᵀ (n_components × L)

Użycie:
    python svd_edr.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.linalg import svd as scipy_svd
from scipy.interpolate import interp1d
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
IN_NPZ   = "../preprocessing/preprocessed/cebsdb_b001.npz"
OUT_NPZ  = "results/svd_edr.npz"
OUT_PLOT = "results/svd_skladowe.png"

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Wczytanie danych z etapu 1
# ---------------------------------------------------------------------------
d         = np.load(IN_NPZ, allow_pickle=False)
X         = d["X"]
r_peaks   = d["r_peaks"]
valid_idx = d["valid_idx"]
resp_ref  = d["resp_ref"]
fs        = float(d["fs"])

print(f"Wczytano: {IN_NPZ}")
print(f"  Macierz cykli:  {X.shape}  (cykle × próbki)")
print(f"  Fs:             {fs:.0f} Hz")

# ---------------------------------------------------------------------------
# SVD
#   X_centered = U · Σ · Vᵀ
#   Kolumna U[:, k] — jak zmienia się k-ta składowa w czasie (1 punkt/cykl)
#   Vt[k, :]        — kształt morfologiczny k-tej składowej
#   S[k]            — "waga" (siła) k-tej składowej
# ---------------------------------------------------------------------------
X_centered = X - X.mean(axis=0)
X_centered = np.nan_to_num(X_centered)

U, S, Vt = scipy_svd(X_centered, full_matrices=False, check_finite=False)

variance_ratio = (S ** 2) / (S ** 2).sum() * 100

print("\nUdział % w wariancji (pierwsze 10 składowych):")
for k in range(10):
    print(f"  SVD-{k+1:02d}: {variance_ratio[k]:.2f}%")

# ---------------------------------------------------------------------------
# Referencja oddechowa w domenie cykli
# ---------------------------------------------------------------------------
r_times = r_peaks[valid_idx] / fs

# Wyznacz średnią referencji w każdym cyklu
resp_per_cycle = np.array([
    resp_ref[r_peaks[valid_idx[i]]:r_peaks[valid_idx[i] + 1]].mean()
    if valid_idx[i] + 1 < len(r_peaks) else np.nan
    for i in range(len(valid_idx))
])

n      = min(U.shape[0], len(resp_per_cycle))
U_n    = U[:n]
resp_n = resp_per_cycle[:n]
r_times = r_times[:n]

# Usuń cykle z NaN w referencji
mask   = ~np.isnan(resp_n)
U_m    = U_n[mask]
resp_m = resp_n[mask]
t_m    = r_times[mask]

# ---------------------------------------------------------------------------
# Interpolacja do regularnej siatki czasowej + filtr pasmowy 0.0666–0.5 Hz
#   Krok 10–11 algorytmu: beat-by-beat (nieregularny) → stała siatka 4 Hz,
#   następnie filtracja w paśmie oddechowym przed doborem składowej.
# ---------------------------------------------------------------------------
FS_INTERP = 4.0      # Hz — standard 4–10 Hz dla sygnałów beat-by-beat
RESP_LO   = 0.0666   # Hz — ~4 odd/min
RESP_HI   = 0.5      # Hz — 30 odd/min

t_reg = np.arange(t_m[0], t_m[-1], 1.0 / FS_INTERP)

def _interp(t_irr, y_irr, t_out):
    f = interp1d(t_irr, y_irr, kind="cubic", bounds_error=False,
                 fill_value=(float(y_irr[0]), float(y_irr[-1])))
    return f(t_out)

def _bandpass(sig):
    sos = butter(4, [RESP_LO, RESP_HI], btype="band", fs=FS_INTERP, output="sos")
    return sosfiltfilt(sos, sig)

resp_reg  = _interp(t_m, resp_m, t_reg)
resp_filt = _bandpass(resp_reg)

n_components = min(20, U_m.shape[1])
U_filt = np.zeros((len(t_reg), n_components))
for k in range(n_components):
    U_filt[:, k] = _bandpass(_interp(t_m, U_m[:, k], t_reg))

# ---------------------------------------------------------------------------
# Wybór składowej EDR — maksymalna korelacja z referencją (sygnały filtrowane)
# ---------------------------------------------------------------------------
correlations = np.array([abs(pearsonr(U_filt[:, k], resp_filt)[0])
                         for k in range(n_components)])

best_idx = int(np.argmax(correlations))
print(f"\nNajlepsza składowa EDR: SVD-{best_idx + 1}  "
      f"(|r| = {correlations[best_idx]:.3f})")

# ---------------------------------------------------------------------------
# Normalizacja i korekta fazy
# ---------------------------------------------------------------------------
def normalize(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

edr_raw  = U_n[:, best_idx]              # beat-domain (archiwum)
edr_filt = U_filt[:, best_idx]           # regularna siatka, przefilt.
edr_norm  = normalize(edr_filt)
resp_norm = normalize(resp_filt)

if pearsonr(edr_norm, resp_norm)[0] < 0:
    edr_norm = 1.0 - edr_norm

final_corr, _ = pearsonr(edr_norm, resp_norm)
print(f"Korelacja EDR z referencją (po filtracji): r = {final_corr:.3f}")

# ---------------------------------------------------------------------------
# Estymacja częstości oddechów (Welch PSD)
# ---------------------------------------------------------------------------
fs_edr  = FS_INTERP
nperseg = min(64, len(t_reg))

f_edr,  psd_edr  = welch(edr_filt,  fs=fs_edr, nperseg=nperseg)
f_resp, psd_resp = welch(resp_filt,  fs=fs_edr, nperseg=nperseg)

mask_f = (f_edr >= 0.0666) & (f_edr <= 0.5)
rr_edr  = f_edr[mask_f][np.argmax(psd_edr[mask_f])]  * 60
rr_resp = f_resp[mask_f][np.argmax(psd_resp[mask_f])] * 60

print(f"Fs EDR (≈ HR):             {fs_edr:.2f} Hz  ({fs_edr*60:.0f} uderzeń/min)")
print(f"Częstość oddechów — EDR:   {rr_edr:.1f} /min")
print(f"Częstość oddechów — ref:   {rr_resp:.1f} /min")

# ---------------------------------------------------------------------------
# Wykres: składowe SVD (pomocniczy)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].bar(range(1, 16), variance_ratio[:15], color="steelblue")
axes[0].bar(best_idx + 1, variance_ratio[best_idx], color="tab:red",
            label=f"EDR = SVD-{best_idx + 1}")
axes[0].set_xlabel("Składowa SVD")
axes[0].set_ylabel("Udział w wariancji [%]")
axes[0].set_title("Energia składowych SVD — CEBSDB b001")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].plot(t_reg, resp_norm, label="Referencja oddechowa", linewidth=1.4)
axes[1].plot(t_reg, edr_norm,  label=f"EDR  SVD-{best_idx+1}  (r={final_corr:.3f})",
             linewidth=1.2, linestyle="--")
axes[1].set_xlabel("Czas [s]")
axes[1].set_ylabel("Amplituda (znorm.)")
axes[1].set_title("Składowa EDR vs referencja oddechowa")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"\nWykres zapisano: {OUT_PLOT}")
plt.close()

# ---------------------------------------------------------------------------
# Zapis wyników do .npz
# ---------------------------------------------------------------------------
np.savez(
    OUT_NPZ,
    edr_signal     = edr_raw,
    edr_filt       = edr_filt,
    edr_norm       = edr_norm,
    resp_norm      = resp_norm,
    r_times        = t_reg,
    r_times_beats  = r_times,
    fs_edr         = np.array(fs_edr),
    best_idx       = np.array(best_idx),
    correlations   = correlations,
    variance_ratio = variance_ratio,
    S              = S,
    U              = U,
    Vt             = Vt,
    rr_edr         = np.array(rr_edr),
    rr_resp        = np.array(rr_resp),
    final_corr     = np.array(final_corr),
)
print(f"Wyniki zapisane do: {OUT_NPZ}")
