"""
run_preprocessing.py  —  ETAP 1: Preprocessing
================================================
Wczytuje rekord CEBSDB b001, filtruje EKG (dwuetapowo: HP + LP),
wykrywa R-piki, delineuje fale PQRST, buduje macierze cykli
i zapisuje wszystko do pliku .npz.

CEBSDB: 5000 Hz, 2 kanały EKG (I i II) — wspiera metody SVD, PCA, ICA.
BIDMC (125 Hz, 1 kanał) jest pominięte — nie spełnia wymagań Hz i liczby kanałów.

Wynik:
    preprocessed/cebsdb_b001.npz  — dane gotowe do analizy metodami SVD/PCA/ICA

Plik .npz zawiera:
    ecg1_raw     — surowy sygnał EKG (odprowadzenie I)
    ecg1_bl      — odprowadzenie I po usunięciu dryfu
    ecg1_filt    — odprowadzenie I po LP < 40 Hz (końcowy)
    ecg_raw      — surowy sygnał EKG (odprowadzenie II)   [= ecg2, backward compat]
    ecg_bl       — odprowadzenie II po usunięciu dryfu
    ecg_filt     — odprowadzenie II po LP < 40 Hz (końcowy)
    r_peaks      — indeksy R-pików [próbki]
    resp_ref     — referencyjny sygnał oddechowy
    fs           — częstotliwość próbkowania [Hz]
    X1           — macierz cykli EKG odprowadzenie I  (N × L)
    X2           — macierz cykli EKG odprowadzenie II (N × L)
    X            — alias X2 (backward compatibility dla SVD)
    cycle_len    — długość jednego cyklu [próbki]
    valid_idx    — indeksy cykli użytych do budowy macierzy X

Użycie:
    python run_preprocessing.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from preprocessing import (
    load_cebsdb,
    remove_baseline,
    lowpass,
    detect_r_peaks,
    delineate_waves,
    build_cycle_matrix,
    plot_preprocessing,
)

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
RECORD_PATH = "../dataset/lk/CEBSDB/b001"
OUT_NPZ     = "preprocessed/cebsdb_b001.npz"
PLOTS_DIR   = "plots"

os.makedirs("preprocessed", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Etap 1.1 — Wczytanie danych
# ---------------------------------------------------------------------------
data     = load_cebsdb(RECORD_PATH)
fs       = data["fs"]
ecg1_raw = data["ecg1"]       # odprowadzenie I
ecg_raw  = data["ecg2"]       # odprowadzenie II  (backward compat: ecg_raw = ecg2)
resp_ref = data["resp_ref"]
r_peaks  = data["r_peaks"]    # R-piki z adnotacji .atr

# ---------------------------------------------------------------------------
# Etap 1.2 — Filtracja dwuetapowa (oba odprowadzenia)
#   Krok 1: dwa filtry medianowe (200 ms + 600 ms) — usunięcie dryfu linii bazowej
#           podejście stosowane w metodach EDR/PCA (Clifford & Tarassenko 2005)
#   Krok 2: LP < 40 Hz — usunięcie szumu mięśniowego i zakłóceń sieciowych
# ---------------------------------------------------------------------------
ecg1_bl   = remove_baseline(ecg1_raw, fs=fs)
ecg1_filt = lowpass(ecg1_bl, cutoff=40.0, fs=fs)

ecg_bl    = remove_baseline(ecg_raw, fs=fs)
ecg_filt  = lowpass(ecg_bl, cutoff=40.0, fs=fs)

# ---------------------------------------------------------------------------
# Etap 1.3 — Delineacja fal (P, Q, R, S, T) — na odprowadzeniu II
# ---------------------------------------------------------------------------
waves = delineate_waves(ecg_filt, r_peaks, fs)
print(f"\nDelineacja fal (odprowadzenie II):")
for wave in ("P", "Q", "R", "S", "T"):
    arr = waves[wave]
    valid = arr[arr >= 0]
    print(f"  {wave}: {len(valid)}/{len(r_peaks)} cykli")

# ---------------------------------------------------------------------------
# Etap 1.4 — Budowa macierzy cykli (wejście dla SVD/PCA/ICA)
#   X1 — odprowadzenie I,  X2 — odprowadzenie II
#   X  — alias X2 (backward compatibility dla istniejącego kodu SVD)
# ---------------------------------------------------------------------------
X1, cycle_len,  valid_idx  = build_cycle_matrix(ecg1_filt, r_peaks)
X2, cycle_len2, valid_idx2 = build_cycle_matrix(ecg_filt,  r_peaks)
X = X2   # backward compat

print(f"\nMacierz cykli I:   {X1.shape}  (cykle × próbki na cykl)")
print(f"Macierz cykli II:  {X2.shape}  (cykle × próbki na cykl)")
print(f"Długość cyklu:     {cycle_len} próbek  ({cycle_len/fs*1000:.0f} ms)")

# ---------------------------------------------------------------------------
# Etap 1.5 — Zapis do .npz
# ---------------------------------------------------------------------------
np.savez(
    OUT_NPZ,
    ecg1_raw  = ecg1_raw,
    ecg1_bl   = ecg1_bl,
    ecg1_filt = ecg1_filt,
    ecg_raw   = ecg_raw,
    ecg_bl    = ecg_bl,
    ecg_filt  = ecg_filt,
    r_peaks   = r_peaks,
    resp_ref  = resp_ref,
    fs        = np.array(fs),
    X1        = X1,
    X2        = X2,
    X         = X,            # alias X2, backward compat
    cycle_len = np.array(cycle_len),
    valid_idx = valid_idx,
)
print(f"\nDane zapisane do: {OUT_NPZ}")

# ---------------------------------------------------------------------------
# Etap 1.6 — Wykresy (01, 02, 03, 03b, 04)  — na odprowadzeniu II
# ---------------------------------------------------------------------------
plot_preprocessing(
    ecg_raw  = ecg_raw,
    ecg_bl   = ecg_bl,
    ecg_filt = ecg_filt,
    resp_ref = resp_ref,
    r_peaks  = r_peaks,
    waves    = waves,
    fs       = fs,
    title    = "CEBSDB b001",
    save_dir = PLOTS_DIR,
)
