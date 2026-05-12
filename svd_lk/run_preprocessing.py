"""
run_preprocessing.py  —  ETAP 1: Preprocessing
================================================
Wczytuje rekord CEBSDB b001, filtruje EKG (dwuetapowo: HP + LP),
wykrywa R-piki, delineuje fale PQRST, buduje macierz cykli
i zapisuje wszystko do pliku .npz.

Wynik:
    preprocessed/cebsdb_b001.npz  — dane gotowe do analizy metodami SVD/PCA/ICA

Plik .npz zawiera:
    ecg_raw      — surowy sygnał EKG (odprowadzenie II)
    ecg_bl       — po usunięciu dryfu linii bazowej (filtry medianowe 200/600 ms)
    ecg_filt     — po filtrze HP + LP < 40 Hz (końcowy)
    r_peaks      — indeksy R-pików [próbki]
    resp_ref     — referencyjny sygnał oddechowy
    fs           — częstotliwość próbkowania [Hz]
    X            — macierz cykli EKG (N × L), gotowa do SVD/PCA/ICA
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
PLOTS_DIR   = "preprocessing"

os.makedirs("preprocessed", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Etap 1.1 — Wczytanie danych
# ---------------------------------------------------------------------------
data     = load_cebsdb(RECORD_PATH)
fs       = data["fs"]
ecg_raw  = data["ecg2"]       # odprowadzenie II
resp_ref = data["resp_ref"]
r_peaks  = data["r_peaks"]    # R-piki z adnotacji .atr

# ---------------------------------------------------------------------------
# Etap 1.2 — Filtracja dwuetapowa
#   Krok 1: dwa filtry medianowe (200 ms + 600 ms) — usunięcie dryfu linii bazowej
#           podejście stosowane w metodach EDR/PCA (Clifford & Tarassenko 2005)
#   Krok 2: LP < 40 Hz — usunięcie szumu mięśniowego i zakłóceń sieciowych
# ---------------------------------------------------------------------------
ecg_bl   = remove_baseline(ecg_raw, fs=fs)       # po usunięciu dryfu
ecg_filt = lowpass(ecg_bl, cutoff=40.0, fs=fs)   # końcowy sygnał

# ---------------------------------------------------------------------------
# Etap 1.3 — Delineacja fal (P, Q, R, S, T)
# ---------------------------------------------------------------------------
waves = delineate_waves(ecg_filt, r_peaks, fs)
print(f"\nDelineacja fal:")
for wave in ("P", "Q", "R", "S", "T"):
    arr = waves[wave]
    valid = arr[arr >= 0]
    print(f"  {wave}: {len(valid)}/{len(r_peaks)} cykli")

# ---------------------------------------------------------------------------
# Etap 1.4 — Budowa macierzy cykli (wejście dla SVD/PCA/ICA)
# ---------------------------------------------------------------------------
X, cycle_len, valid_idx = build_cycle_matrix(ecg_filt, r_peaks)
print(f"\nMacierz cykli:  {X.shape}  (cykle × próbki na cykl)")
print(f"Długość cyklu:  {cycle_len} próbek  ({cycle_len/fs*1000:.0f} ms)")

# ---------------------------------------------------------------------------
# Etap 1.5 — Zapis do .npz
# ---------------------------------------------------------------------------
np.savez(
    OUT_NPZ,
    ecg_raw   = ecg_raw,
    ecg_bl    = ecg_bl,
    ecg_filt  = ecg_filt,
    r_peaks   = r_peaks,
    resp_ref  = resp_ref,
    fs        = np.array(fs),
    X         = X,
    cycle_len = np.array(cycle_len),
    valid_idx = valid_idx,
)
print(f"\nDane zapisane do: {OUT_NPZ}")

# ---------------------------------------------------------------------------
# Etap 1.6 — Wykresy (01, 02, 03, 03b, 04)
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
