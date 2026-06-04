"""
run_preprocessing_bidmc.py  —  ETAP 1: Preprocessing (BIDMC)
=============================================================
Wczytuje rekord BIDMC bidmc01, filtruje EKG (dwuetapowo: HP + LP),
wykrywa R-piki (brak adnotacji .atr — detekcja automatyczna),
delineuje fale PQRST, buduje macierz cykli i zapisuje wszystko do .npz.

Ograniczenia BIDMC vs CEBSDB:
    - Fs = 125 Hz  (CEBSDB: 5000 Hz)  — mniejsza rozdzielczość czasowa
    - 1 kanał EKG  (CEBSDB: 2 kanały) — metody wymagające 2 kanałów nie mogą
      korzystać z BIDMC
    - Brak adnotacji R-pików — R-piki wykrywane automatycznie (detect_r_peaks)

Wynik:
    preprocessed/bidmc_bidmc01.npz  — dane gotowe do analizy metodami SVD/PCA/ICA

Plik .npz zawiera:
    ecg1_raw     — surowy sygnał EKG (odprowadzenie V)
    ecg1_bl      — odprowadzenie V po usunięciu dryfu
    ecg1_filt    — odprowadzenie V po LP < 40 Hz (końcowy)
    ecg_raw      — surowy sygnał EKG (odprowadzenie II)
    ecg_bl       — odprowadzenie II po usunięciu dryfu
    ecg_filt     — odprowadzenie II po LP < 40 Hz (końcowy)
    r_peaks      — indeksy R-pików [próbki] (wykryte automatycznie z odp. II)
    resp_ref     — referencyjny sygnał oddechowy (impedancja)
    breath_samples — ręczne adnotacje momentów oddechów
    fs           — częstotliwość próbkowania [Hz]  (= 125)
    X1           — macierz cykli EKG odprowadzenie V  (N × L)
    X2           — macierz cykli EKG odprowadzenie II (N × L)
    X            — alias X2 (backward compatibility)
    cycle_len    — długość jednego cyklu [próbki]
    valid_idx    — indeksy cykli użytych do budowy macierzy X

Użycie:
    python run_preprocessing_bidmc.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from preprocessing import (
    load_bidmc,
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
RECORD_PATH = "../dataset/lk/BIDMC/bidmc01"
OUT_NPZ     = "preprocessed/bidmc_bidmc01.npz"
PLOTS_DIR   = "plots/bidmc"

os.makedirs("preprocessed", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Etap 1.1 — Wczytanie danych
# ---------------------------------------------------------------------------
data           = load_bidmc(RECORD_PATH)
fs             = data["fs"]
ecg1_raw       = data["ecg1"]          # odprowadzenie V
ecg_raw        = data["ecg2"]          # odprowadzenie II  (backward compat: ecg_raw = ecg2)
resp_ref       = data["resp_ref"]      # referencja oddechowa (impedancja)
breath_samples = data["breath_samples"]

# ---------------------------------------------------------------------------
# Etap 1.2 — Filtracja dwuetapowa (oba odprowadzenia)
#   Krok 1: dwa filtry medianowe (200 ms + 600 ms) — usunięcie dryfu linii bazowej
#   Krok 2: LP < 40 Hz — usunięcie szumu mięśniowego i zakłóceń sieciowych
#
#   Uwaga: przy 125 Hz okna medianowe są krótsze (25 i 75 próbek vs 1000/3000
#   przy 5000 Hz) — filtracja jest mniej precyzyjna, ale nadal wystarczająca
# ---------------------------------------------------------------------------
ecg1_bl   = remove_baseline(ecg1_raw, fs=fs)
ecg1_filt = lowpass(ecg1_bl, cutoff=40.0, fs=fs)

ecg_bl    = remove_baseline(ecg_raw, fs=fs)
ecg_filt  = lowpass(ecg_bl, cutoff=40.0, fs=fs)

# ---------------------------------------------------------------------------
# Etap 1.3 — Detekcja R-pików (automatyczna z odp. II — BIDMC nie ma adnotacji .atr)
# ---------------------------------------------------------------------------
print("\nDetekcja R-pików (automatyczna z odprowadzenia II):")
r_peaks = detect_r_peaks(ecg_filt, fs=fs)

# ---------------------------------------------------------------------------
# Etap 1.4 — Delineacja fal (P, Q, R, S, T) — na odprowadzeniu II
# ---------------------------------------------------------------------------
waves = delineate_waves(ecg_filt, r_peaks, fs)
print(f"\nDelineacja fal (odprowadzenie II):")
for wave in ("P", "Q", "R", "S", "T"):
    arr = waves[wave]
    valid = arr[arr >= 0]
    print(f"  {wave}: {len(valid)}/{len(r_peaks)} cykli")

# ---------------------------------------------------------------------------
# Etap 1.5 — Budowa macierzy cykli (wejście dla SVD/PCA/ICA)
#   X1 — odprowadzenie V,  X2 — odprowadzenie II
#   X  — alias X2 (backward compatibility)
# ---------------------------------------------------------------------------
X1, cycle_len,  valid_idx  = build_cycle_matrix(ecg1_filt, r_peaks)
X2, cycle_len2, valid_idx2 = build_cycle_matrix(ecg_filt,  r_peaks)
X = X2   # backward compat

print(f"\nMacierz cykli V:   {X1.shape}  (cykle × próbki na cykl)")
print(f"Macierz cykli II:  {X2.shape}  (cykle × próbki na cykl)")
print(f"Długość cyklu:     {cycle_len} próbek  ({cycle_len/fs*1000:.0f} ms)")

# ---------------------------------------------------------------------------
# Etap 1.6 — Zapis do .npz
# ---------------------------------------------------------------------------
np.savez(
    OUT_NPZ,
    ecg1_raw       = ecg1_raw,
    ecg1_bl        = ecg1_bl,
    ecg1_filt      = ecg1_filt,
    ecg_raw        = ecg_raw,
    ecg_bl         = ecg_bl,
    ecg_filt       = ecg_filt,
    r_peaks        = r_peaks,
    resp_ref       = resp_ref,
    breath_samples = breath_samples,
    fs             = np.array(fs),
    X1             = X1,
    X2             = X2,
    X              = X,            # alias X2, backward compat
    cycle_len      = np.array(cycle_len),
    valid_idx      = valid_idx,
)
print(f"\nDane zapisane do: {OUT_NPZ}")

# ---------------------------------------------------------------------------
# Etap 1.7 — Wykresy — na odprowadzeniu II
# ---------------------------------------------------------------------------
plot_preprocessing(
    ecg_raw  = ecg_raw,
    ecg_bl   = ecg_bl,
    ecg_filt = ecg_filt,
    resp_ref = resp_ref,
    r_peaks  = r_peaks,
    waves    = waves,
    fs       = fs,
    title    = "BIDMC bidmc01",
    save_dir = PLOTS_DIR,
)
