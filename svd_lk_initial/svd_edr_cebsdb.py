"""
Wyznaczanie składowej oddechowej z EKG metodą SVD
Baza: CEBSDB, rekord b001
Kanały: I, II, RESP, SCG @ 5000 Hz
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, sosfiltfilt, resample, welch
from scipy.linalg import svd as scipy_svd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Ścieżki
# ---------------------------------------------------------------------------
RECORD_PATH = "./dataset/lk/CEBSDB/b001"

# ---------------------------------------------------------------------------
# 1. Wczytanie danych
# ---------------------------------------------------------------------------
record     = wfdb.rdrecord(RECORD_PATH)
annotation = wfdb.rdann(RECORD_PATH, "atr")

fs       = record.fs                    # 5000 Hz
ecg1     = record.p_signal[:, 0]       # odprowadzenie I
ecg2     = record.p_signal[:, 1]       # odprowadzenie II
resp_ref = record.p_signal[:, 2]       # referencja oddechowa (piezoband)

# Adnotacje w CEBSDB są zdublowane (jedna na każde odprowadzenie).
# Bierzemy co drugi wpis (parzyste indeksy = kanał I).
r_peaks = annotation.sample[::2]

print(f"Sygnały:            {record.sig_name}")
print(f"Fs:                 {fs} Hz")
print(f"Czas nagrania:      {record.sig_len / fs:.1f} s")
print(f"Liczba R-pików:     {len(r_peaks)}")
print(f"Mediana RR:         {int(np.median(np.diff(r_peaks)))} próbek  "
      f"({np.median(np.diff(r_peaks))/fs*1000:.0f} ms)")

# ---------------------------------------------------------------------------
# 2. Filtrowanie EKG (pasmowe 0.5–40 Hz)
# ---------------------------------------------------------------------------
def bandpass(signal, lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal)

ecg1_filt = bandpass(ecg1, 0.5, 40, fs)
ecg2_filt = bandpass(ecg2, 0.5, 40, fs)

# ---------------------------------------------------------------------------
# 3. Budowa macierzy cykli EKG
#    Każdy wiersz = jeden cykl R→R, resamplowany do mediany długości cyklu
# ---------------------------------------------------------------------------
def build_cycle_matrix(ecg, r_peaks, cycle_len=None):
    rr_lengths = np.diff(r_peaks)
    if cycle_len is None:
        cycle_len = int(np.median(rr_lengths))

    cycles = []
    valid_indices = []
    for i in range(len(r_peaks) - 1):
        start, end = r_peaks[i], r_peaks[i + 1]
        segment = ecg[start:end]
        if len(segment) < 2:
            continue
        cycles.append(resample(segment, cycle_len))
        valid_indices.append(i)

    return np.array(cycles), cycle_len, np.array(valid_indices)

# Odprowadzenie II — lepsza amplituda fali R
X, L, valid_idx = build_cycle_matrix(ecg2_filt, r_peaks)
print(f"\nMacierz cykli:      {X.shape}  (cykle × próbki na cykl)")

# ---------------------------------------------------------------------------
# 4. SVD
#    X_centered = U · Σ · Vᵀ
#    Kolumny U — współczynniki w czasie (jeden punkt na cykl)
# ---------------------------------------------------------------------------
X_centered = X - X.mean(axis=0)
# Zamień ewentualne NaN/Inf na 0 (ostrożność przy resamplu)
X_centered = np.nan_to_num(X_centered)
U, S, Vt = scipy_svd(X_centered, full_matrices=False, check_finite=False)

variance_ratio = (S ** 2) / (S ** 2).sum() * 100
print("\nUdział % w wariancji (pierwsze 10 składowych):")
for k in range(10):
    print(f"  SVD-{k+1:02d}: {variance_ratio[k]:.2f}%")

# ---------------------------------------------------------------------------
# 5. Referencja oddechowa w domenie cykli
#    Dla każdego cyklu — średnia wartość sygnału oddechowego w tym cyklu
# ---------------------------------------------------------------------------
resp_per_cycle = []
for i in valid_idx:
    if r_peaks[i + 1] <= len(resp_ref):
        resp_per_cycle.append(resp_ref[r_peaks[i]:r_peaks[i + 1]].mean())
    else:
        resp_per_cycle.append(np.nan)

resp_per_cycle = np.array(resp_per_cycle)
n = min(len(U), len(resp_per_cycle))
U_n    = U[:n]
resp_n = resp_per_cycle[:n]

# Czasy R-pików (środek cyklu)
r_times = r_peaks[valid_idx[:n]] / fs

# ---------------------------------------------------------------------------
# 6. Wybór składowej EDR — maksymalna korelacja z referencją
# ---------------------------------------------------------------------------
n_components = min(20, U_n.shape[1])
correlations = [abs(pearsonr(U_n[:, k], resp_n)[0]) for k in range(n_components)]

best_idx = int(np.argmax(correlations))
print(f"\nNajlepsza składowa EDR: SVD-{best_idx + 1}  "
      f"(|r| = {correlations[best_idx]:.3f})")

# ---------------------------------------------------------------------------
# 7. Porównanie EDR z referencją
# ---------------------------------------------------------------------------
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

edr_raw  = U_n[:, best_idx]
edr_norm  = normalize(edr_raw)
resp_norm = normalize(resp_n)

# Korekta fazy (jeśli korelacja ujemna, odwróć sygnał)
if pearsonr(edr_norm, resp_norm)[0] < 0:
    edr_norm = 1.0 - edr_norm

final_corr, _ = pearsonr(edr_norm, resp_norm)
print(f"Korelacja EDR z referencją: r = {final_corr:.3f}")

# ---------------------------------------------------------------------------
# 8. Estymacja częstości oddechów (PSD Welcha)
# ---------------------------------------------------------------------------
fs_edr = 1.0 / np.mean(np.diff(r_times))   # średnia HR w Hz
print(f"Fs EDR (≈ HR):      {fs_edr:.2f} Hz  ({fs_edr*60:.0f} uderzeń/min)")

nperseg = min(64, n)
f_edr,  psd_edr  = welch(edr_raw,  fs=fs_edr, nperseg=nperseg)
f_resp, psd_resp = welch(resp_n,   fs=fs_edr, nperseg=nperseg)

mask = (f_edr >= 0.1) & (f_edr <= 0.5)
rr_edr  = f_edr[mask][np.argmax(psd_edr[mask])]  * 60
rr_resp = f_resp[mask][np.argmax(psd_resp[mask])] * 60

print(f"Częstość oddechów — EDR:        {rr_edr:.1f} /min")
print(f"Częstość oddechów — referencja: {rr_resp:.1f} /min")

# ---------------------------------------------------------------------------
# 9. Wykresy
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# 9a. EKG (pierwsze 10 s)
t = np.arange(len(ecg2_filt)) / fs
seg = slice(0, 10 * fs)
axes[0].plot(t[seg], ecg2_filt[seg], linewidth=0.6)
r_in_seg = r_peaks[(r_peaks < 10 * fs)]
axes[0].scatter(r_in_seg / fs, ecg2_filt[r_in_seg], color="red", s=15, zorder=5, label="R-piki")
axes[0].set_title("EKG — odprowadzenie II (filtrowany, 10 s)")
axes[0].set_xlabel("Czas [s]"); axes[0].set_ylabel("Amplituda [mV]")
axes[0].legend()

# 9b. Energia SVD
axes[1].bar(range(1, 16), variance_ratio[:15], color="steelblue")
axes[1].bar(best_idx + 1, variance_ratio[best_idx], color="tab:red", label=f"EDR = SVD-{best_idx+1}")
axes[1].set_xlabel("Składowa SVD"); axes[1].set_ylabel("Udział w wariancji [%]")
axes[1].set_title("Energia składowych SVD")
axes[1].legend()

# 9c. EDR vs referencja
axes[2].plot(r_times, resp_norm, label="Referencja (oddech)", linewidth=1.5)
axes[2].plot(r_times, edr_norm, label=f"EDR-SVD ({best_idx+1})", linewidth=1.5, linestyle="--")
axes[2].set_xlabel("Czas [s]"); axes[2].set_ylabel("Amplituda (znorm.)")
axes[2].set_title(f"EDR vs referencja  (r = {final_corr:.3f})")
axes[2].legend()

# 9d. PSD w paśmie oddechowym
axes[3].semilogy(f_edr[mask] * 60, psd_edr[mask], label="EDR-SVD")
axes[3].semilogy(f_resp[mask] * 60, psd_resp[mask], label="Referencja", linestyle="--")
axes[3].axvline(rr_edr,  color="tab:blue", linestyle=":", label=f"EDR peak: {rr_edr:.1f} /min")
axes[3].axvline(rr_resp, color="tab:orange", linestyle=":", label=f"Ref peak: {rr_resp:.1f} /min")
axes[3].set_xlabel("Częstość oddechów [/min]")
axes[3].set_ylabel("PSD")
axes[3].set_title("Widmo mocy — zakres oddechowy (Welch)")
axes[3].legend()

plt.tight_layout()
plt.savefig("wyniki_svd.png", dpi=150)
plt.show()

print("\nWykres zapisano do: wyniki_svd_cebsdb.png")
