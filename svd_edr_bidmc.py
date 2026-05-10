"""
Wyznaczanie składowej oddechowej z EKG metodą SVD
Baza: BIDMC, rekord bidmc01
Kanały: RESP, PLETH, V, AVR, II @ 125 Hz
Brak adnotacji R-pików — wykrywanie przez find_peaks (scipy)
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, sosfiltfilt, resample, welch, find_peaks
from scipy.linalg import svd as scipy_svd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Ścieżki
# ---------------------------------------------------------------------------
RECORD_PATH = "./dataset/lk/BIDMC/bidmc01"

# ---------------------------------------------------------------------------
# 1. Wczytanie danych
# ---------------------------------------------------------------------------
record = wfdb.rdrecord(RECORD_PATH)

fs       = record.fs                    # 125 Hz
# Kanały: ['RESP,', 'PLETH,', 'V,', 'AVR,', 'II,']
resp_ref = record.p_signal[:, 0]       # sygnał oddechowy (impedancja)
ecg      = record.p_signal[:, 4]       # EKG odprowadzenie II

# Adnotacje oddechów (do porównania częstości)
ann_breath = wfdb.rdann(RECORD_PATH, "breath")
# Pary próbek — co 2 wpis to kolejny oddech (początek i koniec)
breath_samples = ann_breath.sample[::2]

print(f"Sygnały:         {record.sig_name}")
print(f"Fs:              {fs} Hz")
print(f"Czas nagrania:   {record.sig_len / fs:.1f} s")
print(f"Adnotacje oddechów (pary): {len(ann_breath.sample)}, unikalne: {len(breath_samples)}")

# ---------------------------------------------------------------------------
# 2. Preprocessing EKG
# ---------------------------------------------------------------------------
def bandpass(signal, lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal)

ecg_filt = bandpass(ecg, 0.5, 40, fs)

# ---------------------------------------------------------------------------
# 3. Wykrywanie R-pików (scipy find_peaks)
#    Minimalna odległość: 0.35 s (171 bpm max), próg: 60% max amplitudy
# ---------------------------------------------------------------------------
min_distance = int(0.35 * fs)
threshold    = 0.6 * ecg_filt.max()

r_peaks, _ = find_peaks(ecg_filt, distance=min_distance, height=threshold)

print(f"\nWykryte R-piki:  {len(r_peaks)}")
print(f"Mediana RR:      {int(np.median(np.diff(r_peaks)))} próbek  "
      f"({np.median(np.diff(r_peaks))/fs*1000:.0f} ms)")

# ---------------------------------------------------------------------------
# 4. Budowa macierzy cykli EKG
# ---------------------------------------------------------------------------
def build_cycle_matrix(ecg, r_peaks, cycle_len=None):
    rr_lengths = np.diff(r_peaks)
    if cycle_len is None:
        cycle_len = int(np.median(rr_lengths))

    cycles     = []
    valid_idx  = []
    for i in range(len(r_peaks) - 1):
        start, end = r_peaks[i], r_peaks[i + 1]
        segment = ecg[start:end]
        if len(segment) < 2:
            continue
        cycles.append(resample(segment, cycle_len))
        valid_idx.append(i)

    return np.array(cycles), cycle_len, np.array(valid_idx)

X, L, valid_idx = build_cycle_matrix(ecg_filt, r_peaks)
print(f"\nMacierz cykli:   {X.shape}  (cykle × próbki na cykl)")

# ---------------------------------------------------------------------------
# 5. SVD
# ---------------------------------------------------------------------------
X_centered = X - X.mean(axis=0)
X_centered = np.nan_to_num(X_centered)
U, S, Vt   = scipy_svd(X_centered, full_matrices=False, check_finite=False)

variance_ratio = (S ** 2) / (S ** 2).sum() * 100
print("\nUdział % w wariancji (pierwsze 10 składowych):")
for k in range(10):
    print(f"  SVD-{k+1:02d}: {variance_ratio[k]:.2f}%")

# ---------------------------------------------------------------------------
# 6. Referencja oddechowa w domenie cykli
# ---------------------------------------------------------------------------
resp_per_cycle = []
for i in valid_idx:
    if r_peaks[i + 1] <= len(resp_ref):
        resp_per_cycle.append(resp_ref[r_peaks[i]:r_peaks[i + 1]].mean())
    else:
        resp_per_cycle.append(np.nan)

resp_per_cycle = np.array(resp_per_cycle)
n      = min(len(U), len(resp_per_cycle))
U_n    = U[:n]
resp_n = resp_per_cycle[:n]
r_times = r_peaks[valid_idx[:n]] / fs

# ---------------------------------------------------------------------------
# 7. Wybór składowej EDR
# ---------------------------------------------------------------------------
n_components = min(20, U_n.shape[1])
correlations = [abs(pearsonr(U_n[:, k], resp_n)[0]) for k in range(n_components)]

best_idx = int(np.argmax(correlations))
print(f"\nNajlepsza składowa EDR: SVD-{best_idx + 1}  "
      f"(|r| = {correlations[best_idx]:.3f})")

# ---------------------------------------------------------------------------
# 8. Porównanie EDR z referencją
# ---------------------------------------------------------------------------
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

edr_raw   = U_n[:, best_idx]
edr_norm  = normalize(edr_raw)
resp_norm = normalize(resp_n)

if pearsonr(edr_norm, resp_norm)[0] < 0:
    edr_norm = 1.0 - edr_norm

final_corr, _ = pearsonr(edr_norm, resp_norm)
print(f"Korelacja EDR z referencją: r = {final_corr:.3f}")

# ---------------------------------------------------------------------------
# 9. Estymacja częstości oddechów (PSD Welcha)
# ---------------------------------------------------------------------------
fs_edr = 1.0 / np.mean(np.diff(r_times))
print(f"Fs EDR (≈ HR):   {fs_edr:.2f} Hz  ({fs_edr*60:.0f} uderzeń/min)")

nperseg = min(64, n)
f_edr,  psd_edr  = welch(edr_raw,  fs=fs_edr, nperseg=nperseg)
f_resp, psd_resp = welch(resp_n,   fs=fs_edr, nperseg=nperseg)

mask    = (f_edr >= 0.1) & (f_edr <= 0.5)
rr_edr  = f_edr[mask][np.argmax(psd_edr[mask])]  * 60
rr_resp = f_resp[mask][np.argmax(psd_resp[mask])] * 60

# Referencja z adnotacji oddechów (bezpośrednia)
if len(breath_samples) > 1:
    rr_ann = 60.0 / np.mean(np.diff(breath_samples) / fs)
    print(f"Częstość oddechów — EDR:             {rr_edr:.1f} /min")
    print(f"Częstość oddechów — ref (PSD):       {rr_resp:.1f} /min")
    print(f"Częstość oddechów — adnotacje ręczne: {rr_ann:.1f} /min")
else:
    print(f"Częstość oddechów — EDR:        {rr_edr:.1f} /min")
    print(f"Częstość oddechów — referencja: {rr_resp:.1f} /min")

# ---------------------------------------------------------------------------
# 10. Wykresy
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# 10a. EKG (pierwsze 20 s)
t   = np.arange(len(ecg_filt)) / fs
seg = slice(0, 20 * fs)
axes[0].plot(t[seg], ecg_filt[seg], linewidth=0.6)
r_in_seg = r_peaks[r_peaks < 20 * fs]
axes[0].scatter(r_in_seg / fs, ecg_filt[r_in_seg], color="red", s=20, zorder=5, label="R-piki")
axes[0].set_title("EKG — odprowadzenie II (filtrowany, 20 s)")
axes[0].set_xlabel("Czas [s]"); axes[0].set_ylabel("Amplituda [mV]")
axes[0].legend()

# 10b. Energia SVD
axes[1].bar(range(1, 16), variance_ratio[:15], color="steelblue")
axes[1].bar(best_idx + 1, variance_ratio[best_idx], color="tab:red",
            label=f"EDR = SVD-{best_idx + 1}")
axes[1].set_xlabel("Składowa SVD"); axes[1].set_ylabel("Udział w wariancji [%]")
axes[1].set_title("Energia składowych SVD")
axes[1].legend()

# 10c. EDR vs referencja
axes[2].plot(r_times, resp_norm, label="Referencja (oddech impedancyjny)", linewidth=1.5)
axes[2].plot(r_times, edr_norm, label=f"EDR-SVD (składowa {best_idx + 1})",
             linewidth=1.5, linestyle="--")
axes[2].set_xlabel("Czas [s]"); axes[2].set_ylabel("Amplituda (znorm.)")
axes[2].set_title(f"EDR vs referencja  (r = {final_corr:.3f})")
axes[2].legend()

# 10d. PSD w paśmie oddechowym
axes[3].semilogy(f_edr[mask] * 60, psd_edr[mask], label="EDR-SVD")
axes[3].semilogy(f_resp[mask] * 60, psd_resp[mask], label="Referencja", linestyle="--")
axes[3].axvline(rr_edr,  color="tab:blue",   linestyle=":", label=f"EDR peak: {rr_edr:.1f} /min")
axes[3].axvline(rr_resp, color="tab:orange", linestyle=":", label=f"Ref peak: {rr_resp:.1f} /min")
axes[3].set_xlabel("Częstość oddechów [/min]")
axes[3].set_ylabel("PSD")
axes[3].set_title("Widmo mocy — zakres oddechowy (Welch)")
axes[3].legend()

plt.tight_layout()
plt.savefig("wyniki_svd_bidmc.png", dpi=150)
plt.show()

print("\nWykres zapisano do: wyniki_svd_bidmc.png")
