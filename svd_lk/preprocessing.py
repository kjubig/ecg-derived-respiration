"""
preprocessing.py
================
Wspólny moduł preprocessingu EKG dla projektu EDR (ECG-Derived Respiration).
Używany przez SVD, PCA i ICA — dostarcza identyczne dane wejściowe dla każdej metody.

Funkcje:
    load_cebsdb(record_path)       — wczytaj rekord CEBSDB (zdrowi pacjenci)
    load_bidmc(record_path)        — wczytaj rekord BIDMC (pacjenci kliniczni)
    remove_baseline(signal, fs)    — usunięcie dryfu linii bazowej (2× filtr medianowy 200/600 ms)
    lowpass(signal, cutoff, fs)    — filtr dolnoprzepustowy < 40 Hz (usunięcie szumu)
    bandpass(signal, lo, hi, fs)   — skrót: remove_baseline + lowpass
    detect_r_peaks(ecg_filt, fs)   — detekcja R-pików przez find_peaks
    delineate_waves(ecg_filt, r_peaks, fs) — wyznaczenie P, Q, R, S, T dla każdego cyklu
    build_cycle_matrix(ecg, r_peaks) — macierz cykli N×L → wejście dla SVD/PCA/ICA
    plot_preprocessing(...)        — wykresy etapów 1–4 (surowy, filtracja, R+PQRST, referencja)

Filtracja (wg podejścia EDR/PCA — np. Varon et al. 2020, Clifford & Tarassenko 2005):
    Etap 1 — 2× filtr medianowy (200 ms + 600 ms) → estymacja i odjęcie linii bazowej
             200 ms usuwa QRS i załamek P; 600 ms usuwa załamek T
    Etap 2 — LP Butterworth 4. rzędu, fc = 40 Hz  → usuwa szum mięśniowy / sieciowy

Delineacja fal (na podstawie wykrytego R):
    Q — minimum w oknie (R-100ms, R)
    S — minimum w oknie (R, R+100ms)
    P — maksimum w oknie (R-300ms, R-100ms)
    T — maksimum w oknie (R+100ms, R+400ms)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, sosfiltfilt, find_peaks, resample
from scipy.ndimage import median_filter


# ===========================================================================
# 1. Wczytywanie danych
# ===========================================================================

def load_cebsdb(record_path: str) -> dict:
    """
    Wczytuje rekord z bazy CEBSDB (format WFDB).

    Kanały: ['I', 'II', 'RESP', 'SCG']  @ 5000 Hz
    Adnotacje R-pików w pliku .atr (zdublowane — co drugi wpis).

    Zwraca słownik z kluczami:
        ecg1, ecg2, resp_ref, r_peaks, fs, duration
    """
    record     = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    fs       = record.fs
    ecg1     = record.p_signal[:, 0]   # odprowadzenie I
    ecg2     = record.p_signal[:, 1]   # odprowadzenie II
    resp_ref = record.p_signal[:, 2]   # referencja oddechowa

    # Adnotacje zdublowane (1 na każde odprowadzenie) — bierzemy co drugi
    r_peaks = annotation.sample[::2]

    print("=== CEBSDB ===")
    print(f"  Sygnały:        {record.sig_name}")
    print(f"  Fs:             {fs} Hz")
    print(f"  Czas nagrania:  {record.sig_len / fs:.1f} s")
    print(f"  R-piki (z .atr): {len(r_peaks)}")
    print(f"  Mediana RR:     {int(np.median(np.diff(r_peaks)))} próbek "
          f"({np.median(np.diff(r_peaks))/fs*1000:.0f} ms)")

    return dict(ecg1=ecg1, ecg2=ecg2, resp_ref=resp_ref,
                r_peaks=r_peaks, fs=fs,
                duration=record.sig_len / fs)


def load_bidmc(record_path: str) -> dict:
    """
    Wczytuje rekord z bazy BIDMC (format WFDB).

    Kanały: ['RESP,', 'PLETH,', 'V,', 'AVR,', 'II,']  @ 125 Hz
    Brak adnotacji R-pików — r_peaks = None (wykryj przez detect_r_peaks).

    Zwraca słownik z kluczami:
        ecg, resp_ref, breath_samples, r_peaks (None), fs, duration
    """
    record     = wfdb.rdrecord(record_path)
    ann_breath = wfdb.rdann(record_path, "breath")

    fs            = record.fs
    resp_ref      = record.p_signal[:, 0]   # sygnał oddechowy (impedancja)
    ecg           = record.p_signal[:, 4]   # EKG odprowadzenie II
    breath_samples = ann_breath.sample[::2]  # ręczne adnotacje oddechów

    print("=== BIDMC ===")
    print(f"  Sygnały:        {record.sig_name}")
    print(f"  Fs:             {fs} Hz")
    print(f"  Czas nagrania:  {record.sig_len / fs:.1f} s")
    print(f"  Adnotacje oddechów: {len(breath_samples)}")

    return dict(ecg=ecg, resp_ref=resp_ref,
                breath_samples=breath_samples,
                r_peaks=None, fs=fs,
                duration=record.sig_len / fs)


# ===========================================================================
# 2. Filtracja (dwuetapowa)
# ===========================================================================

def remove_baseline(signal: np.ndarray, fs: float = 5000.0,
                    win1_ms: float = 200.0, win2_ms: float = 600.0) -> np.ndarray:
    """
    Usunięcie dryfu linii bazowej metodą dwóch filtrów medianowych.

    Podejście stosowane w metodach EDR/PCA (Clifford & Tarassenko 2005,
    Varon et al. 2020): zamiast filtru HP, dwa kolejne filtry medianowe
    wyznaczają estymację linii bazowej, która jest następnie odejmowana.

    Etap 1 — mediana 200 ms: usuwa QRS i załamek P (krótkie, ostre zdarzenia)
    Etap 2 — mediana 600 ms: usuwa załamek T (szersze zdarzenie)
    Wynik to czyste tło (baseline); odejmujemy je od oryginału.

    Zalety względem filtru HP Butterwortha:
    - brak zniekształceń fazowych w pobliżu QRS
    - lepsza ochrona morfologii na potrzeby delineacji PQRST
    """
    def _odd(n):
        return n if n % 2 == 1 else n + 1

    win1 = _odd(int(round(win1_ms / 1000.0 * fs)))
    win2 = _odd(int(round(win2_ms / 1000.0 * fs)))
    baseline = median_filter(median_filter(signal, size=win1), size=win2)
    return signal - baseline


def lowpass(signal: np.ndarray, cutoff: float = 40.0,
            fs: float = 5000.0, order: int = 4) -> np.ndarray:
    """
    Filtr dolnoprzepustowy Butterwortha (format SOS).
    Usuwa szum mięśniowy (EMG) i zakłócenia sieciowe (50/60 Hz).

    Parametry
    ----------
    cutoff : częstotliwość odcięcia [Hz]  (domyślnie 40 Hz)
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, signal)


def bandpass(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0,
             fs: float = 5000.0, order: int = 4) -> np.ndarray:
    """
    Filtr pasmowy = HP(0.5 Hz) + LP(40 Hz) — skrót dla wygody.
    Wynik identyczny z kolejnym zastosowaniem highpass i lowpass.
    """
    return lowpass(highpass(signal, lowcut, fs, order), highcut, fs, order)


def filter_frequency_response(fs: float, hp_cut: float = 0.5,
                               lp_cut: float = 40.0, order: int = 4) -> tuple:
    """
    Zwraca charakterystykę częstotliwościową filtru pasmowego (do wykresu Bode).
    Zwraca: (freqs_hz, H_hp, H_lp, H_bp)
    """
    nyq  = fs / 2.0
    npts = 8192
    f    = np.linspace(0, nyq, npts)

    sos_hp = butter(order, hp_cut / nyq, btype="high", output="sos")
    sos_lp = butter(order, lp_cut / nyq, btype="low",  output="sos")
    sos_bp = butter(order, [hp_cut / nyq, lp_cut / nyq], btype="band", output="sos")

    def sos_response(sos, worN):
        # Złożenie odpowiedzi sekcji SOS
        H = np.ones(len(worN), dtype=complex)
        for section in sos:
            b = section[:3]
            a = section[3:]
            _, h = freqz(b, a, worN=worN, fs=fs)
            H *= h
        return np.abs(H)

    w = f / fs * 2 * np.pi   # normalizacja dla freqz
    H_hp = sos_response(sos_hp, w)
    H_lp = sos_response(sos_lp, w)
    H_bp = sos_response(sos_bp, w)

    return f, H_hp, H_lp, H_bp


# ===========================================================================
# 3. Wykrywanie R-pików
# ===========================================================================

def detect_r_peaks(ecg_filt: np.ndarray, fs: float,
                   min_hr_bpm: float = 35,
                   max_hr_bpm: float = 171) -> np.ndarray:
    """
    Wykrywa R-piki na podstawie filtrowanego sygnału EKG (scipy find_peaks).

    Minimalna odległość między pikami wynika z maksymalnej HR.
    Próg amplitudy = 60 % maksimum sygnału.

    Zwraca tablicę indeksów próbek R-pików.
    """
    min_dist  = int(60.0 / max_hr_bpm * fs)   # minimalna odległość [próbki]
    threshold = 0.60 * ecg_filt.max()

    r_peaks, _ = find_peaks(ecg_filt, distance=min_dist, height=threshold)

    print(f"  Wykryte R-piki: {len(r_peaks)}")
    print(f"  Mediana RR:     {int(np.median(np.diff(r_peaks)))} próbek "
          f"({np.median(np.diff(r_peaks))/fs*1000:.0f} ms)")
    return r_peaks


# ===========================================================================
# 4. Delineacja fal EKG (P, Q, R, S, T)
# ===========================================================================

def delineate_waves(ecg_filt: np.ndarray, r_peaks: np.ndarray,
                    fs: float) -> dict:
    """
    Wyznacza pozycje fal P, Q, R, S, T dla każdego cyklu EKG.

    Metoda: przeszukiwanie okien wokół każdego R-piku.

    Okna poszukiwań (względem R):
        Q  — minimum w oknie (-100 ms,    0)
        S  — minimum w oknie (   0,  +100 ms)
        P  — maksimum w oknie (-300 ms, -100 ms)
        T  — maksimum w oknie (+100 ms, +400 ms)

    Zwraca słownik tablic indeksów próbek:
        {P, Q, R, S, T}  — każda tablica długości = liczba ważnych cykli
    Fale nieznalezione (za blisko brzegu sygnału) mają wartość -1.
    """
    win = {
        "Q": (-int(0.100 * fs),  0),
        "S": (0,                 int(0.100 * fs)),
        "P": (-int(0.300 * fs), -int(0.100 * fs)),
        "T": ( int(0.100 * fs),  int(0.400 * fs)),
    }

    P_idx = []
    Q_idx = []
    S_idx = []
    T_idx = []

    n = len(ecg_filt)

    for rp in r_peaks:
        # Q — minimum przed R
        q_start = max(0, rp + win["Q"][0])
        q_end   = max(0, rp + win["Q"][1])
        if q_end > q_start:
            Q_idx.append(q_start + int(np.argmin(ecg_filt[q_start:q_end])))
        else:
            Q_idx.append(-1)

        # S — minimum po R
        s_start = min(n, rp + win["S"][0])
        s_end   = min(n, rp + win["S"][1])
        if s_end > s_start:
            S_idx.append(s_start + int(np.argmin(ecg_filt[s_start:s_end])))
        else:
            S_idx.append(-1)

        # P — maksimum przed Q
        p_start = max(0, rp + win["P"][0])
        p_end   = max(0, rp + win["P"][1])
        if p_end > p_start:
            P_idx.append(p_start + int(np.argmax(ecg_filt[p_start:p_end])))
        else:
            P_idx.append(-1)

        # T — maksimum po S
        t_start = min(n, rp + win["T"][0])
        t_end   = min(n, rp + win["T"][1])
        if t_end > t_start:
            T_idx.append(t_start + int(np.argmax(ecg_filt[t_start:t_end])))
        else:
            T_idx.append(-1)

    return dict(
        R = r_peaks,
        Q = np.array(Q_idx),
        S = np.array(S_idx),
        P = np.array(P_idx),
        T = np.array(T_idx),
    )


# ===========================================================================
# 4. Ekstrakcja cech na cykl
# ===========================================================================

def extract_features(ecg_filt: np.ndarray, ecg_raw: np.ndarray,
                     r_peaks: np.ndarray, resp_ref: np.ndarray,
                     fs: float) -> dict:
    """
    Wyznacza 4 cechy dla każdego cyklu EKG.

    Cechy (każda jest wektorem długości N = liczba cykli):

    1. r_amplitude   — amplituda R-piku (filtrowany sygnał)
                       Oddychanie moduluje amplitudę QRS (respiratory amplitude
                       modulation, RAM).

    2. rr_interval   — odstęp RR [s]
                       Oddychanie moduluje HR (arytmia zatokowa oddechowa, RSA).

    3. qrs_area      — pole powierzchni pod |EKG| w oknie ±60 ms wokół R
                       Odzwierciedla zmiany morfologii QRS.

    4. baseline      — średnia wolnozmiennego sygnału EKG (< 0.5 Hz) w cyklu
                       Odpowiada dryfowi linii izoelektrycznej spowodowanemu
                       ruchem klatki piersiowej.

    Połączenie z SVD
    ----------------
    Te 4 cechy można ułożyć w macierz (N × 4) i wykonać SVD — szybsze, lecz
    traci się szczegóły morfologiczne. Pełna macierz cykli (N × L) dostarcza
    więcej informacji i daje lepsze rezultaty.

    Zwraca słownik: {r_amplitude, rr_interval, qrs_area, baseline, resp_per_cycle}
    """
    # Filtr dolnoprzepustowy do wyznaczenia dryfu linii bazowej (< 0.5 Hz)
    nyq      = fs / 2.0
    sos_base = butter(2, 0.5 / nyq, btype="low", output="sos")
    baseline_signal = sosfiltfilt(sos_base, ecg_raw)

    half_qrs = int(0.060 * fs)   # ±60 ms wokół R-piku

    r_amplitude   = []
    rr_interval   = []
    qrs_area      = []
    baseline_vals = []
    resp_per_cycle = []

    rr_samples = np.diff(r_peaks)

    for i in range(len(r_peaks) - 1):
        rp = r_peaks[i]

        # 1. Amplituda R
        r_amplitude.append(ecg_filt[rp])

        # 2. Odstęp RR [s]
        rr_interval.append(rr_samples[i] / fs)

        # 3. Pole pod |EKG| w oknie QRS
        qrs_start = max(0, rp - half_qrs)
        qrs_end   = min(len(ecg_filt), rp + half_qrs)
        qrs_area.append(np.trapezoid(np.abs(ecg_filt[qrs_start:qrs_end])))

        # 4. Dryf linii bazowej
        baseline_vals.append(baseline_signal[rp])

        # Referencja oddechowa (średnia w cyklu)
        end_samp = r_peaks[i + 1]
        if end_samp <= len(resp_ref):
            resp_per_cycle.append(resp_ref[rp:end_samp].mean())
        else:
            resp_per_cycle.append(np.nan)

    return dict(
        r_amplitude   = np.array(r_amplitude),
        rr_interval   = np.array(rr_interval),
        qrs_area      = np.array(qrs_area),
        baseline      = np.array(baseline_vals),
        resp_per_cycle = np.array(resp_per_cycle),
    )


# ===========================================================================
# 5. Macierz cykli (wejście dla SVD)
# ===========================================================================

def build_cycle_matrix(ecg: np.ndarray, r_peaks: np.ndarray,
                       cycle_len: int = None) -> tuple:
    """
    Buduje macierz cykli EKG X ∈ R^{N × L}.

    Każdy wiersz to jeden cykl R→R resamplowany do wspólnej długości L
    (domyślnie mediana długości RR).

    Połączenie z SVD
    ----------------
    Ta macierz jest bezpośrednim wejściem dla SVD:

        X_centered = U · Σ · Vᵀ

    Wiersze X = obserwacje (cykle), kolumny = cechy (próbki w czasie).
    Im bardziej rytmicznie zmienia się kształt cykli z powodu oddechu,
    tym wyraźniej oddychanie widoczne jest w jednej z kolumn U.

    Zwraca: (X, cycle_len, valid_idx)
    """
    rr_lengths = np.diff(r_peaks)
    if cycle_len is None:
        cycle_len = int(np.median(rr_lengths))

    cycles    = []
    valid_idx = []
    for i in range(len(r_peaks) - 1):
        seg = ecg[r_peaks[i]:r_peaks[i + 1]]
        if len(seg) < 2:
            continue
        cycles.append(resample(seg, cycle_len))
        valid_idx.append(i)

    return np.array(cycles), cycle_len, np.array(valid_idx)


# ===========================================================================
# 6. Wizualizacja analizy wstępnej
# ===========================================================================

def plot_preprocessing(ecg_raw: np.ndarray, ecg_bl: np.ndarray,
                       ecg_filt: np.ndarray, resp_ref: np.ndarray,
                       r_peaks: np.ndarray, waves: dict, fs: float,
                       title: str = "", save_dir: str = None):
    """
    Zapisuje wykresy analizy wstępnej jako osobne pliki PNG.

    Pliki wynikowe:
        01_ecg_surowy.png          — surowy sygnał EKG (10 s)
        02_filtracja_etapy.png     — 3 panele: surowy / po usunięciu baseline / końcowy
        03_detekcja_rpikow.png     — R-piki (10 s) + histogram RR
        03b_delineacja_fal.png     — zoom 1 cyklu: P, Q, R, S, T
        04_referencja_oddechowa.png — referencyjny sygnał oddechowy (60 s)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _save(fig, name):
        if save_dir:
            path = os.path.join(save_dir, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Zapisano: {path}")
        plt.close(fig)

    t_ecg = np.arange(len(ecg_filt)) / fs
    seg10 = slice(0, min(10 * int(fs), len(ecg_filt)))
    t10   = t_ecg[seg10]
    rp_10 = r_peaks[r_peaks < 10 * int(fs)]

    print(f"\nWykresy preprocessingu ({title}):")

    # -----------------------------------------------------------------------
    # 01 — Surowy sygnał EKG
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t10, ecg_raw[seg10], linewidth=0.7, color="tab:gray")
    ax.set_title(f"01: Surowy sygnał EKG (pierwsze 10 s)  —  {title}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda [mV]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "01_ecg_surowy.png")

    # -----------------------------------------------------------------------
    # 02 — Filtracja: 3 s z nakładkami pokazującymi efekt każdego etapu
    # -----------------------------------------------------------------------
    seg3  = slice(0, min(1 * int(fs), len(ecg_filt)))
    t3    = t_ecg[seg3]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"02: Filtracja EKG — usunięcie dryfu + szumu (okno 1 s)  —  {title}",
                 fontsize=12, fontweight="bold")

    # Panel 1 — surowy sygnał: widoczny dryf linii bazowej
    axes[0].plot(t3, ecg_raw[seg3], linewidth=0.9, color="tab:gray")
    axes[0].set_title("Surowy EKG — widoczny dryf linii bazowej (wolna oscylacja tła)")
    axes[0].set_ylabel("Amplituda [mV]")
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — surowy (szary) vs po usunięciu baseline (pomarańczowy)
    axes[1].plot(t3, ecg_raw[seg3], linewidth=0.9, color="tab:gray",
                 alpha=0.45, label="Surowy (przed filtracją)")
    axes[1].plot(t3, ecg_bl[seg3],  linewidth=1.1, color="tab:orange",
                 label="Po usunięciu linii bazowej (filtry medianowe 200/600 ms)")
    axes[1].set_title("Efekt usunięcia dryfu: dwa filtry medianowe (200 ms + 600 ms)")
    axes[1].set_ylabel("Amplituda [mV]")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — po usunięciu baseline (pomarańczowy) vs końcowy (niebieski)
    axes[2].plot(t3, ecg_bl[seg3],   linewidth=0.9, color="tab:orange",
                 alpha=0.45, label="Po usunięciu baseline (przed LP)")
    axes[2].plot(t3, ecg_filt[seg3], linewidth=1.1, color="tab:blue",
                 label="Po filtrze LP < 40 Hz (końcowy)")
    axes[2].set_title("Efekt filtru LP Butterwortha: wygładzenie — usunięcie szumu mięśniowego i sieciowego")
    axes[2].set_xlabel("Czas [s]")
    axes[2].set_ylabel("Amplituda [mV]")
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "02_filtracja_etapy.png")

    # -----------------------------------------------------------------------
    # 03 — Detekcja R-pików (10 s)
    # -----------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(14, 4))
    fig.suptitle(f"03: Detekcja R-pików  —  {title}",
                 fontsize=12, fontweight="bold")

    ax1.plot(t10, ecg_filt[seg10], linewidth=0.9, color="tab:blue")
    ax1.scatter(rp_10 / fs, ecg_filt[rp_10], color="red", s=40, zorder=5,
                label="R-piki")
    ax1.set_title("Filtrowany EKG z wykrytymi R-pikami (pierwsze 10 s)")
    ax1.set_xlabel("Czas [s]")
    ax1.set_ylabel("Amplituda [mV]")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "03_detekcja_rpikow.png")

    # -----------------------------------------------------------------------
    # 03b — Delineacja fal PQRST (zoom na 1 cykl)
    # -----------------------------------------------------------------------
    # Środkowy R-pik — z dala od brzegów sygnału
    mid    = len(r_peaks) // 2
    rp_mid = r_peaks[mid]

    # Okno: 320 ms przed R do 520 ms po R (obejmuje P i T)
    w_start = max(0, rp_mid - int(0.32 * fs))
    w_end   = min(len(ecg_filt), rp_mid + int(0.52 * fs))
    t_zoom  = t_ecg[w_start:w_end] - t_ecg[w_start]   # zacznij od 0
    ecg_zoom = ecg_filt[w_start:w_end]
    amp_range = ecg_zoom.max() - ecg_zoom.min()

    wave_styles = {
        "P": ("tab:green",  "P",  "o",  70),
        "Q": ("tab:purple", "Q",  "v",  70),
        "R": ("tab:red",    "R",  "^", 100),
        "S": ("tab:brown",  "S",  "v",  70),
        "T": ("tab:orange", "T",  "o",  70),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_zoom, ecg_zoom, linewidth=1.3, color="tab:blue", zorder=1)

    # Cieniowanie QRS
    q_idx = waves["Q"]
    s_idx = waves["S"]
    q_near = q_idx[np.argmin(np.abs(q_idx - rp_mid))] if np.any(q_idx >= 0) else max(0, rp_mid - int(0.04*fs))
    s_near = s_idx[np.argmin(np.abs(s_idx - rp_mid))] if np.any(s_idx >= 0) else min(len(ecg_filt)-1, rp_mid + int(0.04*fs))
    ax.axvspan((q_near - w_start) / fs, (s_near - w_start) / fs,
               alpha=0.12, color="tab:red", label="_nolegend_", zorder=0)

    # Punkty fal + adnotacje
    for wave, (color, lbl, marker, size) in wave_styles.items():
        if wave == "R":
            idx = rp_mid
        else:
            arr = waves[wave]
            valid_arr = arr[arr >= 0]
            if len(valid_arr) == 0:
                continue
            idx = valid_arr[np.argmin(np.abs(valid_arr - rp_mid))]
        if idx < w_start or idx >= w_end:
            continue
        x_pos = (idx - w_start) / fs
        y_pos = ecg_filt[idx]
        ax.scatter([x_pos], [y_pos], color=color, s=size, zorder=5,
                   marker=marker, edgecolors="white", linewidths=0.6)
        # etykieta nad lub pod punktem
        above = lbl in ("P", "R", "T")
        dy    = 0.12 * amp_range if above else -0.12 * amp_range
        ax.annotate(lbl,
                    xy=(x_pos, y_pos),
                    xytext=(x_pos, y_pos + dy),
                    fontsize=13, fontweight="bold", color=color, ha="center", va="bottom" if above else "top",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.9))

    # Linia bazowa
    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)

    # Oś X w milisekundach (łatwiej czytać w EKG)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*1000:.0f}"))
    ax.set_xlabel("Czas od początku okna [ms]")
    ax.set_ylabel("Amplituda [mV]")
    ax.set_title(f"03b: Delineacja fal EKG — zoom 1 cykl  —  {title}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "03b_delineacja_fal.png")

    # -----------------------------------------------------------------------
    # 04 — Referencyjny sygnał oddechowy
    # -----------------------------------------------------------------------
    seg60 = slice(0, min(60 * int(fs), len(resp_ref)))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_ecg[seg60], resp_ref[seg60], linewidth=0.9, color="tab:green")
    ax.set_title(f"04: Referencyjny sygnał oddechowy (pierwsze 60 s)  —  {title}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "04_referencja_oddechowa.png")
