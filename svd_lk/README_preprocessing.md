# Preprocessing EKG — opis etapów

Moduł: `preprocessing.py`  
Skrypt wykonawczy: `run_preprocessing.py`  
Wynik: `preprocessed/cebsdb_b001.npz`

---

## Dane wejściowe

Baza **CEBSDB** (Combined Effort Breathing and Sensor Data Base), rekord `b001`.

- 4 kanały: `I`, `II`, `RESP`, `SCG`
- Częstotliwość próbkowania: **5000 Hz**
- Czas nagrania: **272 s** (~4,5 min)
- Adnotacje R-pików z pliku `.atr` (co drugi wpis — brane `[::2]`)
- Do analizy używamy odprowadzenia **II** (najlepsza widoczność QRS)

---

## Etapy preprocessingu

### 1. Wczytanie sygnału
Odczyt rekordu WFDB, ekstrakcja odprowadzenia II i referencyjnego sygnału oddechowego (kanał RESP). R-piki wczytywane z adnotacji `.atr`.

---

### 2. Usunięcie dryfu linii bazowej

Metoda: **dwa kolejne filtry medianowe** (Clifford & Tarassenko 2005, Varon et al. 2020).

| Filtr | Okno | Usuwa |
|-------|------|-------|
| Medianowy 1 | **200 ms** (1000 próbek) | QRS i załamek P — krótkie, ostre zdarzenia |
| Medianowy 2 | **600 ms** (3000 próbek) | Załamek T — szersze zdarzenie |

Wynik filtrów stanowi estymację linii bazowej, która jest **odejmowana** od sygnału.

> Dlaczego nie HP Butterworth?  
> Filtry medianowe nie powodują zniekształceń fazowych w pobliżu QRS, co lepiej chroni morfologię sygnału potrzebną do delineacji fal i analizy SVD.

---

### 3. Filtr dolnoprzepustowy LP 40 Hz

**Butterworth 4. rzędu**, `fc = 40 Hz`, format SOS (`sosfiltfilt` — zero-phase).

Usuwa szum mięśniowy (EMG) i zakłócenia sieciowe (50 Hz i harmoniczne).  
Górna granica 40 Hz zachowuje pełną morfologię QRS istotną dla EDR.

---

### 4. Detekcja R-pików

R-piki wczytywane z adnotacji `.atr` (baza CEBSDB dostarcza ręcznie zweryfikowane adnotacje).  
Dla nagrań bez adnotacji: `scipy.signal.find_peaks` z progiem 60% amplitudy i minimalną odległością 200 ms.

Wyniki dla b001: **298 R-pików**, mediana RR = **914 ms** (65 uderzeń/min).

---

### 5. Delineacja fal PQRST

Dla każdego R-piku, metodą min/max w oknie czasowym:

| Fala | Okno względem R | Kryterium |
|------|-----------------|-----------|
| **Q** | (R − 100 ms, R) | minimum |
| **S** | (R, R + 100 ms) | minimum |
| **P** | (R − 300 ms, R − 100 ms) | maksimum |
| **T** | (R + 100 ms, R + 400 ms) | maksimum |

---

### 6. Budowa macierzy cykli X

Każdy cykl EKG (od R-piku do kolejnego R-piku) jest **resamplowany** do mediany długości RR, tworząc macierz:

$$X \in \mathbb{R}^{N \times L}$$

gdzie $N$ = liczba cykli, $L$ = mediana długości RR w próbkach.

Wyniki dla b001: **297 × 4570** (297 cykli, 914 ms każdy).

Macierz X jest bezpośrednim wejściem dla metod SVD / PCA / ICA.

---

## Wyniki — pliki wyjściowe

### `preprocessed/cebsdb_b001.npz`

| Klucz | Opis |
|-------|------|
| `ecg_raw` | Surowy sygnał EKG (odprowadzenie II) |
| `ecg_bl` | Po usunięciu linii bazowej (filtry medianowe) |
| `ecg_filt` | Końcowy sygnał (po LP 40 Hz) |
| `r_peaks` | Indeksy R-pików [próbki] |
| `resp_ref` | Referencyjny sygnał oddechowy |
| `fs` | Częstotliwość próbkowania [Hz] |
| `X` | Macierz cykli N×L — wejście dla SVD/PCA/ICA |
| `cycle_len` | Długość cyklu [próbki] |
| `valid_idx` | Indeksy użytych cykli |

### `preprocessing/` — wykresy PNG

| Plik | Zawartość |
|------|-----------|
| `01_ecg_surowy.png` | Surowy EKG (pierwsze 10 s) |
| `02_filtracja_etapy.png` | Trzy panele: surowy / po usunięciu baseline / końcowy (okno 1 s) |
| `03_detekcja_rpikow.png` | EKG z zaznaczonymi R-pikami (pierwsze 10 s) |
| `03b_delineacja_fal.png` | Zoom 1 cyklu z delineacją P, Q, R, S, T |
| `04_referencja_oddechowa.png` | Referencyjny sygnał oddechowy (pierwsze 60 s) |

---

## Uruchomienie

```powershell
cd svd_lk
& "../.venv/Scripts/python.exe" run_preprocessing.py
```

---

## Literatura

- Clifford G.D., Tarassenko L. (2005). *Quantifying errors in spectral estimates of HRV due to beat replacement and resampling.* IEEE Trans. Biomed. Eng.
- Varon C. et al. (2020). *A comparative study of ECG-derived respiration techniques.* Medical & Biological Engineering & Computing.
- Pan J., Tompkins W.J. (1985). *A real-time QRS detection algorithm.* IEEE Trans. Biomed. Eng.
