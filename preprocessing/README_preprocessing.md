# Preprocessing EKG — opis etapów

Moduł: `preprocessing.py`  
Skrypt wykonawczy: `run_preprocessing.py`  
Wynik: `preprocessed/cebsdb_b001.npz`

> Wspólny preprocessing dla wszystkich metod: **SVD, PCA, ICA**.  
> Każda metoda wczytuje ten sam plik `.npz` z identycznej macierzy cykli X.

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

Metoda: **dwa kolejne filtry medianowe** (Charlton et al. 2018).

Filtr medianowy zastępuje każdą próbkę medianą z okna o zadanej szerokości. Kluczowa własność: jeśli okno jest **szersze niż trwające zdarzenie**, filtr go nie śledzi — wynik pozostaje zbliżony do linii bazowej bez tego zdarzenia.

**Krok 1 — filtr 200 ms (1000 próbek):**

Zespół QRS trwa ok. 80–120 ms, załamek P ok. 80–120 ms. Oba mieszczą się w całości wewnątrz okna 200 ms. Filtr medianowy nie podąża za tak krótkimi, ostrymi zdarzeniami — wynik zawiera linię bazową z załamkiem T, ale bez QRS i P.

**Krok 2 — filtr 600 ms (3000 próbek):**

Załamek T trwa ok. 150–400 ms. Po usunięciu QRS i P w kroku 1, drugi filtr z oknem 600 ms eliminuje również T — wynik to już tylko wolny dryft linii bazowej (ruch oddechowy, artefakty elektrodowe).

| Filtr | Okno | Eliminuje z sygnału |
|-------|------|----------------------|
| Medianowy 1 | **200 ms** (1000 próbek) | QRS i załamek P |
| Medianowy 2 | **600 ms** (3000 próbek) | Załamek T |
| **Wynik** | — | estymacja dryfu linii bazowej |

Estymacja dryfu jest następnie **odejmowana** od oryginalnego sygnału:

$$\text{EKG}_{\text{bl}} = \text{EKG}_{\text{raw}} - \text{median}_2\bigl(\text{median}_1(\text{EKG}_{\text{raw}})\bigr)$$

> Charlton et al. opisują usuwanie bardzo wolnych składowych przez filtr medianowy lub odjęcie trendu jako standardowy etap poprzedzający ekstrakcję cech oddechowych z EKG.
> Filtry medianowe nie powodują zniekształceń fazowych w pobliżu QRS, co lepiej chroni morfologię sygnału potrzebną do delineacji fal.

---

### 3. Filtr dolnoprzepustowy LP 40 Hz

**Butterworth 4. rzędu**, `fc = 40 Hz`, format SOS (`sosfiltfilt` — zero-phase).

Po usunięciu dryfu sygnał może nadal zawierać:
- **Szum mięśniowy (EMG)** — wysoka częstotliwość, 20–500 Hz, losowy charakter
- **Zakłócenia sieciowe 50 Hz** i harmoniczne (100 Hz, 150 Hz, ...)
- **Szum kwantyzacji** i inne artefakty wysokoczzęstotliwościowe

Filtr Butterwortha 4. rzędu z `fc = 40 Hz` odcina wszystko powyżej, zachowując jednocześnie pełną zawartość spektralną QRS (dominujące częstotliwości 10–40 Hz).

Użycie formatu **SOS** (`sosfiltfilt`) zamiast tradycyjnych współczynników b/a zapobiega niestabilności numerycznej przy wysokich rzędach filtra i niskich częstotliwościach odcięcia względem fs. `sosfiltfilt` stosuje filtr dwa razy (w przód i tył) — efekt zero-phase, brak przesunięcia fazowego.

> Charlton et al. wymieniają 40 Hz jako typową częstotliwość odcięcia LP dla metod EDR opartych na cechach beat-by-beat, zapewniającą zachowanie zawartości zespołu QRS.

---

### 4. Detekcja R-pików

R-piki wczytywane z adnotacji `.atr` (baza CEBSDB dostarcza ręcznie zweryfikowane adnotacje).  
Dla nagrań bez adnotacji: `scipy.signal.find_peaks` z progiem 60% amplitudy i minimalną odległością 200 ms.

> Charlton et al. wskazują detekcję R-pików jako kluczowy etap poprzedzający segmentację beat-by-beat i ekstrakcję cech oddechowych.

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

> Charlton et al. opisują resampling nieregularnego przebiegu beat-by-beat do stałej siatki czasowej jako standardowy krok umożliwiający analizę częstotliwościową. Guaragnella et al. (2019) stosują analogiczną segmentację beat-to-beat i konstrukcję macierzy uderzeń (Lead Beat Matrix) bezpośrednio przed SVD.

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
cd preprocessing
& "../.venv/Scripts/python.exe" run_preprocessing.py
```

---

## Literatura

- Charlton P.H. et al. (2018). *Breathing rate estimation from the electrocardiogram and photoplethysmogram: A review.* IEEE Reviews in Biomedical Engineering, 11, 2–20.
- Guaragnella C. et al. (2019). *ECG Beat-to-Beat Analysis Using Singular Value Decomposition.* (IEEE)
- Kozia J. et al. (2018). *EMD-based QRS complex detection for ECG-derived respiration.*
