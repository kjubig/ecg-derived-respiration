# EDR metodą SVD — opis metody

Moduł: `svd_edr.py`  
Analiza wyników: `analyze_results.py`  
Dane wejściowe: `../preprocessing/preprocessed/cebsdb_b001.npz`  
Wyniki: `results/svd_edr.npz`

> Etap 2 potoku EDR. Etap 1 (preprocessing wspólny) opisany w `../preprocessing/README_preprocessing.md`.

---

## Idea metody

Oddech **może wpływać na rejestrowany sygnał EKG** — zmieniając amplitudę i kształt kolejnych cykli EKG oraz sposób projekcji aktywności serca na odprowadzenia. Każde uderzenie jest trochę inne, a ta zmienność między kolejnymi cyklami niesie informację o oddechu.

SVD pozwala tę zmienność **rozłożyć na ortogonalne składowe** i zidentyfikować tę, która oscyluje z częstotliwością oddechową.

> Budowa macierzy cykli EKG i zastosowanie SVD oparte są na podejściu Guaragnella et al. (2019), gdzie po detekcji R-pików wykonywana jest segmentacja beat-to-beat i konstrukcja macierzy uderzeń (Lead Beat Matrix) poddawanej analizie SVD. W niniejszym projekcie podejście to zostało zaadaptowane do problemu EDR — wydzielania składowej oddechowej z EKG. Etapy interpolacji, filtracji i porównania z referencją są zgodne z ogólną strukturą metod EDR opisywaną przez Charlton et al. (2018) i Kozia et al. (2018).

---

## Etapy metody

### 1. Wczytanie macierzy cykli

Z pliku `.npz` wczytywana jest macierz $X \in \mathbb{R}^{N \times L}$:
- $N$ = liczba cykli RR (uderzenia serca)
- $L$ = długość cyklu w próbkach (mediana RR)

Dla CEBSDB b001: **297 × 4570** (297 uderzeń, ~914 ms każde).

---

### 2. Centrowanie macierzy

$$\tilde{X} = X - \bar{X}$$

gdzie $\bar{X}$ to uśredniony kształt morfologiczny QRS (średnia po wierszach).  
Centrowanie usuwa stałą morfologię "przeciętnego" uderzenia, pozostawiając tylko **zmienność między cyklami**.

> Centrowanie zastosowano jako adaptację metody do analizy zmienności między cyklami. Odjęcie średniego cyklu ogranicza wpływ stałej morfologii EKG i wzmacnia różnice między kolejnymi uderzeniami.

---

### 3. Rozkład SVD

$$\tilde{X} = U \cdot \Sigma \cdot V^T$$

| Symbol | Wymiar | Znaczenie |
|--------|--------|-----------|
| $U$ | $N \times K$ | Lewe wektory osobliwe — zmienność składowych między kolejnymi cyklami (1 pkt/cykl) |
| $S$ | $K$ (diag.) | Wartości osobliwe — energia/siła każdej składowej |
| $V^T$ | $K \times L$ | Prawe wektory osobliwe — wzorce kształtu w obrębie cyklu |

gdzie $K = \min(N, L)$. Dla b001: $N = 297$, $L = 4570$, więc $K = 297$.

Kolumna $U_k$ to sygnał z rozdzielczością 1 punkt/uderzenie. Jeśli oddech moduluje morfologię EKG, jedna z kolumn oscyluje z częstotliwością oddechową.

Implementacja: `scipy.linalg.svd(..., full_matrices=False, check_finite=False)`.

---

### 4. Analiza wariancji

Udział każdej składowej w całkowitej wariancji macierzy:

$$\text{var}_k = \frac{\sigma_k^2}{\sum_j \sigma_j^2} \times 100\%$$

Bez centrowania pierwsza składowa zwykle opisuje dominującą morfologię cyklu EKG. **Po centrowaniu** (jak w tym projekcie) pierwsza składowa opisuje największy wzorzec zmienności między cyklami po usunięciu średniego kształtu. Wykres słupkowy (`results/svd_skladowe.png`) pokazuje rozkład energii wszystkich składowych.

---

### 5. Składowe beat-by-beat

Dla każdej składowej $k$ analizowany jest przebieg kolumny $U[:, k]$ — sygnał opisujący, jak silnie dany wzorzec kształtu morfologicznego $V_k$ występuje w kolejnych uderzeniach serca.

---

### 6. Interpolacja do regularnej siatki czasowej

Sygnał beat-by-beat $U[:, k]$ ma jeden punkt na uderzenie serca — jest **nieregularnie próbkowany** w czasie.
Interpolacja kubiczna (`scipy.interpolate.interp1d`) przekształca go do stałej siatki:

$$f_s^{\text{interp}} = 4\text{ Hz}$$

To samo stosowane jest do referencji `resp_per_cycle`. Wartość 4 Hz daje duży zapas względem Nyquista dla górnej granicy pasma oddechowego (0.5 Hz).

> Charlton et al. (2018) opisują ogólną strukturę algorytmów EDR, w której po ekstrakcji cech beat-by-beat sygnał jest przygotowywany jako przebieg czasowy. Kozia et al. (2018) opisują schemat: detekcja QRS → sygnał EDR → przetwarzanie w paśmie oddechowym.

---

### 7. Filtracja w paśmie oddechowym

**Butterworth 4. rzędu**, pasmowo-przepustowy, format SOS (`sosfiltfilt` — zero-phase):

| Parametr | Wartość |
|---|---|
| Dolna granica | **0.0666 Hz** (~4 odd/min) |
| Górna granica | **0.5 Hz** (30 odd/min) |

Usuwa składowe zbyt wolne (np. dryft) oraz zbyt szybkie (składowe poza pasmem oddechowym) z każdej składowej SVD przed wyborem i estymacją częstości.

---

### 8. Wybór składowej EDR

Dla każdej z 20 pierwszych składowych obliczana jest korelacja Pearsona z referencyjnym sygnałem oddechowym (po interpolacji i filtracji):

$$r_k = |\text{corr}(U_k^{\text{filt}},\ \text{resp}^{\text{filt}})|$$

Wybierana jest składowa o maksymalnej korelacji:

$$k^* = \arg\max_k\, r_k$$

> SVD samo nie wskazuje, która składowa jest oddechowa. Wyboru dokonano na podstawie zgodności z referencyjnym sygnałem oddechowym — zgodnie z podejściem ewaluacyjnym opisanym w Charlton et al. (2018).

---

### 9. Korekta znaku i normalizacja

Znak składowej SVD jest umowny: $U_k$ i $V_k$ mogą być przemnożone przez $-1$, a rozkład nadal pozostaje poprawny.  
Jeśli korelacja z referencją jest ujemna, składowa jest odwracana, aby ułatwić wizualne porównanie z sygnałem oddechowym.  
Następnie sygnał jest normalizowany do przedziału $[0, 1]$.

---

### 10. Estymacja częstości oddechów

Metoda Welcha (PSD) na wybranej składowej EDR, pasmo: **0.0666–0.5 Hz** (4–30 odd/min):

$$f_{\text{resp}} = \arg\max_{f \in [0.0666,\, 0.5]} S_{xx}(f)$$

Częstość oddechów = $f_{\text{resp}} \times 60$ [odd/min].

---

## Wyniki — CEBSDB b001

| Metryka | Wartość |
|---------|---------|
| Najlepsza składowa | **SVD-4** |
| Korelacja z RESP (po filtracji) | **r = 0.671** |
| Częstość EDR | **22.5 odd/min** |
| Częstość referencyjna | **22.5 odd/min** ✓ |

---

## Pliki wyjściowe

### `results/svd_edr.npz`

| Klucz | Opis |
|-------|------|
| `edr_signal` | Surowy sygnał EDR — beat-domain kolumna $U_{k^*}$ (archiwum) |
| `edr_filt` | Sygnał EDR po interpolacji (4 Hz) i filtrze pasmowym 0.0666–0.5 Hz |
| `edr_norm` | Sygnał EDR znormalizowany $[0,1]$, faza skorygowana |
| `resp_norm` | Referencja oddechowa znormalizowana $[0,1]$ |
| `r_times` | Regularna siatka czasowa 4 Hz [s] |
| `r_times_beats` | Czasy cykli w domenie beat-by-beat [s] |
| `fs_edr` | Częstotliwość EDR (≈ HR) [Hz] |
| `best_idx` | Indeks wybranej składowej (0-based) |
| `correlations` | Korelacja każdej z 20 składowych z referencją |
| `variance_ratio` | Udział każdej składowej w wariancji [%] |
| `S` | Wartości osobliwe |
| `U` | Macierz U ($N \times K$) |
| `Vt` | Macierz $V^T$ ($K \times L$) |
| `rr_edr` | Częstość oddechów z EDR [odd/min] |
| `rr_resp` | Częstość oddechów z referencji [odd/min] |
| `final_corr` | Korelacja końcowa EDR vs RESP |

### `results/` — wykresy PNG

| Plik | Zawartość |
|------|-----------|
| `svd_skladowe.png` | Energia składowych (słupki) + EDR vs referencja |
| `01_edr_vs_referencja.png` | Porównanie EDR z referencją oddechową |
| `02_psd_oddechowe.png` | Widmo mocy (Welch) w paśmie oddechowym |
| `03_korelacje_skladowych.png` | Korelacja każdej składowej SVD z RESP |
| `04_morfologie_qrs.png` | Kształty morfologiczne (wiersze $V^T$) |

---

## Uruchomienie

```powershell
# Etap 1 (jeśli jeszcze nie uruchomiony)
cd preprocessing
& "../.venv/Scripts/python.exe" run_preprocessing.py

# Etap 2 — SVD
cd ../svd_lk
& "../.venv/Scripts/python.exe" svd_edr.py

# Etap 3 — analiza wyników
& "../.venv/Scripts/python.exe" analyze_results.py
```

---

## Literatura

- Guaragnella C. et al. (2019). *ECG Beat-to-Beat Analysis Using Singular Value Decomposition.* IEEE.
- Charlton P.H. et al. (2018). *Breathing rate estimation from the electrocardiogram and photoplethysmogram: A review.* IEEE Reviews in Biomedical Engineering, 11, 2–20.
- Kozia J. et al. (2018). *EMD-based QRS complex detection for ECG-derived respiration.*
