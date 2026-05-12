# TODO — Preprocessing (do przeanalizowania)

## 1. Odrzucenie cykli ektopowych / artefaktów
**Źródło:** Clifford & Tarassenko 2005, Varon et al. 2020

Cykle z RR odchylającym się >20% od mediany to skurcze dodatkowe lub artefakty ruchu.

**Problem bez tego kroku:**
- SVD/PCA: outlier dominuje pierwsze składowe i zaburza macierz kowariancji
- ICA: zakłada stacjonarność rozkładów — jeden artefakt może zepsuć zbieżność

**Propozycja implementacji:**
```python
rr = np.diff(r_peaks)
med_rr = np.median(rr)
valid = np.where(np.abs(rr - med_rr) / med_rr <= 0.20)[0]
# valid_idx — indeksy do zachowania przy budowie X
```
Opcja A — **usunięcie** cyklu (prosto, tracimy ciągłość)  
Opcja B — **interpolacja liniowa** sąsiednich cykli (Clifford & Tarassenko — zachowuje ciągłość)

Dla b001: sprawdzić ile cykli by wypadło (spodziewamy się ~0 bo zdrowy pacjent).

---

## 2. Centrowanie macierzy — odjęcie średniego cyklu
**Źródło:** Varon et al. 2020

```python
X_cent = X - X.mean(axis=0)
```

**Dlaczego ważne:**
- Bez centrowania SVD-1 (~68% wariancji) opisuje głównie *średni kształt QRS*, a nie zmienność między cyklami
- Po centrowaniu SVD skupia się na *różnicach* między uderzeniami — czyli właśnie na modulacji oddechowej
- PCA wymaga centrowania z definicji
- FastICA (ICA) robi to wewnętrznie, ale lepiej kontrolować jawnie

**Uwaga:** NIE robić normalizacji wariancji (z-score per cykl) — niszczyłaby cechy amplitudowe (RAM) istotne dla EDR.

**Spodziewany efekt dla SVD:** sygnał oddechowy przesunie się bliżej SVD-1/SVD-2, korelacja z referencją powinna wzrosnąć (aktualnie r=0.542 na SVD-4).

---

## 3. Zapis X vs X_cent do .npz
Jeśli dodamy centrowanie — warto zapisać obie wersje:
- `X` — oryginalna macierz (dla metod które same centrują, np. ICA)
- `X_cent` — wycentrowana (dla SVD/PCA)

Albo parametr przy `build_cycle_matrix(center=True/False)`.

---

## 4. Do sprawdzenia z zespołem
- [ ] Czy PCA/ICA kolegów też korzysta z `preprocessing.py`? Czy centrowanie im nie zaszkodzi?
- [ ] Czy odrzucenie ektopowych ma sens dla CEBSDB (zdrowi pacjenci — mało artefaktów)?
- [ ] Po dodaniu centrowania — powtórzyć SVD i porównać korelacje
