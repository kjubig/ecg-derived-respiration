# Oddychanie wyznaczone z EKG

Repozytorium zawiera projekt studencki poświęcony wyznaczaniu składowej oddechowej z sygnałów EKG.

## Temat projektu

**Ekstrakcja składowej oddechowej z EKG przy użyciu metod dekompozycji macierzowej.**

Główna idea polega na tym, że EKG nie mierzy oddychania bezpośrednio, jednak oddychanie może wpływać na sygnał EKG. Zmiany te mogą przejawiać się w amplitudzie, kształcie, położeniu linii izoelektrycznej lub odstępach czasowych kolejnych cykli EKG.

## Cel

Celem projektu jest estymacja sygnału oddechowego na podstawie zapisów EKG oraz porównanie go z referencyjnym sygnałem oddechu, jeśli jest dostępny.

## Metody

W projekcie zostaną zbadane metody dekompozycji macierzowej, takie jak:

- PCA,
- SVD,
- ICA.

Metody te mogą pomóc zidentyfikować wzorce zmienności w sygnałach EKG, które mogą być związane z oddychaniem.

## Zbiór danych

W projekcie planowane jest wykorzystanie ogólnodostępnych baz sygnałów fizjologicznych, takich jak baza Fantasia Database z serwisu PhysioNet.

W tego typu zbiorze danych:

- EKG jest używane jako sygnał wejściowy,
- sygnał oddechowy może służyć jako odniesienie do porównań.

## Ogólny schemat pracy

1. Wczytanie danych EKG i oddechowych.
2. Przetwarzanie wstępne sygnału EKG.
3. Wykrycie lub użycie zanotowanych uderzeń serca.
4. Podział EKG na kolejne cykle.
5. Zastosowanie metod dekompozycji macierzowej.
6. Ekstrakcja składowej związanej z oddychaniem.
7. Porównanie wyniku z referencyjnym sygnałem oddechowym.

## Oczekiwany wynik

Oczekiwanym rezultatem jest sygnał wyznaczony z EKG, który odzwierciedla aktywność oddechową osoby badanej.

Wynik można ocenić poprzez porównanie z referencyjnym sygnałem oddechowym lub przez estymację rytmu oddechowego.

## Autorzy

Wiktoria Jankowska, Łukasz Kubik, Jan Durawa

