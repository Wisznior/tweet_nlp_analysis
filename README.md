# Predykcja zainteresowania postami w social media z użyciem metod NLP

## Opis projektu
Projekt realizowany w ramach przedmiotu MIO.
Cel: budowa modelu predykcyjnego przewidującego zainteresowanie tweetami Donalda Trumpa (mierzone liczbą retweetów)
z wykorzystaniem metod NLP.

Dane źródłowe: [Trump Tweets – Kaggle](https://www.kaggle.com/datasets/austinreese/trump-tweets)

## Struktura projektu
```
TODO
```

## Jak uruchomić projekt
1. Sklonowanie repozytorium.
2. Instalacja bibliotek: `pip install -r requirements.txt`.
3. Konfiguracja dostępu do Kaggle:
   - Zaloguj się na Kaggle → *Settings* → *API Tokens* → *Create Legacy API Key*.
   - W głównym folderze projektu stwórz plik `.env`.
   - Wpisz w nim swoje dane z pliku `.json` który pobrał się automatycznie po stworzeniu API Key:
```
     KAGGLE_USERNAME=twój_username
     KAGGLE_KEY=twój_klucz_api
```
4. Uruchom skrypt: `py main.py`.

## Pipeline
Na chwilę obecną zaimplementowane są dwa pierwsze kroki:

```
ingestion → preprocessing → modeling (TODO) → interpretation (TODO)
```

| Krok | Plik | Status |
|---|---|---|
| Pobieranie danych z Kaggle | `src/ingestion.py` | Done |
| Czyszczenie tekstu, TF-IDF i cechy metadanych | `src/preprocessing.py` | Done |
| Trening modelu | `src/modeling.py` | ... |
| Analiza SHAP | `src/interpretation.py` | ... |

## Porównanie metod reprezentacji tekstu
Osobny eksperyment porównujący trzy reprezentacje tekstu — Bag of Words (`CountVectorizer`),
TF-IDF unigramy oraz TF-IDF unigramy+bigramy — na każdym z modeli klasyfikacyjnych.
Wszystkie reprezentacje korzystają z tego samego podziału train/test, a wektoryzator jest
dopasowywany wyłącznie na zbiorze treningowym. Wynikiem jest siatka accuracy + F1 dla każdej
kombinacji reprezentacja × model.

Uruchomienie (wymaga wcześniejszego `py main.py`, który buduje bazę z danymi):
```
python -m src.representation_comparison
```
Tabela zapisywana jest do `data/representation_comparison.csv`.

## Cechy metadanych
Główny pipeline (`py main.py`) rozszerza reprezentację tekstu o cechy metadanych tweeta —
długość, liczbę słów, godzinę publikacji, dzień tygodnia oraz flagę obecności URL. Cechy te są
skalowane i łączone z macierzą TF-IDF (`scipy.sparse.hstack`); odpowiada za to parametr
`use_metadata` w `prepare_data` (domyślnie wyłączony, w `main.py` włączony).

Osobny skrypt analityczny porównuje F1 modeli na samym TF-IDF względem TF-IDF + metadane oraz
sprawdza analizą SHAP, jak ważność cech metadanych wypada na tle pojedynczych słów:
```
python -m src.metadata_features
```
Tabela porównawcza zapisywana jest do `data/metadata_comparison.csv`, a wykres ważności SHAP
do `data/plots/metadata_shap_bar.png`.

## Zespół realizujący
Rafał Wiszniowski, Bartosz Amalio, Marcin Oracz
