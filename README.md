# Predykcja zainteresowania postami w social media z użyciem metod NLP

## Opis projektu
Projekt realizowany w ramach przedmiotu MIO.  
Cel: budowa modelu predykcyjnego przewidującego zainteresowanie tweetami Donalda Trumpa (mierzone liczbą retweetów)
z wykorzystaniem metod NLP.

Dane źródłowe: [Trump Tweets – Kaggle](https://www.kaggle.com/datasets/austinreese/trump-tweets)

## Struktura projektu
```
tweet_nlp_analysis/
├── main.py
├── requirements.txt
├── .env.example
├── src/
│   ├── ingestion.py                 # Pobieranie danych z Kaggle
│   ├── database.py                  # Zapis/odczyt SQLite
│   ├── preprocessing.py             # Czyszczenie tekstu, TF-IDF, cechy metadanych
│   ├── modeling.py                  # Trening i ewaluacja modeli
│   ├── interpretation.py            # Analiza SHAP
│   ├── temporal_analysis.py         # Analiza trendu RT w czasie + F1 stare vs nowe tweety
│   ├── metadata_features.py         # Porównanie TF-IDF vs TF-IDF + metadane
│   ├── representation_comparison.py # Porównanie BoW / TF-IDF unigramy / bigramy
│   └── utils/
│       └── logger.py
└── data/                            # Generowany automatycznie po uruchomieniu
    ├── trump_tweets.sqlite
    ├── tfidf_vectorizer.pkl
    ├── best_model.pkl
    ├── shap_model.pkl
    ├── test_data.pkl
    ├── plots/
    │   ├── summary_plot.png
    │   ├── bar_plot.png
    │   ├── waterfall_plot.png
    │   ├── metadata_shap_bar.png
    │   └── rt_per_month.png
    ├── temporal_f1_comparison.csv
    ├── metadata_comparison.csv
    └── representation_comparison.csv
```

## Jak uruchomić projekt
1. Sklonuj repozytorium.
2. Zainstaluj biblioteki: `pip install -r requirements.txt`
3. Skonfiguruj dostęp do Kaggle:
   - Zaloguj się na Kaggle → *Settings* → *API Tokens* → *Create Legacy API Key*
   - W głównym folderze projektu stwórz plik `.env`
   - Wpisz w nim swoje dane z pliku `.json`, który pobrał się automatycznie po stworzeniu API Key:
     ```
     KAGGLE_USERNAME=twój_username
     KAGGLE_KEY=twój_klucz_api
     ```
4. Uruchom główny pipeline: `python main.py`

## Pipeline

```
ingestion → preprocessing → modeling → interpretation → temporal analysis
```

| Krok | Plik | Opis |
|------|------|------|
| Pobieranie danych z Kaggle | `src/ingestion.py` | Kaggle API → CSV → SQLite |
| Czyszczenie tekstu, TF-IDF, cechy metadanych | `src/preprocessing.py` | Czyszczenie, wektoryzacja, parsowanie daty |
| Trening i ewaluacja modeli | `src/modeling.py` | LR, Naive Bayes, Random Forest, XGBoost |
| Analiza SHAP | `src/interpretation.py` | Summary plot, bar plot, waterfall plot |
| Analiza temporalna | `src/temporal_analysis.py` | Trend RT/miesiąc, F1 stare vs nowe tweety |

## Analiza temporalna
Pipeline analizuje jak popularność tweetów zmieniała się w czasie oraz czy model degraduje się
na nowszych danych (concept drift).

Wyniki generowane automatycznie przez `python main.py`:
- `data/plots/rt_per_month.png` — wykres średniej liczby RT per miesiąc
- `data/temporal_f1_comparison.csv` — F1 macro per model na tweetach starszych vs nowszych (podział wg roku)
- wnioski o trendzie i dryfcie logowane do `data/project_run.log`

Uruchomienie oddzielnie (wymaga wcześniejszego `python main.py`, który buduje bazę z danymi):
```
python -m src.temporal_analysis
```

## Porównanie metod reprezentacji tekstu
Eksperyment porównujący trzy reprezentacje tekstu — Bag of Words (`CountVectorizer`),
TF-IDF unigramy oraz TF-IDF unigramy+bigramy — na każdym z modeli klasyfikacyjnych.
Wszystkie reprezentacje korzystają z tego samego podziału train/test, a wektoryzator jest
dopasowywany wyłącznie na zbiorze treningowym. Wynikiem jest siatka accuracy + F1 dla każdej
kombinacji reprezentacja × model.

Uruchomienie (wymaga wcześniejszego `python main.py`):
```
python -m src.representation_comparison
```
Tabela zapisywana jest do `data/representation_comparison.csv`.

## Cechy metadanych
Główny pipeline rozszerza reprezentację tekstu o cechy metadanych tweeta —
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