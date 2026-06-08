import logging
import time

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.database import DatabaseManager
from src.preprocessing import load_and_clean, build_vectorizer
from src.modeling import MODELS

logger = logging.getLogger(__name__)

REPRESENTATIONS = [
    {"name": "BoW (CountVectorizer)", "method": "bow", "ngram_range": (1, 1)},
    {"name": "TF-IDF unigrams", "method": "tfidf", "ngram_range": (1, 1)},
    {"name": "TF-IDF unigrams+bigrams", "method": "tfidf", "ngram_range": (1, 2)},
]


def compare_representations(
    texts: pd.Series,
    y: pd.Series,
    max_features: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Shared split - train: {len(texts_train)}, test: {len(texts_test)}")

    results = []
    for rep in REPRESENTATIONS:
        logger.info(f"=== Representation: {rep['name']} ===")

        vectorizer = build_vectorizer(rep["method"], rep["ngram_range"], max_features)
        X_train = vectorizer.fit_transform(texts_train)
        X_test = vectorizer.transform(texts_test)
        n_features = X_train.shape[1]
        logger.info(f"  features: {n_features}")

        for model_name, base_model in MODELS.items():
            model = clone(base_model)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred)
            logger.info(
                f"  {model_name}: accuracy={acc:.4f}, f1_macro={f1:.4f}, "
                f"train_time={train_time:.1f}s"
            )
            logger.info(f"  confusion matrix:\n{cm}")

            results.append({
                "representation": rep["name"],
                "model": model_name,
                "n_features": n_features,
                "accuracy": acc,
                "f1_macro": f1,
                "train_time_s": round(train_time, 1),
            })

    df = pd.DataFrame(results)

    pivot = df.pivot(index="representation", columns="model", values="f1_macro")
    logger.info(f"\nF1 (macro) - representation x model:\n{pivot.to_string()}")

    best = df.loc[df["f1_macro"].idxmax()]
    logger.info(
        f"Best combination: {best['representation']} + {best['model']} "
        f"(F1={best['f1_macro']:.4f})"
    )
    return df


def run_representation_comparison(
    db_path: str = "data/trump_tweets.sqlite",
    output_csv: str = "data/representation_comparison.csv",
) -> pd.DataFrame:
    db_mgr = DatabaseManager(db_path)
    df, y = load_and_clean(db_mgr.get_engine())

    result = compare_representations(df["clean"], y)

    result.to_csv(output_csv, index=False)
    logger.info(f"Comparison table saved to {output_csv}")
    return result


if __name__ == "__main__":
    from src.utils.logger import setup_logging

    setup_logging(log_level=logging.INFO, log_file="data/representation_comparison.log")
    run_representation_comparison()
