import os
import logging

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from scipy.sparse import hstack, csr_matrix

from src.database import DatabaseManager
from src.preprocessing import load_and_clean, build_vectorizer, build_metadata_features
from src.modeling import MODELS
from src.interpretation import plot_bar

logger = logging.getLogger(__name__)

PLOTS_DIR = "data/plots"


def shap_metadata_importance(model, X_test_combined, feature_names, meta_names, sample_size, plot_path):
    np.random.seed(42)
    idx = np.random.choice(X_test_combined.shape[0], size=min(sample_size, X_test_combined.shape[0]), replace=False)
    X_sample = X_test_combined[idx].toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, approximate=True, check_additivity=False)

    sv = np.array(shap_values)
    if sv.ndim == 3:
        mean_abs = np.abs(sv).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(sv).mean(axis=0)
    mean_abs = mean_abs.flatten()

    order = list(np.argsort(mean_abs)[::-1])
    total = len(feature_names)

    logger.info("SHAP importance of metadata features (mean |SHAP|, rank among all features):")
    ranks = {}
    for name in meta_names:
        i = feature_names.index(name)
        rank = order.index(i) + 1
        ranks[name] = rank
        logger.info(f"  {name}: mean|SHAP|={mean_abs[i]:.5f}, rank {rank}/{total}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_bar(sv, feature_names, plot_path)
    return ranks


def compare_with_metadata(
    df: pd.DataFrame,
    y: pd.Series,
    max_features: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
    shap_sample: int = 200,
) -> pd.DataFrame:
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Shared split - train: {len(df_train)}, test: {len(df_test)}")

    vectorizer = build_vectorizer("tfidf", (1, 2), max_features)
    Xtr_tfidf = vectorizer.fit_transform(df_train["clean"])
    Xte_tfidf = vectorizer.transform(df_test["clean"])
    word_names = list(vectorizer.get_feature_names_out())
    logger.info(f"TF-IDF features: {Xtr_tfidf.shape[1]}")

    Xtr_meta_raw, meta_names = build_metadata_features(df_train)
    Xte_meta_raw, _ = build_metadata_features(df_test)
    scaler = MinMaxScaler()
    Xtr_meta = scaler.fit_transform(Xtr_meta_raw)
    Xte_meta = scaler.transform(Xte_meta_raw)

    Xtr_combined = hstack([Xtr_tfidf, csr_matrix(Xtr_meta)]).tocsr()
    Xte_combined = hstack([Xte_tfidf, csr_matrix(Xte_meta)]).tocsr()
    combined_names = word_names + meta_names
    logger.info(f"Combined features: {Xtr_combined.shape[1]} (TF-IDF + {len(meta_names)} metadata)")

    results = []
    trained_combined = {}
    for model_name, base_model in MODELS.items():
        m_tfidf = clone(base_model)
        m_tfidf.fit(Xtr_tfidf, y_train)
        f1_tfidf = f1_score(y_test, m_tfidf.predict(Xte_tfidf), average="weighted")

        m_combined = clone(base_model)
        m_combined.fit(Xtr_combined, y_train)
        f1_combined = f1_score(y_test, m_combined.predict(Xte_combined), average="weighted")
        trained_combined[model_name] = m_combined

        delta = f1_combined - f1_tfidf
        logger.info(f"  {model_name}: F1 TF-IDF={f1_tfidf:.4f}, F1 combined={f1_combined:.4f}, delta={delta:+.4f}")

        results.append({
            "model": model_name,
            "f1_tfidf": f1_tfidf,
            "f1_combined": f1_combined,
            "delta": round(delta, 4),
        })

    df_res = pd.DataFrame(results).sort_values("f1_combined", ascending=False).reset_index(drop=True)
    logger.info(f"\nMetadata feature comparison (weighted F1):\n{df_res.to_string(index=False)}")

    shap_metadata_importance(
        trained_combined["random_forest"],
        Xte_combined,
        combined_names,
        meta_names,
        shap_sample,
        os.path.join(PLOTS_DIR, "metadata_shap_bar.png"),
    )

    return df_res


def run_metadata_comparison(
    db_path: str = "data/trump_tweets.sqlite",
    output_csv: str = "data/metadata_comparison.csv",
) -> pd.DataFrame:
    db_mgr = DatabaseManager(db_path)
    df, y = load_and_clean(db_mgr.get_engine())

    result = compare_with_metadata(df, y)

    result.to_csv(output_csv, index=False)
    logger.info(f"Comparison table saved to {output_csv}")
    return result


if __name__ == "__main__":
    from src.utils.logger import setup_logging

    setup_logging(log_level=logging.INFO, log_file="data/metadata_features.log")
    run_metadata_comparison()
