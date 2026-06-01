import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from scipy.sparse import spmatrix

logger = logging.getLogger(__name__)

MODELS: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "naive_bayes": MultinomialNB(),
    "linear_svc": LinearSVC(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
}


def train_model(X_train: spmatrix, y_train: pd.Series, model_name: str) -> Any:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}")

    model = MODELS[model_name]
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"{model_name} training completed.")
    return model


def evaluate_model(model: Any, X_test: spmatrix, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
    logger.info(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"{model_name} - Accuracy: {acc:.4f}, F1 (weighted): {f1:.4f}")

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def train_and_evaluate_all(
    X_train: spmatrix, X_test: spmatrix, y_train: pd.Series, y_test: pd.Series
) -> tuple:
    results = []
    trained_models = {}

    for model_name in MODELS:
        try:
            model = train_model(X_train, y_train, model_name)
            trained_models[model_name] = model
            metrics = evaluate_model(model, X_test, y_test, model_name)
            results.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
            })
        except Exception as e:
            logger.error(f"Failed to train/evaluate {model_name}: {e}", exc_info=True)
            results.append({
                "model": model_name,
                "accuracy": np.nan,
                "f1_weighted": np.nan,
            })

    df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False).reset_index(drop=True)
    logger.info(f"\nResults:\n{df.to_string(index=False)}")
    return df, trained_models


def save_model(model: Any, path: str) -> None:
    logger.info(f"Saving model to {path}")
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str) -> Any:
    logger.info(f"Loading model from {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
