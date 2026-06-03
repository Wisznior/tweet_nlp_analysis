import os
import logging
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)

PLOTS_DIR = "data/plots"
CLASS_NAMES = ["low (0)", "medium (1), high (2)"]

def load_artifacts( 
    model_path: str = "data/plots",
    test_data_path: str = "data/test_data.pkl"
):
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading test data from {test_data_path}")
    X_test, y_test, feature_names = joblib.load(test_data_path)

    return model, X_test, y_test, feature_names

def compute_shap_values(model, X_sample, feature_names):
    logger.info("Computing SHAP values")

    model_type = type(model).__name__

    if model_type in ("RandomForestClassifier" ):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    elif model_type in ( "LogisticRegression", "LinearSVC"):
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    else:
        # fallback
        logger.warning(f"Unknown model type {model_type}, using KernelExplainer as fallback ( might be slow )")
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 100))
    
    logger.info("SHAP values computed.")
    return explainer, shap_values