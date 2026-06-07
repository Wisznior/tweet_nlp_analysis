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
        X_dense = X_sample.toarray() if hasattr(X_sample, "toarray") else X_sample
        shap_values = explainer.shap_values(X_dense, approximate=True, check_additivity=False)
    elif model_type in ( "LogisticRegression", "LinearSVC"):
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    else:
        # fallback
        logger.warning(f"Unknown model type {model_type}, using KernelExplainer as fallback ( might be slow )")
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 2000))
    
    logger.info("SHAP values computed.")
    return explainer, shap_values


def plot_summary(shap_values, X_sample, feature_names, output_path: str):
    logger.info("Generating summary plot")

    sv = np.array(shap_values)

    if sv.ndim == 3:
        sv_class2 = sv[:, :, 2]
    else:
        sv_class2 = sv

    if hasattr(X_sample, "toarray"):
        X_dense = X_sample.toarray()
    else:
        X_dense = X_sample

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv_class2,
        X_dense,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP Summary plot - high retweets")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Summary plot saved to {output_path}")

def plot_bar(shap_values, feature_names, output_path: str, top_n: int = 20):
    logger.info("Generating bar plot")
    
    sv = np.abs(shap_values)

    if sv.ndim == 3:
        mean_abs = np.abs(sv).mean(axis=(0,2)) #średnia po klasach i samples 
    else:
        mean_abs = np.abs(sv).mean(axis=0)
    
    mean_abs = mean_abs.flatten()

    indices = np.argsort(mean_abs)[-top_n:][::-1]
    top_features = [str(feature_names[i]) for i in indices]
    top_values = mean_abs[indices]
        
    plt.figure(figsize=(10, 8))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.title(f"Top {top_n} most important words")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(output_path, dpi = 150, bbox_inches="tight")
    plt.close()
    logger.info(f"Bar plot saved to {output_path}")

def plot_waterfall(explainer, shap_values, X_sample, feature_names, output_path: str, sample_idx: int = 0):
    logger.info(f"Generating waterfall plot for sample index {sample_idx}...")

    sv = np.array(shap_values)

    if sv.ndim == 3:
        sv_single = sv[sample_idx, :, 2]
        expected = explainer.expected_value[2] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
    else:
        sv_single = sv[sample_idx]
        expected = explainer.expected_value if not hasattr(explainer.expected_value, '__len__') else explainer.expected_value[0]

    shap_exp = shap.Explanation(
        values=sv_single.flatten(),
        base_values=float(expected),
        feature_names=list(feature_names)
    )

    plt.figure()
    shap.plots.waterfall(shap_exp, max_display=15, show=False)
    plt.title(f"Waterfall plot — sample #{sample_idx} (class: high retweets)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Waterfall plot saved to {output_path}")


def run_interpretation( model_path: str = "data/shap_model.pkl", test_data_path: str = "data/test_data.pkl", sample_size: int = 500 ):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model, X_test, y_test, feature_names = load_artifacts(model_path, test_data_path)

    logger.info(f"Sampling {sample_size} rows for SHAP analysis...")
    np.random.seed(42)
    idx = np.random.choice(X_test.shape[0], size=min(sample_size, X_test.shape[0]), replace=False)
    X_sample = X_test[idx]

    explainer, shap_values = compute_shap_values(model, X_sample, feature_names)

    plot_summary(shap_values, X_sample, feature_names, os.path.join(PLOTS_DIR, "summary_plot.png"))
    plot_bar(shap_values, feature_names, os.path.join(PLOTS_DIR, "bar_plot.png"))
    plot_waterfall(explainer, shap_values, X_sample, feature_names, os.path.join(PLOTS_DIR, "waterfall_plot.png"))

    logger.info("Interpretation completed. Plots saved: data/plots/")