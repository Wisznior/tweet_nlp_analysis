import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score

from src.database import DatabaseManager
from src.preprocessing import load_and_clean, build_vectorizer, create_target
from src.modeling import MODELS

logger = logging.getLogger(__name__)

PLOTS_DIR = "data/plots"
SPLIT_YEAR = 2018


def plot_avg_rt_over_time(df: pd.DataFrame, output_path: str) -> None:
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M')

    monthly = (
        df.groupby('year_month')['retweets']
        .mean()
        .reset_index()
    )
    monthly['year_month_dt'] = monthly['year_month'].dt.to_timestamp()
    monthly = monthly.sort_values('year_month_dt')

    plt.figure(figsize=(14, 5))
    plt.plot(monthly['year_month_dt'], monthly['retweets'], marker='o', linewidth=1.5)
    plt.title('Średnia liczba retweetów per miesiąc')
    plt.xlabel('Miesiąc')
    plt.ylabel('Średnia liczba RT')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_f1_by_year( df: pd.DataFrame, split_year: int = SPLIT_YEAR, max_features: int = 5000, output_csv: str = "data/temporal_f1_comparison.csv") -> pd.DataFrame:
    df = df.copy()
    df['label'] = create_target(df)

    mask_old = df['year'] < split_year
    mask_new = df['year'] >= split_year

    logger.info(f"Podział wg roku {split_year}: stare={mask_old.sum()}, nowe={mask_new.sum()}")

    if mask_old.sum() == 0 or mask_new.sum() == 0:
        logger.warning("Jeden z podzbiorów jest pusty")
        return pd.DataFrame()

    df_train = df[mask_old]
    y_train = df_train['label']

    vectorizer = build_vectorizer("tfidf", (1, 2), max_features)
    X_train = vectorizer.fit_transform(df_train['clean'])

    results = []
    for model_name, base_model in MODELS.items():
        try:
            model = clone(base_model)
            model.fit(X_train, y_train)

            for subset_name, mask in [("old", mask_old), ("new", mask_new)]:
                df_sub = df[mask]
                X_sub = vectorizer.transform(df_sub['clean'])
                y_sub = df_sub['label']
                f1 = f1_score(y_sub, model.predict(X_sub), average='macro')
                logger.info(f"  {model_name} | {subset_name}: F1={f1:.4f} (n={len(df_sub)})")
                results.append({
                    "model": model_name,
                    "subset": subset_name,
                    "n_samples": len(df_sub),
                    "f1_macro": round(f1, 4),
                })
        except Exception as e:
            logger.error(f"Błąd dla modelu {model_name}: {e}", exc_info=True)

    df_res = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_res.to_csv(output_csv, index=False)
    logger.info(f"Wyniki zapisane do {output_csv}")
    return df_res


def log_conclusions(df: pd.DataFrame, f1_df: pd.DataFrame) -> None:
    yearly_mean = df.groupby('year')['retweets'].mean()
    first_year, last_year = yearly_mean.index[0], yearly_mean.index[-1]
    trend = "rosnący" if yearly_mean.iloc[-1] > yearly_mean.iloc[0] else "malejący"

    logger.info("[WNIOSKI] Analiza temporalna")
    logger.info(f"[WNIOSKI] Trend popularności tweetów: {trend}")
    logger.info(f"[WNIOSKI] Średnia RT w {first_year}: {yearly_mean.iloc[0]:.0f}, "
                f"w {last_year}: {yearly_mean.iloc[-1]:.0f}")
    logger.info(f"[WNIOSKI] Średnia RT na rok:\n{yearly_mean.to_string()}")

    if f1_df.empty:
        logger.warning("[WNIOSKI] brak danych F1")
        return

    pivot = f1_df.pivot(index='model', columns='subset', values='f1_macro')
    if 'old' in pivot.columns and 'new' in pivot.columns:
        pivot['degradacja'] = pivot['old'] - pivot['new']
        logger.info(f"[WNIOSKI] Degradacja F1 (old - new) dla modeli:\n{pivot.to_string()}")

        avg_deg = pivot['degradacja'].mean()
        if avg_deg > 0.02:
            logger.info("[WNIOSKI] Modele wyraźnie gorzej klasyfikują nowsze tweety")
        elif avg_deg < -0.02:
            logger.info("[WNIOSKI] Modele lepiej radzą sobie z nowszymi tweetami")
        else:
            logger.info("[WNIOSKI] Brak istotnej degradacji modeli na nowszych tweetach")


def run_temporal_analysis(db_path: str = "data/trump_tweets.sqlite") -> None:
    db_mgr = DatabaseManager(db_path)
    df, _ = load_and_clean(db_mgr.get_engine())

    plot_avg_rt_over_time(df, os.path.join(PLOTS_DIR, "rt_per_month.png"))

    yearly = df.groupby('year')['retweets'].agg(['mean', 'median', 'count'])
    logger.info(f"Statystyki RT na rok:\n{yearly.to_string()}")

    f1_df = compare_f1_by_year(df)
    log_conclusions(df, f1_df)


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging(log_level=logging.INFO, log_file="data/temporal_analysis.log")
    run_temporal_analysis()