import logging
from dotenv import load_dotenv
load_dotenv() 
import pandas as pd
from typing import List
from src.utils.logger import setup_logging
from src.ingestion import DataIngestor
from src.database import DatabaseManager
from src.preprocessing import prepare_data
import joblib
from src.modeling import train_and_evaluate_all, save_model
from src.interpretation import run_interpretation

def main():
    setup_logging(log_level=logging.INFO, log_file="data/project_run.log")

    logger = logging.getLogger(__name__)
    logger.info('Started data processing')

    DATASET_ID: str = "austinreese/trump-tweets"
    DB_PATH: str = "data/trump_tweets.sqlite"
    RAW_DATA_DIR: str = "data/raw"

    try:
        ingestor = DataIngestor(DATASET_ID)
        csv_files: List[str] = ingestor.download_data(download_path=RAW_DATA_DIR)

        if not csv_files:
            logger.error("No data to process")
            return
        target_file: str = next((f for f in csv_files if  "realdonaldtrump" in f.lower()), csv_files[0])

        logger.info(f"Getting data from: {target_file}")
        df: pd.DataFrame = pd.read_csv(target_file)

        db_mgr = DatabaseManager(DB_PATH)
        db_mgr.save_dataframe(df, 'raw_tweets')

        logger.info("Startring preprocessing")
        X_train, X_test, y_train, y_test, feature_names = prepare_data(
            db_engine=db_mgr.get_engine(),
            vectorizer_path="data/tfidf_vectorizer.pkl",
            use_metadata=True
        )
        logger.info("Preprocessing completed.")

        logger.info("Starting model training and evaluation")
        results_df, trained_models = train_and_evaluate_all(X_train, X_test, y_train, y_test)

        best_model_name = results_df.iloc[0]["model"]
        logger.info(f"Best model: {best_model_name} (F1: {results_df.iloc[0]['f1_weighted']:.4f})")

        best_model = trained_models[best_model_name]
        shap_compatible = ["random_forest", "naive_bayes", "logistic_regression"]
        shap_model_name = next((m for m in results_df["model"] if m in shap_compatible), None)
        shap_model = trained_models[shap_model_name]
        joblib.dump(shap_model, "data/shap_model.pkl")
        logger.info(f"SHAP-compatible model saved: {shap_model_name}")

        save_model(best_model, "data/best_model.pkl")

        joblib.dump((X_test, y_test, feature_names), "data/test_data.pkl")
        logger.info("Saved test data for SHAP to data/test_data.pkl")

        logger.info("Starting SHAP interpretation")
        run_interpretation(
            model_path="data/shap_model.pkl",
            test_data_path="data/test_data.pkl",
            sample_size=500
        )
        logger.info("Interpretation completed.")

        logger.info("Succeded!")

    except Exception as e:
        logger.critical(f"Got an error: {e}", exc_info = True)

if __name__ == "__main__":
    main()