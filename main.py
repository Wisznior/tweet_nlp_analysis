import logging
from dotenv import load_dotenv
load_dotenv() 
import pandas as pd
from typing import List
from src.utils.logger import setup_logging
from src.ingestion import DataIngestor
from src.database import DatabaseManager

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

        logger.info("Succeded!")

    except Exception as e:
        logger.critical(f"Got an error: {e}", exc_info = True)

if __name__ == "__main__":
    main()