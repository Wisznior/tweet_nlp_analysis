import os
import pandas as pd
import logging
from sqlalchemy import create_engine, Engine
from typing import Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str) -> None:
        self.db_path: str = db_path

        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok = True)
            logger.info(f"Created folder for data: {db_dir}")
        try:
            self.engine: Engine = create_engine(f"sqlite:///{self.db_path}")
            logger.info(f"Initialized database engine: {self.db_path}")
        except Exception as e:
            logger.critical(f"Couldn't initialize database engine {e}")
            raise
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        if df is None or df.empty:
            logger.warning("Got empty DataFrame. Saving data stopped.")
            return
        try:
            logger.info(f"Started saving {len(df)} rows to database")

            df.to_sql(
                name = table_name,
                con = self.engine,
                if_exists = 'replace',
                index = False,
                chunksize = 1000
            )

            logger.info(f"Data saved to file '{table_name}'")
        except Exception as e:
            logger.error(f"Error while df.to_sql: {e}")
            raise

    def get_engine(self) -> Engine:
        return self.engine
