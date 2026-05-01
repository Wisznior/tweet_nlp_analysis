import os
import logging
from typing import List
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)

class DataIngestor:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name: str = dataset_name

        if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
            logger.error("KAGGLE_USERNAME or KAGGLE KEY missing in .env")
            raise EnvironmentError( "KAGGLE_USERNAME or KAGGLE KEY missing in .env" )
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("Authenticated in Kaggle API")
        except Exception as e:
            logger.critical(f"Error while authenticating with Kaggle API")
            raise

    def download_data(self, download_path: str = 'data/raw') -> List[str]:
        if not os.path.exists(download_path):
            os.makedirs(download_path)
            logger.info(f"Created folder for data: {download_path}")
        
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            self.api.dataset_download_files(self.dataset_name, path=download_path, unzip = True)
            files: List[str] = [ os.path.join(download_path, f) for f in os.listdir(download_path) if f.endswith('.csv')]
            logger.info(f"Downloaded data successfully. Files count: {len(files)}")
            return files
        except Exception as e:
            logger.error(f"Error while downloading data: {e}")
            return []