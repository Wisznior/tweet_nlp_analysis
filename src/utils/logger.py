import logging
import sys
from pathlib import Path

def setup_logging(log_level: int = logging.INFO, log_file: str = "project.log") -> None:
    log_path = Path(log_file)

    log_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    file_handler = logging.FileHandler(log_file, encoding = 'utf-8')
    file_handler.setFormatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info("Logging system initialized")