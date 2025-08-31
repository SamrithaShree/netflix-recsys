import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", log_file=None, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
