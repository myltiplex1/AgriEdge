# logger.py
import logging

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s: %(message)s', "%H:%M:%S")
    ch.setFormatter(formatter)

    # Avoid duplicate logs
    if not logger.handlers:
        logger.addHandler(ch)

    return logger