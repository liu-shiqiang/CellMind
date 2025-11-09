import os
import logging
from logging.handlers import RotatingFileHandler

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE", "logs/app.log")
MAX_BYTES = int(os.environ.get("LOG_MAX_BYTES", 10 * 1024 * 1024))
BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", 5))

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(LOG_LEVEL)
stream_handler.setFormatter(formatter)

file_handler = RotatingFileHandler(
    filename = LOG_FILE,
    maxBytes=MAX_BYTES,
    backupCount=BACKUP_COUNT,
    encoding = "utf-8"

)

file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=LOG_LEVEL,
    handlers = [stream_handler, file_handler],
    force = True
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)