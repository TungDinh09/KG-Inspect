import os
import logging
import logging.config
from dotenv import load_dotenv
from lightrag.utils import logger, set_verbose_debug

# Tải các biến môi trường từ tệp .env
load_dotenv(dotenv_path=".env", override=False)

# Cấu hình Neo4j
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)



def configure_logging():
    """Cấu hình logging cho ứng dụng."""
    log_dir = os.getenv("LOG_DIR", "./logs")
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_app.log"))
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    print(f"\nLog file: {log_file_path}\n")

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "formatter": "detailed",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf-8",
                "formatter": "detailed",
            },
        },
        "loggers": {
            "lightrag": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
    })
    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")