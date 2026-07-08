import logging
import os
from datetime import datetime


def configure_logging(level_name: str = "INFO", log_dir: str = "logs") -> None:
    """
    Configures standard Python logging with a console handler and a file handler.
    Does not use a custom LoggerSingleton class, preferring standard library logging configuration.
    """
    # Create logs directory if it doesn't exist
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create log directory: {e}. Logging to file is disabled.")
        # Configure console-only fallback
        logging.basicConfig(
            level=getattr(logging, level_name.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler()]
        )
        return

    # Create a unique filename for the application session log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"app_{timestamp}.log")

    # Define standard format
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Setup numeric level
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.warning("Could not set up file handler: %s", e)

    logging.info("Logging configured. Level: %s, File: %s", level_name, log_file)
