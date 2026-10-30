
import logging
import os
from pathlib import Path

from commodity_finance_assistance.config_loader import load_config


def get_logger(name: str) -> logging.Logger:
    """Return a configured application logger."""
    config = load_config()
    observability = config.get("observability", {})
    app_config = config.get("app", {})

    level_name = os.getenv(
        "LOG_LEVEL",
        app_config.get("log_level", "INFO"),
    ).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if observability.get("enabled", True):
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / "commodity_finance.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
