"""Shared logging configuration."""
import logging
import sys

from config import settings


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("scholar")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(settings.log_level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
