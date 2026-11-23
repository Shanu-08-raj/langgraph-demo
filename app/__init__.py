"""LangGraph State and Workflow Designer application package."""

import logging
import os
from pathlib import Path


def setup_logging(log_file: str = "logs/workflow.log", level: int = logging.INFO) -> logging.Logger:
    """Configure application-wide logging with both file and console handlers.

    Args:
        log_file: Path to the log file (relative or absolute).
        level: Root logging level.

    Returns:
        logging.Logger: Root logger instance.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if already initialized
    if not root_logger.handlers:
        # File handler
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return root_logger
