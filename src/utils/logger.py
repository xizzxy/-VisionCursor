"""
Logging configuration for VisionCursor.

Provides consistent logging across the application with privacy-safe defaults.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "WARNING",
    log_file: Optional[Path] = None,
    enable_file_logging: bool = False,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Privacy: File logging is OFF by default. Only local logging,
    no network handlers.

    Args:
        name: Logger name (typically __name__)
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_file_logging: Enable file logging (default: False for privacy)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    logger.setLevel(numeric_level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler - always enabled
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Format: timestamp - name - level - message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - only if explicitly enabled (privacy)
    if enable_file_logging and log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"File logging enabled: {log_file}")
        except (OSError, IOError) as e:
            logger.warning(f"Failed to enable file logging: {e}")

    # Don't propagate to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
