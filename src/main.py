"""
VisionCursor - Gaze-controlled cursor application

Main entry point.

Privacy & Security:
- No data sent over network
- All processing local
- No video recording
- Minimal data storage (calibration only)

Usage:
    python -m src.main
"""

import sys
from PyQt6.QtWidgets import QApplication

from src.core.config import get_default_config
from src.gui.app_window import MainWindow
from src.utils.logger import setup_logger, get_logger


def main():
    """Main entry point."""

    # Load configuration
    config = get_default_config()

    # Setup logging
    setup_logger(
        name="src",
        level=config.log_level,
        log_file=config.storage.log_path,
        enable_file_logging=config.storage.enable_file_logging,
    )

    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("VisionCursor Starting")
    logger.info(f"Version: {config.version}")
    logger.info("=" * 60)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("VisionCursor")
    app.setApplicationVersion(config.version)

    # Create and show main window
    window = MainWindow(config)
    window.show()

    logger.info("Application window created")

    # Run event loop
    exit_code = app.exec()

    logger.info("Application exiting")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
