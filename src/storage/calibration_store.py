"""
Calibration data storage with security controls.

Privacy & Security:
- Local-only storage (no network)
- Path traversal protection
- Schema validation
- Safe JSON serialization
"""

import json
from pathlib import Path
from typing import Optional

from src.core.config import StorageConfig
from src.storage.schema import CalibrationData
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationStoreError(Exception):
    """Calibration storage errors."""

    pass


class CalibrationStore:
    """
    Secure storage for calibration data.

    Privacy: Stores only numeric calibration parameters locally.
    No biometric data, no network access.

    Security:
    - Path traversal protection
    - Schema validation before save/load
    - Safe JSON serialization
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize calibration store.

        Args:
            config: Storage configuration

        Raises:
            CalibrationStoreError: If storage path is invalid
        """
        self._config = config

        # Validate and resolve storage path (prevent traversal attacks)
        try:
            self._data_dir = config.data_dir.resolve(strict=False)
            self._data_dir.mkdir(parents=True, exist_ok=True)
        except (RuntimeError, OSError) as e:
            raise CalibrationStoreError(f"Invalid storage path: {e}")

        self._calibration_path = self._data_dir / config.calibration_filename

        # Ensure we're still within the intended directory
        if not self._is_safe_path(self._calibration_path):
            raise CalibrationStoreError("Path traversal detected")

        logger.info(f"CalibrationStore initialized: {self._calibration_path}")

    def save(self, calibration: CalibrationData) -> bool:
        """
        Save calibration data to disk.

        Args:
            calibration: Calibration data to save

        Returns:
            True if successful

        Raises:
            CalibrationStoreError: If save fails
        """
        try:
            # Validate before saving
            calibration.validate()

            # Convert to dictionary
            data_dict = calibration.to_dict()

            # Write to temporary file first (atomic write)
            temp_path = self._calibration_path.with_suffix(".tmp")

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)

            # Atomic replace
            temp_path.replace(self._calibration_path)

            logger.info(f"Calibration saved: {len(calibration.points)} points")
            return True

        except Exception as e:
            error_msg = f"Failed to save calibration: {e}"
            logger.error(error_msg)
            raise CalibrationStoreError(error_msg) from e

    def load(self) -> Optional[CalibrationData]:
        """
        Load calibration data from disk.

        Returns:
            CalibrationData if found and valid, None otherwise

        Raises:
            CalibrationStoreError: If load fails due to corruption
        """
        if not self._calibration_path.exists():
            logger.info("No calibration data found")
            return None

        try:
            with open(self._calibration_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)

            # Parse and validate
            calibration = CalibrationData.from_dict(data_dict)
            calibration.validate()

            logger.info(f"Calibration loaded: {len(calibration.points)} points")
            return calibration

        except json.JSONDecodeError as e:
            error_msg = f"Corrupted calibration file: {e}"
            logger.error(error_msg)
            raise CalibrationStoreError(error_msg) from e

        except ValueError as e:
            error_msg = f"Invalid calibration data: {e}"
            logger.error(error_msg)
            raise CalibrationStoreError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to load calibration: {e}"
            logger.error(error_msg)
            raise CalibrationStoreError(error_msg) from e

    def delete(self) -> bool:
        """
        Delete calibration data.

        Returns:
            True if deleted, False if file didn't exist
        """
        try:
            if self._calibration_path.exists():
                self._calibration_path.unlink()
                logger.info("Calibration data deleted")
                return True
            else:
                logger.info("No calibration data to delete")
                return False

        except Exception as e:
            logger.error(f"Failed to delete calibration: {e}")
            raise CalibrationStoreError(f"Failed to delete calibration: {e}")

    def exists(self) -> bool:
        """
        Check if calibration data exists.

        Returns:
            True if calibration file exists
        """
        return self._calibration_path.exists()

    def _is_safe_path(self, path: Path) -> bool:
        """
        Check if path is safe (within data directory).

        Args:
            path: Path to check

        Returns:
            True if safe, False if potential traversal attack
        """
        try:
            resolved = path.resolve(strict=False)
            return resolved.parent == self._data_dir
        except (RuntimeError, OSError):
            return False
