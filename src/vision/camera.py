"""
Camera capture module with error handling.

Privacy: All frames processed in-memory only, never saved to disk.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from src.core.config import CameraConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def list_available_cameras(max_test: int = 5) -> List[int]:
    """
    List available camera indices.

    Args:
        max_test: Maximum camera index to test

    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()

    logger.info(f"Found {len(available)} available cameras: {available}")
    return available


@dataclass
class CameraFrame:
    """Represents a captured camera frame with metadata."""

    image: np.ndarray  # RGB format (H, W, 3)
    timestamp: float
    frame_number: int


class CameraError(Exception):
    """Camera-related errors."""

    pass


class Camera:
    """
    Camera capture with robust error handling.

    Privacy: Frames are never saved to disk. All processing in-memory only.
    """

    def __init__(self, config: CameraConfig):
        """
        Initialize camera.

        Args:
            config: Camera configuration

        Raises:
            CameraError: If camera cannot be initialized
        """
        self._config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._is_open = False
        self._frame_count = 0

        logger.info(f"Initializing camera {config.camera_index}")

    def open(self) -> bool:
        """
        Open camera and configure capture settings.

        Returns:
            True if successful, False otherwise

        Raises:
            CameraError: If camera cannot be opened
        """
        if self._is_open:
            logger.warning("Camera already open")
            return True

        try:
            # Open camera with DirectShow backend on Windows for better compatibility
            self._capture = cv2.VideoCapture(
                self._config.camera_index, cv2.CAP_DSHOW
            )

            if not self._capture.isOpened():
                raise CameraError(
                    f"Failed to open camera {self._config.camera_index}. "
                    "Check if camera is connected and not used by another application."
                )

            # Configure capture settings
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
            self._capture.set(cv2.CAP_PROP_FPS, self._config.target_fps)

            # Disable auto-exposure for more consistent lighting
            # (not critical, may not work on all cameras)
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

            # Warm up camera (skip first few frames which may be black/corrupted)
            for _ in range(self._config.warmup_frames):
                self._capture.read()

            self._is_open = True
            self._frame_count = 0

            # Log actual settings (may differ from requested)
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f}fps"
            )

            return True

        except Exception as e:
            self._is_open = False
            error_msg = f"Camera initialization failed: {e}"
            logger.error(error_msg)
            raise CameraError(error_msg) from e

    def read_frame(self) -> Optional[CameraFrame]:
        """
        Read a frame from the camera.

        Returns:
            CameraFrame if successful, None if read failed

        Note:
            Returned frame is in RGB format (converted from BGR).
            Frame is never saved to disk (privacy).
        """
        if not self._is_open or self._capture is None:
            logger.warning("Attempted to read from closed camera")
            return None

        try:
            ret, frame = self._capture.read()

            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                return None

            # Convert BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self._frame_count += 1

            return CameraFrame(
                image=frame_rgb,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                frame_number=self._frame_count,
            )

        except Exception as e:
            logger.error(f"Error reading camera frame: {e}")
            return None

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get current frame dimensions.

        Returns:
            (width, height) tuple
        """
        if self._capture is None or not self._is_open:
            return (self._config.frame_width, self._config.frame_height)

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return (width, height)

    def close(self):
        """
        Release camera resources.

        Safe to call multiple times.
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._is_open = False
        logger.info("Camera closed")

    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._is_open

    @property
    def frame_count(self) -> int:
        """Get number of frames read."""
        return self._frame_count

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
