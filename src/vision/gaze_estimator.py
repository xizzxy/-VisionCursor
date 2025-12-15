"""
Gaze direction estimation from eye landmarks.

Computes normalized gaze vectors that can be mapped to screen coordinates
through calibration.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from src.vision.face_tracker import FaceLandmarks, EyeLandmarks
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GazeVector:
    """
    Gaze direction vector in normalized space.

    Coordinates are normalized relative to eye geometry:
    - x: horizontal gaze (-1 = left, 0 = center, +1 = right)
    - y: vertical gaze (-1 = up, 0 = center, +1 = down)
    """

    x: float  # Horizontal component
    y: float  # Vertical component
    confidence: float  # Confidence of estimation (0-1)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y], dtype=np.float32)


class GazeEstimator:
    """
    Estimate gaze direction from eye landmarks.

    Uses iris position relative to eye corners to approximate gaze direction.
    This is a simplified approach suitable for calibration-based cursor control.

    Limitations:
    - Assumes eyes are roughly horizontal
    - Best accuracy within ±30 degrees of screen center
    - Head rotation affects accuracy (requires recalibration if head moves)
    """

    def __init__(self):
        """Initialize gaze estimator."""
        self._last_gaze: Optional[GazeVector] = None
        logger.info("GazeEstimator initialized")

    def estimate(self, face_landmarks: FaceLandmarks) -> Optional[GazeVector]:
        """
        Estimate gaze direction from face landmarks.

        Args:
            face_landmarks: Detected face and eye landmarks

        Returns:
            GazeVector if estimation successful, None otherwise

        Algorithm:
        1. For each eye, compute iris position relative to eye corners
        2. Normalize to [-1, +1] range
        3. Average left and right eye gaze for stability
        """
        if face_landmarks is None:
            return None

        try:
            # Estimate gaze from each eye
            left_gaze = self._estimate_eye_gaze(face_landmarks.left_eye)
            right_gaze = self._estimate_eye_gaze(face_landmarks.right_eye)

            if left_gaze is None or right_gaze is None:
                return None

            # Average both eyes for stability
            # Weight by confidence if needed (currently equal weight)
            gaze_x = (left_gaze[0] + right_gaze[0]) / 2.0
            gaze_y = (left_gaze[1] + right_gaze[1]) / 2.0

            # Use face detection confidence
            confidence = face_landmarks.confidence

            gaze = GazeVector(x=gaze_x, y=gaze_y, confidence=confidence)
            self._last_gaze = gaze

            return gaze

        except Exception as e:
            logger.debug(f"Error estimating gaze: {e}")
            return self._last_gaze  # Return last valid gaze on error

    def _estimate_eye_gaze(self, eye: EyeLandmarks) -> Optional[np.ndarray]:
        """
        Estimate gaze from single eye.

        Args:
            eye: Eye landmarks

        Returns:
            (gaze_x, gaze_y) as numpy array, or None if invalid

        Method:
        - Iris position relative to eye corners gives horizontal gaze
        - Iris position relative to top/bottom gives vertical gaze
        - Normalized to [-1, +1] range
        """
        try:
            # Horizontal gaze: iris position between left and right corners
            eye_width = eye.right_corner[0] - eye.left_corner[0]
            if eye_width <= 0:
                return None

            iris_x_relative = eye.iris_center[0] - eye.left_corner[0]
            gaze_x_normalized = (iris_x_relative / eye_width) * 2.0 - 1.0

            # Vertical gaze: iris position between top and bottom
            eye_height = eye.bottom[1] - eye.top[1]
            if eye_height <= 0:
                return None

            iris_y_relative = eye.iris_center[1] - eye.top[1]
            gaze_y_normalized = (iris_y_relative / eye_height) * 2.0 - 1.0

            # Clamp to valid range
            gaze_x = np.clip(gaze_x_normalized, -1.0, 1.0)
            gaze_y = np.clip(gaze_y_normalized, -1.0, 1.0)

            return np.array([gaze_x, gaze_y], dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error in eye gaze estimation: {e}")
            return None

    @property
    def last_gaze(self) -> Optional[GazeVector]:
        """Get last estimated gaze vector."""
        return self._last_gaze

    def reset(self):
        """Reset estimator state."""
        self._last_gaze = None
