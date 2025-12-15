"""
Face and eye landmark detection using MediaPipe Face Mesh.

Privacy: No facial recognition, no biometric templates stored.
Only extracts geometric landmarks for gaze estimation.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


# MediaPipe Face Mesh landmark indices for eyes
# See: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Left eye landmarks (from user's perspective)
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]  # Iris center and outline

# Right eye landmarks
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]  # Iris center and outline


@dataclass
class EyeLandmarks:
    """Eye landmarks and iris position."""

    # Eye corner positions (normalized 0-1)
    left_corner: np.ndarray  # Shape: (2,)
    right_corner: np.ndarray  # Shape: (2,)
    top: np.ndarray  # Shape: (2,)
    bottom: np.ndarray  # Shape: (2,)

    # Iris center (normalized 0-1)
    iris_center: np.ndarray  # Shape: (2,)

    # All eye contour points
    contour: np.ndarray  # Shape: (N, 2)


@dataclass
class FaceLandmarks:
    """Face detection result with eye landmarks."""

    # Detection confidence
    confidence: float

    # Eye landmarks
    left_eye: EyeLandmarks
    right_eye: EyeLandmarks

    # Face bounding box (normalized 0-1)
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)

    # All face landmarks (for debug visualization)
    all_landmarks: Optional[np.ndarray] = None  # Shape: (478, 2)


class FaceTracker:
    """
    Face and eye tracking using MediaPipe Face Mesh.

    Privacy: Only processes frames in-memory. No data stored.
    Extracts only geometric landmarks, no facial recognition.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
    ):
        """
        Initialize face tracker.

        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            max_num_faces: Maximum number of faces to detect (1 for gaze tracking)
        """
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._max_num_faces = max_num_faces

        # Initialize MediaPipe Face Mesh
        # refine_landmarks=True enables iris tracking
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        logger.info("FaceTracker initialized with MediaPipe Face Mesh")

    def process_frame(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Process a frame and extract face/eye landmarks.

        Args:
            frame: RGB image (H, W, 3) as numpy array

        Returns:
            FaceLandmarks if face detected, None otherwise

        Privacy: Frame is not stored, only landmarks are extracted.
        """
        if frame is None or frame.size == 0:
            return None

        try:
            # Process frame with MediaPipe
            # Input must be RGB (we convert in camera.py)
            results = self._face_mesh.process(frame)

            if not results.multi_face_landmarks:
                return None

            # Use first detected face
            face_landmarks = results.multi_face_landmarks[0]

            # Extract landmarks as numpy array
            height, width = frame.shape[:2]
            landmarks = self._extract_landmarks(face_landmarks, width, height)

            # Extract eye-specific landmarks
            left_eye = self._extract_eye_landmarks(landmarks, LEFT_EYE_INDICES, LEFT_IRIS_INDICES)
            right_eye = self._extract_eye_landmarks(landmarks, RIGHT_EYE_INDICES, RIGHT_IRIS_INDICES)

            # Calculate bounding box
            bbox = self._calculate_bbox(landmarks)

            # Estimate confidence (MediaPipe doesn't provide explicit confidence)
            # Use landmark variance as a proxy
            confidence = self._estimate_confidence(landmarks)

            return FaceLandmarks(
                confidence=confidence,
                left_eye=left_eye,
                right_eye=right_eye,
                bbox=bbox,
                all_landmarks=landmarks,
            )

        except Exception as e:
            logger.debug(f"Error processing frame: {e}")
            return None

    def _extract_landmarks(
        self, face_landmarks, width: int, height: int
    ) -> np.ndarray:
        """
        Extract all landmarks as normalized coordinates.

        Args:
            face_landmarks: MediaPipe face landmarks
            width: Image width
            height: Image height

        Returns:
            Numpy array of shape (478, 2) with normalized coords (0-1)
        """
        landmarks = np.array(
            [[lm.x, lm.y] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )
        return landmarks

    def _extract_eye_landmarks(
        self,
        all_landmarks: np.ndarray,
        eye_indices: List[int],
        iris_indices: List[int],
    ) -> EyeLandmarks:
        """
        Extract eye-specific landmarks.

        Args:
            all_landmarks: All face landmarks (478, 2)
            eye_indices: Indices for eye contour
            iris_indices: Indices for iris

        Returns:
            EyeLandmarks with eye geometry
        """
        # Get eye contour points
        eye_points = all_landmarks[eye_indices]

        # Estimate eye corners and top/bottom
        # Left corner: leftmost point
        # Right corner: rightmost point
        # Top: topmost point
        # Bottom: bottommost point
        left_corner = eye_points[np.argmin(eye_points[:, 0])]
        right_corner = eye_points[np.argmax(eye_points[:, 0])]
        top = eye_points[np.argmin(eye_points[:, 1])]
        bottom = eye_points[np.argmax(eye_points[:, 1])]

        # Iris center (first index is the center point)
        iris_center = all_landmarks[iris_indices[0]]

        return EyeLandmarks(
            left_corner=left_corner,
            right_corner=right_corner,
            top=top,
            bottom=bottom,
            iris_center=iris_center,
            contour=eye_points,
        )

    def _calculate_bbox(self, landmarks: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculate face bounding box from landmarks.

        Args:
            landmarks: All face landmarks (478, 2)

        Returns:
            (x_min, y_min, x_max, y_max) in normalized coords
        """
        x_min = np.min(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        x_max = np.max(landmarks[:, 0])
        y_max = np.max(landmarks[:, 1])

        return (x_min, y_min, x_max, y_max)

    def _estimate_confidence(self, landmarks: np.ndarray) -> float:
        """
        Estimate detection confidence from landmark stability.

        Args:
            landmarks: All face landmarks (478, 2)

        Returns:
            Confidence score 0-1 (higher is better)
        """
        # Use inverse of landmark variance as confidence proxy
        # Stable landmarks = high confidence
        variance = np.var(landmarks)

        # Normalize to 0-1 range (empirical scaling)
        confidence = np.clip(1.0 - variance * 100, 0.0, 1.0)

        return float(confidence)

    def close(self):
        """Release MediaPipe resources."""
        if self._face_mesh:
            self._face_mesh.close()
            logger.info("FaceTracker closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
