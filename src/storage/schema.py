"""
Calibration data schema and validation.

Privacy: Only stores numeric calibration parameters,
no biometric data, no images, no personal information.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationPoint:
    """
    Single calibration point data.

    Privacy: Contains only:
    - Screen coordinates (pixels)
    - Gaze feature vector (normalized numeric values)
    NO biometric templates, NO facial data.
    """

    # Target position on screen (pixels)
    screen_x: float
    screen_y: float

    # Corresponding gaze vector (normalized, -1 to +1)
    gaze_x: float
    gaze_y: float

    # Number of samples averaged for this point
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationPoint":
        """Create from dictionary."""
        return cls(**data)

    def validate(self) -> bool:
        """
        Validate calibration point data.

        Returns:
            True if valid, raises ValueError if invalid
        """
        # Screen coordinates should be positive
        if self.screen_x < 0 or self.screen_y < 0:
            raise ValueError("Screen coordinates must be non-negative")

        # Gaze should be in normalized range (with some tolerance)
        if not -2.0 <= self.gaze_x <= 2.0 or not -2.0 <= self.gaze_y <= 2.0:
            raise ValueError("Gaze values out of expected range")

        # Sample count should be positive
        if self.sample_count <= 0:
            raise ValueError("Sample count must be positive")

        return True


@dataclass
class CalibrationData:
    """
    Complete calibration dataset.

    Privacy: Contains only numeric mapping parameters.
    """

    # Schema version for future compatibility
    version: str = "1.0"

    # Timestamp of calibration
    timestamp: str = ""

    # Screen resolution at time of calibration
    screen_width: int = 0
    screen_height: int = 0

    # Calibration points (typically 5: center, left, right, top, bottom)
    points: List[CalibrationPoint] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.points is None:
            self.points = []

        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "points": [point.to_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationData":
        """Create from dictionary."""
        points = [CalibrationPoint.from_dict(p) for p in data.get("points", [])]

        return cls(
            version=data.get("version", "1.0"),
            timestamp=data.get("timestamp", ""),
            screen_width=data.get("screen_width", 0),
            screen_height=data.get("screen_height", 0),
            points=points,
        )

    def validate(self) -> bool:
        """
        Validate calibration data.

        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check version
        if not self.version:
            raise ValueError("Missing version")

        # Check screen dimensions
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("Invalid screen dimensions")

        # Check points
        if not self.points or len(self.points) < 3:
            raise ValueError("Need at least 3 calibration points")

        # Validate each point
        for i, point in enumerate(self.points):
            try:
                point.validate()
            except ValueError as e:
                raise ValueError(f"Invalid calibration point {i}: {e}")

        # Check timestamp format
        try:
            datetime.fromisoformat(self.timestamp)
        except ValueError:
            raise ValueError("Invalid timestamp format")

        logger.debug(f"Calibration data validated: {len(self.points)} points")
        return True

    def is_compatible_with_screen(self, width: int, height: int) -> bool:
        """
        Check if calibration is compatible with current screen resolution.

        Args:
            width: Current screen width
            height: Current screen height

        Returns:
            True if compatible (same resolution)
        """
        return self.screen_width == width and self.screen_height == height

    def get_gaze_array(self) -> np.ndarray:
        """
        Get gaze vectors as numpy array.

        Returns:
            Array of shape (N, 2) with gaze vectors
        """
        return np.array(
            [[p.gaze_x, p.gaze_y] for p in self.points],
            dtype=np.float32,
        )

    def get_screen_array(self) -> np.ndarray:
        """
        Get screen positions as numpy array.

        Returns:
            Array of shape (N, 2) with screen coordinates
        """
        return np.array(
            [[p.screen_x, p.screen_y] for p in self.points],
            dtype=np.float32,
        )
