"""
Configuration management for VisionCursor.

All application configuration with sensible defaults.
Uses dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from typing import Tuple
import os
from pathlib import Path


@dataclass
class CameraConfig:
    """Camera capture configuration."""

    camera_index: int = 0  # Default camera
    frame_width: int = 640  # Balance between quality and performance
    frame_height: int = 480
    target_fps: int = 30  # Smooth but not excessive CPU usage
    warmup_frames: int = 10  # Frames to skip after camera init


@dataclass
class GazeConfig:
    """Gaze estimation configuration."""

    # Smoothing parameters (exponential moving average)
    smoothing_factor: float = 0.2  # Lower = more smoothing, range: 0.0-1.0

    # Dead zone around center to prevent micro-jitter (proportion of screen)
    dead_zone_radius: float = 0.035  # 3.5% of screen size

    # Maximum cursor movement per frame (pixels)
    max_velocity: float = 35.0  # Prevents sudden jumps

    # Sensitivity multiplier
    sensitivity: float = 0.8  # Range: 0.1-3.0

    # Minimum confidence threshold for face detection
    min_face_confidence: float = 0.6

    # Freeze cursor when face lost (instead of drifting)
    freeze_on_face_lost: bool = True


@dataclass
class CalibrationConfig:
    """Calibration procedure configuration."""

    # Number of samples to collect per calibration point
    samples_per_point: int = 90  # 3 seconds at 30fps (increased for accuracy)

    # Countdown before starting sample collection (seconds)
    countdown_seconds: int = 3  # Increased for user preparation

    # Target positions (normalized 0-1 screen coordinates)
    # Order: center, left, right, top, bottom
    target_positions: Tuple[Tuple[float, float], ...] = (
        (0.5, 0.5),   # Center
        (0.15, 0.5),  # Left (slightly more conservative)
        (0.85, 0.5),  # Right (slightly more conservative)
        (0.5, 0.15),  # Top (slightly more conservative)
        (0.5, 0.85),  # Bottom (slightly more conservative)
    )

    # Target dot size (pixels)
    target_size: int = 25  # Increased for better visibility

    # Use trimmed mean to reject outliers (trim this proportion from each end)
    outlier_trim_percent: float = 0.15  # Trim 15% from each end (more aggressive outlier rejection)

    # Minimum stable samples required (discard frames with high variance)
    min_stable_samples: int = 60  # Require at least this many stable samples


@dataclass
class StorageConfig:
    """Data storage configuration."""

    # User data directory (where calibration data is stored)
    data_dir: Path = field(default_factory=lambda: Path.home() / ".visioncursor")

    # Calibration data filename
    calibration_filename: str = "calibration_data.json"

    # Log filename (optional, off by default)
    log_filename: str = "visioncursor.log"

    # Enable file logging (OFF by default for privacy)
    enable_file_logging: bool = False

    def __post_init__(self):
        """Ensure data directory exists and is secure."""
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Validate path to prevent directory traversal
        try:
            self.data_dir.resolve(strict=True)
        except (RuntimeError, OSError) as e:
            raise ValueError(f"Invalid data directory path: {e}")

    @property
    def calibration_path(self) -> Path:
        """Get full path to calibration data file."""
        return self.data_dir / self.calibration_filename

    @property
    def log_path(self) -> Path:
        """Get full path to log file."""
        return self.data_dir / self.log_filename


@dataclass
class UIConfig:
    """User interface configuration."""

    window_title: str = "VisionCursor - Gaze Tracking System"
    window_width: int = 850
    window_height: int = 700
    window_min_width: int = 750
    window_min_height: int = 600

    # Show debug overlay with landmarks and gaze vector
    show_debug_overlay: bool = False

    # Preview window size
    preview_width: int = 320
    preview_height: int = 240

    # UI spacing and padding
    content_margin: int = 15
    group_spacing: int = 10
    widget_spacing: int = 8

    # Emergency stop keyboard shortcut
    toggle_shortcut: str = "Space"  # Keyboard shortcut to toggle tracking


@dataclass
class AppConfig:
    """Main application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Application version
    version: str = "0.1.0"

    # Log level from environment or default to WARNING
    log_level: str = field(
        default_factory=lambda: os.getenv("VISIONCURSOR_LOG_LEVEL", "WARNING")
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        # Gaze config validation
        if not 0.0 <= self.gaze.smoothing_factor <= 1.0:
            raise ValueError("smoothing_factor must be between 0.0 and 1.0")

        if not 0.0 <= self.gaze.dead_zone_radius <= 0.1:
            raise ValueError("dead_zone_radius must be between 0.0 and 0.1")

        if not 0.1 <= self.gaze.sensitivity <= 5.0:
            raise ValueError("sensitivity must be between 0.1 and 5.0")

        # Calibration config validation
        if self.calibration.samples_per_point < 10:
            raise ValueError("samples_per_point must be at least 10")

        # Camera config validation
        if self.camera.target_fps < 1 or self.camera.target_fps > 60:
            raise ValueError("target_fps must be between 1 and 60")


def get_default_config() -> AppConfig:
    """
    Get default application configuration.

    Returns:
        AppConfig instance with default values
    """
    return AppConfig()
