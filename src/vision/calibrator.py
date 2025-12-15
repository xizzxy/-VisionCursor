"""
Calibration system for mapping gaze to screen coordinates.

Provides guided calibration procedure and gaze-to-screen mapping.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum, auto
from scipy import stats  # For trimmed mean (robust averaging)

from src.core.config import CalibrationConfig
from src.vision.gaze_estimator import GazeVector
from src.storage.schema import CalibrationData, CalibrationPoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationState(Enum):
    """Calibration procedure states."""

    IDLE = auto()
    COUNTDOWN = auto()  # Countdown before collecting samples
    COLLECTING = auto()  # Collecting samples for current target
    COMPLETED = auto()  # All targets done


@dataclass
class CalibrationTarget:
    """Single calibration target information."""

    index: int  # Target number (0-4)
    screen_x: float  # Screen position (pixels)
    screen_y: float  # Screen position (pixels)
    samples: List[GazeVector]  # Collected gaze samples

    def add_sample(self, gaze: GazeVector):
        """Add a gaze sample for this target."""
        self.samples.append(gaze)

    def compute_average(self, trim_percent: float = 0.1) -> Optional[Tuple[float, float]]:
        """
        Compute robust average gaze for this target.

        Uses trimmed mean to reject outliers.

        Args:
            trim_percent: Proportion to trim from each end (0-0.5)

        Returns:
            (avg_gaze_x, avg_gaze_y) or None if insufficient samples
        """
        if len(self.samples) < 10:
            return None

        # Extract x and y components
        gaze_x = np.array([s.x for s in self.samples])
        gaze_y = np.array([s.y for s in self.samples])

        # Trimmed mean (remove outliers)
        avg_x = float(stats.trim_mean(gaze_x, trim_percent))
        avg_y = float(stats.trim_mean(gaze_y, trim_percent))

        return (avg_x, avg_y)


class Calibrator:
    """
    Calibration system with guided procedure.

    Process:
    1. Show target at specific screen position
    2. Countdown (2 seconds)
    3. Collect gaze samples (2 seconds, ~60 samples)
    4. Compute robust average (trimmed mean)
    5. Repeat for all 5 targets
    6. Build mapping from gaze to screen coordinates
    """

    def __init__(
        self,
        config: CalibrationConfig,
        screen_width: int,
        screen_height: int,
    ):
        """
        Initialize calibrator.

        Args:
            config: Calibration configuration
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self._config = config
        self._screen_width = screen_width
        self._screen_height = screen_height

        self._state = CalibrationState.IDLE
        self._current_target_index = 0
        self._targets: List[CalibrationTarget] = []

        # Calibration result
        self._calibration_data: Optional[CalibrationData] = None

        logger.info(f"Calibrator initialized for {screen_width}x{screen_height}")

    def start(self):
        """Start calibration procedure."""
        self._state = CalibrationState.IDLE
        self._current_target_index = 0
        self._targets = []

        # Create calibration targets from config
        for i, (norm_x, norm_y) in enumerate(self._config.target_positions):
            screen_x = norm_x * self._screen_width
            screen_y = norm_y * self._screen_height

            target = CalibrationTarget(
                index=i,
                screen_x=screen_x,
                screen_y=screen_y,
                samples=[],
            )
            self._targets.append(target)

        logger.info(f"Calibration started: {len(self._targets)} targets")

    def add_sample(self, gaze: GazeVector) -> bool:
        """
        Add a gaze sample for the current target.

        Args:
            gaze: Current gaze vector

        Returns:
            True if sample added, False if not in collecting state
        """
        if self._state != CalibrationState.COLLECTING:
            return False

        current_target = self._targets[self._current_target_index]
        current_target.add_sample(gaze)

        # Check if we have enough samples
        if len(current_target.samples) >= self._config.samples_per_point:
            self._complete_current_target()

        return True

    def _complete_current_target(self):
        """Complete current target and move to next."""
        logger.info(
            f"Target {self._current_target_index} completed: "
            f"{len(self._targets[self._current_target_index].samples)} samples"
        )

        # Move to next target
        self._current_target_index += 1

        if self._current_target_index >= len(self._targets):
            # All targets completed
            self._finalize_calibration()
        else:
            # Back to countdown for next target
            self._state = CalibrationState.COUNTDOWN

    def _finalize_calibration(self):
        """Compute final calibration from all targets."""
        try:
            points = []

            for target in self._targets:
                avg_gaze = target.compute_average(self._config.outlier_trim_percent)

                if avg_gaze is None:
                    raise ValueError(f"Insufficient samples for target {target.index}")

                gaze_x, gaze_y = avg_gaze

                point = CalibrationPoint(
                    screen_x=target.screen_x,
                    screen_y=target.screen_y,
                    gaze_x=gaze_x,
                    gaze_y=gaze_y,
                    sample_count=len(target.samples),
                )
                points.append(point)

            # Create calibration data
            self._calibration_data = CalibrationData(
                screen_width=self._screen_width,
                screen_height=self._screen_height,
                points=points,
            )

            # Validate
            self._calibration_data.validate()

            self._state = CalibrationState.COMPLETED
            logger.info("Calibration finalized successfully")

        except Exception as e:
            logger.error(f"Calibration finalization failed: {e}")
            self._state = CalibrationState.IDLE
            self._calibration_data = None

    def get_current_target(self) -> Optional[CalibrationTarget]:
        """Get current calibration target."""
        if 0 <= self._current_target_index < len(self._targets):
            return self._targets[self._current_target_index]
        return None

    def set_state(self, state: CalibrationState):
        """Set calibration state (called by controller)."""
        self._state = state

    @property
    def state(self) -> CalibrationState:
        """Get current state."""
        return self._state

    @property
    def calibration_data(self) -> Optional[CalibrationData]:
        """Get calibration result."""
        return self._calibration_data

    @property
    def progress(self) -> Tuple[int, int]:
        """Get progress (current_target, total_targets)."""
        return (self._current_target_index, len(self._targets))


class GazeMapper:
    """
    Map gaze vectors to screen coordinates using calibration.

    Uses affine transformation learned from calibration points.
    """

    def __init__(self, calibration: CalibrationData):
        """
        Initialize mapper from calibration data.

        Args:
            calibration: Calibration data

        Raises:
            ValueError: If calibration is invalid
        """
        calibration.validate()

        self._calibration = calibration
        self._screen_width = calibration.screen_width
        self._screen_height = calibration.screen_height

        # Compute mapping parameters
        self._compute_mapping()

        logger.info("GazeMapper initialized from calibration")

    def _compute_mapping(self):
        """
        Compute affine mapping from gaze to screen coordinates.

        Uses least-squares fit to find best affine transformation:
        [screen_x]   [a b c]   [gaze_x]
        [screen_y] = [d e f] * [gaze_y]
        [   1    ]   [0 0 1]   [  1   ]

        For simplicity, we use a linear interpolation based on calibration extremes.
        """
        gaze_array = self._calibration.get_gaze_array()  # (N, 2)
        screen_array = self._calibration.get_screen_array()  # (N, 2)

        # Find gaze bounds
        self._gaze_min_x = np.min(gaze_array[:, 0])
        self._gaze_max_x = np.max(gaze_array[:, 0])
        self._gaze_min_y = np.min(gaze_array[:, 1])
        self._gaze_max_y = np.max(gaze_array[:, 1])

        # Find corresponding screen bounds
        idx_min_x = np.argmin(gaze_array[:, 0])
        idx_max_x = np.argmax(gaze_array[:, 0])
        idx_min_y = np.argmin(gaze_array[:, 1])
        idx_max_y = np.argmax(gaze_array[:, 1])

        self._screen_at_min_x = screen_array[idx_min_x, 0]
        self._screen_at_max_x = screen_array[idx_max_x, 0]
        self._screen_at_min_y = screen_array[idx_min_y, 1]
        self._screen_at_max_y = screen_array[idx_max_y, 1]

        logger.debug(
            f"Gaze bounds: x=[{self._gaze_min_x:.2f}, {self._gaze_max_x:.2f}], "
            f"y=[{self._gaze_min_y:.2f}, {self._gaze_max_y:.2f}]"
        )

    def map_gaze_to_screen(self, gaze: GazeVector) -> Tuple[float, float]:
        """
        Map gaze vector to screen coordinates.

        Args:
            gaze: Gaze vector in normalized space

        Returns:
            (screen_x, screen_y) in pixels
        """
        # Linear interpolation
        if self._gaze_max_x != self._gaze_min_x:
            t_x = (gaze.x - self._gaze_min_x) / (self._gaze_max_x - self._gaze_min_x)
            screen_x = self._screen_at_min_x + t_x * (
                self._screen_at_max_x - self._screen_at_min_x
            )
        else:
            screen_x = self._screen_width / 2

        if self._gaze_max_y != self._gaze_min_y:
            t_y = (gaze.y - self._gaze_min_y) / (self._gaze_max_y - self._gaze_min_y)
            screen_y = self._screen_at_min_y + t_y * (
                self._screen_at_max_y - self._screen_at_min_y
            )
        else:
            screen_y = self._screen_height / 2

        # Clamp to screen bounds
        screen_x = np.clip(screen_x, 0, self._screen_width - 1)
        screen_y = np.clip(screen_y, 0, self._screen_height - 1)

        return (float(screen_x), float(screen_y))

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Get calibration screen size."""
        return (self._screen_width, self._screen_height)
