"""
Gaze smoothing and jitter control.

Reduces cursor jitter through exponential moving average,
dead zones, and velocity limiting.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from src.core.config import GazeConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SmoothedGaze:
    """Smoothed gaze position in screen coordinates."""

    x: int  # Screen x coordinate (pixels)
    y: int  # Screen y coordinate (pixels)
    velocity: float  # Movement speed (pixels/frame)


class GazeSmoother:
    """
    Smooth gaze movements to reduce jitter.

    Techniques:
    1. Exponential Moving Average (EMA) - smooth rapid fluctuations
    2. Dead zone - ignore tiny movements around current position
    3. Velocity clamping - limit maximum movement per frame
    """

    def __init__(self, config: GazeConfig, screen_width: int, screen_height: int):
        """
        Initialize smoother.

        Args:
            config: Gaze configuration
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self._config = config
        self._screen_width = screen_width
        self._screen_height = screen_height

        # Current smoothed position (screen coords)
        self._smoothed_x: Optional[float] = None
        self._smoothed_y: Optional[float] = None

        # Dead zone radius in pixels
        self._dead_zone_pixels = int(
            config.dead_zone_radius * min(screen_width, screen_height)
        )

        logger.info(
            f"GazeSmoother initialized: "
            f"smoothing={config.smoothing_factor:.2f}, "
            f"dead_zone={self._dead_zone_pixels}px, "
            f"max_vel={config.max_velocity:.1f}px/frame"
        )

    def smooth(self, screen_x: float, screen_y: float) -> SmoothedGaze:
        """
        Apply smoothing to raw screen coordinates.

        Args:
            screen_x: Raw screen x coordinate
            screen_y: Raw screen y coordinate

        Returns:
            Smoothed gaze position

        Process:
        1. Apply sensitivity multiplier
        2. Dead zone filtering
        3. Exponential moving average
        4. Velocity limiting
        5. Screen bounds clamping
        """
        # Initialize on first call
        if self._smoothed_x is None or self._smoothed_y is None:
            self._smoothed_x = screen_x
            self._smoothed_y = screen_y
            return SmoothedGaze(
                x=int(screen_x), y=int(screen_y), velocity=0.0
            )

        # Calculate displacement from current position
        dx = screen_x - self._smoothed_x
        dy = screen_y - self._smoothed_y
        distance = np.sqrt(dx**2 + dy**2)

        # Apply dead zone: ignore small movements
        if distance < self._dead_zone_pixels:
            return SmoothedGaze(
                x=int(self._smoothed_x),
                y=int(self._smoothed_y),
                velocity=0.0,
            )

        # Apply sensitivity
        dx *= self._config.sensitivity
        dy *= self._config.sensitivity
        distance *= self._config.sensitivity

        # Velocity limiting: clamp maximum movement
        if distance > self._config.max_velocity:
            scale = self._config.max_velocity / distance
            dx *= scale
            dy *= scale
            distance = self._config.max_velocity

        # Exponential moving average smoothing
        # smoothed = alpha * new + (1 - alpha) * old
        alpha = self._config.smoothing_factor
        new_x = self._smoothed_x + alpha * dx
        new_y = self._smoothed_y + alpha * dy

        # Clamp to screen bounds
        new_x = np.clip(new_x, 0, self._screen_width - 1)
        new_y = np.clip(new_y, 0, self._screen_height - 1)

        # Update state
        self._smoothed_x = new_x
        self._smoothed_y = new_y

        return SmoothedGaze(
            x=int(new_x),
            y=int(new_y),
            velocity=float(distance),
        )

    def update_screen_size(self, width: int, height: int):
        """
        Update screen dimensions.

        Args:
            width: New screen width
            height: New screen height
        """
        self._screen_width = width
        self._screen_height = height

        # Recalculate dead zone
        self._dead_zone_pixels = int(
            self._config.dead_zone_radius * min(width, height)
        )

        logger.info(f"Screen size updated: {width}x{height}")

    def update_config(self, config: GazeConfig):
        """
        Update smoothing configuration.

        Args:
            config: New gaze configuration
        """
        self._config = config

        # Recalculate dead zone
        self._dead_zone_pixels = int(
            config.dead_zone_radius * min(self._screen_width, self._screen_height)
        )

        logger.debug(f"Smoothing config updated: factor={config.smoothing_factor:.2f}")

    def reset(self):
        """Reset smoother state (e.g., after calibration)."""
        self._smoothed_x = None
        self._smoothed_y = None
        logger.debug("Smoother reset")

    @property
    def current_position(self) -> Optional[tuple[int, int]]:
        """Get current smoothed position."""
        if self._smoothed_x is None or self._smoothed_y is None:
            return None
        return (int(self._smoothed_x), int(self._smoothed_y))
