"""
Cursor control with safety mechanisms.

Uses pynput for cross-platform cursor control.
Includes rate limiting and bounds checking for safety.

Why pynput over pyautogui:
- More granular control
- Better cross-platform support
- No screenshot dependencies (faster, less memory)
- More suitable for real-time cursor control
"""

import time
from typing import Tuple, Optional
from pynput.mouse import Controller as MouseController

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CursorControlError(Exception):
    """Cursor control errors."""

    pass


class CursorController:
    """
    Safe cursor control with rate limiting and bounds checking.

    Safety features:
    - Screen bounds checking
    - Rate limiting to prevent excessive updates
    - Graceful error handling
    - Emergency stop capability
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        min_update_interval: float = 0.01,  # 100 Hz max
    ):
        """
        Initialize cursor controller.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            min_update_interval: Minimum time between cursor updates (seconds)
        """
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._min_update_interval = min_update_interval

        # pynput mouse controller
        self._mouse = MouseController()

        # Rate limiting
        self._last_update_time = 0.0

        # Emergency stop flag
        self._enabled = True

        # Statistics
        self._total_moves = 0
        self._skipped_moves = 0

        logger.info(
            f"CursorController initialized: {screen_width}x{screen_height}, "
            f"min_interval={min_update_interval*1000:.1f}ms"
        )

    def move_to(self, x: int, y: int) -> bool:
        """
        Move cursor to absolute screen position.

        Args:
            x: Target x coordinate (pixels)
            y: Target y coordinate (pixels)

        Returns:
            True if cursor moved, False if skipped (rate limited or disabled)

        Safety:
        - Bounds checking
        - Rate limiting
        - Respects enabled flag
        """
        if not self._enabled:
            return False

        # Rate limiting
        current_time = time.perf_counter()
        if current_time - self._last_update_time < self._min_update_interval:
            self._skipped_moves += 1
            return False

        # Bounds checking
        x_clamped = max(0, min(x, self._screen_width - 1))
        y_clamped = max(0, min(y, self._screen_height - 1))

        if x != x_clamped or y != y_clamped:
            logger.debug(f"Cursor position clamped: ({x},{y}) -> ({x_clamped},{y_clamped})")

        try:
            # Move cursor
            self._mouse.position = (x_clamped, y_clamped)

            self._last_update_time = current_time
            self._total_moves += 1

            return True

        except Exception as e:
            logger.error(f"Failed to move cursor: {e}")
            return False

    def get_position(self) -> Tuple[int, int]:
        """
        Get current cursor position.

        Returns:
            (x, y) tuple
        """
        try:
            pos = self._mouse.position
            return (int(pos[0]), int(pos[1]))
        except Exception as e:
            logger.error(f"Failed to get cursor position: {e}")
            return (0, 0)

    def enable(self):
        """Enable cursor control."""
        self._enabled = True
        logger.info("Cursor control enabled")

    def disable(self):
        """Disable cursor control (emergency stop)."""
        self._enabled = False
        logger.info("Cursor control disabled")

    def is_enabled(self) -> bool:
        """Check if cursor control is enabled."""
        return self._enabled

    def update_screen_size(self, width: int, height: int):
        """
        Update screen dimensions.

        Args:
            width: New screen width
            height: New screen height
        """
        self._screen_width = width
        self._screen_height = height
        logger.info(f"Screen size updated: {width}x{height}")

    def reset_statistics(self):
        """Reset move counters."""
        self._total_moves = 0
        self._skipped_moves = 0

    @property
    def statistics(self) -> dict:
        """Get cursor control statistics."""
        return {
            "total_moves": self._total_moves,
            "skipped_moves": self._skipped_moves,
            "effective_rate": (
                self._total_moves / (self._total_moves + self._skipped_moves)
                if (self._total_moves + self._skipped_moves) > 0
                else 0.0
            ),
        }

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions."""
        return (self._screen_width, self._screen_height)
