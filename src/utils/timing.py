"""
Timing utilities for performance monitoring and FPS control.
"""

import time
from typing import Optional


class FPSCounter:
    """
    Track and calculate frames per second.

    Useful for monitoring camera capture and processing performance.
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average over
        """
        self._window_size = window_size
        self._frame_times: list[float] = []
        self._last_time: Optional[float] = None

    def tick(self) -> float:
        """
        Register a frame and return current FPS.

        Returns:
            Current FPS (frames per second)
        """
        current_time = time.perf_counter()

        if self._last_time is not None:
            frame_time = current_time - self._last_time
            self._frame_times.append(frame_time)

            # Keep only last N frames
            if len(self._frame_times) > self._window_size:
                self._frame_times.pop(0)

        self._last_time = current_time

        return self.fps

    @property
    def fps(self) -> float:
        """
        Get current FPS.

        Returns:
            Current FPS, or 0.0 if no frames recorded
        """
        if not self._frame_times:
            return 0.0

        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        if avg_frame_time <= 0:
            return 0.0

        return 1.0 / avg_frame_time

    def reset(self):
        """Reset FPS counter."""
        self._frame_times.clear()
        self._last_time = None


class FrameRateLimiter:
    """
    Limit frame processing rate to target FPS.

    Helps prevent excessive CPU usage.
    """

    def __init__(self, target_fps: float):
        """
        Initialize frame rate limiter.

        Args:
            target_fps: Target frames per second
        """
        self._target_fps = target_fps
        self._min_frame_time = 1.0 / target_fps if target_fps > 0 else 0.0
        self._last_frame_time: Optional[float] = None

    def wait(self):
        """
        Wait to maintain target frame rate.

        Call this at the end of each frame processing cycle.
        """
        current_time = time.perf_counter()

        if self._last_frame_time is not None:
            elapsed = current_time - self._last_frame_time
            sleep_time = self._min_frame_time - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)
                self._last_frame_time = time.perf_counter()
            else:
                self._last_frame_time = current_time
        else:
            self._last_frame_time = current_time

    @property
    def target_fps(self) -> float:
        """Get target FPS."""
        return self._target_fps

    @target_fps.setter
    def target_fps(self, fps: float):
        """
        Set target FPS.

        Args:
            fps: Target frames per second
        """
        self._target_fps = fps
        self._min_frame_time = 1.0 / fps if fps > 0 else 0.0

    def reset(self):
        """Reset timing."""
        self._last_frame_time = None


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        """
        Initialize timer.

        Args:
            name: Optional name for the timer
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Stop timer and calculate elapsed time."""
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time

    def __str__(self) -> str:
        """String representation."""
        name_str = f"{self.name}: " if self.name else ""
        return f"{name_str}{self.elapsed * 1000:.2f}ms"
