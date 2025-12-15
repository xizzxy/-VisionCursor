"""
Tests for gaze-to-screen mapping mathematics.
"""

import pytest
import numpy as np

from src.vision.gaze_estimator import GazeVector
from src.vision.calibrator import GazeMapper
from src.storage.schema import CalibrationData, CalibrationPoint


class TestGazeMapper:
    """Tests for gaze-to-screen coordinate mapping."""

    @pytest.fixture
    def simple_calibration(self):
        """Create a simple 5-point calibration for testing."""
        # 1920x1080 screen
        # 5 points: center, left, right, top, bottom
        points = [
            CalibrationPoint(960, 540, 0.0, 0.0, 60),   # Center
            CalibrationPoint(192, 540, -0.8, 0.0, 60),  # Left
            CalibrationPoint(1728, 540, 0.8, 0.0, 60),  # Right
            CalibrationPoint(960, 108, 0.0, -0.8, 60),  # Top
            CalibrationPoint(960, 972, 0.0, 0.8, 60),   # Bottom
        ]

        return CalibrationData(
            version="1.0",
            screen_width=1920,
            screen_height=1080,
            points=points,
        )

    def test_mapper_initialization(self, simple_calibration):
        """Test that mapper initializes correctly."""
        mapper = GazeMapper(simple_calibration)

        assert mapper is not None
        assert mapper.screen_size == (1920, 1080)

    def test_center_mapping(self, simple_calibration):
        """Test that center gaze maps near screen center."""
        mapper = GazeMapper(simple_calibration)

        # Gaze at center (0, 0)
        gaze = GazeVector(x=0.0, y=0.0, confidence=0.9)
        screen_x, screen_y = mapper.map_gaze_to_screen(gaze)

        # Should map close to center (960, 540)
        assert 800 < screen_x < 1120  # Within ~160px of center
        assert 400 < screen_y < 680   # Within ~140px of center

    def test_left_mapping(self, simple_calibration):
        """Test that left gaze maps to left side of screen."""
        mapper = GazeMapper(simple_calibration)

        # Gaze left (-0.8, 0)
        gaze = GazeVector(x=-0.8, y=0.0, confidence=0.9)
        screen_x, screen_y = mapper.map_gaze_to_screen(gaze)

        # Should map to left side (closer to 192 than to 1728)
        assert screen_x < 960  # Left of center

    def test_right_mapping(self, simple_calibration):
        """Test that right gaze maps to right side of screen."""
        mapper = GazeMapper(simple_calibration)

        # Gaze right (0.8, 0)
        gaze = GazeVector(x=0.8, y=0.0, confidence=0.9)
        screen_x, screen_y = mapper.map_gaze_to_screen(gaze)

        # Should map to right side
        assert screen_x > 960  # Right of center

    def test_bounds_clamping(self, simple_calibration):
        """Test that extreme gaze values are clamped to screen bounds."""
        mapper = GazeMapper(simple_calibration)

        # Extreme left gaze
        gaze = GazeVector(x=-5.0, y=0.0, confidence=0.9)
        screen_x, screen_y = mapper.map_gaze_to_screen(gaze)

        assert screen_x >= 0
        assert screen_x < 1920

        # Extreme right gaze
        gaze = GazeVector(x=5.0, y=0.0, confidence=0.9)
        screen_x, screen_y = mapper.map_gaze_to_screen(gaze)

        assert screen_x >= 0
        assert screen_x < 1920

        # Extreme top gaze
        gaze = GazeVector(x=0.0, y=-5.0, confidence=0.9)
        screen_x, screen_y = mapper.map_gaze_to_screen(gaze)

        assert screen_y >= 0
        assert screen_y < 1080

        # Extreme bottom gaze
        gaze = GazeVector(x=0.0, y=5.0, confidence=0.9)
        screen_x, screen_y = mapper.map_gaze_to_screen(gaze)

        assert screen_y >= 0
        assert screen_y < 1080

    def test_monotonicity_horizontal(self, simple_calibration):
        """Test that increasing gaze_x monotonically increases screen_x."""
        mapper = GazeMapper(simple_calibration)

        gaze_values = [-0.8, -0.4, 0.0, 0.4, 0.8]
        screen_x_values = []

        for gaze_x in gaze_values:
            gaze = GazeVector(x=gaze_x, y=0.0, confidence=0.9)
            screen_x, _ = mapper.map_gaze_to_screen(gaze)
            screen_x_values.append(screen_x)

        # Check monotonicity: each value should be >= previous
        for i in range(1, len(screen_x_values)):
            assert screen_x_values[i] >= screen_x_values[i - 1], \
                f"Horizontal mapping not monotonic: {screen_x_values}"

    def test_monotonicity_vertical(self, simple_calibration):
        """Test that increasing gaze_y monotonically increases screen_y."""
        mapper = GazeMapper(simple_calibration)

        gaze_values = [-0.8, -0.4, 0.0, 0.4, 0.8]
        screen_y_values = []

        for gaze_y in gaze_values:
            gaze = GazeVector(x=0.0, y=gaze_y, confidence=0.9)
            _, screen_y = mapper.map_gaze_to_screen(gaze)
            screen_y_values.append(screen_y)

        # Check monotonicity
        for i in range(1, len(screen_y_values)):
            assert screen_y_values[i] >= screen_y_values[i - 1], \
                f"Vertical mapping not monotonic: {screen_y_values}"


class TestGazeVector:
    """Tests for GazeVector."""

    def test_to_array(self):
        """Test conversion to numpy array."""
        gaze = GazeVector(x=0.5, y=-0.3, confidence=0.9)
        array = gaze.to_array()

        assert isinstance(array, np.ndarray)
        assert array.shape == (2,)
        assert array[0] == 0.5
        assert array[1] == -0.3
