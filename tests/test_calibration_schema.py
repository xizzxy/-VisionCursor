"""
Tests for calibration schema and validation.
"""

import pytest
from datetime import datetime

from src.storage.schema import CalibrationPoint, CalibrationData


class TestCalibrationPoint:
    """Tests for CalibrationPoint."""

    def test_valid_point(self):
        """Test creating a valid calibration point."""
        point = CalibrationPoint(
            screen_x=100.0,
            screen_y=200.0,
            gaze_x=0.5,
            gaze_y=-0.3,
            sample_count=60,
        )

        assert point.validate() is True

    def test_negative_screen_coordinates(self):
        """Test that negative screen coordinates are invalid."""
        point = CalibrationPoint(
            screen_x=-100.0,
            screen_y=200.0,
            gaze_x=0.5,
            gaze_y=-0.3,
            sample_count=60,
        )

        with pytest.raises(ValueError, match="Screen coordinates must be non-negative"):
            point.validate()

    def test_gaze_out_of_range(self):
        """Test that gaze values far outside range are invalid."""
        point = CalibrationPoint(
            screen_x=100.0,
            screen_y=200.0,
            gaze_x=5.0,  # Way out of range
            gaze_y=-0.3,
            sample_count=60,
        )

        with pytest.raises(ValueError, match="Gaze values out of expected range"):
            point.validate()

    def test_zero_samples(self):
        """Test that zero samples is invalid."""
        point = CalibrationPoint(
            screen_x=100.0,
            screen_y=200.0,
            gaze_x=0.5,
            gaze_y=-0.3,
            sample_count=0,
        )

        with pytest.raises(ValueError, match="Sample count must be positive"):
            point.validate()

    def test_to_dict_and_back(self):
        """Test serialization roundtrip."""
        point = CalibrationPoint(
            screen_x=100.0,
            screen_y=200.0,
            gaze_x=0.5,
            gaze_y=-0.3,
            sample_count=60,
        )

        # Convert to dict and back
        point_dict = point.to_dict()
        point_restored = CalibrationPoint.from_dict(point_dict)

        assert point_restored.screen_x == point.screen_x
        assert point_restored.screen_y == point.screen_y
        assert point_restored.gaze_x == point.gaze_x
        assert point_restored.gaze_y == point.gaze_y
        assert point_restored.sample_count == point.sample_count


class TestCalibrationData:
    """Tests for CalibrationData."""

    def test_valid_calibration(self):
        """Test creating valid calibration data."""
        points = [
            CalibrationPoint(960, 540, 0.0, 0.0, 60),  # Center
            CalibrationPoint(192, 540, -0.8, 0.0, 60),  # Left
            CalibrationPoint(1728, 540, 0.8, 0.0, 60),  # Right
            CalibrationPoint(960, 108, 0.0, -0.8, 60),  # Top
            CalibrationPoint(960, 972, 0.0, 0.8, 60),  # Bottom
        ]

        calibration = CalibrationData(
            version="1.0",
            screen_width=1920,
            screen_height=1080,
            points=points,
        )

        assert calibration.validate() is True

    def test_insufficient_points(self):
        """Test that too few points is invalid."""
        points = [
            CalibrationPoint(960, 540, 0.0, 0.0, 60),
            CalibrationPoint(192, 540, -0.8, 0.0, 60),
        ]

        calibration = CalibrationData(
            version="1.0",
            screen_width=1920,
            screen_height=1080,
            points=points,
        )

        with pytest.raises(ValueError, match="Need at least 3 calibration points"):
            calibration.validate()

    def test_invalid_screen_dimensions(self):
        """Test that invalid screen dimensions are caught."""
        points = [
            CalibrationPoint(960, 540, 0.0, 0.0, 60),
            CalibrationPoint(192, 540, -0.8, 0.0, 60),
            CalibrationPoint(1728, 540, 0.8, 0.0, 60),
        ]

        calibration = CalibrationData(
            version="1.0",
            screen_width=0,  # Invalid
            screen_height=1080,
            points=points,
        )

        with pytest.raises(ValueError, match="Invalid screen dimensions"):
            calibration.validate()

    def test_screen_compatibility(self):
        """Test screen compatibility check."""
        points = [
            CalibrationPoint(960, 540, 0.0, 0.0, 60),
            CalibrationPoint(192, 540, -0.8, 0.0, 60),
            CalibrationPoint(1728, 540, 0.8, 0.0, 60),
        ]

        calibration = CalibrationData(
            version="1.0",
            screen_width=1920,
            screen_height=1080,
            points=points,
        )

        # Same resolution: compatible
        assert calibration.is_compatible_with_screen(1920, 1080) is True

        # Different resolution: not compatible
        assert calibration.is_compatible_with_screen(2560, 1440) is False

    def test_timestamp_auto_generation(self):
        """Test that timestamp is auto-generated."""
        calibration = CalibrationData(
            screen_width=1920,
            screen_height=1080,
        )

        # Should have a timestamp
        assert calibration.timestamp != ""

        # Should be parseable as ISO format
        timestamp = datetime.fromisoformat(calibration.timestamp)
        assert timestamp is not None

    def test_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        points = [
            CalibrationPoint(960, 540, 0.0, 0.0, 60),
            CalibrationPoint(192, 540, -0.8, 0.0, 60),
            CalibrationPoint(1728, 540, 0.8, 0.0, 60),
            CalibrationPoint(960, 108, 0.0, -0.8, 60),
            CalibrationPoint(960, 972, 0.0, 0.8, 60),
        ]

        calibration = CalibrationData(
            version="1.0",
            screen_width=1920,
            screen_height=1080,
            points=points,
        )

        # Convert to dict and back
        cal_dict = calibration.to_dict()
        cal_restored = CalibrationData.from_dict(cal_dict)

        # Verify all fields match
        assert cal_restored.version == calibration.version
        assert cal_restored.screen_width == calibration.screen_width
        assert cal_restored.screen_height == calibration.screen_height
        assert len(cal_restored.points) == len(calibration.points)

        # Verify first point as example
        assert cal_restored.points[0].screen_x == points[0].screen_x
        assert cal_restored.points[0].gaze_x == points[0].gaze_x
