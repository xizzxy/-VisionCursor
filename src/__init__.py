"""
VisionCursor - Privacy-focused gaze tracking cursor control.

A Windows desktop application that uses webcam-based gaze estimation
to control the mouse cursor.

Privacy First:
- All processing happens locally
- No data sent over network
- No video recording
- Minimal data storage (numeric calibration parameters only)

Security:
- Path traversal protection
- Schema validation
- Safe JSON serialization
- No default telemetry

Architecture:
- Modular design with dependency injection
- Clean separation of concerns
- Type hints throughout
- Comprehensive error handling
"""

__version__ = "0.1.0"
__author__ = "VisionCursor Team"
__license__ = "MIT"
