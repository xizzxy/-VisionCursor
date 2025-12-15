# VisionCursor

A privacy-focused Windows desktop application that uses webcam-based gaze tracking to control your mouse cursor.

##  Privacy & Security First

**VisionCursor is designed with your privacy in mind:**
-  **All processing happens locally** - No data ever leaves your computer
-  **No video recording** - Frames are processed in-memory only
-  **No telemetry** - Zero network communication
-  **Minimal data storage** - Only calibration parameters (numeric values) are saved
-  **No biometric templates** - We don't store facial recognition data
-  **Easy data deletion** - One-click calibration data removal in the UI
-  **Open source** - Full transparency, audit the code yourself

### What Data is Processed?
- **Webcam frames** - Processed in real-time, never saved to disk
- **Face landmarks** - Computed on-the-fly using MediaPipe, discarded immediately
- **Gaze vectors** - Mathematical coordinates derived from eye positions

### What Data is Stored?
- **Calibration parameters only** - A local JSON file (`calibration_data.json`) containing:
  - Screen resolution at time of calibration
  - 5 calibration target positions (screen coordinates)
  - 5 corresponding gaze feature vectors (numeric arrays)
  - Timestamp of calibration
  - **No images, no video, no biometric identifiers**

### What is NEVER Stored?
-  Webcam images or video
-  Facial recognition templates
-  Personal identifiable information
-  Usage telemetry or analytics

## Features

-  **Gaze-based cursor control** - Move your mouse cursor by looking at different areas of the screen
-  **Guided calibration** - Face ID-style setup with 5 calibration points
-  **Adjustable sensitivity** - Fine-tune cursor speed and smoothing
-  **Safety controls** - Dead zones, velocity limiting, and emergency stop
-  **Debug overlay** - Visual feedback for development and troubleshooting
-  **Pause/Resume** - Quick toggle to disable tracking temporarily

## System Requirements

- **OS**: Windows 10 or Windows 11
- **Python**: 3.10 or higher
- **Webcam**: Any USB or built-in webcam (720p or better recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **CPU**: Modern multi-core processor (2015 or newer)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/visioncursor.git
cd visioncursor
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python -m src.main
```

## Quick Start Guide

### First Time Setup

1. **Launch the application**
   ```bash
   python -m src.main
   ```

2. **Allow camera access** - Windows will prompt for camera permission

3. **Run calibration**
   - Click "Calibrate" button
   - Follow on-screen instructions
   - Look at each target dot when it appears (5 targets total)
   - Keep your head relatively still during calibration

4. **Start tracking**
   - Click "Start Tracking" button
   - Move your eyes to control the cursor
   - Adjust sensitivity slider as needed

5. **Pause/Stop**
   - Click "Pause" to temporarily disable tracking
   - Click "Stop" to end the session and release the camera

### Tips for Best Results

-  **Consistent seating position** - Sit at similar distance from screen during calibration and use
-  **Good lighting** - Ensure your face is well-lit (avoid backlighting)
-  **Glasses OK** - Works with most eyeglasses (avoid heavily tinted lenses)
-  **Screen angle** - Keep screen perpendicular to your gaze
-  **Re-calibrate** - Run calibration again if you change seating position or lighting

## Architecture

VisionCursor uses a modular architecture with clear separation of concerns:

```
┌─────────────┐
│   PyQt6 GUI │  (UI Layer - app_window.py, widgets.py)
└──────┬──────┘
       │
┌──────▼──────────────┐
│    Controller       │  (State Machine - controller.py)
└──────┬──────────────┘
       │
       ├─► Camera (camera.py)
       │
       ├─► FaceTracker (face_tracker.py) ─► MediaPipe Face Mesh
       │
       ├─► GazeEstimator (gaze_estimator.py)
       │
       ├─► Calibrator (calibrator.py) ──► CalibrationStore (storage/)
       │
       ├─► Smoother (smoothing.py)
       │
       └─► CursorController (cursor_controller.py) ─► pynput
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Security & Threat Model

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for:
- Threat analysis
- Attack surface
- Mitigation strategies
- Privacy guarantees

## Development

### Project Structure
```
visioncursor/
├── src/
│   ├── main.py                 # Entry point
│   ├── gui/                    # PyQt6 UI components
│   ├── core/                   # Controller, state, config
│   ├── vision/                 # Camera, face tracking, gaze estimation
│   ├── os_control/             # Cursor control
│   ├── storage/                # Calibration data persistence
│   └── utils/                  # Logging, timing utilities
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── requirements.txt
└── README.md
```

### Running Tests
```bash
python -m pytest tests/
```

### Debug Mode
Set environment variable to enable debug logging:
```bash
set VISIONCURSOR_LOG_LEVEL=DEBUG
python -m src.main
```

## Known Limitations

- **Head movement** - Works best with minimal head movement; primarily tracks eye/gaze direction
- **Multiple monitors** - Currently optimized for primary monitor
- **Precision** - Not suitable for pixel-perfect tasks; best for general navigation
- **Calibration persistence** - Re-calibration needed if seating position changes significantly

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request


## Disclaimer

VisionCursor is experimental assistive technology. Not recommended for:
- Security-critical operations
- Tasks requiring high precision
- Replacement for accessibility features in critical applications

Always have a fallback input method available.



---

**Built with privacy, security, and user control as core principles.**
