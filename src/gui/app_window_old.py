"""
Main application window with worker thread architecture.

Threading model:
- Main thread: UI event loop (PyQt6)
- Worker thread: Camera capture and processing
- Communication: Qt signals/slots
"""

import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QGroupBox,
    QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QScreen
import time

from src.core.controller import Controller, FrameProcessingResult
from src.core.config import AppConfig
from src.core.state import AppState
from src.vision.calibrator import CalibrationState
from src.gui.widgets import CameraPreviewWidget, CalibrationTargetWidget
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingWorker(QThread):
    """
    Worker thread for camera processing.

    Runs the processing loop and emits signals to update UI.
    NEVER updates UI directly from this thread.
    """

    # Signals (thread-safe communication with main thread)
    frame_processed = pyqtSignal(FrameProcessingResult)
    error_occurred = pyqtSignal(str)

    def __init__(self, controller: Controller, parent=None):
        """
        Initialize worker.

        Args:
            controller: Controller instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self._controller = controller
        self._running = False

    def run(self):
        """
        Worker thread main loop.

        Processes frames and emits results via signals.
        """
        logger.info("Processing worker started")
        self._running = True

        try:
            while self._running:
                # Check if we should be processing
                state = self._controller.state

                if state in (AppState.TRACKING, AppState.CALIBRATING):
                    # Process frame
                    result = self._controller.process_frame()

                    # Emit result to UI thread
                    self.frame_processed.emit(result)

                elif state == AppState.PAUSED:
                    # Camera open but not processing
                    # Just sleep to avoid busy loop
                    time.sleep(0.03)  # ~30fps polling

                else:
                    # Idle or error state
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Worker thread error: {e}")
            self.error_occurred.emit(str(e))

        logger.info("Processing worker stopped")

    def stop(self):
        """Stop the worker thread."""
        self._running = False


class MainWindow(QMainWindow):
    """
    Main application window.

    Privacy notice displayed prominently.
    All controls for calibration and tracking.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize main window.

        Args:
            config: Application configuration
        """
        super().__init__()

        self._config = config

        # Get screen size
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        self._screen_width = screen_geometry.width()
        self._screen_height = screen_geometry.height()

        logger.info(f"Screen size: {self._screen_width}x{self._screen_height}")

        # Initialize controller
        self._controller = Controller(
            config,
            self._screen_width,
            self._screen_height,
        )

        if not self._controller.initialize():
            QMessageBox.critical(
                self,
                "Initialization Error",
                "Failed to initialize VisionCursor. Check logs for details.",
            )
            sys.exit(1)

        # Worker thread
        self._worker: ProcessingWorker = None

        # Calibration overlay
        self._calibration_overlay: CalibrationTargetWidget = None

        # UI setup
        self._init_ui()

        # Start worker thread
        self._start_worker()

        # Timer for UI updates during calibration
        self._calibration_timer = QTimer(self)
        self._calibration_timer.timeout.connect(self._update_calibration_ui)
        self._calibration_timer.setInterval(100)  # 10 Hz

    def _init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle(self._config.ui.window_title)
        self.setGeometry(100, 100, self._config.ui.window_width, self._config.ui.window_height)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Privacy notice (prominent)
        privacy_label = QLabel(
            "🔒 PRIVACY MODE: No video is saved or sent. All processing is local."
        )
        privacy_label.setStyleSheet(
            "background-color: #2e7d32; color: white; padding: 10px; "
            "font-weight: bold; border-radius: 5px;"
        )
        privacy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(privacy_label)

        # Camera preview
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()
        self._preview_widget = CameraPreviewWidget(
            self._config.ui.preview_width,
            self._config.ui.preview_height,
        )
        preview_layout.addWidget(self._preview_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self._status_label = QLabel("Status: Idle")
        self._fps_label = QLabel("FPS: 0.0")
        self._face_label = QLabel("Face: Not detected")

        status_layout.addWidget(self._status_label)
        status_layout.addWidget(self._fps_label)
        status_layout.addWidget(self._face_label)

        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        # Calibration button
        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        controls_layout.addWidget(self._calibrate_btn)

        # Tracking buttons
        tracking_layout = QHBoxLayout()

        self._start_btn = QPushButton("Start Tracking")
        self._start_btn.clicked.connect(self._on_start_clicked)
        self._start_btn.setEnabled(False)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._pause_btn.setEnabled(False)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._stop_btn.setEnabled(False)

        tracking_layout.addWidget(self._start_btn)
        tracking_layout.addWidget(self._pause_btn)
        tracking_layout.addWidget(self._stop_btn)

        controls_layout.addLayout(tracking_layout)

        # Sensitivity slider
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("Sensitivity:"))

        self._sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self._sensitivity_slider.setMinimum(10)  # 0.1
        self._sensitivity_slider.setMaximum(30)  # 3.0
        self._sensitivity_slider.setValue(10)  # 1.0
        self._sensitivity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._sensitivity_slider.setTickInterval(5)
        self._sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)

        self._sensitivity_value_label = QLabel("1.0")

        sensitivity_layout.addWidget(self._sensitivity_slider)
        sensitivity_layout.addWidget(self._sensitivity_value_label)

        controls_layout.addLayout(sensitivity_layout)

        # Delete calibration button
        self._delete_cal_btn = QPushButton("Delete Calibration Data")
        self._delete_cal_btn.clicked.connect(self._on_delete_calibration_clicked)
        controls_layout.addWidget(self._delete_cal_btn)

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # Stretch to push everything up
        main_layout.addStretch()

        # Update initial button states
        self._update_button_states()

    def _start_worker(self):
        """Start processing worker thread."""
        self._worker = ProcessingWorker(self._controller)
        self._worker.frame_processed.connect(self._on_frame_processed)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.start()

        logger.info("Worker thread started")

    def _on_calibrate_clicked(self):
        """Handle calibrate button click."""
        if self._controller.start_calibration():
            self._show_calibration_overlay()
            self._calibration_timer.start()
            self._update_button_states()

    def _on_start_clicked(self):
        """Handle start tracking button click."""
        if self._controller.start_tracking():
            self._update_button_states()
        else:
            # Show error if no calibration
            if not self._controller.has_calibration:
                QMessageBox.warning(
                    self,
                    "No Calibration",
                    "Please calibrate first before starting tracking.",
                )

    def _on_pause_clicked(self):
        """Handle pause button click."""
        if self._controller.state == AppState.TRACKING:
            self._controller.pause_tracking()
        elif self._controller.state == AppState.PAUSED:
            self._controller.resume_tracking()

        self._update_button_states()

    def _on_stop_clicked(self):
        """Handle stop button click."""
        self._controller.stop_tracking()
        self._update_button_states()

    def _on_sensitivity_changed(self, value: int):
        """Handle sensitivity slider change."""
        sensitivity = value / 10.0
        self._sensitivity_value_label.setText(f"{sensitivity:.1f}")
        self._controller.update_sensitivity(sensitivity)

    def _on_delete_calibration_clicked(self):
        """Handle delete calibration button click."""
        reply = QMessageBox.question(
            self,
            "Delete Calibration",
            "Are you sure you want to delete calibration data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._controller.delete_calibration()
            QMessageBox.information(
                self,
                "Calibration Deleted",
                "Calibration data has been deleted.",
            )
            self._update_button_states()

    def _on_frame_processed(self, result: FrameProcessingResult):
        """
        Handle frame processing result from worker.

        Called in main thread via signal.

        Args:
            result: Processing result
        """
        # Update FPS
        self._fps_label.setText(f"FPS: {result.fps:.1f}")

        # Update face detection status
        if result.face_detected:
            self._face_label.setText("Face: ✓ Detected")
            self._face_label.setStyleSheet("color: green;")
        else:
            self._face_label.setText("Face: ✗ Not detected")
            self._face_label.setStyleSheet("color: red;")

    def _on_worker_error(self, error_msg: str):
        """Handle worker thread error."""
        logger.error(f"Worker error: {error_msg}")
        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred during processing:\n{error_msg}",
        )

    def _show_calibration_overlay(self):
        """Show full-screen calibration overlay."""
        if self._calibration_overlay is None:
            self._calibration_overlay = CalibrationTargetWidget()
            self._calibration_overlay.setWindowFlags(
                Qt.WindowType.Window
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
            )
            self._calibration_overlay.showFullScreen()

        self._calibration_overlay.show()

    def _hide_calibration_overlay(self):
        """Hide calibration overlay."""
        if self._calibration_overlay:
            self._calibration_overlay.hide()

    def _update_calibration_ui(self):
        """Update calibration UI (called by timer)."""
        if self._controller.state != AppState.CALIBRATING:
            self._calibration_timer.stop()
            self._hide_calibration_overlay()
            self._update_button_states()
            return

        calibrator = self._controller.calibrator
        if not calibrator:
            return

        target = calibrator.get_current_target()
        if not target:
            return

        # Update overlay
        self._calibration_overlay.set_target(
            target.screen_x,
            target.screen_y,
            self._config.calibration.target_size,
        )

        # Update instruction
        current, total = calibrator.progress
        instruction = f"Calibration: Point {current + 1} of {total}"
        self._calibration_overlay.set_instruction(instruction)

        # Update countdown
        if calibrator.state == CalibrationState.COUNTDOWN:
            # Show countdown
            # This is simplified - you'd track exact countdown time
            self._calibration_overlay.set_countdown("Ready...")
        elif calibrator.state == CalibrationState.COLLECTING:
            self._calibration_overlay.set_countdown("Look Here!")
        else:
            self._calibration_overlay.set_countdown("")

    def _update_button_states(self):
        """Update button enabled/disabled states based on current state."""
        state = self._controller.state
        has_calibration = self._controller.has_calibration

        # Calibrate button: enabled when idle
        self._calibrate_btn.setEnabled(state == AppState.IDLE)

        # Start button: enabled when idle and has calibration
        self._start_btn.setEnabled(state == AppState.IDLE and has_calibration)

        # Pause button: enabled when tracking or paused
        if state == AppState.TRACKING:
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("Pause")
        elif state == AppState.PAUSED:
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("Resume")
        else:
            self._pause_btn.setEnabled(False)
            self._pause_btn.setText("Pause")

        # Stop button: enabled when tracking or paused
        self._stop_btn.setEnabled(state in (AppState.TRACKING, AppState.PAUSED))

        # Update status label
        status_text = {
            AppState.IDLE: "Idle",
            AppState.CALIBRATING: "Calibrating...",
            AppState.TRACKING: "Tracking Active",
            AppState.PAUSED: "Paused",
            AppState.ERROR: "Error",
        }
        self._status_label.setText(f"Status: {status_text.get(state, 'Unknown')}")

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Application closing")

        # Stop worker thread
        if self._worker:
            self._worker.stop()
            self._worker.wait(5000)  # Wait up to 5 seconds

        # Shutdown controller
        self._controller.shutdown()

        event.accept()
