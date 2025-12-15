"""
Main application window with worker thread architecture.

Professional UI with safety controls, camera selection, and improved calibration flow.

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
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QProgressBar,
    QTextEdit,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QEvent
from PyQt6.QtGui import QScreen, QKeySequence, QShortcut, QFont
import time

from src.core.controller import Controller, FrameProcessingResult
from src.core.config import AppConfig
from src.core.state import AppState
from src.vision.calibrator import CalibrationState
from src.vision.camera import list_available_cameras
from src.gui.widgets import CameraPreviewWidget, CalibrationTargetWidget
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationDialog(QDialog):
    """
    Modal dialog explaining calibration process before it starts.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.setMinimumWidth(450)
        self.setMinimumHeight(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Title
        title = QLabel("Calibration Process")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Simplified explanation
        explanation = QTextEdit()
        explanation.setReadOnly(True)
        explanation.setPlainText(
            "This process maps your eye movements to screen positions.\n\n"
            "What happens:\n"
            "• Five target dots will appear (center, left, right, top, bottom)\n"
            "• Look directly at each target when it appears\n"
            "• Duration: approximately 30 seconds\n\n"
            "Instructions:\n"
            "• Sit comfortably at your normal distance from the screen\n"
            "• Keep your head still (move only your eyes)\n"
            "• Ensure good lighting on your face\n"
            "• Press ESC to cancel at any time\n\n"
            "Note: Tracking will not move your cursor until you enable it after calibration."
        )
        explanation.setMinimumHeight(240)
        layout.addWidget(explanation)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Rename OK button to "Start Calibration"
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setText("Start Calibration")

        layout.addWidget(buttons)


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

    Professional UI with comprehensive safety controls.
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

        # Initialize controller (camera index set later)
        self._controller: Controller = None

        # Available cameras
        self._available_cameras = list_available_cameras()

        # Worker thread
        self._worker: ProcessingWorker = None

        # Calibration overlay
        self._calibration_overlay: CalibrationTargetWidget = None

        # UI setup
        self._init_ui()

        # Timer for UI updates during calibration
        self._calibration_timer = QTimer(self)
        self._calibration_timer.timeout.connect(self._update_calibration_ui)
        self._calibration_timer.setInterval(100)  # 10 Hz

        # Keyboard shortcut for emergency toggle
        self._setup_keyboard_shortcuts()

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for safety."""
        # Ctrl+Q to toggle tracking (safe, non-intrusive)
        self._toggle_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self._toggle_shortcut.activated.connect(self._on_toggle_tracking_shortcut)

    def _init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle(self._config.ui.window_title)
        self.setGeometry(100, 100, self._config.ui.window_width, self._config.ui.window_height)
        self.setMinimumSize(self._config.ui.window_min_width, self._config.ui.window_min_height)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(
            self._config.ui.content_margin,
            self._config.ui.content_margin,
            self._config.ui.content_margin,
            self._config.ui.content_margin,
        )
        main_layout.setSpacing(self._config.ui.group_spacing)

        # Tracking status and control (PRIMARY - most important)
        tracking_section = self._create_tracking_section()
        main_layout.addWidget(tracking_section)

        # Status display (face detection, FPS)
        status_section = self._create_status_section()
        main_layout.addWidget(status_section)

        # Camera selection
        camera_section = self._create_camera_section()
        main_layout.addWidget(camera_section)

        # Calibration and controls
        controls_section = self._create_controls_section()
        main_layout.addWidget(controls_section)

        # Sensitivity control
        sensitivity_section = self._create_sensitivity_section()
        main_layout.addWidget(sensitivity_section)

        # Data management (less prominent)
        data_section = self._create_data_section()
        main_layout.addWidget(data_section)

        # Stretch to push everything up
        main_layout.addStretch()

        # Subtle privacy notice at bottom (dismissible)
        self._privacy_notice = self._create_privacy_notice()
        main_layout.addWidget(self._privacy_notice)

    def _create_privacy_notice(self) -> QWidget:
        """Create subtle dismissible privacy notice at bottom."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 3, 5, 3)

        notice = QLabel("Privacy: All processing is local. No images or video are stored or transmitted.")
        notice.setStyleSheet("color: #888; font-size: 9pt;")
        notice.setWordWrap(True)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(16, 16)
        close_btn.setStyleSheet("QPushButton { border: none; color: #888; font-size: 14pt; padding: 0; }")
        close_btn.clicked.connect(lambda: container.setVisible(False))

        layout.addWidget(notice)
        layout.addWidget(close_btn)

        return container

    def _create_camera_section(self) -> QWidget:
        """Create camera selection controls (simplified, no group box)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Title
        title = QLabel("Camera")
        title.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(title)

        # Camera selection
        camera_layout = QHBoxLayout()

        self._camera_combo = QComboBox()
        if self._available_cameras:
            for cam_idx in self._available_cameras:
                self._camera_combo.addItem(f"Camera {cam_idx}", cam_idx)
        else:
            self._camera_combo.addItem("No camera detected", -1)

        camera_layout.addWidget(self._camera_combo)

        self._test_camera_btn = QPushButton("Test")
        self._test_camera_btn.setStyleSheet("color: #666; font-size: 9pt;")
        self._test_camera_btn.clicked.connect(self._on_test_camera_clicked)
        camera_layout.addWidget(self._test_camera_btn)

        camera_layout.addStretch()
        layout.addLayout(camera_layout)

        return container

    def _create_status_section(self) -> QWidget:
        """Create status display (no group box, cleaner)."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(20)

        self._face_label = QLabel("Face: Not detected")
        self._face_label.setStyleSheet("color: #666;")

        self._fps_label = QLabel("FPS: 0.0")
        self._fps_label.setStyleSheet("color: #999; font-size: 9pt;")

        layout.addWidget(self._face_label)
        layout.addWidget(self._fps_label)
        layout.addStretch()

        return container

    def _create_tracking_section(self) -> QWidget:
        """Create tracking status and control (PRIMARY - most prominent)."""
        container = QWidget()
        container.setStyleSheet("background-color: #f5f5f5; border-radius: 4px; padding: 12px;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Tracking status (large, clear)
        self._tracking_status_label = QLabel("Tracking: OFF")
        self._tracking_status_label.setStyleSheet(
            "font-weight: bold; font-size: 14pt; color: #888;"
        )

        # Toggle button
        self._toggle_tracking_btn = QPushButton("Enable Tracking")
        self._toggle_tracking_btn.setEnabled(False)
        self._toggle_tracking_btn.clicked.connect(self._on_toggle_tracking_clicked)
        self._toggle_tracking_btn.setMinimumHeight(35)

        # Subtle shortcut hint
        shortcut_hint = QLabel("Ctrl+Q to toggle")
        shortcut_hint.setStyleSheet("color: #aaa; font-size: 8pt;")

        layout.addWidget(self._tracking_status_label)
        layout.addWidget(self._toggle_tracking_btn)
        layout.addWidget(shortcut_hint)

        return container

    def _create_controls_section(self) -> QWidget:
        """Create main control buttons (simplified)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(8)

        # Title
        title = QLabel("Controls")
        title.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(title)

        # Calibration button
        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        layout.addWidget(self._calibrate_btn)

        # Tracking buttons
        tracking_layout = QHBoxLayout()

        self._start_btn = QPushButton("Start")
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

        layout.addLayout(tracking_layout)

        return container

    def _create_sensitivity_section(self) -> QWidget:
        """Create sensitivity slider (simplified)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(4)

        # Title
        title_layout = QHBoxLayout()
        title = QLabel("Sensitivity")
        title.setStyleSheet("font-weight: bold; color: #555;")
        title_layout.addWidget(title)

        self._sensitivity_value_label = QLabel("0.8")
        self._sensitivity_value_label.setStyleSheet("color: #999; font-size: 9pt;")
        title_layout.addWidget(self._sensitivity_value_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Low"))

        self._sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self._sensitivity_slider.setMinimum(1)  # 0.1
        self._sensitivity_slider.setMaximum(30)  # 3.0
        self._sensitivity_slider.setValue(8)  # 0.8 (default)
        self._sensitivity_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)

        slider_layout.addWidget(self._sensitivity_slider)
        slider_layout.addWidget(QLabel("High"))

        layout.addLayout(slider_layout)

        return container

    def _create_data_section(self) -> QWidget:
        """Create data management controls (de-emphasized)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(4)

        self._delete_cal_btn = QPushButton("Delete Calibration")
        self._delete_cal_btn.setStyleSheet("color: #999; font-size: 9pt;")
        self._delete_cal_btn.clicked.connect(self._on_delete_calibration_clicked)
        layout.addWidget(self._delete_cal_btn)

        return container

    def _initialize_controller(self, camera_index: int) -> bool:
        """
        Initialize controller with selected camera.

        Args:
            camera_index: Camera index to use

        Returns:
            True if successful
        """
        # Update config with selected camera
        self._config.camera.camera_index = camera_index

        # Create controller
        self._controller = Controller(
            self._config,
            self._screen_width,
            self._screen_height,
        )

        if not self._controller.initialize():
            QMessageBox.critical(
                self,
                "Initialization Error",
                "Failed to initialize VisionCursor. Check that the camera is available and not in use by another application.",
            )
            return False

        # Start worker thread
        self._start_worker()

        return True

    def _start_worker(self):
        """Start processing worker thread."""
        self._worker = ProcessingWorker(self._controller)
        self._worker.frame_processed.connect(self._on_frame_processed)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.start()

        logger.info("Worker thread started")

    def _on_test_camera_clicked(self):
        """Test camera access."""
        camera_index = self._camera_combo.currentData()

        if camera_index < 0:
            QMessageBox.warning(
                self,
                "No Camera",
                "No cameras were detected. Please connect a camera and restart the application.",
            )
            return

        # Initialize controller if not already done
        if self._controller is None:
            if not self._initialize_controller(camera_index):
                return

        QMessageBox.information(
            self,
            "Camera Test",
            f"Camera {camera_index} is accessible and ready for use.\n\n"
            "You can now proceed with calibration.",
        )

    def _on_calibrate_clicked(self):
        """Handle calibrate button click."""
        # Ensure controller is initialized
        if self._controller is None:
            camera_index = self._camera_combo.currentData()
            if camera_index < 0:
                QMessageBox.warning(
                    self,
                    "No Camera",
                    "Please select a valid camera before calibrating.",
                )
                return

            if not self._initialize_controller(camera_index):
                return

        # Show calibration explanation dialog
        dialog = CalibrationDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return  # User cancelled

        # Start calibration
        if self._controller.start_calibration():
            self._show_calibration_overlay()
            self._calibration_timer.start()
            self._update_button_states()

    def _on_start_clicked(self):
        """Handle start tracking button click."""
        if self._controller.start_tracking():
            self._update_button_states()
            self._update_tracking_status()
        else:
            # Show error if no calibration
            if not self._controller.has_calibration:
                QMessageBox.warning(
                    self,
                    "No Calibration",
                    "Please calibrate the system before starting tracking.",
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
        self._update_tracking_status()

    def _on_toggle_tracking_clicked(self):
        """Handle toggle tracking button click."""
        if self._controller and self._controller.state == AppState.TRACKING:
            enabled = self._controller.toggle_tracking()
            self._update_tracking_status()

    def _on_toggle_tracking_shortcut(self):
        """Handle keyboard shortcut to toggle tracking."""
        if self._controller and self._controller.state == AppState.TRACKING:
            enabled = self._controller.toggle_tracking()
            self._update_tracking_status()

    def _update_tracking_status(self):
        """Update tracking status display (neutral colors, calm presentation)."""
        if self._controller and self._controller.state == AppState.TRACKING:
            enabled = self._controller.is_tracking_enabled()

            if enabled:
                self._tracking_status_label.setText("Tracking: ON")
                self._tracking_status_label.setStyleSheet(
                    "font-weight: bold; font-size: 14pt; color: #4caf50;"  # Neutral green
                )
                self._toggle_tracking_btn.setText("Disable")
            else:
                self._tracking_status_label.setText("Tracking: OFF")
                self._tracking_status_label.setStyleSheet(
                    "font-weight: bold; font-size: 14pt; color: #888;"  # Neutral grey, not red
                )
                self._toggle_tracking_btn.setText("Enable")

            self._toggle_tracking_btn.setEnabled(True)
        else:
            self._tracking_status_label.setText("Tracking: OFF")
            self._tracking_status_label.setStyleSheet(
                "font-weight: bold; font-size: 14pt; color: #888;"
            )
            self._toggle_tracking_btn.setEnabled(False)
            self._toggle_tracking_btn.setText("Enable Tracking")

    def _on_sensitivity_changed(self, value: int):
        """Handle sensitivity slider change."""
        sensitivity = value / 10.0
        self._sensitivity_value_label.setText(f"{sensitivity:.1f}")

        if self._controller:
            self._controller.update_sensitivity(sensitivity)

    def _on_delete_calibration_clicked(self):
        """Handle delete calibration button click."""
        reply = QMessageBox.question(
            self,
            "Delete Calibration",
            "Are you sure you want to delete calibration data?\n\n"
            "You will need to calibrate again before using cursor tracking.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self._controller:
                self._controller.delete_calibration()

            QMessageBox.information(
                self,
                "Calibration Deleted",
                "Calibration data has been deleted successfully.",
            )
            self._update_button_states()

    def _on_frame_processed(self, result: FrameProcessingResult):
        """
        Handle frame processing result from worker.

        Called in main thread via signal.
        """
        # Update FPS
        self._fps_label.setText(f"FPS: {result.fps:.1f}")

        # Update face detection status (neutral colors)
        if result.face_detected:
            self._face_label.setText("Face: Detected")
            self._face_label.setStyleSheet("color: #666;")  # Neutral, not bright green
        else:
            self._face_label.setText("Face: Not detected")
            self._face_label.setStyleSheet("color: #999;")  # Grey, not red (not an error)

    def _on_worker_error(self, error_msg: str):
        """Handle worker thread error."""
        logger.error(f"Worker error: {error_msg}")
        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred during processing:\n{error_msg}\n\n"
            "Please check that the camera is connected and accessible.",
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
        instruction = f"Calibration: Target {current + 1} of {total}"
        self._calibration_overlay.set_instruction(instruction)

        # Update countdown/status
        if calibrator.state == CalibrationState.COUNTDOWN:
            self._calibration_overlay.set_countdown("Get ready...")
        elif calibrator.state == CalibrationState.COLLECTING:
            # Show progress
            samples = len(target.samples)
            total_samples = self._config.calibration.samples_per_point
            progress_text = f"Collecting... ({samples}/{total_samples})"
            self._calibration_overlay.set_countdown(progress_text)
        else:
            self._calibration_overlay.set_countdown("")

    def _update_button_states(self):
        """Update button enabled/disabled states based on current state."""
        if self._controller is None:
            # Before controller initialization
            self._calibrate_btn.setEnabled(True)
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            return

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

        # Update tracking status (no separate status label needed - state is clear from buttons and tracking status)
        self._update_tracking_status()

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Application closing")

        # Stop worker thread
        if self._worker:
            self._worker.stop()
            self._worker.wait(5000)  # Wait up to 5 seconds

        # Shutdown controller
        if self._controller:
            self._controller.shutdown()

        event.accept()
