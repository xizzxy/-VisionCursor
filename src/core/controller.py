"""
Central controller orchestrating the gaze tracking pipeline.

Manages state transitions and coordinates all components.
"""

from typing import Optional, Callable
from dataclasses import dataclass
import time

from src.core.config import AppConfig
from src.core.state import StateMachine, AppState, ErrorInfo
from src.vision.camera import Camera, CameraFrame, CameraError
from src.vision.face_tracker import FaceTracker, FaceLandmarks
from src.vision.gaze_estimator import GazeEstimator, GazeVector
from src.vision.calibrator import Calibrator, CalibrationState, GazeMapper
from src.vision.smoothing import GazeSmoother, SmoothedGaze
from src.os_control.cursor_controller import CursorController
from src.storage.calibration_store import CalibrationStore, CalibrationStoreError
from src.storage.schema import CalibrationData
from src.utils.timing import FPSCounter
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameProcessingResult:
    """Result of processing a single frame."""

    success: bool
    face_detected: bool
    gaze: Optional[GazeVector] = None
    cursor_pos: Optional[tuple[int, int]] = None
    fps: float = 0.0


class Controller:
    """
    Central controller for VisionCursor.

    Manages the complete pipeline:
    camera → face detection → gaze estimation → calibration mapping → smoothing → cursor

    Thread-safe state management with dependency injection.
    """

    def __init__(
        self,
        config: AppConfig,
        screen_width: int,
        screen_height: int,
    ):
        """
        Initialize controller.

        Args:
            config: Application configuration
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self._config = config
        self._screen_width = screen_width
        self._screen_height = screen_height

        # State machine
        self._state_machine = StateMachine(initial_state=AppState.IDLE)

        # Components (initialized lazily)
        self._camera: Optional[Camera] = None
        self._face_tracker: Optional[FaceTracker] = None
        self._gaze_estimator: Optional[GazeEstimator] = None
        self._calibrator: Optional[Calibrator] = None
        self._gaze_mapper: Optional[GazeMapper] = None
        self._smoother: Optional[GazeSmoother] = None
        self._cursor_controller: Optional[CursorController] = None
        self._calibration_store: Optional[CalibrationStore] = None

        # Performance monitoring
        self._fps_counter = FPSCounter()

        # Calibration countdown tracking
        self._calibration_countdown_start: Optional[float] = None

        # Tracking control (for safety)
        self._tracking_enabled = False  # User must explicitly enable

        # Last valid cursor position (for freeze on face lost)
        self._last_cursor_pos: Optional[tuple[int, int]] = None

        logger.info(f"Controller initialized for {screen_width}x{screen_height}")

    def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if successful

        Raises:
            Exception: If initialization fails
        """
        try:
            # Camera
            self._camera = Camera(self._config.camera)

            # Face tracker
            self._face_tracker = FaceTracker(
                min_detection_confidence=self._config.gaze.min_face_confidence,
                min_tracking_confidence=self._config.gaze.min_face_confidence,
            )

            # Gaze estimator
            self._gaze_estimator = GazeEstimator()

            # Calibrator
            self._calibrator = Calibrator(
                self._config.calibration,
                self._screen_width,
                self._screen_height,
            )

            # Smoother
            self._smoother = GazeSmoother(
                self._config.gaze,
                self._screen_width,
                self._screen_height,
            )

            # Cursor controller
            self._cursor_controller = CursorController(
                self._screen_width,
                self._screen_height,
            )

            # Calibration storage
            self._calibration_store = CalibrationStore(self._config.storage)

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            error_msg = f"Initialization failed: {e}"
            logger.error(error_msg)
            self._state_machine.set_error(
                ErrorInfo(
                    error_type="InitializationError",
                    message=error_msg,
                    recoverable=False,
                )
            )
            return False

    def start_tracking(self) -> bool:
        """
        Start gaze tracking.

        Returns:
            True if started successfully

        Requires:
        - Valid calibration loaded or completed
        - Camera available
        """
        if not self._state_machine.can_transition_to(AppState.TRACKING):
            logger.warning(f"Cannot start tracking from state {self._state_machine.current_state}")
            return False

        # Check if calibration exists
        if self._gaze_mapper is None:
            # Try to load calibration
            if not self._load_calibration():
                self._state_machine.set_error(
                    ErrorInfo(
                        error_type="CalibrationError",
                        message="No calibration found. Please calibrate first.",
                        recoverable=True,
                    )
                )
                return False

        # Open camera
        try:
            if not self._camera.is_open:
                self._camera.open()

            self._cursor_controller.enable()
            self._smoother.reset()
            self._fps_counter.reset()
            self._tracking_enabled = True  # Enable tracking by default

            self._state_machine.transition_to(AppState.TRACKING)
            logger.info("Tracking started")
            return True

        except CameraError as e:
            self._state_machine.set_error(
                ErrorInfo(
                    error_type="CameraError",
                    message=str(e),
                    recoverable=True,
                )
            )
            return False

    def stop_tracking(self) -> bool:
        """
        Stop gaze tracking.

        Returns:
            True if stopped successfully
        """
        if self._state_machine.current_state == AppState.TRACKING:
            self._tracking_enabled = False  # Disable tracking
            self._cursor_controller.disable()
            self._camera.close()
            self._state_machine.transition_to(AppState.IDLE)
            logger.info("Tracking stopped")
            return True

        return False

    def pause_tracking(self) -> bool:
        """Pause tracking (keep camera active)."""
        if self._state_machine.current_state == AppState.TRACKING:
            self._cursor_controller.disable()
            self._state_machine.transition_to(AppState.PAUSED)
            logger.info("Tracking paused")
            return True

        return False

    def resume_tracking(self) -> bool:
        """Resume tracking from paused state."""
        if self._state_machine.current_state == AppState.PAUSED:
            self._cursor_controller.enable()
            self._smoother.reset()
            self._state_machine.transition_to(AppState.TRACKING)
            logger.info("Tracking resumed")
            return True

        return False

    def start_calibration(self) -> bool:
        """
        Start calibration procedure.

        Returns:
            True if started successfully
        """
        if not self._state_machine.can_transition_to(AppState.CALIBRATING):
            logger.warning(f"Cannot start calibration from state {self._state_machine.current_state}")
            return False

        try:
            # Open camera if not already open
            if not self._camera.is_open:
                self._camera.open()

            # Start calibrator
            self._calibrator.start()
            self._calibrator.set_state(CalibrationState.COUNTDOWN)
            self._calibration_countdown_start = time.time()

            self._state_machine.transition_to(AppState.CALIBRATING)
            logger.info("Calibration started")
            return True

        except CameraError as e:
            self._state_machine.set_error(
                ErrorInfo(
                    error_type="CameraError",
                    message=str(e),
                    recoverable=True,
                )
            )
            return False

    def process_frame(self) -> FrameProcessingResult:
        """
        Process one frame of the pipeline.

        Returns:
            FrameProcessingResult with processing details

        Call this from the main loop / worker thread.
        """
        result = FrameProcessingResult(
            success=False,
            face_detected=False,
            fps=self._fps_counter.tick(),
        )

        current_state = self._state_machine.current_state

        # Read camera frame
        frame = self._camera.read_frame()
        if frame is None:
            return result

        # Detect face
        face_landmarks = self._face_tracker.process_frame(frame.image)
        if face_landmarks is None:
            # Face not detected - freeze cursor if configured
            if current_state == AppState.TRACKING and self._config.gaze.freeze_on_face_lost:
                result.cursor_pos = self._last_cursor_pos
            return result

        result.face_detected = True

        # Estimate gaze
        gaze = self._gaze_estimator.estimate(face_landmarks)
        if gaze is None:
            # Gaze estimation failed - freeze cursor if configured
            if current_state == AppState.TRACKING and self._config.gaze.freeze_on_face_lost:
                result.cursor_pos = self._last_cursor_pos
            return result

        result.gaze = gaze

        # State-specific processing
        if current_state == AppState.CALIBRATING:
            self._process_calibration_frame(gaze)

        elif current_state == AppState.TRACKING:
            cursor_pos = self._process_tracking_frame(gaze)
            result.cursor_pos = cursor_pos

        result.success = True
        return result

    def _process_calibration_frame(self, gaze: GazeVector):
        """Process frame during calibration."""
        cal_state = self._calibrator.state

        # Handle countdown
        if cal_state == CalibrationState.COUNTDOWN:
            elapsed = time.time() - self._calibration_countdown_start

            if elapsed >= self._config.calibration.countdown_seconds:
                # Start collecting samples
                self._calibrator.set_state(CalibrationState.COLLECTING)
                logger.debug("Calibration sample collection started")

        # Collect samples
        elif cal_state == CalibrationState.COLLECTING:
            self._calibrator.add_sample(gaze)

            # Check if target completed (handled internally by calibrator)
            if self._calibrator.state == CalibrationState.COUNTDOWN:
                # Target completed, moved to next target
                self._calibration_countdown_start = time.time()

            elif self._calibrator.state == CalibrationState.COMPLETED:
                # All targets completed, save calibration
                self._finalize_calibration()

    def _process_tracking_frame(self, gaze: GazeVector) -> Optional[tuple[int, int]]:
        """Process frame during tracking."""
        if self._gaze_mapper is None:
            return None

        # Check if tracking is enabled (safety control)
        if not self._tracking_enabled:
            return self._last_cursor_pos

        # Map gaze to screen coordinates
        screen_x, screen_y = self._gaze_mapper.map_gaze_to_screen(gaze)

        # Apply smoothing
        smoothed = self._smoother.smooth(screen_x, screen_y)

        # Move cursor
        self._cursor_controller.move_to(smoothed.x, smoothed.y)

        # Remember last position
        self._last_cursor_pos = (smoothed.x, smoothed.y)

        return (smoothed.x, smoothed.y)

    def _finalize_calibration(self):
        """Finalize and save calibration."""
        calibration_data = self._calibrator.calibration_data

        if calibration_data is None:
            self._state_machine.set_error(
                ErrorInfo(
                    error_type="CalibrationError",
                    message="Calibration failed to finalize",
                    recoverable=True,
                )
            )
            return

        try:
            # Save to disk
            self._calibration_store.save(calibration_data)

            # Create mapper
            self._gaze_mapper = GazeMapper(calibration_data)

            # Return to idle
            self._camera.close()
            self._state_machine.transition_to(AppState.IDLE)

            logger.info("Calibration completed and saved successfully")

        except CalibrationStoreError as e:
            self._state_machine.set_error(
                ErrorInfo(
                    error_type="StorageError",
                    message=f"Failed to save calibration: {e}",
                    recoverable=True,
                )
            )

    def _load_calibration(self) -> bool:
        """Load calibration from disk."""
        try:
            calibration_data = self._calibration_store.load()

            if calibration_data is None:
                logger.info("No saved calibration found")
                return False

            # Check if compatible with current screen
            if not calibration_data.is_compatible_with_screen(
                self._screen_width, self._screen_height
            ):
                logger.warning(
                    f"Calibration screen size mismatch: "
                    f"calibrated for {calibration_data.screen_width}x{calibration_data.screen_height}, "
                    f"current {self._screen_width}x{self._screen_height}"
                )
                return False

            # Create mapper
            self._gaze_mapper = GazeMapper(calibration_data)
            logger.info("Calibration loaded successfully")
            return True

        except CalibrationStoreError as e:
            logger.error(f"Failed to load calibration: {e}")
            return False

    def delete_calibration(self) -> bool:
        """Delete saved calibration data."""
        try:
            self._calibration_store.delete()
            self._gaze_mapper = None
            logger.info("Calibration deleted")
            return True

        except CalibrationStoreError as e:
            logger.error(f"Failed to delete calibration: {e}")
            return False

    def update_sensitivity(self, sensitivity: float):
        """Update cursor sensitivity."""
        self._config.gaze.sensitivity = sensitivity
        self._smoother.update_config(self._config.gaze)
        logger.debug(f"Sensitivity updated: {sensitivity:.2f}")

    def enable_tracking(self):
        """Enable cursor tracking (safety control)."""
        self._tracking_enabled = True
        logger.info("Cursor tracking enabled")

    def disable_tracking(self):
        """Disable cursor tracking (safety control)."""
        self._tracking_enabled = False
        logger.info("Cursor tracking disabled")

    def toggle_tracking(self) -> bool:
        """
        Toggle tracking enabled state.

        Returns:
            New tracking state (True = enabled)
        """
        self._tracking_enabled = not self._tracking_enabled
        state_str = "enabled" if self._tracking_enabled else "disabled"
        logger.info(f"Cursor tracking toggled: {state_str}")
        return self._tracking_enabled

    def is_tracking_enabled(self) -> bool:
        """Check if tracking is currently enabled."""
        return self._tracking_enabled

    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down controller")

        if self._camera and self._camera.is_open:
            self._camera.close()

        if self._face_tracker:
            self._face_tracker.close()

        if self._cursor_controller:
            self._cursor_controller.disable()

        self._state_machine.reset()

    # Properties
    @property
    def state(self) -> AppState:
        """Get current application state."""
        return self._state_machine.current_state

    @property
    def error(self):
        """Get error information if in ERROR state."""
        return self._state_machine.error

    @property
    def calibrator(self) -> Optional[Calibrator]:
        """Get calibrator (for UI to access calibration targets)."""
        return self._calibrator

    @property
    def has_calibration(self) -> bool:
        """Check if calibration exists."""
        return self._gaze_mapper is not None or self._calibration_store.exists()

    @property
    def fps(self) -> float:
        """Get current processing FPS."""
        return self._fps_counter.fps
