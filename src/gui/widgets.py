"""
Custom GUI widgets for VisionCursor.
"""

from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import numpy as np


class CameraPreviewWidget(QWidget):
    """
    Widget to display camera preview with optional debug overlays.

    Privacy: Only displays in-memory frames, never saves to disk.
    """

    def __init__(self, width: int = 320, height: int = 240, parent=None):
        """
        Initialize preview widget.

        Args:
            width: Preview width
            height: Preview height
            parent: Parent widget
        """
        super().__init__(parent)

        self._preview_width = width
        self._preview_height = height

        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: black;")

        self._current_pixmap: QPixmap = None

    def update_frame(self, frame: np.ndarray):
        """
        Update preview with new frame.

        Args:
            frame: RGB frame (H, W, 3) as numpy array

        Privacy: Frame is only displayed, never saved.
        """
        if frame is None or frame.size == 0:
            return

        try:
            height, width, channels = frame.shape

            # Convert numpy array to QImage
            bytes_per_line = channels * width
            q_image = QImage(
                frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )

            # Scale to preview size
            scaled_image = q_image.scaled(
                self._preview_width,
                self._preview_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            self._current_pixmap = QPixmap.fromImage(scaled_image)
            self.update()  # Trigger repaint

        except Exception as e:
            # Silently fail on invalid frames
            pass

    def paintEvent(self, event):
        """Paint the preview."""
        painter = QPainter(self)

        if self._current_pixmap:
            # Center the pixmap
            x = (self.width() - self._current_pixmap.width()) // 2
            y = (self.height() - self._current_pixmap.height()) // 2
            painter.drawPixmap(x, y, self._current_pixmap)
        else:
            # Draw placeholder text
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No Camera Feed",
            )


class CalibrationTargetWidget(QWidget):
    """
    Full-screen overlay widget showing calibration target.
    """

    def __init__(self, parent=None):
        """Initialize calibration target widget."""
        super().__init__(parent)

        # Semi-transparent overlay
        self.setStyleSheet("background-color: rgba(0, 0, 0, 180);")

        # Target position and size
        self._target_x = 0
        self._target_y = 0
        self._target_size = 20
        self._target_visible = False

        # Instruction text
        self._instruction_text = ""
        self._countdown_text = ""

    def set_target(self, x: float, y: float, size: int = 20):
        """
        Set target position.

        Args:
            x: Target x position (pixels)
            y: Target y position (pixels)
            size: Target size (pixels)
        """
        self._target_x = int(x)
        self._target_y = int(y)
        self._target_size = size
        self._target_visible = True
        self.update()

    def set_instruction(self, text: str):
        """Set instruction text."""
        self._instruction_text = text
        self.update()

    def set_countdown(self, text: str):
        """Set countdown text."""
        self._countdown_text = text
        self.update()

    def hide_target(self):
        """Hide the target."""
        self._target_visible = False
        self.update()

    def paintEvent(self, event):
        """Paint the calibration overlay."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw target if visible
        if self._target_visible:
            # Outer circle (white)
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(
                self._target_x - self._target_size,
                self._target_y - self._target_size,
                self._target_size * 2,
                self._target_size * 2,
            )

            # Inner circle (red)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(
                self._target_x - self._target_size // 2,
                self._target_y - self._target_size // 2,
                self._target_size,
                self._target_size,
            )

        # Draw instruction text at top
        if self._instruction_text:
            painter.setPen(QColor(255, 255, 255))
            font = painter.font()
            font.setPointSize(16)
            painter.setFont(font)

            painter.drawText(
                0,
                50,
                self.width(),
                100,
                Qt.AlignmentFlag.AlignCenter,
                self._instruction_text,
            )

        # Draw countdown text near target
        if self._countdown_text and self._target_visible:
            painter.setPen(QColor(255, 255, 0))
            font = painter.font()
            font.setPointSize(48)
            font.setBold(True)
            painter.setFont(font)

            painter.drawText(
                self._target_x - 100,
                self._target_y - 100,
                200,
                80,
                Qt.AlignmentFlag.AlignCenter,
                self._countdown_text,
            )
