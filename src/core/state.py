"""
Application state management.

Defines the state machine for VisionCursor with clear transitions.
"""

from enum import Enum, auto
from typing import Optional, Set
from dataclasses import dataclass


class AppState(Enum):
    """
    Application states.

    State transitions:
        IDLE -> CALIBRATING -> IDLE
        IDLE -> TRACKING -> PAUSED -> TRACKING
        TRACKING -> IDLE
        Any -> ERROR -> IDLE
    """

    IDLE = auto()           # Camera off, no tracking
    CALIBRATING = auto()    # Running calibration procedure
    TRACKING = auto()       # Active cursor tracking
    PAUSED = auto()         # Tracking paused, camera still active
    ERROR = auto()          # Error state, requires user intervention


@dataclass
class StateTransition:
    """Represents a state transition with validation."""

    from_state: AppState
    to_state: AppState

    def __post_init__(self):
        """Validate transition is allowed."""
        if not is_valid_transition(self.from_state, self.to_state):
            raise ValueError(
                f"Invalid state transition: {self.from_state.name} -> {self.to_state.name}"
            )


# Define valid state transitions
_VALID_TRANSITIONS: dict[AppState, Set[AppState]] = {
    AppState.IDLE: {
        AppState.CALIBRATING,
        AppState.TRACKING,
        AppState.ERROR,
    },
    AppState.CALIBRATING: {
        AppState.IDLE,
        AppState.ERROR,
    },
    AppState.TRACKING: {
        AppState.PAUSED,
        AppState.IDLE,
        AppState.ERROR,
    },
    AppState.PAUSED: {
        AppState.TRACKING,
        AppState.IDLE,
        AppState.ERROR,
    },
    AppState.ERROR: {
        AppState.IDLE,
    },
}


def is_valid_transition(from_state: AppState, to_state: AppState) -> bool:
    """
    Check if a state transition is valid.

    Args:
        from_state: Current state
        to_state: Target state

    Returns:
        True if transition is allowed, False otherwise
    """
    # Same state is always valid (no-op)
    if from_state == to_state:
        return True

    return to_state in _VALID_TRANSITIONS.get(from_state, set())


@dataclass
class ErrorInfo:
    """Information about an error that occurred."""

    error_type: str
    message: str
    recoverable: bool = True
    details: Optional[str] = None


class StateMachine:
    """
    State machine for managing application state transitions.

    Thread-safe state management with validation.
    """

    def __init__(self, initial_state: AppState = AppState.IDLE):
        """
        Initialize state machine.

        Args:
            initial_state: Starting state (default: IDLE)
        """
        self._current_state = initial_state
        self._previous_state: Optional[AppState] = None
        self._error: Optional[ErrorInfo] = None

    @property
    def current_state(self) -> AppState:
        """Get current state."""
        return self._current_state

    @property
    def previous_state(self) -> Optional[AppState]:
        """Get previous state."""
        return self._previous_state

    @property
    def error(self) -> Optional[ErrorInfo]:
        """Get error information if in ERROR state."""
        return self._error

    def transition_to(self, new_state: AppState) -> bool:
        """
        Transition to a new state.

        Args:
            new_state: Target state

        Returns:
            True if transition succeeded, False if invalid
        """
        if not is_valid_transition(self._current_state, new_state):
            return False

        self._previous_state = self._current_state
        self._current_state = new_state

        # Clear error when leaving ERROR state
        if self._previous_state == AppState.ERROR and new_state != AppState.ERROR:
            self._error = None

        return True

    def set_error(self, error_info: ErrorInfo) -> bool:
        """
        Set error state with error information.

        Args:
            error_info: Information about the error

        Returns:
            True if transition to ERROR succeeded
        """
        self._error = error_info
        return self.transition_to(AppState.ERROR)

    def can_transition_to(self, new_state: AppState) -> bool:
        """
        Check if can transition to a state without actually transitioning.

        Args:
            new_state: Target state to check

        Returns:
            True if transition would be valid
        """
        return is_valid_transition(self._current_state, new_state)

    def reset(self):
        """Reset to IDLE state, clearing error."""
        self._previous_state = self._current_state
        self._current_state = AppState.IDLE
        self._error = None
