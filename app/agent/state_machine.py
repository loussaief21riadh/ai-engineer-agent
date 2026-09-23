"""State Machine Hardening for V3.2 — deterministic transitions with validation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionState(str, Enum):
    """All possible execution states in the agent lifecycle."""
    IDLE = "IDLE"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    INSPECTING = "INSPECTING"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    SECURITY_CHECKING = "SECURITY_CHECKING"
    DIAGNOSING = "DIAGNOSING"
    FIXING = "FIXING"
    RETESTING = "RETESTING"
    REVIEWING = "REVIEWING"
    VALIDATING = "VALIDATING"
    REPLAYING = "REPLAYING"
    REPLANNING = "REPLANNING"
    REPORTING = "REPORTING"
    DONE = "DONE"
    FAILED = "FAILED"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"


VALID_TRANSITIONS: dict[ExecutionState, set[ExecutionState]] = {
    ExecutionState.IDLE: {
        ExecutionState.UNDERSTANDING,
        ExecutionState.CANCELLED,
    },
    ExecutionState.UNDERSTANDING: {
        ExecutionState.PLANNING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.PLANNING: {
        ExecutionState.INSPECTING,
        ExecutionState.IMPLEMENTING,
        ExecutionState.REPLANNING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.INSPECTING: {
        ExecutionState.IMPLEMENTING,
        ExecutionState.TESTING,
        ExecutionState.REPORTING,
        ExecutionState.REPLANNING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.IMPLEMENTING: {
        ExecutionState.TESTING,
        ExecutionState.DIAGNOSING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.TESTING: {
        ExecutionState.SECURITY_CHECKING,
        ExecutionState.DIAGNOSING,
        ExecutionState.REVIEWING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.SECURITY_CHECKING: {
        ExecutionState.IMPLEMENTING,
        ExecutionState.REVIEWING,
        ExecutionState.DIAGNOSING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.DIAGNOSING: {
        ExecutionState.FIXING,
        ExecutionState.REPLANNING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.FIXING: {
        ExecutionState.RETESTING,
        ExecutionState.TESTING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.RETESTING: {
        ExecutionState.IMPLEMENTING,
        ExecutionState.TESTING,
        ExecutionState.DIAGNOSING,
        ExecutionState.SECURITY_CHECKING,
        ExecutionState.REVIEWING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.REVIEWING: {
        ExecutionState.VALIDATING,
        ExecutionState.FIXING,
        ExecutionState.INSPECTING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.VALIDATING: {
        ExecutionState.REPORTING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.REPLAYING: {
        ExecutionState.TESTING,
        ExecutionState.IMPLEMENTING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.REPLANNING: {
        ExecutionState.INSPECTING,
        ExecutionState.IMPLEMENTING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.REPORTING: {
        ExecutionState.DONE,
        ExecutionState.FAILED,
    },
    ExecutionState.DONE: set(),
    ExecutionState.FAILED: set(),
    ExecutionState.PAUSED: {
        ExecutionState.UNDERSTANDING,
        ExecutionState.PLANNING,
        ExecutionState.INSPECTING,
        ExecutionState.IMPLEMENTING,
        ExecutionState.TESTING,
        ExecutionState.SECURITY_CHECKING,
        ExecutionState.DIAGNOSING,
        ExecutionState.FIXING,
        ExecutionState.RETESTING,
        ExecutionState.REVIEWING,
        ExecutionState.VALIDATING,
        ExecutionState.REPLAYING,
        ExecutionState.REPLANNING,
        ExecutionState.REPORTING,
        ExecutionState.CANCELLED,
    },
    ExecutionState.CANCELLED: set(),
}


class TransitionResult(BaseModel):
    """Result of a state transition attempt."""
    success: bool
    from_state: str
    to_state: str
    error: str = ""
    timestamp: float = 0.0


class StateMachine:
    """Deterministic state machine with transition validation and history."""

    def __init__(self, initial_state: ExecutionState = ExecutionState.IDLE) -> None:
        self._current_state: ExecutionState = initial_state
        self._history: list[dict[str, Any]] = []
        self._transition_count: int = 0
        self._max_transitions: int = 100

    @property
    def current_state(self) -> ExecutionState:
        return self._current_state

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def can_transition(self, target: ExecutionState) -> bool:
        """Check if transition to target state is valid."""
        allowed = VALID_TRANSITIONS.get(self._current_state, set())
        return target in allowed

    def get_valid_targets(self) -> set[ExecutionState]:
        """Get all valid target states from current state."""
        return set(VALID_TRANSITIONS.get(self._current_state, set()))

    def transition_to(self, target: ExecutionState) -> TransitionResult:
        """Attempt transition to target state. Returns TransitionResult."""
        import time

        if self._transition_count >= self._max_transitions:
            return TransitionResult(
                success=False,
                from_state=self._current_state.value,
                to_state=target.value,
                error=f"Maximum transitions ({self._max_transitions}) exceeded",
                timestamp=time.time(),
            )

        if not self.can_transition(target):
            valid = [s.value for s in self.get_valid_targets()]
            return TransitionResult(
                success=False,
                from_state=self._current_state.value,
                to_state=target.value,
                error=f"Invalid transition from {self._current_state.value} to {target.value}. Valid targets: {valid}",
                timestamp=time.time(),
            )

        old_state = self._current_state
        self._current_state = target
        self._transition_count += 1

        entry = {
            "from": old_state.value,
            "to": target.value,
            "transition_number": self._transition_count,
            "timestamp": time.time(),
        }
        self._history.append(entry)

        return TransitionResult(
            success=True,
            from_state=old_state.value,
            to_state=target.value,
            timestamp=time.time(),
        )

    def force_state(self, state: ExecutionState) -> None:
        """Force state (for checkpoint resume). Use sparingly."""
        self._current_state = state

    def is_terminal(self) -> bool:
        """Check if current state is terminal (no transitions possible)."""
        return len(self.get_valid_targets()) == 0

    def is_active(self) -> bool:
        """Check if state machine is in an active (non-terminal) state."""
        return not self.is_terminal() and self._current_state not in (
            ExecutionState.CANCELLED,
        )

    def reset(self, initial_state: ExecutionState = ExecutionState.IDLE) -> None:
        """Reset state machine to initial state."""
        self._current_state = initial_state
        self._history.clear()
        self._transition_count = 0

    def get_state_summary(self) -> dict[str, Any]:
        """Get current state machine summary."""
        return {
            "current_state": self._current_state.value,
            "transition_count": self._transition_count,
            "is_terminal": self.is_terminal(),
            "is_active": self.is_active(),
            "valid_targets": [s.value for s in self.get_valid_targets()],
            "history_length": len(self._history),
        }

    def validate_history(self) -> list[str]:
        """Validate that all transitions in history are valid."""
        errors: list[str] = []
        state = ExecutionState.IDLE

        for i, entry in enumerate(self._history):
            from_state = ExecutionState(entry["from"])
            to_state = ExecutionState(entry["to"])

            if state != from_state:
                errors.append(
                    f"Entry {i}: history state mismatch. Expected {state.value}, got {from_state.value}"
                )

            if not self.can_transition_from(from_state, to_state):
                errors.append(
                    f"Entry {i}: invalid transition {from_state.value} -> {to_state.value}"
                )

            state = to_state

        return errors

    def can_transition_from(self, source: ExecutionState, target: ExecutionState) -> bool:
        """Check if transition from source to target is valid."""
        allowed = VALID_TRANSITIONS.get(source, set())
        return target in allowed


def validate_all_transitions() -> list[str]:
    """Validate that all transitions in the state machine are consistent."""
    errors: list[str] = []

    for state, targets in VALID_TRANSITIONS.items():
        for target in targets:
            if target not in VALID_TRANSITIONS:
                errors.append(f"Target {target.value} not defined in VALID_TRANSITIONS")
            elif state not in VALID_TRANSITIONS.get(target, set()) and target not in (
                ExecutionState.DONE, ExecutionState.FAILED, ExecutionState.CANCELLED
            ):
                pass  # Not all transitions need to be bidirectional

    terminal_states = {ExecutionState.DONE, ExecutionState.FAILED, ExecutionState.CANCELLED}
    for state in terminal_states:
        if VALID_TRANSITIONS.get(state, set()):
            errors.append(f"Terminal state {state.value} has outgoing transitions")

    return errors
