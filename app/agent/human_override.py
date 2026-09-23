"""Human Override for V5 — STOP/CANCEL/PAUSE/RESUME controls."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class OverrideCommand(str, Enum):
    STOP = "STOP"
    CANCEL = "CANCEL"
    PAUSE = "PAUSE"
    RESUME = "RESUME"


class OverrideState(BaseModel):
    """Current override state."""
    command: OverrideCommand = OverrideCommand.RESUME
    active: bool = False
    reason: str = ""
    timestamp: float = 0.0


class HumanOverride:
    """Human override controls for autonomous execution."""

    def __init__(self) -> None:
        self._state = OverrideState()
        self._history: list[OverrideState] = []

    def stop(self, reason: str = "") -> None:
        import time
        self._state = OverrideState(
            command=OverrideCommand.STOP,
            active=True,
            reason=reason,
            timestamp=time.time(),
        )
        self._history.append(self._state)

    def cancel(self, reason: str = "") -> None:
        import time
        self._state = OverrideState(
            command=OverrideCommand.CANCEL,
            active=True,
            reason=reason,
            timestamp=time.time(),
        )
        self._history.append(self._state)

    def pause(self, reason: str = "") -> None:
        import time
        self._state = OverrideState(
            command=OverrideCommand.PAUSE,
            active=True,
            reason=reason,
            timestamp=time.time(),
        )
        self._history.append(self._state)

    def resume(self) -> None:
        import time
        self._state = OverrideState(
            command=OverrideCommand.RESUME,
            active=False,
            timestamp=time.time(),
        )
        self._history.append(self._state)

    def should_stop(self) -> bool:
        return self._state.command == OverrideCommand.STOP and self._state.active

    def should_cancel(self) -> bool:
        return self._state.command == OverrideCommand.CANCEL and self._state.active

    def should_pause(self) -> bool:
        return self._state.command == OverrideCommand.PAUSE and self._state.active

    def is_paused(self) -> bool:
        return self._state.command == OverrideCommand.PAUSE and self._state.active

    def get_state(self) -> OverrideState:
        return self._state

    def get_history(self) -> list[OverrideState]:
        return list(self._history)

    def summary(self) -> dict[str, Any]:
        return {
            "current_command": self._state.command.value,
            "active": self._state.active,
            "history_count": len(self._history),
        }
