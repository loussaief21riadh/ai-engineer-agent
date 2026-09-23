"""Quota and budget management for V2.0 — local protection limits."""

from __future__ import annotations

import os
import time
from typing import Any

from pydantic import BaseModel


class BudgetLimits(BaseModel):
    max_llm_calls: int = 50
    max_tool_calls: int = 100
    max_retry_cycles: int = 3
    max_task_duration: float = 600.0
    max_agent_steps: int = 15


class BudgetTracker:
    def __init__(self, limits: BudgetLimits | None = None) -> None:
        self.limits = limits or BudgetLimits(
            max_llm_calls=int(os.getenv("MAX_LLM_CALLS", "50")),
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "100")),
            max_retry_cycles=int(os.getenv("MAX_RETRY_CYCLES", "3")),
            max_task_duration=float(os.getenv("MAX_TASK_DURATION", "600")),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "15")),
        )
        self.llm_calls: int = 0
        self.tool_calls: int = 0
        self.retry_cycles: int = 0
        self.start_time: float = 0.0
        self._active: bool = False

    def start(self) -> None:
        self.llm_calls = 0
        self.tool_calls = 0
        self.retry_cycles = 0
        self.start_time = time.monotonic()
        self._active = True

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore budget counters from a checkpoint snapshot without resetting."""
        self.llm_calls = snapshot.get("llm_calls", 0)
        self.tool_calls = snapshot.get("tool_calls", 0)
        self.retry_cycles = snapshot.get("retry_cycles", 0)
        self.start_time = time.monotonic()
        self._active = True

    def stop(self) -> None:
        self._active = False

    def record_llm_call(self) -> None:
        self.llm_calls += 1

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_retry(self) -> None:
        self.retry_cycles += 1

    def elapsed(self) -> float:
        if not self._active:
            return 0.0
        return time.monotonic() - self.start_time

    def is_within_budget(self) -> bool:
        if not self._active:
            return True

        if self.llm_calls >= self.limits.max_llm_calls:
            return False

        if self.tool_calls >= self.limits.max_tool_calls:
            return False

        if self.retry_cycles >= self.limits.max_retry_cycles:
            return False

        if self.elapsed() >= self.limits.max_task_duration:
            return False

        return True

    def budget_violation(self) -> str | None:
        if not self._active:
            return None

        if self.llm_calls >= self.limits.max_llm_calls:
            return f"LLM call limit reached ({self.limits.max_llm_calls})"

        if self.tool_calls >= self.limits.max_tool_calls:
            return f"Tool call limit reached ({self.limits.max_tool_calls})"

        if self.retry_cycles >= self.limits.max_retry_cycles:
            return f"Retry cycle limit reached ({self.limits.max_retry_cycles})"

        if self.elapsed() >= self.limits.max_task_duration:
            return f"Task duration limit reached ({self.limits.max_task_duration}s)"

        return None

    def status(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.limits.max_llm_calls,
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.limits.max_tool_calls,
            "retry_cycles": self.retry_cycles,
            "max_retry_cycles": self.limits.max_retry_cycles,
            "elapsed": round(self.elapsed(), 1),
            "max_task_duration": self.limits.max_task_duration,
            "within_budget": self.is_within_budget(),
        }
