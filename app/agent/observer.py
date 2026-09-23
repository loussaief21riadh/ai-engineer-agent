from __future__ import annotations

import time
import uuid
from typing import Any

from app.models.schemas import TokenUsage, TraceEvent


class ObserverEngine:
    """Structured trace event observer for observability."""

    def __init__(self, task_id: str = "") -> None:
        self.task_id = task_id or str(uuid.uuid4())[:8]
        self._events: list[TraceEvent] = []
        self._phase_start_times: dict[str, float] = {}
        self._accumulated_tokens = TokenUsage()
        self._accumulated_cost: float = 0.0

    def emit(
        self,
        event_type: str,
        phase: str = "",
        tool: str | None = None,
        model: str | None = None,
        duration_ms: float | None = None,
        tokens: TokenUsage | None = None,
        cost_estimate: float | None = None,
        success: bool = True,
        subtask_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            event_id=str(uuid.uuid4())[:12],
            task_id=self.task_id,
            subtask_id=subtask_id,
            timestamp=time.time(),
            event_type=event_type,
            phase=phase,
            tool=tool,
            model=model,
            duration_ms=duration_ms,
            tokens=tokens,
            cost_estimate=cost_estimate,
            success=success,
            details=details or {},
        )
        self._events.append(event)

        if tokens:
            self._accumulated_tokens = self._accumulated_tokens.add(tokens)
        if cost_estimate is not None:
            self._accumulated_cost += cost_estimate

        return event

    def phase_start(self, phase: str) -> TraceEvent:
        self._phase_start_times[phase] = time.time()
        return self.emit("phase_start", phase=phase)

    def phase_end(self, phase: str, success: bool = True) -> TraceEvent:
        start = self._phase_start_times.pop(phase, None)
        duration_ms = (time.time() - start) * 1000 if start else None
        return self.emit("phase_end", phase=phase, duration_ms=duration_ms, success=success)

    def tool_call(
        self,
        tool: str,
        phase: str = "",
        duration_ms: float | None = None,
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> TraceEvent:
        return self.emit(
            "tool_call",
            phase=phase,
            tool=tool,
            duration_ms=duration_ms,
            success=success,
            details=details,
        )

    def llm_call(
        self,
        model: str,
        phase: str = "",
        duration_ms: float | None = None,
        tokens: TokenUsage | None = None,
        cost_estimate: float | None = None,
        success: bool = True,
    ) -> TraceEvent:
        return self.emit(
            "llm_call",
            phase=phase,
            model=model,
            duration_ms=duration_ms,
            tokens=tokens,
            cost_estimate=cost_estimate,
            success=success,
        )

    def security_check(
        self,
        phase: str = "SECURITY_CHECK",
        findings_count: int = 0,
        critical_count: int = 0,
        success: bool = True,
    ) -> TraceEvent:
        return self.emit(
            "security_check",
            phase=phase,
            success=success,
            details={"findings_count": findings_count, "critical_count": critical_count},
        )

    def review(
        self,
        phase: str = "REVIEW",
        verdict: str = "",
        success: bool = True,
    ) -> TraceEvent:
        return self.emit(
            "review",
            phase=phase,
            success=success,
            details={"verdict": verdict},
        )

    def validation(
        self,
        phase: str = "VALIDATE",
        passed: bool = True,
        success: bool = True,
    ) -> TraceEvent:
        return self.emit(
            "validation",
            phase=phase,
            success=success,
            details={"passed": passed},
        )

    def query(
        self,
        event_type: str | None = None,
        phase: str | None = None,
        tool: str | None = None,
        model: str | None = None,
        success_only: bool = False,
    ) -> list[TraceEvent]:
        results = self._events
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if phase:
            results = [e for e in results if e.phase == phase]
        if tool:
            results = [e for e in results if e.tool == tool]
        if model:
            results = [e for e in results if e.model == model]
        if success_only:
            results = [e for e in results if e.success]
        return results

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    @property
    def total_tokens(self) -> TokenUsage:
        return self._accumulated_tokens

    @property
    def total_cost(self) -> float:
        return self._accumulated_cost

    def summarize(self) -> dict[str, Any]:
        phases_seen = list(dict.fromkeys(e.phase for e in self._events if e.phase))
        tools_used = list(dict.fromkeys(
            e.tool for e in self._events if e.tool
        ))
        models_used = list(dict.fromkeys(
            e.model for e in self._events if e.model
        ))
        return {
            "task_id": self.task_id,
            "total_events": len(self._events),
            "phases_seen": phases_seen,
            "tools_used": tools_used,
            "models_used": models_used,
            "total_tokens": self._accumulated_tokens.model_dump(),
            "total_cost": self._accumulated_cost,
            "failed_events": sum(1 for e in self._events if not e.success),
        }
