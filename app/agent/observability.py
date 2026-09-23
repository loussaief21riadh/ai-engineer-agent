"""Observability 2.0 for V4 — complete execution trace reconstruction."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class TraceSpan(BaseModel):
    """A span in the execution trace."""
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    parent_id: str = ""
    name: str
    category: str = ""  # phase, subtask, llm_call, tool_call, test, security, etc.
    start_time: float = Field(default_factory=time.time)
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "OK"  # OK, ERROR, TIMEOUT
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)

    def finish(self, status: str = "OK") -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status

    def add_event(self, name: str, details: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "details": details or {},
        })


class ExecutionTrace:
    """Complete execution trace for observability."""

    def __init__(self, task_id: str = "") -> None:
        self.task_id = task_id or str(uuid.uuid4())[:8]
        self._spans: list[TraceSpan] = []
        self._active_spans: list[TraceSpan] = []
        self._root_spans: list[TraceSpan] = []
        self._metadata: dict[str, Any] = {}

    def start_span(
        self,
        name: str,
        category: str = "",
        parent_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Start a new trace span."""
        span = TraceSpan(
            name=name,
            category=category,
            parent_id=parent_id or (self._active_spans[-1].span_id if self._active_spans else ""),
            metadata=metadata or {},
        )
        self._spans.append(span)
        self._active_spans.append(span)

        if not span.parent_id:
            self._root_spans.append(span)

        return span

    def finish_span(self, span: TraceSpan, status: str = "OK") -> None:
        """Finish a trace span."""
        span.finish(status)
        if span in self._active_spans:
            self._active_spans.remove(span)

    def add_event(self, name: str, details: dict[str, Any] | None = None) -> None:
        """Add event to the current active span."""
        if self._active_spans:
            self._active_spans[-1].add_event(name, details)

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_spans(self) -> list[TraceSpan]:
        return list(self._spans)

    def get_root_spans(self) -> list[TraceSpan]:
        return list(self._root_spans)

    def get_children(self, parent_id: str) -> list[TraceSpan]:
        return [s for s in self._spans if s.parent_id == parent_id]

    def get_spans_by_category(self, category: str) -> list[TraceSpan]:
        return [s for s in self._spans if s.category == category]

    def get_total_duration_ms(self) -> float:
        if not self._spans:
            return 0.0
        start = min(s.start_time for s in self._spans)
        end = max(s.end_time for s in self._spans if s.end_time > 0)
        return (end - start) * 1000 if end > start else 0.0

    def get_error_spans(self) -> list[TraceSpan]:
        return [s for s in self._spans if s.status == "ERROR"]

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_spans": len(self._spans),
            "root_spans": len(self._root_spans),
            "error_spans": len(self.get_error_spans()),
            "total_duration_ms": round(self.get_total_duration_ms(), 1),
            "categories": list(set(s.category for s in self._spans if s.category)),
            "metadata": self._metadata,
        }

    def to_timeline(self) -> list[dict[str, Any]]:
        """Export trace as a timeline for debugging."""
        timeline: list[dict[str, Any]] = []
        for span in sorted(self._spans, key=lambda s: s.start_time):
            timeline.append({
                "span_id": span.span_id,
                "name": span.name,
                "category": span.category,
                "start": span.start_time,
                "duration_ms": round(span.duration_ms, 1),
                "status": span.status,
                "parent_id": span.parent_id,
                "events": len(span.events),
            })
        return timeline

    def clear(self) -> None:
        self._spans.clear()
        self._active_spans.clear()
        self._root_spans.clear()
        self._metadata.clear()
