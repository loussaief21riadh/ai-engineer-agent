from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str = ""


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    tool_call_id: str = ""


class ToolExecution(BaseModel):
    step: int
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None
    tool_call_id: str = ""


class AgentMessage(BaseModel):
    role: str
    content: str


class AgentHistory(BaseModel):
    messages: list[AgentMessage] = Field(default_factory=list)
    executions: list[ToolExecution] = Field(default_factory=list)


class ReviewSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ReviewFinding(BaseModel):
    severity: ReviewSeverity
    category: str
    description: str
    file: str | None = None
    line: int | None = None


class ReviewResult(BaseModel):
    approved: bool
    findings: list[ReviewFinding] = Field(default_factory=list)
    summary: str = ""


class TaskReport(BaseModel):
    task: str
    mode: str
    files_inspected: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    commands_executed: list[str] = Field(default_factory=list)
    test_results: dict[str, Any] | None = None
    review: ReviewResult | None = None
    final_response: str
    steps_taken: int
    executions: list[ToolExecution] = Field(default_factory=list)

    final_phase: str = "DONE"
    phase_history: list[str] = Field(default_factory=list)
    iteration_count: int = 0
    retry_count: int = 0
    diagnoses: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)
    stop_reason: str = "completed"
