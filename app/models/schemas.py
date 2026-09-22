from __future__ import annotations

import time
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


class ExecutionEvidence(BaseModel):
    """Structured evidence from a single tool execution."""
    tool: str
    command: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    exit_code: int | None = None
    success: bool = False
    stdout_summary: str = ""
    stderr_summary: str = ""
    trust_level: str = "TOOL_VERIFIED"
    duration_ms: float | None = None

    @classmethod
    def from_tool_execution(cls, ex: ToolExecution) -> ExecutionEvidence:
        result_data = ex.result if isinstance(ex.result, dict) else {}
        return cls(
            tool=ex.tool_name,
            command=ex.arguments.get("command", ""),
            arguments=ex.arguments,
            timestamp=time.time(),
            exit_code=result_data.get("exit_code"),
            success=ex.success,
            stdout_summary=str(result_data.get("stdout", ""))[:500],
            stderr_summary=str(result_data.get("stderr", ex.error or ""))[:500],
            trust_level="TOOL_VERIFIED",
            duration_ms=ex.duration_ms,
        )


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


class ReviewVerdict(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"


class ReviewFinding(BaseModel):
    severity: ReviewSeverity
    category: str
    description: str
    file: str | None = None
    line: int | None = None


class ReviewResult(BaseModel):
    approved: bool
    verdict: ReviewVerdict = ReviewVerdict.APPROVE
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
