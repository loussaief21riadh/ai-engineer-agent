"""Execution Controller for V3.2 — subtask lifecycle, bounded execution, failure classification."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SubtaskLifecycle(str, Enum):
    """Lifecycle states for a subtask."""
    PENDING = "PENDING"
    READY = "READY"
    EXECUTING = "EXECUTING"
    TESTING = "TESTING"
    SECURITY_CHECKING = "SECURITY_CHECKING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"
    REPLANNING = "REPLANNING"


class SubtaskRecord(BaseModel):
    """Detailed record of a subtask's execution lifecycle."""
    subtask_id: str
    description: str = ""
    lifecycle: SubtaskLifecycle = SubtaskLifecycle.PENDING
    dependencies: list[str] = Field(default_factory=list)
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    error_message: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "description": self.description[:100],
            "lifecycle": self.lifecycle.value,
            "dependencies": self.dependencies,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "error_message": self.error_message[:200],
            "files_modified": self.files_modified,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
        }


class ExecutionConstraints(BaseModel):
    """Constraints on execution."""
    max_retries_per_subtask: int = 3
    max_total_retries: int = 10
    max_subtask_duration_ms: float = 300_000.0  # 5 minutes
    max_total_duration_ms: float = 600_000.0  # 10 minutes
    max_concurrent_subtasks: int = 1  # Sequential execution
    max_evidence_per_subtask: int = 50
    max_files_per_subtask: int = 20


class ExecutionController:
    """Controls subtask lifecycle with bounded execution and failure classification."""

    def __init__(self, constraints: ExecutionConstraints | None = None) -> None:
        self.constraints = constraints or ExecutionConstraints()
        self._subtasks: dict[str, SubtaskRecord] = {}
        self._execution_order: list[str] = []
        self._total_retries: int = 0
        self._start_time: float = 0.0
        self._active_subtask_id: str | None = None

    def start_execution(self) -> None:
        """Mark execution start."""
        self._start_time = time.monotonic()
        self._total_retries = 0

    def register_subtask(
        self,
        subtask_id: str,
        description: str = "",
        dependencies: list[str] | None = None,
        max_retries: int | None = None,
    ) -> SubtaskRecord:
        """Register a subtask for execution tracking."""
        record = SubtaskRecord(
            subtask_id=subtask_id,
            description=description,
            dependencies=dependencies or [],
            max_retries=max_retries or self.constraints.max_retries_per_subtask,
        )
        self._subtasks[subtask_id] = record
        self._execution_order.append(subtask_id)
        return record

    def can_start_subtask(self, subtask_id: str) -> tuple[bool, str]:
        """Check if a subtask can be started."""
        if subtask_id not in self._subtasks:
            return False, f"Subtask {subtask_id} not registered"

        record = self._subtasks[subtask_id]

        if record.lifecycle not in (SubtaskLifecycle.PENDING, SubtaskLifecycle.READY):
            return False, f"Subtask {subtask_id} is in state {record.lifecycle.value}"

        if record.retry_count >= record.max_retries:
            return False, f"Subtask {subtask_id} exceeded max retries ({record.max_retries})"

        if self._active_subtask_id is not None and self._active_subtask_id != subtask_id:
            return False, f"Another subtask {self._active_subtask_id} is active"

        elapsed = (time.monotonic() - self._start_time) * 1000 if self._start_time else 0
        if elapsed > self.constraints.max_total_duration_ms:
            return False, f"Total execution time exceeded ({elapsed:.0f}ms > {self.constraints.max_total_duration_ms}ms)"

        if self._total_retries >= self.constraints.max_total_retries:
            return False, f"Total retries exceeded ({self._total_retries} >= {self.constraints.max_total_retries})"

        for dep_id in record.dependencies:
            if dep_id in self._subtasks:
                dep_record = self._subtasks[dep_id]
                if dep_record.lifecycle != SubtaskLifecycle.COMPLETED:
                    return False, f"Dependency {dep_id} not completed (state: {dep_record.lifecycle.value})"

        return True, ""

    def start_subtask(self, subtask_id: str) -> bool:
        """Start a subtask. Returns True if successful."""
        can_start, reason = self.can_start_subtask(subtask_id)
        if not can_start:
            return False

        record = self._subtasks[subtask_id]
        record.lifecycle = SubtaskLifecycle.EXECUTING
        record.started_at = time.monotonic()
        self._active_subtask_id = subtask_id
        return True

    def complete_subtask(
        self,
        subtask_id: str,
        files_modified: list[str] | None = None,
        evidence_ids: list[str] | None = None,
    ) -> bool:
        """Mark a subtask as completed."""
        if subtask_id not in self._subtasks:
            return False

        record = self._subtasks[subtask_id]
        record.lifecycle = SubtaskLifecycle.COMPLETED
        record.completed_at = time.monotonic()
        record.duration_ms = (record.completed_at - record.started_at) * 1000 if record.started_at else 0
        record.files_modified = files_modified or []
        record.evidence_ids = evidence_ids or []

        if self._active_subtask_id == subtask_id:
            self._active_subtask_id = None

        return True

    def fail_subtask(
        self,
        subtask_id: str,
        error_message: str = "",
        retry: bool = True,
    ) -> bool:
        """Mark a subtask as failed. Optionally retry."""
        if subtask_id not in self._subtasks:
            return False

        record = self._subtasks[subtask_id]
        record.error_message = error_message

        if retry and record.retry_count < record.max_retries:
            record.retry_count += 1
            self._total_retries += 1
            record.lifecycle = SubtaskLifecycle.PENDING
            record.started_at = 0.0
            return True

        record.lifecycle = SubtaskLifecycle.FAILED
        record.completed_at = time.monotonic()
        record.duration_ms = (record.completed_at - record.started_at) * 1000 if record.started_at else 0

        if self._active_subtask_id == subtask_id:
            self._active_subtask_id = None

        return False

    def block_subtask(self, subtask_id: str, reason: str = "") -> None:
        """Block a subtask (e.g., due to dependency failure)."""
        if subtask_id in self._subtasks:
            self._subtasks[subtask_id].lifecycle = SubtaskLifecycle.BLOCKED
            self._subtasks[subtask_id].error_message = reason

    def skip_subtask(self, subtask_id: str, reason: str = "") -> None:
        """Skip a subtask."""
        if subtask_id in self._subtasks:
            self._subtasks[subtask_id].lifecycle = SubtaskLifecycle.SKIPPED
            self._subtasks[subtask_id].error_message = reason

    def start_testing(self, subtask_id: str) -> None:
        """Mark subtask as testing phase."""
        if subtask_id in self._subtasks:
            self._subtasks[subtask_id].lifecycle = SubtaskLifecycle.TESTING

    def start_security_check(self, subtask_id: str) -> None:
        """Mark subtask as security checking phase."""
        if subtask_id in self._subtasks:
            self._subtasks[subtask_id].lifecycle = SubtaskLifecycle.SECURITY_CHECKING

    def record_test_results(
        self,
        subtask_id: str,
        tests_run: int,
        tests_passed: int,
        tests_failed: int,
    ) -> None:
        """Record test results for a subtask."""
        if subtask_id in self._subtasks:
            record = self._subtasks[subtask_id]
            record.tests_run = tests_run
            record.tests_passed = tests_passed
            record.tests_failed = tests_failed

    def get_subtask(self, subtask_id: str) -> SubtaskRecord | None:
        return self._subtasks.get(subtask_id)

    def get_all_subtasks(self) -> list[SubtaskRecord]:
        return list(self._subtasks.values())

    def get_completed_subtasks(self) -> list[SubtaskRecord]:
        return [s for s in self._subtasks.values() if s.lifecycle == SubtaskLifecycle.COMPLETED]

    def get_failed_subtasks(self) -> list[SubtaskRecord]:
        return [s for s in self._subtasks.values() if s.lifecycle == SubtaskLifecycle.FAILED]

    def get_pending_subtasks(self) -> list[SubtaskRecord]:
        return [s for s in self._subtasks.values() if s.lifecycle in (
            SubtaskLifecycle.PENDING, SubtaskLifecycle.READY
        )]

    def get_next_ready_subtask(self) -> SubtaskRecord | None:
        """Get the next subtask that can be executed."""
        for subtask_id in self._execution_order:
            record = self._subtasks.get(subtask_id)
            if record is None:
                continue
            if record.lifecycle in (SubtaskLifecycle.PENDING, SubtaskLifecycle.READY):
                can_start, _ = self.can_start_subtask(subtask_id)
                if can_start:
                    return record
        return None

    def is_execution_complete(self) -> bool:
        """Check if all subtasks are in terminal states."""
        for record in self._subtasks.values():
            if record.lifecycle not in (
                SubtaskLifecycle.COMPLETED,
                SubtaskLifecycle.FAILED,
                SubtaskLifecycle.SKIPPED,
                SubtaskLifecycle.BLOCKED,
            ):
                return False
        return True

    def has_failures(self) -> bool:
        return any(s.lifecycle == SubtaskLifecycle.FAILED for s in self._subtasks.values())

    def summary(self) -> dict[str, Any]:
        total = len(self._subtasks)
        completed = sum(1 for s in self._subtasks.values() if s.lifecycle == SubtaskLifecycle.COMPLETED)
        failed = sum(1 for s in self._subtasks.values() if s.lifecycle == SubtaskLifecycle.FAILED)
        blocked = sum(1 for s in self._subtasks.values() if s.lifecycle == SubtaskLifecycle.BLOCKED)
        skipped = sum(1 for s in self._subtasks.values() if s.lifecycle == SubtaskLifecycle.SKIPPED)
        pending = sum(1 for s in self._subtasks.values() if s.lifecycle in (
            SubtaskLifecycle.PENDING, SubtaskLifecycle.READY
        ))
        executing = sum(1 for s in self._subtasks.values() if s.lifecycle in (
            SubtaskLifecycle.EXECUTING, SubtaskLifecycle.TESTING, SubtaskLifecycle.SECURITY_CHECKING
        ))

        elapsed = (time.monotonic() - self._start_time) * 1000 if self._start_time else 0

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "blocked": blocked,
            "skipped": skipped,
            "pending": pending,
            "executing": executing,
            "total_retries": self._total_retries,
            "elapsed_ms": round(elapsed, 1),
            "execution_complete": self.is_execution_complete(),
            "has_failures": self.has_failures(),
        }

    def reset(self) -> None:
        """Reset the execution controller."""
        self._subtasks.clear()
        self._execution_order.clear()
        self._total_retries = 0
        self._start_time = 0.0
        self._active_subtask_id = None
