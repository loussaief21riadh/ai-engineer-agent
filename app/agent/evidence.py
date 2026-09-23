"""Evidence Engine for V3.2 — centralized structured evidence with provenance."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    FILE = "FILE"
    COMMAND = "COMMAND"
    TEST = "TEST"
    SECURITY = "SECURITY"
    LLM = "LLM"
    REVIEW = "REVIEW"
    REGRESSION = "REGRESSION"
    CHECKPOINT = "CHECKPOINT"
    PLAN = "PLAN"
    DIAGNOSIS = "DIAGNOSIS"
    FIX = "FIX"
    OBSERVATION = "OBSERVATION"


class EvidenceStatus(str, Enum):
    CAPTURED = "CAPTURED"
    VERIFIED = "VERIFIED"
    DISPUTED = "DISPUTED"
    SUPERSEDED = "SUPERSEDED"
    INVALID = "INVALID"


class Evidence(BaseModel):
    """Structured evidence record with full provenance."""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    evidence_type: EvidenceType
    source: str
    task_id: str = ""
    subtask_id: str | None = None
    phase: str = ""
    timestamp: float = Field(default_factory=time.time)
    status: EvidenceStatus = EvidenceStatus.CAPTURED
    tool: str = ""
    success: bool = False
    payload_summary: str = ""
    payload_detail: dict[str, Any] = Field(default_factory=dict)
    trust_level: str = "TOOL_VERIFIED"
    related_evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceStore:
    """Centralized evidence store with query capabilities."""

    def __init__(self, task_id: str = "") -> None:
        self.task_id = task_id
        self._evidence: list[Evidence] = []

    def record(
        self,
        evidence_type: EvidenceType,
        source: str,
        subtask_id: str | None = None,
        phase: str = "",
        tool: str = "",
        success: bool = False,
        payload_summary: str = "",
        payload_detail: dict[str, Any] | None = None,
        trust_level: str = "TOOL_VERIFIED",
        related_evidence_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        ev = Evidence(
            evidence_type=evidence_type,
            source=source,
            task_id=self.task_id,
            subtask_id=subtask_id,
            phase=phase,
            tool=tool,
            success=success,
            payload_summary=payload_summary[:500],
            payload_detail=payload_detail or {},
            trust_level=trust_level,
            related_evidence_ids=related_evidence_ids or [],
            metadata=metadata or {},
        )
        self._evidence.append(ev)
        return ev

    def record_file(
        self, path: str, operation: str, content_summary: str = "",
        success: bool = True, subtask_id: str | None = None, phase: str = "",
    ) -> Evidence:
        return self.record(
            evidence_type=EvidenceType.FILE,
            source=path,
            subtask_id=subtask_id,
            phase=phase,
            tool=f"file_{operation}",
            success=success,
            payload_summary=content_summary,
            trust_level="TOOL_VERIFIED",
        )

    def record_command(
        self, command: str, exit_code: int, stdout: str = "", stderr: str = "",
        success: bool = True, subtask_id: str | None = None, phase: str = "",
    ) -> Evidence:
        return self.record(
            evidence_type=EvidenceType.COMMAND,
            source=command,
            subtask_id=subtask_id,
            phase=phase,
            tool="run_command",
            success=success,
            payload_summary=f"exit_code={exit_code}",
            payload_detail={"exit_code": exit_code, "stdout": stdout[:500], "stderr": stderr[:500]},
            trust_level="TOOL_VERIFIED",
        )

    def record_test(
        self, test_command: str, exit_code: int, passed: bool,
        stdout: str = "", subtask_id: str | None = None, phase: str = "",
    ) -> Evidence:
        return self.record(
            evidence_type=EvidenceType.TEST,
            source=test_command,
            subtask_id=subtask_id,
            phase=phase,
            tool="run_tests",
            success=passed,
            payload_summary=f"exit_code={exit_code}, passed={passed}",
            payload_detail={"exit_code": exit_code, "stdout": stdout[:1000]},
            trust_level="TOOL_VERIFIED",
        )

    def record_security(
        self, check_type: str, findings_count: int, critical_count: int,
        findings: list[str] | None = None, subtask_id: str | None = None, phase: str = "",
    ) -> Evidence:
        return self.record(
            evidence_type=EvidenceType.SECURITY,
            source=check_type,
            subtask_id=subtask_id,
            phase=phase,
            tool="security_check",
            success=critical_count == 0,
            payload_summary=f"findings={findings_count}, critical={critical_count}",
            payload_detail={"findings": findings or []},
            trust_level="SYSTEM_DERIVED",
        )

    def record_regression(
        self, baseline_tests: int, after_tests: int, regressions: list[str],
        new_tests: int = 0, subtask_id: str | None = None, phase: str = "",
    ) -> Evidence:
        has_regression = len(regressions) > 0
        return self.record(
            evidence_type=EvidenceType.REGRESSION,
            source="regression_engine",
            subtask_id=subtask_id,
            phase=phase,
            tool="regression_engine",
            success=not has_regression,
            payload_summary=f"baseline={baseline_tests}, after={after_tests}, regressions={len(regressions)}",
            payload_detail={
                "baseline_count": baseline_tests,
                "after_count": after_tests,
                "regressions": regressions,
                "new_tests": new_tests,
            },
            trust_level="SYSTEM_DERIVED",
        )

    def record_review(
        self, verdict: str, approved: bool, summary: str = "",
        findings: list[dict[str, Any]] | None = None,
        subtask_id: str | None = None, phase: str = "",
    ) -> Evidence:
        return self.record(
            evidence_type=EvidenceType.REVIEW,
            source="reviewer",
            subtask_id=subtask_id,
            phase=phase,
            tool="reviewer",
            success=approved,
            payload_summary=f"verdict={verdict}, approved={approved}",
            payload_detail={"summary": summary, "findings": findings or []},
            trust_level="MODEL_INFERRED",
        )

    def record_checkpoint(
        self, action: str, task_id: str, phase: str = "",
        subtask_id: str | None = None,
    ) -> Evidence:
        return self.record(
            evidence_type=EvidenceType.CHECKPOINT,
            source=task_id,
            subtask_id=subtask_id,
            phase=phase,
            tool="checkpoint",
            success=True,
            payload_summary=f"action={action}",
            trust_level="SYSTEM_DERIVED",
        )

    def record_diagnosis(
        self, category: str, hypothesis: str, confidence: float,
        suggested_fix: str = "", subtask_id: str | None = None, phase: str = "",
    ) -> Evidence:
        return self.record(
            evidence_type=EvidenceType.DIAGNOSIS,
            source="diagnostics",
            subtask_id=subtask_id,
            phase=phase,
            tool="failure_analyzer",
            success=True,
            payload_summary=f"category={category}, confidence={confidence:.2f}",
            payload_detail={
                "hypothesis": hypothesis,
                "suggested_fix": suggested_fix,
            },
            trust_level="MODEL_INFERRED",
            metadata={"confidence": confidence},
        )

    def record_fix(
        self, description: str, files_changed: list[str] | None = None,
        subtask_id: str | None = None, phase: str = "",
    ) -> Evidence:
        return self.record(
            evidence_type=EvidenceType.FIX,
            source="fix_phase",
            subtask_id=subtask_id,
            phase=phase,
            tool="edit_file",
            success=True,
            payload_summary=description[:200],
            payload_detail={"files_changed": files_changed or []},
            trust_level="MODEL_PROPOSED",
        )

    def record_plan(
        self, action: str, plan_version: int, subtask_count: int,
        plan_summary: str = "", subtask_id: str | None = None, phase: str = "",
    ) -> Evidence:
        return self.record(
            evidence_type=EvidenceType.PLAN,
            source="planner",
            subtask_id=subtask_id,
            phase=phase,
            tool="planner",
            success=True,
            payload_summary=f"action={action}, version={plan_version}, subtasks={subtask_count}",
            payload_detail={"plan_summary": plan_summary},
            trust_level="SYSTEM_DERIVED",
        )

    def get_by_type(self, evidence_type: EvidenceType) -> list[Evidence]:
        return [e for e in self._evidence if e.evidence_type == evidence_type]

    def get_by_subtask(self, subtask_id: str) -> list[Evidence]:
        return [e for e in self._evidence if e.subtask_id == subtask_id]

    def get_by_phase(self, phase: str) -> list[Evidence]:
        return [e for e in self._evidence if e.phase == phase]

    def get_by_status(self, status: EvidenceStatus) -> list[Evidence]:
        return [e for e in self._evidence if e.status == status]

    def get_successful(self) -> list[Evidence]:
        return [e for e in self._evidence if e.success]

    def get_failed(self) -> list[Evidence]:
        return [e for e in self._evidence if not e.success]

    def get_critical_failures(self) -> list[Evidence]:
        return [e for e in self._evidence
                if not e.success and e.evidence_type in (
                    EvidenceType.SECURITY, EvidenceType.REGRESSION, EvidenceType.TEST
                )]

    def has_regression(self) -> bool:
        regression_evidence = self.get_by_type(EvidenceType.REGRESSION)
        return any(not e.success for e in regression_evidence)

    def has_critical_security(self) -> bool:
        security_evidence = self.get_by_type(EvidenceType.SECURITY)
        return any(not e.success for e in security_evidence)

    def mark_disputed(self, evidence_id: str, reason: str = "") -> bool:
        for ev in self._evidence:
            if ev.evidence_id == evidence_id:
                ev.status = EvidenceStatus.DISPUTED
                ev.metadata["dispute_reason"] = reason
                return True
        return False

    def mark_superseded(self, evidence_id: str) -> bool:
        for ev in self._evidence:
            if ev.evidence_id == evidence_id:
                ev.status = EvidenceStatus.SUPERSEDED
                return True
        return False

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for ev in self._evidence:
            key = ev.evidence_type.value
            by_type[key] = by_type.get(key, 0) + 1

        return {
            "total": len(self._evidence),
            "by_type": by_type,
            "successful": sum(1 for e in self._evidence if e.success),
            "failed": sum(1 for e in self._evidence if not e.success),
            "disputed": sum(1 for e in self._evidence if e.status == EvidenceStatus.DISPUTED),
            "has_regression": self.has_regression(),
            "has_critical_security": self.has_critical_security(),
        }

    def to_list(self) -> list[dict[str, Any]]:
        return [e.model_dump() for e in self._evidence]

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        for ev in self._evidence:
            if ev.evidence_id == evidence_id:
                return ev
        return None

    def __len__(self) -> int:
        return len(self._evidence)

    def __iter__(self):
        return iter(self._evidence)
