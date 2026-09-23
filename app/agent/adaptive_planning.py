"""Adaptive Planning for V5 — plan revision from evidence."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlanRevision(BaseModel):
    """A revision to the plan based on evidence."""
    reason: str
    evidence_source: str
    changes: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    timestamp: float = 0.0


class AdaptivePlanner:
    """Revises plans based on execution evidence."""

    def __init__(self) -> None:
        self._revisions: list[PlanRevision] = []

    def should_revise(
        self,
        test_passed: bool,
        security_issues: int = 0,
        regressions: int = 0,
        replan_count: int = 0,
        max_replans: int = 3,
    ) -> tuple[bool, str]:
        """Determine if plan should be revised."""
        if replan_count >= max_replans:
            return False, "Max replans reached"

        if not test_passed:
            return True, "Tests failed — plan may need adjustment"

        if security_issues > 0:
            return True, f"Security issues found ({security_issues})"

        if regressions > 0:
            return True, f"Regressions detected ({regressions})"

        return False, "No revision needed"

    def create_revision(
        self,
        reason: str,
        evidence_source: str = "",
        changes: list[str] | None = None,
        confidence: float = 0.5,
    ) -> PlanRevision:
        import time
        revision = PlanRevision(
            reason=reason,
            evidence_source=evidence_source,
            changes=changes or [],
            confidence=confidence,
            timestamp=time.time(),
        )
        self._revisions.append(revision)
        return revision

    def get_revisions(self) -> list[PlanRevision]:
        return list(self._revisions)

    def summary(self) -> dict[str, Any]:
        return {
            "total_revisions": len(self._revisions),
            "reasons": [r.reason for r in self._revisions],
        }
