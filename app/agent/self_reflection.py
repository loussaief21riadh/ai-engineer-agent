"""Self-Reflection for V5 — structured decision-making after failures."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReflectionDecision(str, Enum):
    RETRY = "RETRY"
    FIX = "FIX"
    REPLAN = "REPLAN"
    ABORT = "ABORT"
    ESCALATE = "ESCALATE"


class ReflectionEntry(BaseModel):
    """A structured self-reflection entry."""
    what_failed: str
    evidence: str
    wrong_assumption: str
    what_should_change: str
    decision: ReflectionDecision
    confidence: float = 0.5
    reasoning: str = ""


class SelfReflectionEngine:
    """Structured self-reflection after failures."""

    def __init__(self) -> None:
        self._reflections: list[ReflectionEntry] = []

    def reflect(
        self,
        failure_output: str,
        evidence: str = "",
        context: str = "",
    ) -> ReflectionEntry:
        """Analyze a failure and produce a structured reflection."""
        what_failed = self._identify_what_failed(failure_output)
        wrong_assumption = self._identify_wrong_assumption(failure_output, context)
        what_changed = self._identify_what_should_change(failure_output, evidence)
        decision = self._make_decision(failure_output, evidence)
        confidence = self._assess_confidence(failure_output, evidence)
        reasoning = self._build_reasoning(what_failed, wrong_assumption, what_changed, decision)

        entry = ReflectionEntry(
            what_failed=what_failed,
            evidence=evidence[:500],
            wrong_assumption=wrong_assumption,
            what_should_change=what_changed,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
        )

        self._reflections.append(entry)
        return entry

    def _identify_what_failed(self, output: str) -> str:
        if "SyntaxError" in output:
            return "Syntax error introduced in code"
        elif "FAILED" in output or "AssertionError" in output:
            return "Test assertion failed"
        elif "TypeError" in output:
            return "Type error in code"
        elif "ImportError" in output or "ModuleNotFoundError" in output:
            return "Import error"
        elif "PermissionError" in output:
            return "Permission denied"
        elif "timeout" in output.lower():
            return "Operation timed out"
        return "Unknown failure"

    def _identify_wrong_assumption(self, output: str, context: str) -> str:
        if "SyntaxError" in output:
            return "Assumed code was syntactically valid"
        elif "FAILED" in output:
            return "Assumed implementation would pass existing tests"
        elif "TypeError" in output:
            return "Assumed types were compatible"
        elif "ImportError" in output:
            return "Assumed module/dependency was available"
        return "Assumption unclear from output"

    def _identify_what_should_change(self, output: str, evidence: str) -> str:
        if "SyntaxError" in output:
            return "Validate syntax before proceeding"
        elif "FAILED" in output:
            return "Run tests earlier or validate logic before testing"
        elif "TypeError" in output:
            return "Add type checking or handle edge cases"
        elif "ImportError" in output:
            return "Verify dependencies exist before using"
        return "Gather more evidence before deciding"

    def _make_decision(self, output: str, evidence: str) -> ReflectionDecision:
        if "SyntaxError" in output:
            return ReflectionDecision.FIX
        elif "FAILED" in output:
            if "replan" in evidence.lower():
                return ReflectionDecision.REPLAN
            return ReflectionDecision.FIX
        elif "timeout" in output.lower():
            return ReflectionDecision.RETRY
        elif "budget" in output.lower():
            return ReflectionDecision.ABORT
        return ReflectionDecision.FIX

    def _assess_confidence(self, output: str, evidence: str) -> float:
        confidence = 0.5

        if "SyntaxError" in output or "TypeError" in output:
            confidence = 0.8
        elif "FAILED" in output:
            confidence = 0.6
        elif "timeout" in output.lower():
            confidence = 0.4

        if evidence:
            confidence = min(confidence + 0.1, 1.0)

        return round(confidence, 2)

    def _build_reasoning(
        self,
        what_failed: str,
        wrong_assumption: str,
        what_changed: str,
        decision: ReflectionDecision,
    ) -> str:
        return (
            f"Failed: {what_failed}. "
            f"Wrong assumption: {wrong_assumption}. "
            f"Change: {what_changed}. "
            f"Decision: {decision.value}."
        )

    def get_reflections(self) -> list[ReflectionEntry]:
        return list(self._reflections)

    def get_last_reflection(self) -> ReflectionEntry | None:
        return self._reflections[-1] if self._reflections else None

    def summary(self) -> dict[str, Any]:
        decisions: dict[str, int] = {}
        for r in self._reflections:
            key = r.decision.value
            decisions[key] = decisions.get(key, 0) + 1

        return {
            "total_reflections": len(self._reflections),
            "decisions": decisions,
            "avg_confidence": (
                sum(r.confidence for r in self._reflections) / len(self._reflections)
                if self._reflections else 0.0
            ),
        }

    def clear(self) -> None:
        self._reflections.clear()
