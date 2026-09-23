"""Final Validation Gate for V5 — comprehensive validation before DONE."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GateCheck(str, Enum):
    POLICY = "POLICY"
    TESTS = "TESTS"
    SECURITY = "SECURITY"
    REGRESSION = "REGRESSION"
    SCOPE = "SCOPE"
    EVIDENCE = "EVIDENCE"
    REVIEW = "REVIEW"


class GateResult(BaseModel):
    """Result of a single gate check."""
    check: GateCheck
    passed: bool
    details: str = ""
    severity: str = "INFO"


class ValidationGateResult(BaseModel):
    """Result of the final validation gate."""
    overall_passed: bool = True
    results: list[GateResult] = Field(default_factory=list)
    blocked_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "results": [r.model_dump() for r in self.results],
            "blocked_reason": self.blocked_reason,
        }


class FinalValidationGate:
    """Comprehensive validation gate before task finalization."""

    def validate(
        self,
        policy_passed: bool = True,
        policy_reason: str = "",
        tests_passed: bool = True,
        test_details: str = "",
        security_passed: bool = True,
        security_details: str = "",
        no_regressions: bool = True,
        regression_details: str = "",
        scope_valid: bool = True,
        scope_details: str = "",
        has_evidence: bool = True,
        evidence_details: str = "",
        review_approved: bool = True,
        review_details: str = "",
    ) -> ValidationGateResult:
        results: list[GateResult] = []

        results.append(GateResult(
            check=GateCheck.POLICY,
            passed=policy_passed,
            details=policy_reason or "Policy check passed",
            severity="CRITICAL" if not policy_passed else "INFO",
        ))

        results.append(GateResult(
            check=GateCheck.TESTS,
            passed=tests_passed,
            details=test_details or "Tests passed",
            severity="CRITICAL" if not tests_passed else "INFO",
        ))

        results.append(GateResult(
            check=GateCheck.SECURITY,
            passed=security_passed,
            details=security_details or "Security check passed",
            severity="CRITICAL" if not security_passed else "INFO",
        ))

        results.append(GateResult(
            check=GateCheck.REGRESSION,
            passed=no_regressions,
            details=regression_details or "No regressions detected",
            severity="HIGH" if not no_regressions else "INFO",
        ))

        results.append(GateResult(
            check=GateCheck.SCOPE,
            passed=scope_valid,
            details=scope_details or "Scope valid",
            severity="HIGH" if not scope_valid else "INFO",
        ))

        results.append(GateResult(
            check=GateCheck.EVIDENCE,
            passed=has_evidence,
            details=evidence_details or "Evidence present",
            severity="MEDIUM" if not has_evidence else "INFO",
        ))

        results.append(GateResult(
            check=GateCheck.REVIEW,
            passed=review_approved,
            details=review_details or "Review approved",
            severity="HIGH" if not review_approved else "INFO",
        ))

        overall_passed = all(r.passed for r in results)
        blocked_reason = ""
        if not overall_passed:
            failed = [r for r in results if not r.passed]
            blocked_reason = "; ".join(f"{r.check.value}: {r.details}" for r in failed)

        return ValidationGateResult(
            overall_passed=overall_passed,
            results=results,
            blocked_reason=blocked_reason,
        )

    def summary(self) -> dict[str, Any]:
        return {"gate_type": "FinalValidationGate", "checks": len(GateCheck)}
