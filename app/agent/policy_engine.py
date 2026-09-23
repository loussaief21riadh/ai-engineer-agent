"""Policy Engine for V5 — deterministic permission decisions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class PolicyViolation(BaseModel):
    """A policy violation."""
    policy: str
    decision: PolicyDecision
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyEngine:
    """Deterministic policy engine for permission decisions."""

    def __init__(self) -> None:
        self._policies: dict[str, dict[str, Any]] = {}
        self._violations: list[PolicyViolation] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("budget_check", {
            "description": "Check budget before LLM calls",
            "enabled": True,
        })
        self.register("scope_check", {
            "description": "Check file scope before modifications",
            "enabled": True,
        })
        self.register("security_check", {
            "description": "Block on CRITICAL security findings",
            "enabled": True,
        })
        self.register("regression_check", {
            "description": "Block on regressions",
            "enabled": True,
        })
        self.register("checkpoint_check", {
            "description": "Validate checkpoints before resume",
            "enabled": True,
        })

    def register(self, name: str, config: dict[str, Any]) -> None:
        self._policies[name] = config

    def can_execute_tool(
        self,
        tool_name: str,
        budget_within: bool = True,
        mode: str = "READ_ONLY",
    ) -> tuple[PolicyDecision, str]:
        """Check if tool execution is allowed."""
        if not budget_within:
            return PolicyDecision.DENY, "Budget exceeded"

        write_tools = {"write_file", "edit_file"}
        if tool_name in write_tools and mode == "READ_ONLY":
            return PolicyDecision.DENY, "Write tools not allowed in READ_ONLY mode"

        return PolicyDecision.ALLOW, ""

    def can_modify_file(
        self,
        filepath: str,
        allowed_paths: list[str] | None = None,
        is_secret: bool = False,
    ) -> tuple[PolicyDecision, str]:
        """Check if file modification is allowed."""
        if is_secret:
            return PolicyDecision.DENY, "Cannot modify secret files"

        if allowed_paths:
            in_scope = any(filepath.startswith(p) for p in allowed_paths)
            if not in_scope:
                return PolicyDecision.DENY, f"File {filepath} is outside allowed scope"

        return PolicyDecision.ALLOW, ""

    def can_call_llm(
        self,
        budget_within: bool = True,
        retry_count: int = 0,
        max_retries: int = 3,
    ) -> tuple[PolicyDecision, str]:
        """Check if LLM call is allowed."""
        if not budget_within:
            return PolicyDecision.DENY, "Budget exceeded"

        if retry_count >= max_retries:
            return PolicyDecision.DENY, f"Max retries ({max_retries}) reached"

        return PolicyDecision.ALLOW, ""

    def can_replan(
        self,
        replan_count: int = 0,
        max_replans: int = 3,
    ) -> tuple[PolicyDecision, str]:
        """Check if replanning is allowed."""
        if replan_count >= max_replans:
            return PolicyDecision.DENY, f"Max replans ({max_replans}) reached"

        return PolicyDecision.ALLOW, ""

    def can_resume(
        self,
        checkpoint_valid: bool = True,
        checkpoint_stale: bool = False,
        schema_valid: bool = True,
    ) -> tuple[PolicyDecision, str]:
        """Check if checkpoint resume is allowed."""
        if not checkpoint_valid:
            return PolicyDecision.DENY, "Checkpoint is invalid"

        if checkpoint_stale:
            return PolicyDecision.DENY, "Checkpoint is stale"

        if not schema_valid:
            return PolicyDecision.DENY, "Checkpoint schema mismatch"

        return PolicyDecision.ALLOW, ""

    def can_finalize(
        self,
        has_critical_security: bool = False,
        has_regression: bool = False,
        tests_passed: bool = True,
        budget_within: bool = True,
    ) -> tuple[PolicyDecision, str]:
        """Check if task finalization is allowed."""
        if has_critical_security:
            return PolicyDecision.DENY, "CRITICAL security findings must be resolved"

        if has_regression:
            return PolicyDecision.DENY, "Regressions must be resolved"

        if not tests_passed:
            return PolicyDecision.DENY, "Tests must pass before finalization"

        if not budget_within:
            return PolicyDecision.DENY, "Budget exceeded"

        return PolicyDecision.ALLOW, ""

    def get_violations(self) -> list[PolicyViolation]:
        return list(self._violations)

    def record_violation(self, violation: PolicyViolation) -> None:
        self._violations.append(violation)

    def summary(self) -> dict[str, Any]:
        return {
            "policies": len(self._policies),
            "violations": len(self._violations),
            "active_policies": [k for k, v in self._policies.items() if v.get("enabled", True)],
        }
