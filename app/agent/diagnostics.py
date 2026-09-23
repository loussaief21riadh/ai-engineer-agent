"""Failure diagnostics for V2.0 — structured error classification."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    SYNTAX_ERROR = "SYNTAX_ERROR"
    TYPE_ERROR = "TYPE_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    TEST_FAILURE = "TEST_FAILURE"
    LOGIC_ERROR = "LOGIC_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    TIMEOUT = "TIMEOUT"
    TOOL_ERROR = "TOOL_ERROR"
    LLM_ERROR = "LLM_ERROR"
    SECURITY_ERROR = "SECURITY_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    BUDGET_ERROR = "BUDGET_ERROR"
    CHECKPOINT_ERROR = "CHECKPOINT_ERROR"
    REGRESSION = "REGRESSION"
    INVALID_PLAN = "INVALID_PLAN"
    UNKNOWN = "UNKNOWN"


class FailureStrategy(str, Enum):
    """Recommended strategy for handling each failure category."""
    RETRY = "RETRY"
    DIAGNOSE = "DIAGNOSE"
    FIX = "FIX"
    RETEST = "RETEST"
    REPLAN = "REPLAN"
    BLOCK = "BLOCK"
    ABORT = "ABORT"
    ESCALATE = "ESCALATE"


FAILURE_STRATEGY_MAP: dict[ErrorCategory, FailureStrategy] = {
    ErrorCategory.SYNTAX_ERROR: FailureStrategy.FIX,
    ErrorCategory.TYPE_ERROR: FailureStrategy.FIX,
    ErrorCategory.IMPORT_ERROR: FailureStrategy.FIX,
    ErrorCategory.TEST_FAILURE: FailureStrategy.DIAGNOSE,
    ErrorCategory.LOGIC_ERROR: FailureStrategy.DIAGNOSE,
    ErrorCategory.CONFIGURATION_ERROR: FailureStrategy.FIX,
    ErrorCategory.DEPENDENCY_ERROR: FailureStrategy.BLOCK,
    ErrorCategory.ENVIRONMENT_ERROR: FailureStrategy.DIAGNOSE,
    ErrorCategory.PERMISSION_ERROR: FailureStrategy.BLOCK,
    ErrorCategory.TIMEOUT: FailureStrategy.RETRY,
    ErrorCategory.TOOL_ERROR: FailureStrategy.DIAGNOSE,
    ErrorCategory.LLM_ERROR: FailureStrategy.RETRY,
    ErrorCategory.SECURITY_ERROR: FailureStrategy.BLOCK,
    ErrorCategory.NETWORK_ERROR: FailureStrategy.RETRY,
    ErrorCategory.BUDGET_ERROR: FailureStrategy.ABORT,
    ErrorCategory.CHECKPOINT_ERROR: FailureStrategy.REPLAN,
    ErrorCategory.REGRESSION: FailureStrategy.REPLAN,
    ErrorCategory.INVALID_PLAN: FailureStrategy.REPLAN,
    ErrorCategory.UNKNOWN: FailureStrategy.DIAGNOSE,
}


class FailureDiagnosis(BaseModel):
    failure: str
    evidence: str = ""
    category: ErrorCategory = ErrorCategory.UNKNOWN
    hypothesis: str = ""
    confidence: float = 0.0
    suggested_fix: str = ""
    recommended_action: str = ""
    strategy: FailureStrategy = FailureStrategy.DIAGNOSE

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure": self.failure,
            "evidence": self.evidence,
            "category": self.category.value,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "suggested_fix": self.suggested_fix,
            "recommended_action": self.recommended_action,
            "strategy": self.strategy.value,
        }


class FailureAnalyzer:
    PATTERN_MAP: dict[ErrorCategory, list[str]] = {
        ErrorCategory.SYNTAX_ERROR: [
            "SyntaxError",
            "unexpected EOF",
            "invalid syntax",
            "IndentationError",
            "TabError",
            "unterminated string",
        ],
        ErrorCategory.TYPE_ERROR: [
            "TypeError",
            "unsupported operand",
            "unexpected type",
            "NoneType",
            "got an unexpected keyword argument",
            "argument must be",
        ],
        ErrorCategory.IMPORT_ERROR: [
            "ImportError",
            "ModuleNotFoundError",
            "No module named",
            "cannot import name",
        ],
        ErrorCategory.TEST_FAILURE: [
            "FAILED",
            "AssertionError",
            "assert equal",
            "assert ==",
            "assert !=",
            "assert True",
            "assert False",
            "FAIL ",
            "short test summary",
            "FAILED tests/",
        ],
        ErrorCategory.PERMISSION_ERROR: [
            "PermissionError",
            "Permission denied",
            "Access denied",
            "EACCES",
        ],
        ErrorCategory.TIMEOUT: [
            "timed out",
            "TimeoutExpired",
            "TIMEOUT",
            "deadline exceeded",
        ],
        ErrorCategory.TOOL_ERROR: [
            "ToolValidationError",
            "Unknown tool",
            "tool execution failed",
        ],
        ErrorCategory.LLM_ERROR: [
            "OpenRouterError",
            "LLM error",
            "rate limit",
            "quota",
        ],
        ErrorCategory.DEPENDENCY_ERROR: [
            "No matching distribution",
            "Could not install",
            "pip install failed",
            "requires",
        ],
        ErrorCategory.CONFIGURATION_ERROR: [
            "ConfigError",
            "missing key",
            "invalid config",
            "OPENROUTER_API_KEY",
            "not configured",
        ],
        ErrorCategory.ENVIRONMENT_ERROR: [
            "FileNotFoundError",
            "No such file",
            "directory",
            "ENOENT",
        ],
        ErrorCategory.LOGIC_ERROR: [
            "IndexError",
            "KeyError",
            "AttributeError",
            "ValueError",
            "RuntimeError",
            "RecursionError",
            "infinite loop",
            "stack overflow",
        ],
        ErrorCategory.SECURITY_ERROR: [
            "security",
            "blocked",
            "forbidden",
            "unauthorized",
            "injection",
            "malicious",
        ],
        ErrorCategory.NETWORK_ERROR: [
            "ConnectionError",
            "ConnectionRefused",
            "DNS resolution",
            "network",
            "unreachable",
            "ENOTCONN",
        ],
        ErrorCategory.BUDGET_ERROR: [
            "budget exceeded",
            "limit reached",
            "quota exceeded",
            "max_llm_calls",
            "max_tool_calls",
        ],
        ErrorCategory.CHECKPOINT_ERROR: [
            "checkpoint",
            "corrupt",
            "stale",
            "schema mismatch",
        ],
        ErrorCategory.REGRESSION: [
            "regression",
            "previously passing",
            "was working",
            "PASS → FAIL",
        ],
        ErrorCategory.INVALID_PLAN: [
            "invalid plan",
            "plan validation",
            "missing subtask",
            "duplicate id",
            "unknown dependency",
        ],
    }

    def classify(self, output: str) -> ErrorCategory:
        if not output:
            return ErrorCategory.UNKNOWN

        for category, patterns in self.PATTERN_MAP.items():
            for pattern in patterns:
                if pattern.lower() in output.lower():
                    return category

        return ErrorCategory.UNKNOWN

    def analyze(self, test_output: str, command_output: str = "") -> FailureDiagnosis:
        combined = f"{test_output}\n{command_output}".strip()

        category = self.classify(combined)

        evidence_lines: list[str] = []
        for line in combined.split("\n"):
            stripped = line.strip()
            if stripped and (
                "error" in stripped.lower()
                or "fail" in stripped.lower()
                or "traceback" in stripped.lower()
                or "assert" in stripped.lower()
            ):
                evidence_lines.append(stripped)
            if len(evidence_lines) >= 10:
                break

        evidence = "\n".join(evidence_lines) if evidence_lines else combined[:500]

        hypothesis = self._generate_hypothesis(category, combined)
        suggested_fix = self._generate_suggested_fix(category, combined)
        strategy = FAILURE_STRATEGY_MAP.get(category, FailureStrategy.DIAGNOSE)
        recommended_action = self._generate_recommended_action(category, strategy)

        return FailureDiagnosis(
            failure=combined[:1000],
            evidence=evidence,
            category=category,
            hypothesis=hypothesis,
            confidence=0.5 if category != ErrorCategory.UNKNOWN else 0.2,
            suggested_fix=suggested_fix,
            recommended_action=recommended_action,
            strategy=strategy,
        )

    def _generate_hypothesis(self, category: ErrorCategory, output: str) -> str:
        if category == ErrorCategory.SYNTAX_ERROR:
            return "Likely a syntax error in Python code. Check for missing colons, parentheses, or indentation."
        elif category == ErrorCategory.TYPE_ERROR:
            return "Type mismatch in function call or operation. Check argument types and None handling."
        elif category == ErrorCategory.IMPORT_ERROR:
            return "Missing or incorrectly named import. Verify module exists and name is correct."
        elif category == ErrorCategory.TEST_FAILURE:
            return "Test assertion failed. Compare expected vs actual values in the failing test."
        elif category == ErrorCategory.PERMISSION_ERROR:
            return "Insufficient permissions. Check file/directory ownership and access rights."
        elif category == ErrorCategory.TIMEOUT:
            return "Operation exceeded time limit. Check for infinite loops or slow operations."
        elif category == ErrorCategory.TOOL_ERROR:
            return "Tool execution error. Check tool arguments and tool availability."
        elif category == ErrorCategory.LLM_ERROR:
            return "LLM API error. Check API key, quota, and network connectivity."
        elif category == ErrorCategory.DEPENDENCY_ERROR:
            return "Missing dependency. Install required package."
        elif category == ErrorCategory.CONFIGURATION_ERROR:
            return "Configuration issue. Check environment variables and config files."
        elif category == ErrorCategory.ENVIRONMENT_ERROR:
            return "File or directory not found. Verify path exists and is correct."
        elif category == ErrorCategory.LOGIC_ERROR:
            return "Runtime logic error. Check for incorrect indexing, missing keys, or None attribute access."
        return "Unable to determine root cause from output. Manual inspection may be needed."

    def _generate_suggested_fix(self, category: ErrorCategory, output: str) -> str:
        if category == ErrorCategory.SYNTAX_ERROR:
            return "Fix syntax: check colons, parentheses, indentation, and string quoting."
        elif category == ErrorCategory.TYPE_ERROR:
            return "Fix types: add type checks, handle None, verify function signatures."
        elif category == ErrorCategory.IMPORT_ERROR:
            return "Fix import: add missing dependency or correct module name."
        elif category == ErrorCategory.TEST_FAILURE:
            return "Fix assertion: compare expected vs actual, update test or implementation."
        elif category == ErrorCategory.PERMISSION_ERROR:
            return "Fix permissions: check file ownership or run with appropriate user."
        elif category == ErrorCategory.TIMEOUT:
            return "Fix timeout: optimize loops, reduce computation, or increase timeout."
        elif category == ErrorCategory.TOOL_ERROR:
            return "Fix tool: verify arguments match tool schema."
        elif category == ErrorCategory.LLM_ERROR:
            return "Fix LLM: check API key, wait for rate limit, or use fallback model."
        elif category == ErrorCategory.DEPENDENCY_ERROR:
            return "Fix dependency: pip install the missing package."
        elif category == ErrorCategory.CONFIGURATION_ERROR:
            return "Fix config: set required environment variables in .env."
        elif category == ErrorCategory.ENVIRONMENT_ERROR:
            return "Fix path: verify file/directory exists before accessing."
        elif category == ErrorCategory.LOGIC_ERROR:
            return "Fix logic: add bounds checking, handle edge cases, validate inputs."
        return "Review error output and trace to the originating code."

    def _generate_recommended_action(self, category: ErrorCategory, strategy: FailureStrategy) -> str:
        if strategy == FailureStrategy.ABORT:
            return "Stop execution. Budget or critical limit exceeded."
        elif strategy == FailureStrategy.BLOCK:
            return "Do not retry. This failure requires manual intervention or a different approach."
        elif strategy == FailureStrategy.REPLAN:
            return "Current plan is invalid. Generate a new plan with different approach."
        elif strategy == FailureStrategy.RETRY:
            return "Retry the operation. Transient error may resolve on retry."
        elif strategy == FailureStrategy.FIX:
            return "Apply a targeted fix to the identified issue."
        elif strategy == FailureStrategy.DIAGNOSE:
            return "Gather more evidence before deciding on action."
        elif strategy == FailureStrategy.RETEST:
            return "Re-run tests to confirm the fix resolves the issue."
        return "Analyze the failure and decide on next steps."

    @staticmethod
    def get_strategy(category: ErrorCategory) -> FailureStrategy:
        """Get the recommended strategy for a given error category."""
        return FAILURE_STRATEGY_MAP.get(category, FailureStrategy.DIAGNOSE)
