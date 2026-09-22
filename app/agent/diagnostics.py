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
    UNKNOWN = "UNKNOWN"


class FailureDiagnosis(BaseModel):
    failure: str
    evidence: str = ""
    category: ErrorCategory = ErrorCategory.UNKNOWN
    hypothesis: str = ""
    confidence: float = 0.0
    suggested_fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure": self.failure,
            "evidence": self.evidence,
            "category": self.category.value,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "suggested_fix": self.suggested_fix,
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

        return FailureDiagnosis(
            failure=combined[:1000],
            evidence=evidence,
            category=category,
            hypothesis=hypothesis,
            confidence=0.5 if category != ErrorCategory.UNKNOWN else 0.2,
            suggested_fix="",
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
        return "Unable to determine root cause from output. Manual inspection may be needed."
