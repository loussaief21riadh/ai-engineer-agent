"""Validation engine for V2.0 — programmatic validation beyond reviewer LLM."""

from __future__ import annotations

import ast
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ValidationCheck(str, Enum):
    SYNTAX = "SYNTAX"
    TESTS = "TESTS"
    TEST_EXIT_CODE = "TEST_EXIT_CODE"
    GIT_DIFF = "GIT_DIFF"
    EXPECTED_FILES = "EXPECTED_FILES"
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION"


class ValidationResult(BaseModel):
    check: ValidationCheck
    passed: bool
    details: str = ""
    evidence: str = ""


class ValidationReport(BaseModel):
    results: list[ValidationResult] = Field(default_factory=list)
    overall_passed: bool = True

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)
        if not result.passed:
            self.overall_passed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "checks": [r.model_dump() for r in self.results],
        }


class Validator:
    def validate_all(
        self,
        test_results: dict[str, Any] | None = None,
        files_modified: list[str] | None = None,
        expected_files: list[str] | None = None,
        executions: list[dict[str, Any]] | None = None,
    ) -> ValidationReport:
        report = ValidationReport()

        if files_modified:
            syntax_result = self.validate_syntax(files_modified)
            report.add(syntax_result)

        if test_results is not None:
            report.add(self.validate_test_exit_code(test_results))

        if files_modified is not None and expected_files is not None:
            report.add(self.validate_expected_files(files_modified, expected_files))

        if executions is not None:
            report.add(self.validate_execution_success(executions))

        return report

    def validate_test_exit_code(self, test_results: dict[str, Any]) -> ValidationResult:
        exit_code = test_results.get("exit_code", -1)
        success = test_results.get("success", False)

        if exit_code == -1 and isinstance(test_results.get("result"), dict):
            exit_code = test_results["result"].get("exit_code", -1)

        passed = exit_code == 0 and success
        details = f"exit_code={exit_code}, success={success}"

        stdout = test_results.get("stdout", "")
        if not stdout and isinstance(test_results.get("result"), dict):
            stdout = test_results["result"].get("stdout", "")
        evidence = stdout[-500:] if stdout else ""

        return ValidationResult(
            check=ValidationCheck.TEST_EXIT_CODE,
            passed=passed,
            details=details,
            evidence=evidence,
        )

    def validate_expected_files(
        self,
        files_modified: list[str],
        expected_files: list[str],
    ) -> ValidationResult:
        missing = [f for f in expected_files if f not in files_modified]
        passed = len(missing) == 0
        details = f"Modified: {files_modified}, Expected: {expected_files}"
        if missing:
            details += f", Missing: {missing}"

        return ValidationResult(
            check=ValidationCheck.EXPECTED_FILES,
            passed=passed,
            details=details,
        )

    BLOCKED_PATTERNS = ["not allowed", "blocked git", "unknown executable", "shell metacharacters"]

    def validate_execution_success(
        self,
        executions: list[dict[str, Any]],
    ) -> ValidationResult:
        failures = []
        for ex in executions:
            if ex.get("success", False):
                continue
            error = (ex.get("error") or "").lower()
            if any(pat in error for pat in self.BLOCKED_PATTERNS):
                continue
            failures.append(ex)
        passed = len(failures) == 0

        failure_details = []
        for ex in failures:
            tool = ex.get("tool_name", "?")
            error = ex.get("error", "unknown error")
            failure_details.append(f"{tool}: {error}")

        details = f"Total: {len(executions)}, Failed: {len(failures)}"
        evidence = "\n".join(failure_details) if failure_details else "All executions succeeded"

        return ValidationResult(
            check=ValidationCheck.TESTS,
            passed=passed,
            details=details,
            evidence=evidence,
        )

    def validate_syntax(self, files_modified: list[str]) -> ValidationResult:
        syntax_errors: list[str] = []
        py_files = [f for f in files_modified if f.endswith(".py")]
        for filepath in py_files:
            try:
                with open(filepath) as f:
                    source = f.read()
                ast.parse(source, filename=filepath)
            except SyntaxError as e:
                syntax_errors.append(f"{filepath}:{e.lineno}: {e.msg}")
            except (OSError, UnicodeDecodeError):
                pass

        passed = len(syntax_errors) == 0
        details = f"Checked {len(py_files)} Python files"
        evidence = "\n".join(syntax_errors) if syntax_errors else "All files have valid syntax"

        return ValidationResult(
            check=ValidationCheck.SYNTAX,
            passed=passed,
            details=details,
            evidence=evidence,
        )
