"""Regression Engine for V3.2 — baseline/after-state comparison for regression detection."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"
    NEW = "NEW"
    REMOVED = "REMOVED"


class RegressionSeverity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TestResult(BaseModel):
    """Result of a single test."""
    test_id: str
    name: str
    status: TestStatus
    duration_ms: float = 0.0
    error_message: str = ""
    file: str = ""


class RegressionResult(BaseModel):
    """Result of comparing baseline vs after-state."""
    baseline_test_count: int = 0
    after_test_count: int = 0
    passed_to_failed: list[str] = Field(default_factory=list)
    failed_to_passed: list[str] = Field(default_factory=list)
    new_tests: list[str] = Field(default_factory=list)
    removed_tests: list[str] = Field(default_factory=list)
    unchanged_pass: list[str] = Field(default_factory=list)
    unchanged_fail: list[str] = Field(default_factory=list)
    severity: RegressionSeverity = RegressionSeverity.NONE
    has_regression: bool = False
    summary: str = ""
    timestamp: float = Field(default_factory=time.time)


class TestBaseline(BaseModel):
    """Baseline test state captured before implementation."""
    task_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    test_results: list[TestResult] = Field(default_factory=list)
    test_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    test_ids: set[str] = Field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "test_count": self.test_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "test_ids": list(self.test_ids),
        }


class RegressionEngine:
    """Detects regressions by comparing baseline vs after-state test results."""

    def __init__(self) -> None:
        self._baseline: TestBaseline | None = None
        self._history: list[RegressionResult] = []

    def capture_baseline(
        self,
        test_results: dict[str, Any],
        task_id: str = "",
    ) -> TestBaseline:
        """Capture baseline test state from pytest-like output."""
        tests = self._parse_test_results(test_results)

        baseline = TestBaseline(
            task_id=task_id,
            test_results=tests,
            test_count=len(tests),
            pass_count=sum(1 for t in tests if t.status == TestStatus.PASS),
            fail_count=sum(1 for t in tests if t.status == TestStatus.FAIL),
            skip_count=sum(1 for t in tests if t.status == TestStatus.SKIP),
            test_ids={t.test_id for t in tests},
        )

        self._baseline = baseline
        return baseline

    def capture_after_state(
        self,
        test_results: dict[str, Any],
        task_id: str = "",
    ) -> list[TestResult]:
        """Capture after-state test results."""
        return self._parse_test_results(test_results)

    def compare(
        self,
        after_results: dict[str, Any],
        task_id: str = "",
    ) -> RegressionResult:
        """Compare baseline vs after-state and detect regressions."""
        if self._baseline is None:
            return RegressionResult(
                severity=RegressionSeverity.NONE,
                summary="No baseline captured. Cannot detect regressions.",
            )

        after_tests = self._parse_test_results(after_results)
        after_map = {t.test_id: t for t in after_tests}
        baseline_map = {t.test_id: t for t in self._baseline.test_results}

        passed_to_failed: list[str] = []
        failed_to_passed: list[str] = []
        new_tests: list[str] = []
        removed_tests: list[str] = []
        unchanged_pass: list[str] = []
        unchanged_fail: list[str] = []

        for test_id, baseline_test in baseline_map.items():
            if test_id not in after_map:
                removed_tests.append(test_id)
                continue

            after_test = after_map[test_id]

            if baseline_test.status == TestStatus.PASS and after_test.status == TestStatus.FAIL:
                passed_to_failed.append(test_id)
            elif baseline_test.status == TestStatus.FAIL and after_test.status == TestStatus.PASS:
                failed_to_passed.append(test_id)
            elif baseline_test.status == TestStatus.PASS and after_test.status == TestStatus.PASS:
                unchanged_pass.append(test_id)
            elif baseline_test.status == TestStatus.FAIL and after_test.status == TestStatus.FAIL:
                unchanged_fail.append(test_id)

        for test_id in after_map:
            if test_id not in baseline_map:
                new_tests.append(test_id)

        severity = self._assess_severity(passed_to_failed, removed_tests)
        has_regression = len(passed_to_failed) > 0 or (
            severity in (RegressionSeverity.HIGH, RegressionSeverity.CRITICAL)
        )

        result = RegressionResult(
            baseline_test_count=self._baseline.test_count,
            after_test_count=len(after_tests),
            passed_to_failed=passed_to_failed,
            failed_to_passed=failed_to_passed,
            new_tests=new_tests,
            removed_tests=removed_tests,
            unchanged_pass=unchanged_pass,
            unchanged_fail=unchanged_fail,
            severity=severity,
            has_regression=has_regression,
            summary=self._build_summary(
                passed_to_failed, failed_to_passed, new_tests,
                removed_tests, severity, has_regression,
            ),
        )

        self._history.append(result)
        return result

    def _assess_severity(
        self,
        passed_to_failed: list[str],
        removed_tests: list[str],
    ) -> RegressionSeverity:
        regression_count = len(passed_to_failed)
        removal_count = len(removed_tests)

        if regression_count == 0 and removal_count == 0:
            return RegressionSeverity.NONE
        if regression_count == 0 and removal_count <= 2:
            return RegressionSeverity.LOW
        if regression_count <= 2:
            return RegressionSeverity.MEDIUM
        if regression_count <= 5:
            return RegressionSeverity.HIGH
        return RegressionSeverity.CRITICAL

    def _build_summary(
        self,
        passed_to_failed: list[str],
        failed_to_passed: list[str],
        new_tests: list[str],
        removed_tests: list[str],
        severity: RegressionSeverity,
        has_regression: bool,
    ) -> str:
        parts: list[str] = []

        if has_regression:
            parts.append(f"REGRESSION DETECTED ({severity.value})")
            parts.append(f"  Tests that REGRESSED (PASS→FAIL): {len(passed_to_failed)}")
            for t in passed_to_failed[:5]:
                parts.append(f"    - {t}")
        else:
            parts.append("No regressions detected")

        if failed_to_passed:
            parts.append(f"  Tests FIXED (FAIL→PASS): {len(failed_to_passed)}")
            for t in failed_to_passed[:5]:
                parts.append(f"    + {t}")

        if new_tests:
            parts.append(f"  New tests: {len(new_tests)}")

        if removed_tests:
            parts.append(f"  Removed tests: {len(removed_tests)}")
            for t in removed_tests[:5]:
                parts.append(f"    - {t}")

        return "\n".join(parts)

    def _parse_test_results(self, results: dict[str, Any]) -> list[TestResult]:
        """Parse test results from various formats into TestResult list."""
        tests: list[TestResult] = []

        if "tests" in results and isinstance(results["tests"], list):
            for t in results["tests"]:
                if isinstance(t, dict):
                    tests.append(TestResult(
                        test_id=t.get("test_id", t.get("name", "")),
                        name=t.get("name", ""),
                        status=TestStatus(t.get("status", "PASS")),
                        duration_ms=t.get("duration_ms", 0),
                        error_message=t.get("error_message", ""),
                        file=t.get("file", ""),
                    ))
            return tests

        stdout = results.get("stdout", "")
        if not stdout and isinstance(results.get("result"), dict):
            stdout = results["result"].get("stdout", "")

        if stdout:
            tests = self._parse_pytest_output(stdout)

        if not tests:
            success = results.get("success", False)
            exit_code = results.get("exit_code", -1)
            if not success and isinstance(results.get("result"), dict):
                exit_code = results["result"].get("exit_code", exit_code)

            tests.append(TestResult(
                test_id="__overall__",
                name="Overall test result",
                status=TestStatus.PASS if exit_code == 0 else TestStatus.FAIL,
            ))

        return tests

    def _parse_pytest_output(self, stdout: str) -> list[TestResult]:
        """Parse pytest verbose output to extract individual test results."""
        import re
        tests: list[TestResult] = []

        test_pattern = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(\S+)", re.MULTILINE)
        for match in test_pattern.finditer(stdout):
            status_str = match.group(1)
            test_name = match.group(2)

            status_map = {
                "PASSED": TestStatus.PASS,
                "FAILED": TestStatus.FAIL,
                "ERROR": TestStatus.ERROR,
                "SKIPPED": TestStatus.SKIP,
            }

            tests.append(TestResult(
                test_id=test_name,
                name=test_name,
                status=status_map.get(status_str, TestStatus.FAIL),
            ))

        return tests

    def get_baseline(self) -> TestBaseline | None:
        return self._baseline

    def get_history(self) -> list[RegressionResult]:
        return list(self._history)

    def has_active_regression(self) -> bool:
        if not self._history:
            return False
        return self._history[-1].has_regression

    def clear_baseline(self) -> None:
        self._baseline = None

    def summary(self) -> dict[str, Any]:
        return {
            "has_baseline": self._baseline is not None,
            "baseline_tests": self._baseline.test_count if self._baseline else 0,
            "comparisons": len(self._history),
            "active_regression": self.has_active_regression(),
            "last_severity": self._history[-1].severity.value if self._history else "NONE",
        }
