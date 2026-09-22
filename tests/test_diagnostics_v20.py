"""Tests for V2.0-C Failure Diagnostics and Validator."""

from __future__ import annotations

import pytest

from app.agent.diagnostics import ErrorCategory, FailureAnalyzer, FailureDiagnosis
from app.agent.validator import ValidationCheck, ValidationReport, ValidationResult, Validator


class TestErrorCategory:
    def test_all_categories_defined(self):
        cats = [c.value for c in ErrorCategory]
        assert "SYNTAX_ERROR" in cats
        assert "TYPE_ERROR" in cats
        assert "IMPORT_ERROR" in cats
        assert "TEST_FAILURE" in cats
        assert "LOGIC_ERROR" in cats
        assert "CONFIGURATION_ERROR" in cats
        assert "DEPENDENCY_ERROR" in cats
        assert "ENVIRONMENT_ERROR" in cats
        assert "PERMISSION_ERROR" in cats
        assert "TIMEOUT" in cats
        assert "TOOL_ERROR" in cats
        assert "LLM_ERROR" in cats
        assert "UNKNOWN" in cats

    def test_category_count(self):
        assert len(ErrorCategory) == 13


class TestFailureDiagnosis:
    def test_diagnosis_creation(self):
        d = FailureDiagnosis(failure="test failed", category=ErrorCategory.TEST_FAILURE)
        assert d.failure == "test failed"
        assert d.category == ErrorCategory.TEST_FAILURE

    def test_diagnosis_to_dict(self):
        d = FailureDiagnosis(
            failure="test failed",
            evidence="FAIL test_x",
            category=ErrorCategory.TEST_FAILURE,
            hypothesis="assertion mismatch",
            confidence=0.7,
        )
        result = d.to_dict()
        assert result["category"] == "TEST_FAILURE"
        assert result["confidence"] == 0.7


class TestFailureAnalyzer:
    def setup_method(self):
        self.analyzer = FailureAnalyzer()

    def test_classify_syntax_error(self):
        assert self.analyzer.classify("SyntaxError: invalid syntax") == ErrorCategory.SYNTAX_ERROR

    def test_classify_type_error(self):
        assert self.analyzer.classify("TypeError: unsupported operand") == ErrorCategory.TYPE_ERROR

    def test_classify_import_error(self):
        assert self.analyzer.classify("ModuleNotFoundError: No module named 'foo'") == ErrorCategory.IMPORT_ERROR

    def test_classify_test_failure(self):
        assert self.analyzer.classify("FAILED tests/test_foo.py::test_bar") == ErrorCategory.TEST_FAILURE

    def test_classify_permission_error(self):
        assert self.analyzer.classify("PermissionError: [Errno 13] Permission denied") == ErrorCategory.PERMISSION_ERROR

    def test_classify_timeout(self):
        assert self.analyzer.classify("Command timed out after 30s") == ErrorCategory.TIMEOUT

    def test_classify_tool_error(self):
        assert self.analyzer.classify("ToolValidationError: Invalid arguments") == ErrorCategory.TOOL_ERROR

    def test_classify_llm_error(self):
        assert self.analyzer.classify("OpenRouterError: rate limit exceeded") == ErrorCategory.LLM_ERROR

    def test_classify_dependency_error(self):
        assert self.analyzer.classify("No matching distribution found for foo") == ErrorCategory.DEPENDENCY_ERROR

    def test_classify_config_error(self):
        assert self.analyzer.classify("OPENROUTER_API_KEY is not configured") == ErrorCategory.CONFIGURATION_ERROR

    def test_classify_environment_error(self):
        assert self.analyzer.classify("FileNotFoundError: [Errno 2] No such file") == ErrorCategory.ENVIRONMENT_ERROR

    def test_classify_unknown(self):
        assert self.analyzer.classify("") == ErrorCategory.UNKNOWN

    def test_classify_case_insensitive(self):
        assert self.analyzer.classify("SYNTAXERROR: something") == ErrorCategory.SYNTAX_ERROR

    def test_analyze_test_output(self):
        output = "FAILED tests/test_calc.py::test_add - AssertionError: 2 != 3"
        diag = self.analyzer.analyze(output)
        assert diag.category == ErrorCategory.TEST_FAILURE
        assert diag.confidence > 0
        assert len(diag.hypothesis) > 0
        assert "FAIL" in diag.evidence or "assert" in diag.evidence

    def test_analyze_empty_output(self):
        diag = self.analyzer.analyze("")
        assert diag.category == ErrorCategory.UNKNOWN
        assert diag.confidence == 0.2

    def test_analyze_with_command_output(self):
        test_out = "FAILED test_x"
        cmd_out = "exit code 1"
        diag = self.analyzer.analyze(test_out, cmd_out)
        assert diag.category == ErrorCategory.TEST_FAILURE

    def test_hypothesis_for_syntax(self):
        diag = self.analyzer.analyze("SyntaxError: invalid syntax at line 5")
        assert "syntax" in diag.hypothesis.lower()

    def test_hypothesis_for_type_error(self):
        diag = self.analyzer.analyze("TypeError: expected str, got int")
        assert "type" in diag.hypothesis.lower()

    def test_hypothesis_for_import_error(self):
        diag = self.analyzer.analyze("ModuleNotFoundError: No module named 'requests'")
        assert "import" in diag.hypothesis.lower()

    def test_hypothesis_for_timeout(self):
        diag = self.analyzer.analyze("Command timed out after 30s")
        assert "time" in diag.hypothesis.lower()


class TestValidationCheck:
    def test_all_checks_defined(self):
        checks = [c.value for c in ValidationCheck]
        assert "SYNTAX" in checks
        assert "TESTS" in checks
        assert "TEST_EXIT_CODE" in checks
        assert "GIT_DIFF" in checks
        assert "EXPECTED_FILES" in checks
        assert "SCHEMA_VALIDATION" in checks


class TestValidationResult:
    def test_result_creation(self):
        r = ValidationResult(check=ValidationCheck.TESTS, passed=True, details="all good")
        assert r.passed is True
        assert r.check == ValidationCheck.TESTS

    def test_result_failed(self):
        r = ValidationResult(check=ValidationCheck.SYNTAX, passed=False, details="error found")
        assert r.passed is False


class TestValidationReport:
    def test_report_default(self):
        report = ValidationReport()
        assert report.overall_passed is True
        assert len(report.results) == 0

    def test_report_add_passed(self):
        report = ValidationReport()
        report.add(ValidationResult(check=ValidationCheck.TESTS, passed=True))
        assert report.overall_passed is True

    def test_report_add_failed(self):
        report = ValidationReport()
        report.add(ValidationResult(check=ValidationCheck.TESTS, passed=False))
        assert report.overall_passed is False

    def test_report_mixed(self):
        report = ValidationReport()
        report.add(ValidationResult(check=ValidationCheck.TESTS, passed=True))
        report.add(ValidationResult(check=ValidationCheck.SYNTAX, passed=False))
        assert report.overall_passed is False

    def test_report_to_dict(self):
        report = ValidationReport()
        report.add(ValidationResult(check=ValidationCheck.TESTS, passed=True, details="ok"))
        d = report.to_dict()
        assert d["overall_passed"] is True
        assert len(d["checks"]) == 1


class TestValidator:
    def setup_method(self):
        self.validator = Validator()

    def test_validate_test_exit_code_pass(self):
        result = self.validator.validate_test_exit_code({"exit_code": 0, "success": True})
        assert result.passed is True
        assert result.check == ValidationCheck.TEST_EXIT_CODE

    def test_validate_test_exit_code_fail(self):
        result = self.validator.validate_test_exit_code({"exit_code": 1, "success": False})
        assert result.passed is False

    def test_validate_test_exit_code_with_stdout(self):
        result = self.validator.validate_test_exit_code({
            "exit_code": 1,
            "success": False,
            "stdout": "FAILED test_x\nassert 1 == 2",
        })
        assert result.passed is False
        assert "FAILED" in result.evidence

    def test_validate_expected_files_pass(self):
        result = self.validator.validate_expected_files(
            ["app/main.py", "app/config.py"],
            ["app/main.py"],
        )
        assert result.passed is True

    def test_validate_expected_files_fail(self):
        result = self.validator.validate_expected_files(
            ["app/main.py"],
            ["app/main.py", "app/config.py"],
        )
        assert result.passed is False
        assert "app/config.py" in result.details

    def test_validate_execution_success_pass(self):
        result = self.validator.validate_execution_success([
            {"tool_name": "read_file", "success": True},
            {"tool_name": "write_file", "success": True},
        ])
        assert result.passed is True

    def test_validate_execution_success_fail(self):
        result = self.validator.validate_execution_success([
            {"tool_name": "read_file", "success": True},
            {"tool_name": "write_file", "success": False, "error": "Permission denied"},
        ])
        assert result.passed is False
        assert "Permission denied" in result.evidence

    def test_validate_all_with_test_results(self):
        report = self.validator.validate_all(
            test_results={"exit_code": 0, "success": True},
        )
        assert report.overall_passed is True

    def test_validate_all_with_expected_files(self):
        report = self.validator.validate_all(
            files_modified=["app/main.py"],
            expected_files=["app/main.py"],
        )
        assert report.overall_passed is True

    def test_validate_all_with_executions(self):
        report = self.validator.validate_all(
            executions=[{"tool_name": "read_file", "success": True}],
        )
        assert report.overall_passed is True

    def test_validate_all_mixed(self):
        report = self.validator.validate_all(
            test_results={"exit_code": 0, "success": True},
            executions=[{"tool_name": "write_file", "success": False, "error": "fail"}],
        )
        assert report.overall_passed is False
