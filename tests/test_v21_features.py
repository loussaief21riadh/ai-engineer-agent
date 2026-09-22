"""Tests for V2.1 new features: security check, ExecutionEvidence, memory persistence,
reviewer verdict, validator improvements, diagnostics improvements."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.agent.context import ContextBuilder, TaskContext, TrustLevel
from app.agent.diagnostics import ErrorCategory, FailureAnalyzer
from app.agent.memory import ProjectMemory
from app.agent.phases import TaskPhase, can_transition, next_phases
from app.agent.planner import PlanValidator, SubtaskStatus, TaskPlan
from app.agent.reviewer import Reviewer
from app.agent.router import ModelRouter, TaskCategory
from app.agent.validator import ValidationCheck, Validator
from app.llm.openrouter import ChatResponse
from app.models.schemas import ExecutionEvidence, ReviewResult, ReviewVerdict, ToolExecution


# ============================================================
# ExecutionEvidence
# ============================================================

class TestExecutionEvidence:
    def test_from_tool_execution_success(self):
        ex = ToolExecution(
            step=1,
            tool_name="run_command",
            arguments={"command": "ls"},
            success=True,
            result={"exit_code": 0, "stdout": "file.py\n", "stderr": ""},
            duration_ms=50.0,
        )
        ev = ExecutionEvidence.from_tool_execution(ex)
        assert ev.tool == "run_command"
        assert ev.command == "ls"
        assert ev.success is True
        assert ev.exit_code == 0
        assert ev.trust_level == "TOOL_VERIFIED"
        assert ev.duration_ms == 50.0

    def test_from_tool_execution_failure(self):
        ex = ToolExecution(
            step=1,
            tool_name="run_tests",
            arguments={},
            success=False,
            error="Test failed",
        )
        ev = ExecutionEvidence.from_tool_execution(ex)
        assert ev.success is False
        assert "Test failed" in ev.stderr_summary

    def test_from_tool_execution_truncates_long_output(self):
        ex = ToolExecution(
            step=1,
            tool_name="run_command",
            arguments={},
            success=True,
            result={"stdout": "x" * 1000, "stderr": ""},
        )
        ev = ExecutionEvidence.from_tool_execution(ex)
        assert len(ev.stdout_summary) <= 500


# ============================================================
# ReviewVerdict
# ============================================================

class TestReviewVerdict:
    def test_verdict_values(self):
        assert ReviewVerdict.APPROVE.value == "APPROVE"
        assert ReviewVerdict.REJECT.value == "REJECT"
        assert ReviewVerdict.NEEDS_MORE_EVIDENCE.value == "NEEDS_MORE_EVIDENCE"

    def test_review_result_default_verdict(self):
        r = ReviewResult(approved=True)
        assert r.verdict == ReviewVerdict.APPROVE

    def test_review_result_needs_more_evidence(self):
        r = ReviewResult(
            approved=False,
            verdict=ReviewVerdict.NEEDS_MORE_EVIDENCE,
            summary="Need more tests",
        )
        assert r.verdict == ReviewVerdict.NEEDS_MORE_EVIDENCE
        assert r.approved is False


# ============================================================
# Security check phase
# ============================================================

class TestSecurityCheckPhase:
    def test_security_check_phase_exists(self):
        assert hasattr(TaskPhase, "SECURITY_CHECK")
        assert TaskPhase.SECURITY_CHECK.value == "SECURITY_CHECK"

    def test_security_check_transitions_from_test(self):
        assert can_transition(TaskPhase.TEST, TaskPhase.SECURITY_CHECK)

    def test_security_check_transitions_to_review(self):
        assert can_transition(TaskPhase.SECURITY_CHECK, TaskPhase.REVIEW)

    def test_security_check_transitions_to_diagnose(self):
        assert can_transition(TaskPhase.SECURITY_CHECK, TaskPhase.DIAGNOSE)

    def test_test_transitions_to_security_check(self):
        valid = next_phases(TaskPhase.TEST)
        assert TaskPhase.SECURITY_CHECK in valid

    def test_security_check_not_from_implement(self):
        assert not can_transition(TaskPhase.IMPLEMENT, TaskPhase.SECURITY_CHECK)


# ============================================================
# Memory persistence
# ============================================================

class TestMemoryPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            mem = ProjectMemory()
            mem.add_important_file("main.py", "Entry point")
            mem.add_convention("Use type hints")
            mem.add_successful_fix("Fixed import error")
            mem.save(path)

            loaded = ProjectMemory.load(path)
            assert loaded.important_files["main.py"] == "Entry point"
            assert "Use type hints" in loaded.project_conventions
            assert "Fixed import error" in loaded.successful_fixes

    def test_load_nonexistent_returns_empty(self):
        loaded = ProjectMemory.load("/nonexistent/path/memory.json")
        assert loaded.important_files == {}
        assert loaded.entries == []

    def test_load_corrupt_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            path.write_text("not valid json {{{")
            loaded = ProjectMemory.load(path)
            assert loaded.important_files == {}

    def test_to_context_string(self):
        mem = ProjectMemory()
        mem.add_important_file("app.py", "Main app")
        mem.add_convention("PEP8")
        ctx = mem.to_context_string()
        assert "app.py" in ctx
        assert "PEP8" in ctx

    def test_to_context_string_empty(self):
        mem = ProjectMemory()
        ctx = mem.to_context_string()
        assert "No project memory" in ctx

    def test_bounded_collections(self):
        mem = ProjectMemory()
        for i in range(60):
            mem.add_architecture_note(f"Note {i}")
        assert len(mem.architecture_notes) == 50

        for i in range(40):
            mem.add_decision(f"Decision {i}")
        assert len(mem.important_decisions) == 30


# ============================================================
# ModelRouter
# ============================================================

class TestModelRouter:
    def test_classify_debugging(self):
        r = ModelRouter()
        assert r.classify_task("Fix the bug in main.py") == TaskCategory.DEBUGGING
        assert r.classify_task("Debug the error") == TaskCategory.DEBUGGING

    def test_classify_coding(self):
        r = ModelRouter()
        assert r.classify_task("Implement a new feature") == TaskCategory.CODING
        assert r.classify_task("Write a test") == TaskCategory.CODING

    def test_classify_review(self):
        r = ModelRouter()
        assert r.classify_task("Review the code changes") == TaskCategory.REVIEW

    def test_classify_simple(self):
        r = ModelRouter()
        assert r.classify_task("What time is it?") == TaskCategory.SIMPLE

    def test_route_returns_model(self):
        r = ModelRouter(default_model="test-model")
        model = r.route(TaskCategory.CODING)
        assert isinstance(model, str)


# ============================================================
# Validator improvements
# ============================================================

class TestValidatorImprovements:
    def setup_method(self):
        self.validator = Validator()

    def test_validate_test_exit_code_nested_result(self):
        result = self.validator.validate_test_exit_code({
            "success": True,
            "result": {"exit_code": 0, "stdout": "All passed"},
        })
        assert result.passed is True

    def test_validate_test_exit_code_nested_failure(self):
        result = self.validator.validate_test_exit_code({
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL"},
        })
        assert result.passed is False

    def test_validate_execution_success_blocks_security_rejections(self):
        result = self.validator.validate_execution_success([
            {"tool_name": "run_command", "success": False, "error": "Unknown executable: 'rm'"},
        ])
        assert result.passed is True

    def test_validate_execution_success_real_failure(self):
        result = self.validator.validate_execution_success([
            {"tool_name": "write_file", "success": False, "error": "Permission denied"},
        ])
        assert result.passed is False

    def test_validate_execution_success_mixed(self):
        result = self.validator.validate_execution_success([
            {"tool_name": "run_command", "success": False, "error": "Unknown executable: 'curl'"},
            {"tool_name": "write_file", "success": False, "error": "Disk full"},
        ])
        assert result.passed is False

    def test_validate_execution_success_all_blocked(self):
        result = self.validator.validate_execution_success([
            {"tool_name": "run_command", "success": False, "error": "Unknown executable: 'rm'"},
            {"tool_name": "run_command", "success": False, "error": "Shell metacharacters are not allowed"},
            {"tool_name": "run_command", "success": False, "error": "python -c is not allowed"},
        ])
        assert result.passed is True


# ============================================================
# Diagnostics improvements
# ============================================================

class TestDiagnosticsImprovements:
    def setup_method(self):
        self.analyzer = FailureAnalyzer()

    def test_logic_error_patterns(self):
        assert self.analyzer.classify("IndexError: list index out of range") == ErrorCategory.LOGIC_ERROR
        assert self.analyzer.classify("KeyError: 'missing_key'") == ErrorCategory.LOGIC_ERROR
        assert self.analyzer.classify("RecursionError: maximum recursion") == ErrorCategory.LOGIC_ERROR
        assert self.analyzer.classify("AttributeError: custom object has no attribute 'foo'") == ErrorCategory.LOGIC_ERROR

    def test_suggested_fix_populated(self):
        diag = self.analyzer.analyze("SyntaxError: invalid syntax")
        assert diag.suggested_fix != ""

    def test_suggested_fix_logic_error(self):
        diag = self.analyzer.analyze("IndexError: list index out of range")
        assert "bounds" in diag.suggested_fix.lower() or "edge" in diag.suggested_fix.lower()

    def test_suggested_fix_import_error(self):
        diag = self.analyzer.analyze("ModuleNotFoundError: No module named 'foo'")
        assert "install" in diag.suggested_fix.lower() or "import" in diag.suggested_fix.lower()

    def test_hypothesis_logic_error(self):
        diag = self.analyzer.analyze("KeyError: 'foo'")
        assert "Runtime logic error" in diag.hypothesis


# ============================================================
# Reviewer curly brace safety
# ============================================================

class TestReviewerCurlyBraceSafety:
    def test_reviewer_handles_curly_braces_in_context(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        reviewer = Reviewer(client=mock_client)

        result = reviewer.review(
            task="Fix {broken} code",
            changes="Added {variable} handling",
            diff="diff --git a/file.py\n+{x = 1}",
            test_results='{"key": "value with {braces}"}',
            context="Context with { interpolation }",
        )
        assert result.approved is True
        assert result.verdict == ReviewVerdict.APPROVE

    def test_reviewer_needs_more_evidence(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(content=json.dumps({
            "approved": False,
            "verdict": "NEEDS_MORE_EVIDENCE",
            "findings": [],
            "summary": "Insufficient evidence to approve.",
        }))
        reviewer = Reviewer(client=mock_client)

        result = reviewer.review(task="Review changes")
        assert result.verdict == ReviewVerdict.NEEDS_MORE_EVIDENCE
        assert result.approved is False


# ============================================================
# Context trust levels surfaced
# ============================================================

class TestContextTrustSurfaced:
    def test_observations_include_trust_in_prompt(self):
        ctx = TaskContext()
        ctx.add_observation("File read", trust=TrustLevel.TOOL_VERIFIED)
        ctx.add_observation("User claim", trust=TrustLevel.USER_ASSERTED)

        builder = ContextBuilder()
        prompt = builder.build_phase_prompt(ctx, "TEST", "test task")
        assert "[TOOL_VERIFIED]" in prompt
        assert "[USER_ASSERTED]" in prompt

    def test_review_context_includes_trust(self):
        ctx = TaskContext()
        ctx.add_observation("Tool output", trust=TrustLevel.TOOL_VERIFIED)

        builder = ContextBuilder()
        context = builder.build_review_context(ctx, "task", "changes", "diff", "tests")
        assert "[TOOL_VERIFIED]" in context


# ============================================================
# Validation blocks on failure
# ============================================================

class TestValidationBlocks:
    def test_validate_blocks_on_test_failure(self):
        from app.agent.validator import Validator, ValidationReport, ValidationResult, ValidationCheck

        validator = Validator()
        report = validator.validate_all(
            test_results={"success": False, "result": {"exit_code": 1, "stdout": "FAIL"}},
        )
        assert report.overall_passed is False

        phase_history = []
        from app.agent.phases import TaskPhase, can_transition
        if not report.overall_passed:
            phase_history.append(TaskPhase.FAILED.value)
        assert "FAILED" in phase_history
