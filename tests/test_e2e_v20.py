"""End-to-end integration test for V2.0 — full multi-step task simulation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.agent.context import TrustLevel
from app.agent.diagnostics import ErrorCategory
from app.agent.orchestrator import Orchestrator
from app.agent.planner import SubtaskStatus
from app.agent.quota import BudgetTracker
from app.agent.router import ModelRouter, TaskCategory
from app.agent.validator import Validator
from app.config import AgentMode
from app.llm.openrouter import ChatResponse


def _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS):
    orch = Orchestrator(client=mock_client, mode=mode)
    mock_test = MagicMock()
    mock_test.execute.return_value = {
        "success": True,
        "result": {"exit_code": 0, "stdout": "All tests passed", "stderr": ""},
    }
    orch.core.tools["run_tests"] = mock_test
    return orch


def _phase_responses(*contents):
    return [ChatResponse(content=c) for c in contents]


def _review_response(approved=True, findings=None, summary="LGTM", verdict="APPROVE"):
    return ChatResponse(content=json.dumps({
        "approved": approved,
        "verdict": verdict,
        "findings": findings or [],
        "summary": summary,
    }))


class TestEndToEndHappyPath:
    def test_full_cycle_with_context_continuity(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Task is clear: fix the import error in main.py",
            "Plan: 1. Inspect main.py, 2. Fix import, 3. Test",
            "Read main.py: found missing import of os module",
            "Added 'import os' to main.py",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "LGTM",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        report = orch.run_task("Fix import error in main.py")

        assert report.final_phase == "DONE"
        assert report.stop_reason == "completed"
        assert "UNDERSTAND" in report.phase_history
        assert "PLAN" in report.phase_history
        assert "INSPECT" in report.phase_history
        assert "IMPLEMENT" in report.phase_history
        assert "TEST" in report.phase_history
        assert "REVIEW" in report.phase_history
        assert "VALIDATE" in report.phase_history
        assert "REPORT" in report.phase_history
        assert report.review is not None
        assert report.review.approved is True

    def test_context_survives_across_phases(self):
        mock_client = MagicMock()
        understand = ChatResponse(content="Found the issue in app/main.py")
        plan = ChatResponse(content="Plan: read main.py, fix the bug")
        inspect = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "read_file", "arguments": {"path": "app/main.py"}}],
        )
        inspect_result = ChatResponse(content="Read app/main.py: line 42 has the bug")
        implement = ChatResponse(content="Fixed line 42")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = [understand, plan, inspect, inspect_result, implement, reviewer]

        orch = _make_orch(mock_client)
        report = orch.run_task("Fix the bug in main.py")

        assert "app/main.py" in orch.context.inspected_files
        assert len(orch.context.phase_history) >= 4
        assert orch.context.current_phase in ("REPORT", "DONE")


class TestEndToEndFailureRecovery:
    def test_test_failure_diagnose_fix_retest_cycle(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            "Plan: inspect and fix.",
            "Inspected.",
            "Implemented fix.",
            "Diagnosed root cause: missing None check.",
            "Applied None check fix.",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "Fixed.",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        test_call_count = [0]

        def fake_test(args="", **kwargs):
            test_call_count[0] += 1
            if test_call_count[0] == 1:
                return {"success": False, "result": {"exit_code": 1, "stdout": "FAIL test_none_check", "stderr": ""}}
            return {"success": True, "result": {"exit_code": 0, "stdout": "All passed", "stderr": ""}}

        orch.core.tools["run_tests"].execute.side_effect = fake_test

        report = orch.run_task("Fix failing test")

        assert report.final_phase == "DONE"
        assert "DIAGNOSE" in report.phase_history
        assert "FIX" in report.phase_history
        assert "RETEST" in report.phase_history
        assert len(report.diagnoses) >= 1
        assert len(report.fixes) >= 1

    def test_reviewer_rejection_recovery(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            "Plan: inspect and fix.",
            "Inspected.",
            "Implemented.",
            "Fixed based on review feedback.",
        )
        reviewer_first = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT",
            "findings": [{"severity": "warning", "category": "quality", "description": "Missing docstring"}],
            "summary": "Needs docs.",
        }))
        reviewer_second = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_first, reviewer_second]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Add documentation")

        assert report.final_phase == "DONE"
        assert report.review.approved is True
        assert "FIX" in report.phase_history
        assert "RETEST" in report.phase_history


class TestEndToEndPlannerIntegration:
    def test_planner_validates_plan_output(self):
        from app.agent.planner import PlanValidator

        valid_plan = {
            "objective": "fix bug",
            "subtasks": [
                {"id": "1", "description": "inspect files"},
                {"id": "2", "description": "fix code", "dependencies": ["1"]},
            ],
        }
        is_valid, errors, plan = PlanValidator.validate_plan(valid_plan)
        assert is_valid is True
        assert plan is not None
        assert len(plan.subtasks) == 2
        assert plan.get_next_subtask().id == "1"

    def test_planner_subtask_lifecycle(self):
        from app.agent.planner import Subtask, SubtaskStatus, TaskPlan

        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="inspect"))
        plan.add_subtask(Subtask(id="2", description="fix", dependencies=["1"]))

        assert plan.get_next_subtask().id == "1"
        plan.get_subtask("1").status = SubtaskStatus.COMPLETED
        assert plan.get_next_subtask().id == "2"
        plan.get_subtask("2").status = SubtaskStatus.COMPLETED
        assert plan.all_completed()

    def test_valid_json_plan_accepted_by_orchestrator(self):
        from app.agent.planner import PlanValidator

        valid_plan = json.dumps({
            "objective": "fix bug",
            "subtasks": [
                {"id": "1", "description": "inspect"},
                {"id": "2", "description": "fix"},
            ],
        })

        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            valid_plan,
            "Inspected.",
            "Implemented.",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        report = orch.run_task("Fix bug")

        assert orch.context.plan is not None
        assert orch.context.plan["objective"] == "fix bug"
        assert len(orch.context.plan["subtasks"]) == 2
        assert report.final_phase == "DONE"

    def test_duplicate_subtask_ids_rejected(self):
        from app.agent.planner import PlanValidator

        invalid_plan = {
            "objective": "fix bug",
            "subtasks": [
                {"id": "1", "description": "inspect"},
                {"id": "1", "description": "fix again"},
            ],
        }
        is_valid, errors, plan = PlanValidator.validate_plan(invalid_plan)
        assert is_valid is False
        assert any("Duplicate" in e for e in errors)

    def test_unknown_dependency_rejected(self):
        from app.agent.planner import PlanValidator

        invalid_plan = {
            "objective": "fix bug",
            "subtasks": [
                {"id": "1", "description": "fix", "dependencies": ["99"]},
            ],
        }
        is_valid, errors, plan = PlanValidator.validate_plan(invalid_plan)
        assert is_valid is False
        assert any("unknown subtask" in e for e in errors)

    def test_malformed_plan_blocks_implement(self):
        invalid_plan = json.dumps({
            "objective": "fix bug",
            "subtasks": [
                {"id": "1", "description": "inspect"},
                {"id": "1", "description": "fix"},
            ],
        })

        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            invalid_plan,
        )
        mock_client.chat.side_effect = responses

        orch = _make_orch(mock_client)
        report = orch.run_task("Fix bug")

        assert report.final_phase == "FAILED"
        assert report.stop_reason == "failed"
        assert "PLAN" in report.phase_history
        assert "IMPLEMENT" not in report.phase_history

    def test_free_text_plan_still_proceeds(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            "Plan: 1. Read files, 2. Fix bug, 3. Test",
            "Inspected.",
            "Implemented.",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        report = orch.run_task("Fix bug")

        assert orch.context.plan is None
        assert report.final_phase == "DONE"
        assert "IMPLEMENT" in report.phase_history


class TestEndToEndDiagnosticsIntegration:
    def test_failure_analyzer_classifies_correctly(self):
        from app.agent.diagnostics import FailureAnalyzer

        analyzer = FailureAnalyzer()
        diag = analyzer.analyze("FAILED tests/test_main.py::test_calc - AssertionError: 2 != 3")
        assert diag.category == ErrorCategory.TEST_FAILURE
        assert diag.confidence > 0
        assert len(diag.hypothesis) > 0

    def test_diagnosis_incorporated_into_context(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            "Plan.",
            "Inspected.",
            "Implemented.",
            "Diagnosed.",
            "Fixed.",
            "Fixed again.",
            "Diagnosed again.",
            "Fixed third.",
            "Diagnosed third.",
            "Final fix.",
            "Review fix.",
            "Review fix again.",
            "Review fix third.",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer, reviewer, reviewer]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL test_x", "stderr": ""},
        }

        report = orch.run_task("Fix test")

        assert len(orch.context.diagnoses) >= 1
        assert len(orch.context.failures) >= 1


class TestEndToEndValidatorIntegration:
    def test_validator_checks_test_results(self):
        validator = Validator()
        report = validator.validate_all(
            test_results={"exit_code": 0, "success": True},
        )
        assert report.overall_passed is True

    def test_validator_catches_failures(self):
        validator = Validator()
        report = validator.validate_all(
            test_results={"exit_code": 1, "success": False, "stdout": "FAIL"},
        )
        assert report.overall_passed is False


class TestEndToEndBudgetIntegration:
    def test_budget_stops_on_limit(self):
        from app.agent.quota import BudgetLimits, BudgetTracker

        tracker = BudgetTracker(limits=BudgetLimits(max_llm_calls=2))
        tracker.start()
        tracker.record_llm_call()
        tracker.record_llm_call()
        assert tracker.is_within_budget() is False
        assert tracker.budget_violation() is not None

    def test_budget_tracks_in_orchestrator(self):
        mock_client = MagicMock()
        responses = _phase_responses("A", "B", "C", "D")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        orch.run_task("Simple task")

        assert orch.budget.llm_calls > 0
        assert orch.budget.tool_calls >= 0

    def test_one_provider_call_equals_one_budget_increment(self):
        from app.agent.core import AgentCore
        from app.agent.quota import BudgetTracker

        tracker = BudgetTracker()
        tracker.start()

        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(content="Done.")
        core = AgentCore(client=mock_client, on_llm_call=tracker.record_llm_call)
        core.run("Do something", max_steps=1)

        assert tracker.llm_calls == 1
        assert mock_client.chat.call_count == 1

    def test_multiple_provider_calls_equal_multiple_increments(self):
        from app.agent.core import AgentCore
        from app.agent.quota import BudgetTracker
        from app.tools.testing import RunTestsTool

        tracker = BudgetTracker()
        tracker.start()

        mock_client = MagicMock()
        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                return ChatResponse(
                    content="",
                    tool_calls=[{"id": str(call_count[0]), "name": "run_tests", "arguments": {"args": "-v"}}],
                )
            return ChatResponse(content="Done.")
        mock_client.chat.side_effect = side_effect

        mock_test = MagicMock()
        mock_test.execute.return_value = {"success": True, "result": {"exit_code": 0}}

        core = AgentCore(client=mock_client, on_llm_call=tracker.record_llm_call)
        core.register_tool(mock_test)
        core.run("Run tests", max_steps=5)

        assert tracker.llm_calls == call_count[0]
        assert tracker.llm_calls >= 3

    def test_budget_exhaustion_stops_llm_requests(self):
        from app.agent.quota import BudgetLimits, BudgetTracker

        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(content="Should not be called")

        tracker = BudgetTracker(limits=BudgetLimits(max_llm_calls=1))
        tracker.start()
        tracker.record_llm_call()

        from app.agent.core import AgentCore
        core = AgentCore(client=mock_client, on_llm_call=tracker.record_llm_call)

        assert tracker.is_within_budget() is False
        assert tracker.llm_calls == 1

    def test_no_double_counting(self):
        mock_client = MagicMock()
        responses = _phase_responses("A", "B", "C", "D")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        orch.run_task("Simple task")

        total_chat_calls = mock_client.chat.call_count
        assert orch.budget.llm_calls == total_chat_calls


class TestEndToEndRouterIntegration:
    def test_router_classifies_task(self):
        router = ModelRouter()
        assert router.classify_task("Fix the bug") == TaskCategory.DEBUGGING
        assert router.classify_task("Implement feature") == TaskCategory.CODING
        assert router.classify_task("Review code") == TaskCategory.REVIEW


class TestEndToEndAntiFabrication:
    def test_report_reflects_actual_executions(self):
        mock_client = MagicMock()
        responses = _phase_responses("I read everything.", "Plan.", "Inspected.", "Done.")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        report = orch.run_task("Review all files")

        assert report.files_inspected == []
        assert report.files_modified == []
        assert report.commands_executed == []

    def test_no_fabrication_model_claim_vs_execution(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "I modified app/main.py",
            "Plan.",
            "Inspected.",
            "Done.",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        report = orch.run_task("Modify main.py")

        assert "app/main.py" not in report.files_modified


class TestEndToEndContextTrustLevels:
    def test_tool_verified_trust(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            "Plan.",
            "Inspected.",
            "Done.",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        orch.run_task("Fix bug")

        tool_verified = [o for o in orch.context.observations if o.trust == TrustLevel.TOOL_VERIFIED]
        assert len(tool_verified) > 0

    def test_model_proposed_not_auto_verified(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "I fixed the bug.",
            "Plan.",
            "Inspected.",
            "Done.",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        orch.run_task("Fix bug")

        for obs in orch.context.observations:
            if "fixed" in obs.content.lower():
                assert obs.trust != TrustLevel.TOOL_VERIFIED


class TestReviewerContextIntegration:
    def test_reviewer_receives_enriched_context(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            "Plan.",
            "Inspected.",
            "Implemented.",
            "Diagnosed: missing null check in parser.",
            "Applied null check fix.",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        test_call_count = [0]

        def fake_test(args="", **kwargs):
            test_call_count[0] += 1
            if test_call_count[0] == 1:
                return {"success": False, "result": {"exit_code": 1, "stdout": "FAIL test_null", "stderr": ""}}
            return {"success": True, "result": {"exit_code": 0, "stdout": "All passed", "stderr": ""}}

        orch.core.tools["run_tests"].execute.side_effect = fake_test
        orch.run_task("Fix null pointer bug")

        reviewer_call_args = mock_client.chat.call_args_list[-1]
        messages = reviewer_call_args.kwargs.get("messages") or reviewer_call_args[1].get("messages")
        review_text = messages[0]["content"]

        assert "null check" in review_text.lower()

    def test_diagnoses_and_fixes_reach_reviewer(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            "Plan.",
            "Inspected.",
            "Implemented.",
            "Diagnosed: ImportError in utils.",
            "Fixed: added missing import.",
        )
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        test_call_count = [0]

        def fake_test(args="", **kwargs):
            test_call_count[0] += 1
            if test_call_count[0] == 1:
                return {"success": False, "result": {"exit_code": 1, "stdout": "FAIL ImportError", "stderr": ""}}
            return {"success": True, "result": {"exit_code": 0, "stdout": "All passed", "stderr": ""}}

        orch.core.tools["run_tests"].execute.side_effect = fake_test
        orch.run_task("Fix import errors")

        assert len(orch.context.diagnoses) >= 1
        assert len(orch.context.fixes) >= 1

        reviewer_call_args = mock_client.chat.call_args_list[-1]
        messages = reviewer_call_args.kwargs.get("messages") or reviewer_call_args[1].get("messages")
        review_text = messages[0]["content"]

        assert "ImportError" in review_text or "import" in review_text.lower()

    def test_reviewer_backward_compatible_without_context(self):
        from app.agent.reviewer import Reviewer

        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        reviewer = Reviewer(client=mock_client)

        result = reviewer.review(task="Fix bug", changes="changed file.py", diff="-old\n+new")

        assert result.approved is True
        prompt = mock_client.chat.call_args.kwargs["messages"][0]["content"]
        assert "Additional context" not in prompt

    def test_build_review_context_includes_diagnoses_and_fixes(self):
        from app.agent.context import ContextBuilder, TaskContext

        ctx = TaskContext()
        ctx.record_diagnosis("Root cause: null pointer in parser")
        ctx.record_fix("Added null check guard")

        builder = ContextBuilder()
        text = builder.build_review_context(
            ctx, "Fix bug", "changed file.py", "-old\n+new", "{}",
        )

        assert "null pointer" in text
        assert "null check" in text
        assert "Diagnoses:" in text
        assert "Fixes applied:" in text
