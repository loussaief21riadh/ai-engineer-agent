from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from app.agent.orchestrator import Orchestrator, _build_report_from_executions
from app.agent.phases import TaskPhase, can_transition, next_phases
from app.config import AgentMode
from app.llm.openrouter import ChatResponse, OpenRouterError
from app.models.schemas import TaskReport, ToolExecution


class TestTaskPhaseTransitions:
    def test_valid_transitions(self):
        assert can_transition(TaskPhase.UNDERSTAND, TaskPhase.PLAN)
        assert can_transition(TaskPhase.PLAN, TaskPhase.INSPECT)
        assert can_transition(TaskPhase.INSPECT, TaskPhase.IMPLEMENT)
        assert can_transition(TaskPhase.IMPLEMENT, TaskPhase.TEST)
        assert can_transition(TaskPhase.TEST, TaskPhase.DIAGNOSE)
        assert can_transition(TaskPhase.TEST, TaskPhase.SECURITY_CHECK)
        assert not can_transition(TaskPhase.TEST, TaskPhase.DONE)
        assert can_transition(TaskPhase.SECURITY_CHECK, TaskPhase.REVIEW)
        assert can_transition(TaskPhase.DIAGNOSE, TaskPhase.FIX)
        assert can_transition(TaskPhase.FIX, TaskPhase.RETEST)
        assert can_transition(TaskPhase.RETEST, TaskPhase.TEST)
        assert can_transition(TaskPhase.RETEST, TaskPhase.DIAGNOSE)
        assert can_transition(TaskPhase.RETEST, TaskPhase.SECURITY_CHECK)
        assert can_transition(TaskPhase.RETEST, TaskPhase.REVIEW)
        assert can_transition(TaskPhase.REVIEW, TaskPhase.VALIDATE)
        assert can_transition(TaskPhase.VALIDATE, TaskPhase.REPORT)
        assert can_transition(TaskPhase.REPORT, TaskPhase.DONE)

    def test_invalid_transitions(self):
        assert not can_transition(TaskPhase.UNDERSTAND, TaskPhase.TEST)
        assert not can_transition(TaskPhase.PLAN, TaskPhase.FIX)
        assert not can_transition(TaskPhase.DONE, TaskPhase.UNDERSTAND)
        assert not can_transition(TaskPhase.FAILED, TaskPhase.UNDERSTAND)

    def test_next_phases_returns_successors(self):
        successors = next_phases(TaskPhase.PLAN)
        assert TaskPhase.INSPECT in successors

    def test_next_phases_empty_for_done(self):
        successors = next_phases(TaskPhase.DONE)
        assert successors == []

    def test_all_phases_defined(self):
        phases = [p.value for p in TaskPhase]
        assert "UNDERSTAND" in phases
        assert "PLAN" in phases
        assert "INSPECT" in phases
        assert "IMPLEMENT" in phases
        assert "TEST" in phases
        assert "DIAGNOSE" in phases
        assert "FIX" in phases
        assert "RETEST" in phases
        assert "REVIEW" in phases
        assert "VALIDATE" in phases
        assert "REPORT" in phases
        assert "DONE" in phases
        assert "FAILED" in phases


def _make_orch(mock_client, mode=AgentMode.READ_ONLY):
    orch = Orchestrator(client=mock_client, mode=mode)
    mock_test = MagicMock()
    mock_test.execute.return_value = {
        "success": True,
        "result": {"exit_code": 0, "stdout": "All tests passed", "stderr": ""},
    }
    orch.core.tools["run_tests"] = mock_test
    return orch


_CHANGE_REQUIRED_RESPONSE = json.dumps({
    "decision": "CHANGE_REQUIRED",
    "confidence": 0.95,
    "reason": "Inspection identified work required for the requested task.",
    "evidence": ["inspection completed"],
})


def _phase_responses(*contents):
    result = []
    for i, c in enumerate(contents):
        if i == 2:
            result.append(ChatResponse(content=_CHANGE_REQUIRED_RESPONSE))
        else:
            result.append(ChatResponse(content=c))
    return result


class TestHappyPath:
    def test_full_cycle_all_phases_traversed(self):
        mock_client = MagicMock()
        responses = _phase_responses("Understood.", "Plan ready.", "Inspected.", "Implemented.")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "LGTM",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Fix the bug")
        assert report.stop_reason == "completed"
        assert "UNDERSTAND" in report.phase_history
        assert "PLAN" in report.phase_history
        assert "INSPECT" in report.phase_history
        assert "IMPLEMENT" in report.phase_history
        assert "TEST" in report.phase_history
        assert "REVIEW" in report.phase_history
        assert "VALIDATE" in report.phase_history
        assert "REPORT" in report.phase_history
        assert "DONE" in report.phase_history
        assert report.review is not None
        assert report.review.approved is True

    def test_report_has_correct_structure(self):
        mock_client = MagicMock()
        responses = _phase_responses("A", "B", "C", "D")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client)
        report = orch.run_task("Simple task")

        assert isinstance(report, TaskReport)
        assert report.task == "Simple task"
        assert report.mode == "READ_ONLY"
        assert report.final_response is not None
        assert report.iteration_count >= 1
        assert report.retry_count == 0


class TestRecovery:
    def test_first_fail_then_pass(self):
        mock_client = MagicMock()

        understand = ChatResponse(content="Understood.")
        plan = ChatResponse(content="Plan.")
        inspect = ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
        implement = ChatResponse(content="Implemented.")
        diagnose = ChatResponse(content="Found the root cause.")
        fix = ChatResponse(content="Applied fix.")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "Fixed.",
        }))
        mock_client.chat.side_effect = [
            understand, plan, inspect, implement,
            diagnose, fix,
            reviewer,
        ]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        test_call_count = [0]

        def fake_test(args="", **kwargs):
            test_call_count[0] += 1
            if test_call_count[0] == 1:
                return {"success": False, "result": {"exit_code": 1, "stdout": "FAIL test_one", "stderr": ""}}
            return {"success": True, "result": {"exit_code": 0, "stdout": "All passed", "stderr": ""}}

        orch.core.tools["run_tests"].execute.side_effect = fake_test

        report = orch.run_task("Fix failing test")

        assert report.final_phase == "DONE"
        assert report.retry_count == 0
        assert len(report.diagnoses) == 1
        assert len(report.fixes) == 1
        assert "RETEST" in report.phase_history
        assert "DIAGNOSE" in report.phase_history

    def test_direct_test_calls_recorded_in_executions(self):
        mock_client = MagicMock()

        understand = ChatResponse(content="Understood.")
        plan = ChatResponse(content="Plan.")
        inspect = ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
        implement = ChatResponse(content="Implemented.")
        diagnose = ChatResponse(content="Found the root cause.")
        fix = ChatResponse(content="Applied fix.")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "Fixed.",
        }))
        mock_client.chat.side_effect = [
            understand, plan, inspect, implement,
            diagnose, fix,
            reviewer,
        ]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        test_call_count = [0]

        def fake_test(args="", **kwargs):
            test_call_count[0] += 1
            if test_call_count[0] == 1:
                return {"success": False, "result": {"exit_code": 1, "stdout": "FAIL test_one", "stderr": ""}}
            return {"success": True, "result": {"exit_code": 0, "stdout": "All passed", "stderr": ""}}

        orch.core.tools["run_tests"].execute.side_effect = fake_test

        report = orch.run_task("Fix failing test")

        test_execs = [ex for ex in report.executions if ex.tool_name == "run_tests"]
        assert len(test_execs) == 2
        assert test_execs[0].success is False
        assert test_execs[0].result["stdout"] == "FAIL test_one"
        assert test_execs[1].success is True

    def test_diagnose_receives_failure_output(self):
        mock_client = MagicMock()

        understand = ChatResponse(content="Understood.")
        plan = ChatResponse(content="Plan.")
        inspect = ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
        implement = ChatResponse(content="Implemented.")
        diagnose = ChatResponse(content="Found the root cause.")
        fix = ChatResponse(content="Applied fix.")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "Fixed.",
        }))

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 4:
                return [understand, plan, inspect, implement][call_count[0] - 1]
            elif call_count[0] % 2 == 1:
                return diagnose
            else:
                return fix

        mock_client.chat.side_effect = side_effect

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL test_widget::test_calc", "stderr": ""},
        }

        report = orch.run_task("Fix test")

        diagnose_calls = [
            c for c in mock_client.chat.call_args_list
            if len(c[1].get("messages", [])) > 1
            and "Phase: DIAGNOSE" in c[1]["messages"][1]["content"]
        ]
        assert len(diagnose_calls) >= 1
        prompt = diagnose_calls[0][1]["messages"][1]["content"]
        assert "FAIL test_widget::test_calc" in prompt


class TestRepeatedFailure:
    def test_stops_after_max_retries(self):
        mock_client = MagicMock()

        responses = _phase_responses("Understood.", "Plan.", "Inspected.", "Implemented.",
                                      "Diagnosed.", "Fixed.",
                                      "Diagnosed again.", "Fixed again.",
                                      "Diagnosed third.", "Fixed third.",
                                      "Still broken.", "Fix attempt.",
                                      "Fix attempt again.", "Fix attempt third.")
        reviewer_reject = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT", "findings": [], "summary": "Still broken.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_reject, reviewer_reject, reviewer_reject]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""},
        }

        report = orch.run_task("Broken task")

        assert report.final_phase == "FAILED"
        assert report.stop_reason == "failed"


class TestRegression:
    def test_test_passes_reviewer_rejects(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", "Plan.", "Inspected.", "Implemented.",
            "Applied fix based on reviewer feedback.",
        )
        reviewer_first = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT",
            "findings": [{"severity": "warning", "category": "quality", "description": "Needs cleanup"}],
            "summary": "Needs cleanup.",
        }))
        reviewer_second = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE",
            "findings": [],
            "summary": "Approved after cleanup.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_first, reviewer_second]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Quick fix")

        assert report.final_phase == "DONE"
        assert report.review is not None
        assert report.review.approved is True
        assert "FIX" in report.phase_history
        assert "RETEST" in report.phase_history
        assert "REVIEW" in report.phase_history


class TestReadOnly:
    def test_cannot_write_in_read_only(self):
        mock_client = MagicMock()
        responses = _phase_responses("Understood.", "Plan.", "Inspected.", "Cannot write in read-only mode.")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = responses + [reviewer]

        orch = _make_orch(mock_client, mode=AgentMode.READ_ONLY)
        report = orch.run_task("Write a new file")

        assert "write_file" not in orch.core.tools
        assert "INSPECT" in report.phase_history
        assert "REPORT" in report.phase_history
        assert "IMPLEMENT" not in report.phase_history


class TestToolRejection:
    def test_blocked_command_handled(self):
        mock_client = MagicMock()
        understand = ChatResponse(content="Understood.")
        plan = ChatResponse(content="Plan.")
        inspect = ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
        implement = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "run_command", "arguments": {"command": "rm -rf /"}}],
        )
        implement_result = ChatResponse(content="Command blocked.")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = [understand, plan, inspect, implement, implement_result, reviewer]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Delete everything")

        assert report.final_phase == "DONE"


class TestToolBudget:
    def test_max_steps_exceeded(self):
        mock_client = MagicMock()
        tool_resp = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "run_command", "arguments": {"command": "ls"}}],
        )
        mock_client.chat.return_value = tool_resp

        orch = _make_orch(mock_client)
        report = orch.run_task("List files")

        assert report.final_phase in ("DONE", "FAILED", "REPORT")
        assert report.stop_reason in ("completed", "failed", "iteration_limit_reached", "budget_exceeded")


class TestIterationLimit:
    def test_stops_at_max_iterations(self):
        mock_client = MagicMock()

        diagnose_response = ChatResponse(content="Still broken.")
        fix_response = ChatResponse(content="Trying fix.")

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return diagnose_response
            return fix_response

        mock_client.chat.side_effect = side_effect

        orch = _make_orch(mock_client)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""},
        }

        report = orch.run_task("Impossible task")

        assert report.iteration_count > 0
        assert report.stop_reason in ("failed", "iteration_limit_reached")


class TestRealityVsClaim:
    def test_report_reflects_actual_executions(self):
        mock_client = MagicMock()
        understand = ChatResponse(content="Understood.")
        plan = ChatResponse(content="Plan.")
        inspect = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "read_file", "arguments": {"path": "app/main.py"}}],
        )
        inspect_result = ChatResponse(content="Read main.py.")
        implement = ChatResponse(content="Done.")
        reviewer = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = [understand, plan, inspect, inspect_result, implement, reviewer]

        orch = _make_orch(mock_client)
        report = orch.run_task("Review main.py")

        assert "app/main.py" in report.files_inspected
        assert report.files_modified == []

    def test_no_fabrication(self):
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


class TestReviewerRejection:
    def test_reviewer_rejection_triggers_recovery(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", "Plan.", "Inspected.", "Implemented.",
            "Applied fix.",
        )
        reviewer_first = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT",
            "findings": [{"severity": "critical", "category": "security", "description": "Exposed key"}],
            "summary": "Rejected.",
        }))
        reviewer_second = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE",
            "findings": [],
            "summary": "Fixed.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_first, reviewer_second]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Security audit")

        assert report.review is not None
        assert report.review.approved is True
        assert "FIX" in report.phase_history
        assert "RETEST" in report.phase_history

    def test_reviewer_rejection_stops_after_max_retries(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", "Plan.", "Inspected.", "Implemented.",
            "Fix attempt 1.", "Fix attempt 2.", "Fix attempt 3.", "Fix attempt 4.",
        )
        reviewer_reject = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT",
            "findings": [{"severity": "critical", "category": "security", "description": "Exposed key"}],
            "summary": "Still rejected.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_reject] * 4

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Security audit")

        assert report.final_phase == "FAILED"
        assert report.review is not None
        assert report.review.approved is False
        assert orch._review_retry_count >= 3

    def test_rejection_causes_fix_phase(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", "Plan.", "Inspected.", "Implemented.",
            "Fixed security issue.",
        )
        reviewer_reject = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT",
            "findings": [{"severity": "critical", "category": "security", "description": "Key exposed"}],
            "summary": "Security issue.",
        }))
        reviewer_approve = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "Fixed.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_reject, reviewer_approve]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Fix security")

        fix_idx = report.phase_history.index("FIX")
        review_idx = report.phase_history.index("REVIEW")
        assert fix_idx > review_idx

    def test_rejection_causes_retest_phase(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", "Plan.", "Inspected.", "Implemented.",
            "Fixed issue.",
        )
        reviewer_reject = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT",
            "findings": [], "summary": "Rejected.",
        }))
        reviewer_approve = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_reject, reviewer_approve]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Fix thing")

        assert "RETEST" in report.phase_history
        retest_idx = report.phase_history.index("RETEST")
        fix_idx = report.phase_history.index("FIX")
        assert retest_idx > fix_idx

    def test_successful_retest_returns_to_review(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", "Plan.", "Inspected.", "Implemented.",
            "Fixed.",
        )
        reviewer_reject = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT", "findings": [], "summary": "Rejected.",
        }))
        reviewer_approve = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "Approved.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_reject, reviewer_approve]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Fix")

        review_indices = [i for i, p in enumerate(report.phase_history) if p == "REVIEW"]
        assert len(review_indices) == 2
        assert review_indices[1] > review_indices[0]

    def test_failed_retest_goes_to_diagnose(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", "Plan.", "Inspected.", "Implemented.",
            "Diagnosed.", "Fixed.",
            "Diagnosed again.", "Fixed again.",
            "Diagnosed third.", "Fixed third.",
            "Fix attempt.", "Fix attempt 2.", "Fix attempt 3.",
            "Fix attempt 4.", "Fix attempt 5.", "Fix attempt 6.",
        )
        reviewer_reject = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT", "findings": [], "summary": "Rejected.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_reject] * 4

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False, "result": {"exit_code": 1, "stdout": "FAIL test_x", "stderr": ""},
        }

        report = orch.run_task("Fix failing test")

        assert "DIAGNOSE" in report.phase_history
        assert "FIX" in report.phase_history
        assert "RETEST" in report.phase_history
        retest_indices = [i for i, p in enumerate(report.phase_history) if p == "RETEST"]
        for idx in retest_indices:
            assert report.phase_history[idx + 1] in ("DIAGNOSE", "REVIEW", "FAILED")

    def test_reviewer_feedback_passed_to_fix(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", "Plan.", "Inspected.", "Implemented.",
            "Fixed based on feedback.",
        )
        reviewer_reject = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT",
            "findings": [{"severity": "warning", "category": "quality", "description": "Missing docstring"}],
            "summary": "Needs docs.",
        }))
        reviewer_approve = ChatResponse(content=json.dumps({
            "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_reject, reviewer_approve]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Add docs")

        fix_calls = [
            c for c in mock_client.chat.call_args_list
            if any("Phase: FIX (reviewer rejection)" in msg["content"] for msg in c[1].get("messages", []) if isinstance(msg, dict))
        ]
        assert len(fix_calls) >= 1
        fix_prompt = fix_calls[0][1]["messages"][1]["content"]
        assert "reviewer rejected" in fix_prompt.lower() or "Needs docs" in fix_prompt

    def test_read_only_rejection_stops_immediately(self):
        mock_client = MagicMock()
        responses = _phase_responses("Understood.", "Plan.", "Inspected.", "Cannot write in read-only mode.")
        reviewer_reject = ChatResponse(content=json.dumps({
            "approved": False, "verdict": "REJECT", "findings": [], "summary": "Rejected.",
        }))
        mock_client.chat.side_effect = responses + [reviewer_reject]

        orch = _make_orch(mock_client, mode=AgentMode.READ_ONLY)
        report = orch.run_task("Write file")

        assert "INSPECT" in report.phase_history
        assert "REPORT" in report.phase_history
        assert "IMPLEMENT" not in report.phase_history
        assert report.review is None


class TestBuildReportFromExecutions:
    def test_populates_files_inspected(self):
        execs = [
            ToolExecution(step=1, tool_name="read_file", arguments={"path": "a.py"}, success=True, result="content"),
            ToolExecution(step=2, tool_name="read_file", arguments={"path": "b.py"}, success=True, result="content"),
        ]
        report = _build_report_from_executions(
            task="read files", mode="READ_ONLY", final_response="Done.",
            steps_taken=2, executions=execs,
        )
        assert "a.py" in report.files_inspected
        assert "b.py" in report.files_inspected

    def test_populates_files_modified(self):
        execs = [
            ToolExecution(step=1, tool_name="write_file", arguments={"path": "out.py"}, success=True, result="ok"),
        ]
        report = _build_report_from_executions(
            task="write file", mode="ALLOW_EDITS", final_response="Done.",
            steps_taken=1, executions=execs,
        )
        assert "out.py" in report.files_modified

    def test_populates_commands(self):
        execs = [
            ToolExecution(step=1, tool_name="run_command", arguments={"command": "ls"}, success=True, result="ok"),
        ]
        report = _build_report_from_executions(
            task="list", mode="READ_ONLY", final_response="Done.",
            steps_taken=1, executions=execs,
        )
        assert "ls" in report.commands_executed

    def test_no_duplicates(self):
        execs = [
            ToolExecution(step=1, tool_name="read_file", arguments={"path": "a.py"}, success=True, result="c"),
            ToolExecution(step=2, tool_name="read_file", arguments={"path": "a.py"}, success=True, result="c"),
        ]
        report = _build_report_from_executions(
            task="read twice", mode="READ_ONLY", final_response="Done.",
            steps_taken=2, executions=execs,
        )
        assert report.files_inspected.count("a.py") == 1

    def test_v15_fields_populated(self):
        report = _build_report_from_executions(
            task="test", mode="READ_ONLY", final_response="Done.",
            steps_taken=1, executions=[],
            final_phase="DONE",
            phase_history=["UNDERSTAND", "DONE"],
            iteration_count=1,
            retry_count=0,
            diagnoses=["diagnosis"],
            fixes=["fix"],
            stop_reason="completed",
        )
        assert report.final_phase == "DONE"
        assert report.phase_history == ["UNDERSTAND", "DONE"]
        assert report.iteration_count == 1
        assert report.retry_count == 0
        assert report.diagnoses == ["diagnosis"]
        assert report.fixes == ["fix"]
        assert report.stop_reason == "completed"
