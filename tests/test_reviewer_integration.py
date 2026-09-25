from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from app.agent.orchestrator import Orchestrator
from app.config import AgentMode
from app.llm.openrouter import ChatResponse, OpenRouterError
from app.models.schemas import ReviewResult, ReviewFinding, ReviewSeverity


@pytest.fixture
def mock_client():
    return MagicMock()


def _make_approved_review() -> ReviewResult:
    return ReviewResult(
        approved=True,
        findings=[],
        summary="All good.",
    )


def _make_rejected_review() -> ReviewResult:
    return ReviewResult(
        approved=False,
        findings=[
            ReviewFinding(
                severity=ReviewSeverity.WARNING,
                category="testing",
                description="Missing test coverage",
            )
        ],
        summary="Needs tests.",
    )


_CHANGE_REQUIRED_RESPONSE = json.dumps({
    "decision": "CHANGE_REQUIRED",
    "confidence": 0.95,
    "reason": "Inspection identified work required for the requested task.",
    "evidence": ["inspection completed"],
})


def _phase_responses(*contents: str) -> list[ChatResponse]:
    result = []
    for i, c in enumerate(contents):
        if i == 2:
            result.append(ChatResponse(content=_CHANGE_REQUIRED_RESPONSE))
        else:
            result.append(ChatResponse(content=c))
    return result


def _make_orch(mock_client, mode=AgentMode.READ_ONLY):
    orch = Orchestrator(client=mock_client, mode=mode)
    mock_test = MagicMock()
    mock_test.execute.return_value = {
        "success": True,
        "result": {"exit_code": 0, "stdout": "All tests passed", "stderr": ""},
    }
    orch.core.tools["run_tests"] = mock_test
    return orch


class TestReviewerIntegration:
    def test_reviewer_called_during_run_task(self, mock_client):
        agent_responses = _phase_responses("Understood.", "Plan ready.", "Inspected.", "Implemented.")
        reviewer_response = ChatResponse(content=json.dumps({
            "approved": True, "findings": [], "summary": "LGTM",
        }))
        mock_client.chat.side_effect = agent_responses + [reviewer_response]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Do something")

        assert report.review is not None
        assert report.review.approved is True

    def test_actual_executions_passed_to_reviewer(self, mock_client):
        understand = ChatResponse(content="Understood.")
        plan = ChatResponse(content="Plan ready.")
        inspect = ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
        implement = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "run_command", "arguments": {"command": "echo hello"}}],
        )
        implement_result = ChatResponse(content="Ran command.")
        reviewer_response = ChatResponse(content=json.dumps({
            "approved": True, "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = [understand, plan, inspect, implement, implement_result, reviewer_response]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Run echo")

        review_call = mock_client.chat.call_args_list[-1]
        prompt_content = review_call[1]["messages"][0]["content"]
        assert "echo hello" in prompt_content

    def test_review_result_in_task_report(self, mock_client):
        agent_responses = _phase_responses("Understood.", "Plan ready.", "Inspected.", "Implemented.", "Fixed.")
        reviewer_reject = ChatResponse(content=json.dumps({
            "approved": False,
            "findings": [{"severity": "critical", "category": "security", "description": "Exposed key"}],
            "summary": "Critical issues found.",
        }))
        reviewer_approve = ChatResponse(content=json.dumps({
            "approved": True,
            "findings": [],
            "summary": "Fixed.",
        }))
        mock_client.chat.side_effect = agent_responses + [reviewer_reject, reviewer_approve]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Check security")

        assert report.review is not None
        assert report.review.approved is True
        assert "FIX" in report.phase_history
        assert "RETEST" in report.phase_history

    def test_reviewer_failure_safe(self, mock_client):
        understand = ChatResponse(content="Understood.")
        plan = ChatResponse(content="Plan ready.")
        inspect = ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
        implement = ChatResponse(content="Done.")
        reviewer_fail = OpenRouterError("Reviewer down")
        fix = ChatResponse(content="Fixed.")
        reviewer_approve = ChatResponse(content=json.dumps({
            "approved": True, "findings": [], "summary": "OK after fix.",
        }))
        mock_client.chat.side_effect = [
            understand, plan, inspect, implement,
            reviewer_fail, fix, reviewer_approve,
        ]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Do task")

        assert report.review is not None
        assert report.review.approved is True
        assert report.final_response is not None
        assert report.steps_taken > 0

    def test_no_fabrication_reviewer_gets_real_data(self, mock_client):
        understand = ChatResponse(content="Understood.")
        plan = ChatResponse(content="Plan ready.")
        inspect = ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
        implement = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "write_file", "arguments": {"path": "output.py", "content": "x=1"}}],
        )
        implement_result = ChatResponse(content="Wrote file.")
        reviewer_response = ChatResponse(content=json.dumps({
            "approved": True, "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = [understand, plan, inspect, implement, implement_result, reviewer_response]

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        real_write_tool = orch.core.tools["write_file"]
        real_write_tool.execute = MagicMock(
            return_value={"success": True, "result": "Wrote 3 bytes to output.py"}
        )
        report = orch.run_task("Create output.py")

        review_call = mock_client.chat.call_args_list[-1]
        prompt = review_call[1]["messages"][0]["content"]
        assert "output.py" in prompt
        assert "Wrote" in prompt or "wrote" in prompt
        assert report.review is not None

    def test_cli_print_report_shows_review(self, mock_client, capsys):
        from app.main import print_report
        from app.models.schemas import TaskReport

        report = TaskReport(
            task="Check code",
            mode="read_only",
            final_response="Reviewed.",
            steps_taken=3,
            review=ReviewResult(
                approved=True,
                findings=[],
                summary="All clear.",
            ),
        )
        print_report(report)
        captured = capsys.readouterr()
        assert "All clear." in captured.out
