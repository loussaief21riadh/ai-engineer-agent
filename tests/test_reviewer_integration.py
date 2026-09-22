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


@pytest.fixture
def mock_reviewer():
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


class TestReviewerIntegration:
    def test_reviewer_called_during_run_task(self, mock_client):
        agent_response = ChatResponse(content="Done.")
        reviewer_response = ChatResponse(content=json.dumps({
            "approved": True, "findings": [], "summary": "LGTM",
        }))
        mock_client.chat.side_effect = [agent_response, reviewer_response]

        orch = Orchestrator(client=mock_client, mode=AgentMode.READ_ONLY)
        report = orch.run_task("Do something")

        assert mock_client.chat.call_count == 2
        assert report.review is not None
        assert report.review.approved is True

    def test_actual_executions_passed_to_reviewer(self, mock_client):
        tool_call_response = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "run_command", "arguments": {"command": "echo hello"}}],
        )
        final_response = ChatResponse(content="Ran command.")
        reviewer_response = ChatResponse(content=json.dumps({
            "approved": True, "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = [tool_call_response, final_response, reviewer_response]

        orch = Orchestrator(client=mock_client, mode=AgentMode.READ_ONLY)
        report = orch.run_task("Run echo")

        review_call = mock_client.chat.call_args_list[2]
        prompt_content = review_call[1]["messages"][0]["content"]
        assert "echo hello" in prompt_content

    def test_review_result_in_task_report(self, mock_client):
        agent_response = ChatResponse(content="Finished.")
        reviewer_response = ChatResponse(content=json.dumps({
            "approved": False,
            "findings": [{"severity": "critical", "category": "security", "description": "Exposed key"}],
            "summary": "Critical issues found.",
        }))
        mock_client.chat.side_effect = [agent_response, reviewer_response]

        orch = Orchestrator(client=mock_client, mode=AgentMode.READ_ONLY)
        report = orch.run_task("Check security")

        assert report.review is not None
        assert report.review.approved is False
        assert len(report.review.findings) == 1
        assert report.review.findings[0].severity == ReviewSeverity.CRITICAL
        assert report.review.summary == "Critical issues found."

    def test_reviewer_failure_safe(self, mock_client):
        agent_response = ChatResponse(content="Done.")
        mock_client.chat.side_effect = [agent_response, OpenRouterError("Reviewer down")]

        orch = Orchestrator(client=mock_client, mode=AgentMode.READ_ONLY)
        report = orch.run_task("Do task")

        assert report.review is not None
        assert report.review.approved is False
        assert "Reviewer LLM error" in report.review.summary
        assert report.final_response == "Done."
        assert report.steps_taken > 0

    def test_no_fabrication_reviewer_gets_real_data(self, mock_client):
        tool_call_response = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "write_file", "arguments": {"path": "test.py", "content": "x=1"}}],
        )
        final_response = ChatResponse(content="Wrote file.")
        reviewer_response = ChatResponse(content=json.dumps({
            "approved": True, "findings": [], "summary": "OK",
        }))
        mock_client.chat.side_effect = [tool_call_response, final_response, reviewer_response]

        orch = Orchestrator(client=mock_client, mode=AgentMode.ALLOW_EDITS)
        report = orch.run_task("Create test.py")

        review_call = mock_client.chat.call_args_list[2]
        prompt = review_call[1]["messages"][0]["content"]
        assert "test.py" in prompt
        assert "write_file" in prompt.lower() or "Wrote" in prompt or "wrote" in prompt
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
