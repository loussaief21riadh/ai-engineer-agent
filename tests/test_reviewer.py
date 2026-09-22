from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from app.agent.reviewer import Reviewer
from app.llm.openrouter import ChatResponse, OpenRouterError
from app.models.schemas import ReviewResult, ReviewFinding, ReviewSeverity


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def reviewer(mock_client):
    return Reviewer(client=mock_client)


class TestReviewerParseReview:
    def test_parses_valid_json(self, reviewer):
        response = json.dumps({
            "approved": True,
            "findings": [
                {
                    "severity": "info",
                    "category": "style",
                    "description": "Looks good",
                    "file": None,
                    "line": None,
                }
            ],
            "summary": "All clear.",
        })
        result = reviewer._parse_review(response)
        assert result.approved is True
        assert len(result.findings) == 1
        assert result.findings[0].severity == ReviewSeverity.INFO
        assert result.summary == "All clear."

    def test_handles_invalid_json(self, reviewer):
        result = reviewer._parse_review("not json at all")
        assert result.approved is False
        assert "Could not parse review response" in result.summary

    def test_handles_empty_json_object(self, reviewer):
        result = reviewer._parse_review("{}")
        assert result.approved is False
        assert result.findings == []
        assert result.summary == ""

    def test_handles_markdown_wrapped_json(self, reviewer):
        response = '```json\n{"approved": true, "findings": [], "summary": "OK"}\n```'
        result = reviewer._parse_review(response)
        assert result.approved is True
        assert result.summary == "OK"

    def test_handles_invalid_json_with_trailing_text(self, reviewer):
        response = '{"approved": false, "findings": [], "summary": "No"}\nDone.'
        result = reviewer._parse_review(response)
        assert result.approved is False
        assert "Could not parse review response" in result.summary

    def test_returns_error_for_broken_markdown_json(self, reviewer):
        response = '```json\n{broken}\n```'
        result = reviewer._parse_review(response)
        assert result.approved is False
        assert "Could not parse review response" in result.summary

    def test_handles_findings_with_different_severities(self, reviewer):
        response = json.dumps({
            "approved": False,
            "findings": [
                {"severity": "critical", "category": "security", "description": "SQL injection", "file": "db.py", "line": 42},
                {"severity": "warning", "category": "logic", "description": "Off by one", "file": None, "line": None},
            ],
            "summary": "Issues found.",
        })
        result = reviewer._parse_review(response)
        assert result.approved is False
        assert len(result.findings) == 2
        assert result.findings[0].severity == ReviewSeverity.CRITICAL
        assert result.findings[0].file == "db.py"
        assert result.findings[0].line == 42
        assert result.findings[1].severity == ReviewSeverity.WARNING


class TestReviewerReview:
    def test_calls_llm_with_formatted_prompt(self, reviewer, mock_client):
        mock_client.chat.return_value = ChatResponse(content=json.dumps({
            "approved": True,
            "findings": [],
            "summary": "LGTM",
        }))
        result = reviewer.review(
            task="Fix bug",
            changes="Fixed line 10",
            diff="- old\n+ new",
            test_results="All passed",
        )
        assert result.approved is True
        assert result.summary == "LGTM"
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Fix bug" in prompt_content
        assert "Fixed line 10" in prompt_content

    def test_handles_llm_error(self, reviewer, mock_client):
        mock_client.chat.side_effect = OpenRouterError("Timeout")
        result = reviewer.review(task="Do something")
        assert result.approved is False
        assert "Reviewer LLM error" in result.summary

    def test_handles_missing_test_results(self, reviewer, mock_client):
        mock_client.chat.return_value = ChatResponse(content=json.dumps({
            "approved": True,
            "findings": [],
            "summary": "OK",
        }))
        reviewer.review(task="Test", test_results="")
        call_args = mock_client.chat.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Test results:" in prompt_content
        assert "Test" in prompt_content


class TestReviewResultModel:
    def test_review_result_defaults(self):
        result = ReviewResult(approved=False)
        assert result.approved is False
        assert result.findings == []
        assert result.summary == ""

    def test_review_result_with_findings(self):
        finding = ReviewFinding(
            severity=ReviewSeverity.WARNING,
            category="testing",
            description="Missing test",
        )
        result = ReviewResult(
            approved=False,
            findings=[finding],
            summary="Needs tests.",
        )
        assert len(result.findings) == 1
        assert result.findings[0].category == "testing"
