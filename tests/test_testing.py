from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.tools.testing import RunTestsTool


@patch("app.tools.testing.resolve_executable")
@patch("app.tools.testing.subprocess.run")
def test_run_tests_tool_runs_pytest(mock_run: MagicMock, mock_resolve: MagicMock) -> None:
    mock_resolve.return_value = None
    mock_run.return_value = MagicMock(
        returncode=0, stdout="3 passed", stderr=""
    )

    tool = RunTestsTool()
    result = tool.execute()

    assert result["success"] is True
    assert result["result"]["command"] == "pytest"
    assert result["result"]["exit_code"] == 0
    mock_run.assert_called_once()


@patch("app.tools.testing.resolve_executable")
@patch("app.tools.testing.subprocess.run")
def test_run_tests_tool_passes_extra_args(mock_run: MagicMock, mock_resolve: MagicMock) -> None:
    mock_resolve.return_value = None
    mock_run.return_value = MagicMock(
        returncode=0, stdout="1 passed", stderr=""
    )

    tool = RunTestsTool()
    result = tool.execute(args="-v")

    assert result["success"] is True
    assert result["result"]["command"] == "pytest -v"
    mock_run.assert_called_once()


@patch("app.tools.testing.resolve_executable")
@patch("app.tools.testing.subprocess.run")
def test_run_tests_tool_handles_timeout(mock_run: MagicMock, mock_resolve: MagicMock) -> None:
    mock_resolve.return_value = None
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=10)

    tool = RunTestsTool()
    result = tool.execute()

    assert result["success"] is False
    assert "timed out" in result["error"]


# ---------------------------------------------------------------------------
# Phase 17: Testing environment resolution
# ---------------------------------------------------------------------------

class TestTestingEnvironmentResolution:
    @patch("app.tools.testing.resolve_executable")
    @patch("app.tools.testing.subprocess.run")
    def test_pytest_resolves_to_project_venv(
        self, mock_run: MagicMock, mock_resolve: MagicMock, fake_project: Path
    ) -> None:
        mock_resolve.return_value = str(fake_project / ".venv" / "bin" / "pytest")
        mock_run.return_value = MagicMock(returncode=0, stdout="passed", stderr="")

        tool = RunTestsTool()
        result = tool.execute()

        assert result["success"] is True
        call_args = mock_run.call_args[0][0]
        assert "pytest" in call_args[0]
        assert ".venv" in call_args[0]

    @patch("app.tools.testing.resolve_executable")
    @patch("app.tools.testing.subprocess.run")
    def test_pytest_args_preserved_with_resolved_executable(
        self, mock_run: MagicMock, mock_resolve: MagicMock, fake_project: Path
    ) -> None:
        mock_resolve.return_value = str(fake_project / ".venv" / "bin" / "pytest")
        mock_run.return_value = MagicMock(returncode=0, stdout="1 passed", stderr="")

        tool = RunTestsTool()
        result = tool.execute(args="tests/test_foo.py -v")

        assert result["success"] is True
        call_args = mock_run.call_args[0][0]
        assert ".venv" in call_args[0]
        assert "tests/test_foo.py" in call_args
        assert "-v" in call_args

    @patch("app.tools.testing.resolve_executable")
    @patch("app.tools.testing.subprocess.run")
    def test_cwd_is_project_root(
        self, mock_run: MagicMock, mock_resolve: MagicMock
    ) -> None:
        mock_resolve.return_value = None
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        tool = RunTestsTool()
        tool.execute()

        _, kwargs = mock_run.call_args
        assert "cwd" in kwargs
        assert kwargs["cwd"] is not None

    @patch("app.tools.testing.resolve_executable")
    @patch("app.tools.testing.subprocess.run")
    def test_shell_false_enforced(
        self, mock_run: MagicMock, mock_resolve: MagicMock
    ) -> None:
        mock_resolve.return_value = None
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        tool = RunTestsTool()
        tool.execute()

        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is not True

    def test_malicious_args_blocked(self) -> None:
        tool = RunTestsTool()
        result = tool.execute(args="-c 'import os'")
        assert result["success"] is False

    def test_shell_metacharacters_blocked(self) -> None:
        tool = RunTestsTool()
        result = tool.execute(args="tests/; rm -rf /")
        assert result["success"] is False
