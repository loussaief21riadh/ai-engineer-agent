from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.tools.terminal import RunCommandTool, _validate_command


@patch("app.tools.terminal.subprocess.run")
def test_run_command_executes_simple_command(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="hello", stderr="")

    tool = RunCommandTool()
    result = tool.execute(command="echo hello")

    assert result["success"] is True
    assert result["result"]["stdout"] == "hello"
    assert result["result"]["exit_code"] == 0
    mock_run.assert_called_once()


def test_blocked_commands_are_rejected() -> None:
    tool = RunCommandTool()

    blocked = [
        "rm -rf /tmp/foo",
        "sudo rm -rf /",
        "sudo apt install foo",
        "dd if=/dev/zero of=/dev/sda",
        "shutdown -h now",
        "reboot",
        "mkfs.ext4 /dev/sda1",
        "diskutil eraseDisk",
    ]

    for cmd in blocked:
        result = tool.execute(command=cmd)
        assert result["success"] is False, f"Expected blocked for: {cmd}"


def test_empty_command_returns_error() -> None:
    tool = RunCommandTool()
    result = tool.execute(command="")
    assert result["success"] is False
    assert result["error"] == "command is required"

    result = tool.execute()
    assert result["success"] is False
    assert result["error"] == "command is required"


@patch("app.tools.terminal.subprocess.run")
def test_command_timeout_handling(mock_run: MagicMock) -> None:
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="date", timeout=1)

    tool = RunCommandTool()
    result = tool.execute(command="date", timeout=1)

    assert result["success"] is False
    assert "timed out" in result["error"]


@patch("app.tools.terminal.subprocess.run")
def test_stdout_and_stderr_capture(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=1, stdout="out data", stderr="err data"
    )

    tool = RunCommandTool()
    result = tool.execute(command="ls")

    assert result["result"]["stdout"] == "out data"
    assert result["result"]["stderr"] == "err data"


@patch("app.tools.terminal.subprocess.run")
def test_exit_code_capture(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=42, stdout="", stderr="")

    tool = RunCommandTool()
    result = tool.execute(command="ls")

    assert result["success"] is False
    assert result["result"]["exit_code"] == 42


def test_validate_command_allows_normal_commands() -> None:
    valid, reason, parts = _validate_command("ls -la")
    assert valid is True
    assert reason == ""
    assert parts == ["ls", "-la"]


def test_validate_command_rejects_fork_bomb() -> None:
    valid, reason, parts = _validate_command(":(){ :|:& };:")
    assert valid is False


# ---------------------------------------------------------------------------
# Phase 17: Terminal environment resolution
# ---------------------------------------------------------------------------

class TestTerminalEnvironmentResolution:
    @patch("app.tools.terminal.resolve_executable")
    @patch("app.tools.terminal.subprocess.run")
    def test_python_resolves_to_project_venv(
        self, mock_run: MagicMock, mock_resolve: MagicMock, fake_project: Path
    ) -> None:
        mock_resolve.return_value = str(fake_project / ".venv" / "bin" / "python3")
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        tool = RunCommandTool()
        result = tool.execute(command="python --version")

        assert result["success"] is True
        call_args = mock_run.call_args[0][0]
        assert "python3" in call_args[0]
        assert ".venv" in call_args[0]

    @patch("app.tools.terminal.resolve_executable")
    @patch("app.tools.terminal.subprocess.run")
    def test_pytest_resolves_to_project_venv(
        self, mock_run: MagicMock, mock_resolve: MagicMock, fake_project: Path
    ) -> None:
        mock_resolve.return_value = str(fake_project / ".venv" / "bin" / "pytest")
        mock_run.return_value = MagicMock(returncode=0, stdout="passed", stderr="")

        tool = RunCommandTool()
        result = tool.execute(command="pytest tests/")

        assert result["success"] is True
        call_args = mock_run.call_args[0][0]
        assert "pytest" in call_args[0]
        assert ".venv" in call_args[0]

    @patch("app.tools.terminal.resolve_executable")
    @patch("app.tools.terminal.subprocess.run")
    def test_ls_does_not_resolve(
        self, mock_run: MagicMock, mock_resolve: MagicMock
    ) -> None:
        mock_resolve.return_value = None
        mock_run.return_value = MagicMock(returncode=0, stdout="files", stderr="")

        tool = RunCommandTool()
        result = tool.execute(command="ls")

        assert result["success"] is True
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ls"

    @patch("app.tools.terminal.resolve_executable")
    @patch("app.tools.terminal.subprocess.run")
    def test_cwd_is_project_root(
        self, mock_run: MagicMock, mock_resolve: MagicMock
    ) -> None:
        mock_resolve.return_value = None
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        tool = RunCommandTool()
        tool.execute(command="pwd")

        _, kwargs = mock_run.call_args
        assert "cwd" in kwargs
        assert kwargs["cwd"] is not None

    @patch("app.tools.terminal.resolve_executable")
    @patch("app.tools.terminal.subprocess.run")
    def test_shell_false_enforced(
        self, mock_run: MagicMock, mock_resolve: MagicMock
    ) -> None:
        mock_resolve.return_value = None
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        tool = RunCommandTool()
        tool.execute(command="echo test")

        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is not True
        assert "shell" not in kwargs or kwargs.get("shell") is not True

    def test_python_c_still_blocked(self) -> None:
        tool = RunCommandTool()
        result = tool.execute(command="python -c 'import os'")
        assert result["success"] is False
        assert "not allowed" in result["error"]
