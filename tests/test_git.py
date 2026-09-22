from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from app.tools.git import GitDiffTool, GitStatusTool, _git_read_only


@patch("app.tools.git.subprocess.run")
def test_git_status_tool_returns_status(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=0, stdout="On branch main\nnothing to commit", stderr=""
    )

    tool = GitStatusTool()
    result = tool.execute()

    assert result["success"] is True
    assert "On branch main" in result["result"]


@patch("app.tools.git.subprocess.run")
def test_git_status_tool_handles_errors(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=128, stdout="", stderr="fatal: not a git repo"
    )

    tool = GitStatusTool()
    result = tool.execute()

    assert result["success"] is False
    assert "fatal: not a git repo" in result["error"]


@patch("app.tools.git.subprocess.run")
def test_git_diff_tool_returns_diff(mock_run: MagicMock) -> None:
    diff_output = "diff --git a/file.txt b/file.txt\n+new line"
    mock_run.return_value = MagicMock(
        returncode=0, stdout=diff_output, stderr=""
    )

    tool = GitDiffTool()
    result = tool.execute()

    assert result["success"] is True
    assert result["result"] == diff_output


@patch("app.tools.git.subprocess.run")
def test_git_diff_tool_handles_no_changes(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    tool = GitDiffTool()
    result = tool.execute()

    assert result["success"] is True
    assert result["result"] == "No changes."


@patch("app.tools.git.subprocess.run")
def test_git_diff_tool_handles_errors(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=128, stdout="", stderr="fatal: not a git repo"
    )

    tool = GitDiffTool()
    result = tool.execute()

    assert result["success"] is False
    assert "fatal: not a git repo" in result["error"]
