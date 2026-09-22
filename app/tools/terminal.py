from __future__ import annotations

import shlex
import subprocess
from typing import Any

from app.config import COMMAND_TIMEOUT, PROJECT_ROOT
from app.tools.base import BaseTool, ToolSchema
from app.tools.env import resolve_executable
from app.tools.security import contains_shell_metacharacters


ALLOWED_COMMANDS: dict[str, list[str]] = {
    "python": ["python"],
    "python3": ["python3"],
    "pytest": ["pytest"],
    "git": ["git"],
    "ls": ["ls"],
    "pwd": ["pwd"],
    "find": ["find"],
    "cat": ["cat"],
    "head": ["head"],
    "tail": ["tail"],
    "wc": ["wc"],
    "grep": ["grep"],
    "echo": ["echo"],
    "date": ["date"],
    "whoami": ["whoami"],
    "which": ["which"],
}

BLOCKED_GIT_SUBCOMMANDS: set[str] = {
    "push",
    "commit",
    "reset",
    "checkout",
    "branch",
    "merge",
    "rebase",
    "stash",
    "cherry-pick",
    "revert",
    "tag",
    "config",
    "clean",
    "restore",
    "switch",
}


def _validate_command(command: str) -> tuple[bool, str, list[str]]:
    if not command.strip():
        return False, "command is required", []

    if contains_shell_metacharacters(command):
        return False, "Shell metacharacters are not allowed (;, &&, ||, |, $(), backticks, >, >>, <, newlines)", []

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return False, f"Cannot parse command: {exc}", []

    if not parts:
        return False, "command is required", []

    executable = parts[0]

    if executable not in ALLOWED_COMMANDS:
        allowed = sorted(ALLOWED_COMMANDS.keys())
        return False, f"Unknown executable: '{executable}'. Allowed: {', '.join(allowed)}", []

    if executable == "git" and len(parts) >= 2:
        subcmd = parts[1]
        if subcmd in BLOCKED_GIT_SUBCOMMANDS:
            return False, f"Blocked git subcommand: '{subcmd}'", []

    if executable == "python" or executable == "python3":
        if len(parts) >= 2 and parts[1] == "-c":
            return False, "python -c is not allowed", []

    return True, "", parts


class RunCommandTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_command",
            description="Execute a safe developer command from the project root. Captures stdout, stderr, and exit code.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute (e.g. 'ls -la', 'git status')",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: from config)",
                    },
                },
                "required": ["command"],
            },
        )

    def execute(self, command: str = "", timeout: int | None = None, **kwargs: Any) -> dict[str, Any]:
        valid, reason, parts = _validate_command(command)
        if not valid:
            return {"success": False, "error": reason}

        executable = parts[0]
        resolved = resolve_executable(executable)
        if resolved:
            parts[0] = resolved

        effective_timeout = timeout or COMMAND_TIMEOUT

        try:
            result = subprocess.run(
                parts,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )

            return {
                "success": result.returncode == 0,
                "result": {
                    "command": command,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {effective_timeout}s",
            }
        except Exception as exc:
            return {"success": False, "error": f"Command failed: {exc}"}
