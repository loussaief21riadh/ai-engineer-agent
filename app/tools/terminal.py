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
    "add",
    "push",
    "commit",
    "reset",
    "checkout",
    "merge",
    "rebase",
    "cherry-pick",
    "revert",
    "config",
    "clean",
    "restore",
    "switch",
    "clone",
    "init",
    "pull",
    "fetch",
    "rm",
    "mv",
}

GIT_WRITE_SUBCOMMANDS: set[str] = {
    "push",
    "commit",
    "reset",
    "checkout",
    "merge",
    "rebase",
    "cherry-pick",
    "revert",
    "config",
    "clean",
    "restore",
    "switch",
    "clone",
    "init",
    "pull",
    "fetch",
    "rm",
    "mv",
    "add",
}

GIT_CONDITIONAL_SUBCOMMANDS: dict[str, set[str]] = {
    "branch": {"feature", "new", "main", "master", "dev", "develop"},
    "stash": {"push", "save", "pop", "apply", "drop", "clear", "branch", "create"},
    "tag": {"-d", "-f", "-a", "-s", "-u", "--delete", "--force"},
}


def _is_write_git_subcommand(subcmd: str, args: list[str]) -> bool:
    """Check if a git subcommand is a write/destructive operation."""
    if subcmd in GIT_WRITE_SUBCOMMANDS:
        return True

    if subcmd in BLOCKED_GIT_SUBCOMMANDS:
        return True

    if subcmd == "branch":
        write_flags = {"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy"}
        for arg in args:
            if arg in write_flags:
                return True
        if args:
            first = args[0]
            if not first.startswith("-"):
                return True
        return False

    if subcmd == "stash":
        write_stash = {"push", "save", "pop", "apply", "drop", "clear", "branch", "create"}
        if not args:
            return True
        if args[0] in write_stash:
            return True
        return False

    if subcmd == "tag":
        if not args:
            return False
        if args[0] in ("-l", "--list", "-n", "--contains", "--sort", "--format"):
            return False
        return True

    return False

GIT_GLOBAL_OPTIONS: set[str] = {
    "-C",
    "--git-dir",
    "--work-tree",
    "--exec-path",
}


def _extract_git_subcommand(parts: list[str]) -> str | None:
    """Extract the actual git subcommand, skipping global options."""
    if len(parts) < 2:
        return None
    i = 1
    while i < len(parts):
        part = parts[i]
        if part in GIT_GLOBAL_OPTIONS:
            i += 2
            continue
        if part.startswith("-C") and len(part) > 2:
            i += 1
            continue
        if part.startswith("--git-dir=") or part.startswith("--work-tree=") or part.startswith("--exec-path="):
            i += 1
            continue
        return part
    return None


def _get_git_args(parts: list[str]) -> list[str]:
    """Extract the arguments after the git subcommand, skipping global options."""
    if len(parts) < 2:
        return []
    i = 1
    subcmd_found = False
    args: list[str] = []
    while i < len(parts):
        part = parts[i]
        if not subcmd_found:
            if part in GIT_GLOBAL_OPTIONS:
                i += 2
                continue
            if part.startswith("-C") and len(part) > 2:
                i += 1
                continue
            if part.startswith("--git-dir=") or part.startswith("--work-tree=") or part.startswith("--exec-path="):
                i += 1
                continue
            subcmd_found = True
            i += 1
            continue
        args.append(part)
        i += 1
    return args


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

    if executable == "git":
        subcmd = _extract_git_subcommand(parts)
        if subcmd is None:
            return False, "git command with no subcommand", []
        git_args = _get_git_args(parts)
        if _is_write_git_subcommand(subcmd, git_args):
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
