from __future__ import annotations

import shlex
import subprocess
from typing import Any

from app.config import COMMAND_TIMEOUT, PROJECT_ROOT
from app.tools.base import BaseTool, ToolSchema
from app.tools.env import resolve_executable
from app.tools.security import contains_shell_metacharacters


SAFE_PYTEST_FLAGS: set[str] = {
    "-v",
    "-vv",
    "-q",
    "--verbose",
    "--quiet",
    "-x",
    "--exitfirst",
    "--tb=short",
    "--tb=long",
    "--tb=line",
    "--tb=no",
    "--no-header",
    "-s",
    "--capture=no",
    "--co",
    "--collect-only",
    "-k",
    "-m",
    "--lf",
    "--last-failed",
    "--ff",
    "--failed-first",
    "-p",
}


def _parse_pytest_args(args: str) -> tuple[bool, str, list[str]]:
    if not args.strip():
        return True, "", ["pytest"]

    if contains_shell_metacharacters(args):
        return False, "Shell metacharacters are not allowed in test arguments", []

    try:
        parts = shlex.split(args)
    except ValueError as exc:
        return False, f"Cannot parse test arguments: {exc}", []

    cmd = ["pytest"]

    i = 0
    while i < len(parts):
        part = parts[i]

        if part.startswith("-"):
            flag = part.split("=")[0] if "=" in part else part

            if flag in SAFE_PYTEST_FLAGS:
                cmd.append(part)
                if flag in ("-k", "-m", "-p") and i + 1 < len(parts):
                    i += 1
                    cmd.append(parts[i])
            else:
                return False, f"Blocked pytest flag: '{part}'", []
        else:
            if "/" in part or part.endswith(".py") or part.endswith("/"):
                cmd.append(part)
            else:
                return False, f"Blocked pytest argument: '{part}'", []

        i += 1

    return True, "", cmd


class RunTestsTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_tests",
            description="Run pytest from the project root and return structured results.",
            parameters={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Pytest arguments (e.g. 'tests/test_foo.py -v')",
                    },
                },
                "required": [],
            },
        )

    def execute(self, args: str = "", **kwargs: Any) -> dict[str, Any]:
        valid, reason, cmd = _parse_pytest_args(args)
        if not valid:
            return {"success": False, "error": reason}

        resolved = resolve_executable("pytest")
        if resolved:
            cmd[0] = resolved

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT * 3,
            )

            return {
                "success": result.returncode == 0,
                "result": {
                    "command": " ".join(cmd),
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Tests timed out after {COMMAND_TIMEOUT * 3}s",
            }
        except Exception as exc:
            return {"success": False, "error": f"Test execution failed: {exc}"}
