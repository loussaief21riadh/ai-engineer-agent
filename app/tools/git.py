from __future__ import annotations

import subprocess
from typing import Any

from app.config import PROJECT_ROOT
from app.tools.base import BaseTool, ToolSchema


def _git_read_only(subcommand: str, *extra_args: str) -> tuple[int, str, str]:
    ALLOWED_GIT_SUBCOMMANDS: set[str] = {"status", "diff", "log"}

    if subcommand not in ALLOWED_GIT_SUBCOMMANDS:
        return 128, "", f"Blocked: git '{subcommand}' is not a read-only operation"

    parts = ["git", subcommand, *extra_args]

    result = subprocess.run(
        parts,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


class GitStatusTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_status",
            description="Show the working tree status.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        code, stdout, stderr = _git_read_only("status")

        if code != 0:
            return {"success": False, "error": stderr.strip() or "git status failed"}

        return {"success": True, "result": stdout}


class GitDiffTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_diff",
            description="Show changes between working directory and last commit.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        code, stdout, stderr = _git_read_only("diff")

        if code != 0:
            return {"success": False, "error": stderr.strip() or "git diff failed"}

        if not stdout.strip():
            return {"success": True, "result": "No changes."}

        return {"success": True, "result": stdout}
