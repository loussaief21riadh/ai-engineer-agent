"""Safe edit tool for V2.0 — targeted file modifications instead of full replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT
from app.tools.base import BaseTool, ToolSchema


def _safe_path(path: str) -> Path:
    from app.tools.security import safe_path
    return safe_path(path, PROJECT_ROOT)


class EditFileTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="edit_file",
            description=(
                "Apply a targeted text replacement in a file. "
                "The old_text must match exactly once in the file. "
                "Returns an error if old_text is not found or is ambiguous."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find and replace (must be unique in file)",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        )

    def execute(
        self,
        path: str = "",
        old_text: str = "",
        new_text: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not path:
            return {"success": False, "error": "path is required"}

        if not old_text:
            return {"success": False, "error": "old_text is required"}

        if old_text == new_text:
            return {"success": False, "error": "old_text and new_text are identical"}

        try:
            target = _safe_path(path)
        except PermissionError as exc:
            return {"success": False, "error": str(exc)}

        if not target.exists():
            return {"success": False, "error": f"File not found: {path}"}

        if not target.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"success": False, "error": f"File is not UTF-8 text: {path}"}

        count = content.count(old_text)

        if count == 0:
            return {
                "success": False,
                "error": f"old_text not found in {path}. The text may have changed or the match is incorrect.",
            }

        if count > 1:
            return {
                "success": False,
                "error": f"old_text is ambiguous — found {count} occurrences in {path}. Provide more context to make it unique.",
            }

        new_content = content.replace(old_text, new_text, 1)

        try:
            target.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            return {"success": False, "error": f"Write failed: {exc}"}

        return {
            "success": True,
            "result": f"Replaced 1 occurrence in {path}",
        }
