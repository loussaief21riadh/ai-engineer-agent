from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT
from app.tools.base import BaseTool, ToolSchema
from app.tools.security import is_secret_path


def _safe_path(path: str) -> Path:
    from app.tools.security import safe_path
    return safe_path(path, PROJECT_ROOT)


class ListFilesTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_files",
            description="List files and directories at the given path within the project root.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to list (default: project root)",
                    }
                },
                "required": [],
            },
        )

    def execute(self, path: str = ".", **kwargs: Any) -> dict[str, Any]:
        try:
            target = _safe_path(path)
        except PermissionError as exc:
            return {"success": False, "error": str(exc)}

        if not target.exists():
            return {"success": False, "error": f"Path not found: {path}"}

        if not target.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}

        entries = []
        for entry in sorted(target.iterdir()):
            if is_secret_path(entry, PROJECT_ROOT):
                continue

            rel = entry.relative_to(PROJECT_ROOT)
            entries.append({
                "name": entry.name,
                "path": str(rel),
                "type": "directory" if entry.is_dir() else "file",
            })

        return {"success": True, "result": entries}


class ReadFileTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read the contents of a text file within the project root.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file",
                    }
                },
                "required": ["path"],
            },
        )

    def execute(self, path: str = "", **kwargs: Any) -> dict[str, Any]:
        if not path:
            return {"success": False, "error": "path is required"}

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

        return {"success": True, "result": content}


class WriteFileTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Write content to a file within the project root. Creates parent directories if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        )

    def execute(self, path: str = "", content: str = "", **kwargs: Any) -> dict[str, Any]:
        if not path:
            return {"success": False, "error": "path is required"}

        try:
            target = _safe_path(path)
        except PermissionError as exc:
            return {"success": False, "error": str(exc)}

        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"success": False, "error": f"Write failed: {exc}"}

        return {"success": True, "result": f"Wrote {len(content)} bytes to {path}"}
