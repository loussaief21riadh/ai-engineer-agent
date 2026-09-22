"""Tests for V2.0-D Safe Edit Tool."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.tools.edit import EditFileTool, _safe_path


class TestSafePath:
    def test_relative_path_resolves_within_project(self, tmp_path: Path) -> None:
        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = _safe_path("app/main.py")
            assert result == (tmp_path / "app/main.py").resolve()

    def test_absolute_path_outside_project_rejected(self, tmp_path: Path) -> None:
        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            with pytest.raises(PermissionError, match="outside project root"):
                _safe_path("/etc/passwd")

    def test_relative_path_traversal_rejected(self, tmp_path: Path) -> None:
        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            with pytest.raises(PermissionError, match="outside project root"):
                _safe_path("../../../etc/passwd")

    def test_secret_path_rejected(self, tmp_path: Path) -> None:
        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            with pytest.raises(PermissionError, match="protected secret"):
                _safe_path(".env")


class TestEditFileTool:
    def setup_method(self):
        self.tool = EditFileTool()

    def test_schema(self):
        schema = self.tool.schema
        assert schema.name == "edit_file"
        assert "old_text" in schema.parameters["properties"]
        assert "new_text" in schema.parameters["properties"]
        assert "path" in schema.parameters["properties"]
        assert "old_text" in schema.parameters["required"]
        assert "new_text" in schema.parameters["required"]
        assert "path" in schema.parameters["required"]

    def test_edit_success(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    pass\n")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="test.py", old_text="pass", new_text="return 42")

        assert result["success"] is True
        assert "Replaced" in result["result"]
        assert test_file.read_text() == "def foo():\n    return 42\n"

    def test_edit_not_found(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("hello world")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="test.py", old_text="goodbye", new_text="hello")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_edit_ambiguous(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\nx = 2\nx = 3\n")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="test.py", old_text="x = ", new_text="y = ")

        assert result["success"] is False
        assert "ambiguous" in result["error"]
        assert "3 occurrences" in result["error"]

    def test_edit_file_not_found(self, tmp_path: Path) -> None:
        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="nonexistent.py", old_text="a", new_text="b")

        assert result["success"] is False
        assert "File not found" in result["error"]

    def test_edit_path_outside_project(self, tmp_path: Path) -> None:
        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="../etc/passwd", old_text="root", new_text="hacked")

        assert result["success"] is False
        assert "outside project root" in result["error"]

    def test_edit_secret_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=123")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path=".env", old_text="SECRET=123", new_text="SECRET=456")

        assert result["success"] is False
        assert "protected secret" in result["error"]

    def test_edit_empty_path(self) -> None:
        result = self.tool.execute(path="", old_text="a", new_text="b")
        assert result["success"] is False
        assert "path is required" in result["error"]

    def test_edit_empty_old_text(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="test.py", old_text="", new_text="new")

        assert result["success"] is False
        assert "old_text is required" in result["error"]

    def test_edit_same_text(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="test.py", old_text="content", new_text="content")

        assert result["success"] is False
        assert "identical" in result["error"]

    def test_edit_multiline(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(
                path="test.py",
                old_text="def foo():\n    pass",
                new_text="def foo():\n    return 1",
            )

        assert result["success"] is True
        content = test_file.read_text()
        assert "return 1" in content
        assert "def bar():" in content

    def test_edit_non_utf8(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x80\x81\x82")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="test.bin", old_text="a", new_text="b")

        assert result["success"] is False
        assert "not UTF-8" in result["error"]

    def test_edit_directory(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "mydir"
        test_dir.mkdir()

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="mydir", old_text="a", new_text="b")

        assert result["success"] is False
        assert "Not a file" in result["error"]

    def test_edit_preserves_surrounding_content(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        original = "line1\nline2\nline3\nline4\n"
        test_file.write_text(original)

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(path="test.py", old_text="line2", new_text="LINE2")

        assert result["success"] is True
        content = test_file.read_text()
        assert "line1" in content
        assert "LINE2" in content
        assert "line3" in content
        assert "line4" in content

    def test_edit_addition(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    pass\n")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(
                path="test.py",
                old_text="def foo():\n    pass\n",
                new_text="def foo():\n    pass\n\ndef bar():\n    pass\n",
            )

        assert result["success"] is True
        content = test_file.read_text()
        assert "def bar():" in content

    def test_edit_deletion(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nDELETE_ME\nline3\n")

        with patch("app.tools.edit.PROJECT_ROOT", tmp_path):
            result = self.tool.execute(
                path="test.py",
                old_text="DELETE_ME\n",
                new_text="",
            )

        assert result["success"] is True
        content = test_file.read_text()
        assert "DELETE_ME" not in content
        assert "line1" in content
        assert "line3" in content
