import os
from pathlib import Path

import pytest

import app.tools.filesystem as fs_mod
from app.config import PROJECT_ROOT
from app.tools.filesystem import (
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    _safe_path,
)


@pytest.fixture(autouse=True)
def _patch_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(fs_mod, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def root(_patch_project_root):
    return _patch_project_root


# ── _safe_path ────────────────────────────────────────────────────────────


class TestSafePath:
    def test_relative_path_resolves_inside_root(self, root):
        target = _safe_path("foo/bar.txt")
        assert target == (root / "foo/bar.txt").resolve()

    def test_absolute_path_inside_root_accepted(self, root):
        inner = root / "inside.txt"
        target = _safe_path(str(inner))
        assert target == inner.resolve()

    def test_absolute_path_outside_root_rejected(self, root):
        outside = root.parent / "outside.txt"
        with pytest.raises(PermissionError, match="outside project root"):
            _safe_path(str(outside))

    def test_relative_traversal_outside_root_rejected(self, root):
        with pytest.raises(PermissionError, match="outside project root"):
            _safe_path("../../etc/passwd")

    def test_dotdot_segments_resolved(self, root):
        sub = root / "a" / "b"
        sub.mkdir(parents=True)
        target = _safe_path("a/b/../b/file.txt")
        assert target == (root / "a/b/file.txt").resolve()

    def test_single_dot_resolves_to_root(self, root):
        target = _safe_path(".")
        assert target == root.resolve()


# ── ListFilesTool ─────────────────────────────────────────────────────────


class TestListFilesTool:
    def test_lists_files_and_dirs(self, root):
        (root / "file.txt").write_text("hi")
        (root / "subdir").mkdir()

        tool = ListFilesTool()
        result = tool.execute(path=".")

        assert result["success"] is True
        entries = result["result"]
        names = [e["name"] for e in entries]
        assert "file.txt" in names
        assert "subdir" in names

        file_entry = next(e for e in entries if e["name"] == "file.txt")
        assert file_entry["type"] == "file"

        dir_entry = next(e for e in entries if e["name"] == "subdir")
        assert dir_entry["type"] == "directory"

    def test_empty_directory(self, root):
        tool = ListFilesTool()
        result = tool.execute(path=".")
        assert result["success"] is True
        assert result["result"] == []

    def test_non_existent_path(self, root):
        tool = ListFilesTool()
        result = tool.execute(path="no_such_dir")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_file_not_directory(self, root):
        (root / "a_file").write_text("data")
        tool = ListFilesTool()
        result = tool.execute(path="a_file")
        assert result["success"] is False
        assert "not a directory" in result["error"].lower()

    def test_lists_subdirectory_contents(self, root):
        sub = root / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("a")
        (sub / "b.txt").write_text("b")

        tool = ListFilesTool()
        result = tool.execute(path="sub")
        assert result["success"] is True
        names = sorted(e["name"] for e in result["result"])
        assert names == ["a.txt", "b.txt"]

    def test_entries_are_sorted(self, root):
        for name in ["z.txt", "a.txt", "m.txt"]:
            (root / name).write_text("x")

        tool = ListFilesTool()
        result = tool.execute(path=".")
        names = [e["name"] for e in result["result"]]
        assert names == sorted(names)

    def test_schema_properties(self):
        schema = ListFilesTool().schema
        assert schema.name == "list_files"
        assert "path" in schema.parameters["properties"]


# ── ReadFileTool ──────────────────────────────────────────────────────────


class TestReadFileTool:
    def test_reads_file_correctly(self, root):
        (root / "hello.txt").write_text("Hello, world!", encoding="utf-8")
        tool = ReadFileTool()
        result = tool.execute(path="hello.txt")
        assert result["success"] is True
        assert result["result"] == "Hello, world!"

    def test_handles_missing_file(self, root):
        tool = ReadFileTool()
        result = tool.execute(path="nope.txt")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_handles_directory_path(self, root):
        (root / "adir").mkdir()
        tool = ReadFileTool()
        result = tool.execute(path="adir")
        assert result["success"] is False
        assert "not a file" in result["error"].lower()

    def test_rejects_empty_path(self):
        tool = ReadFileTool()
        result = tool.execute(path="")
        assert result["success"] is False
        assert "required" in result["error"].lower()

    def test_rejects_non_utf8_file(self, root):
        binary_path = root / "binary.bin"
        binary_path.write_bytes(b"\x80\x81\x82\xff")
        tool = ReadFileTool()
        result = tool.execute(path="binary.bin")
        assert result["success"] is False
        assert "utf-8" in result["error"].lower()

    def test_reads_nested_path(self, root):
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "c.txt").write_text("deep")
        tool = ReadFileTool()
        result = tool.execute(path="a/b/c.txt")
        assert result["success"] is True
        assert result["result"] == "deep"

    def test_schema_properties(self):
        schema = ReadFileTool().schema
        assert schema.name == "read_file"
        assert "path" in schema.parameters["properties"]
        assert "path" in schema.parameters["required"]

    def test_path_traversal_blocked(self, root):
        tool = ReadFileTool()
        result = tool.execute(path="../../etc/passwd")
        assert result["success"] is False
        assert "outside project root" in result["error"]


# ── WriteFileTool ─────────────────────────────────────────────────────────


class TestWriteFileTool:
    def test_writes_file_correctly(self, root):
        tool = WriteFileTool()
        result = tool.execute(path="out.txt", content="some data")
        assert result["success"] is True
        assert (root / "out.txt").read_text(encoding="utf-8") == "some data"

    def test_creates_parent_directories(self, root):
        tool = WriteFileTool()
        result = tool.execute(path="a/b/c/deep.txt", content="nested")
        assert result["success"] is True
        assert (root / "a/b/c/deep.txt").read_text(encoding="utf-8") == "nested"

    def test_overwrites_existing_file(self, root):
        (root / "over.txt").write_text("old")
        tool = WriteFileTool()
        result = tool.execute(path="over.txt", content="new")
        assert result["success"] is True
        assert (root / "over.txt").read_text(encoding="utf-8") == "new"

    def test_empty_path_rejected(self):
        tool = WriteFileTool()
        result = tool.execute(path="", content="x")
        assert result["success"] is False
        assert "required" in result["error"].lower()

    def test_empty_content_succeeds(self, root):
        tool = WriteFileTool()
        result = tool.execute(path="empty.txt", content="")
        assert result["success"] is True
        assert (root / "empty.txt").read_text() == ""

    def test_returns_byte_count(self, root):
        tool = WriteFileTool()
        result = tool.execute(path="c.txt", content="hello")
        assert "5 bytes" in result["result"]

    def test_schema_properties(self):
        schema = WriteFileTool().schema
        assert schema.name == "write_file"
        assert "path" in schema.parameters["properties"]
        assert "content" in schema.parameters["properties"]
        assert set(schema.parameters["required"]) == {"path", "content"}

    def test_path_traversal_blocked(self, root):
        tool = WriteFileTool()
        result = tool.execute(path="../../etc/cron", content="evil")
        assert result["success"] is False
        assert "outside project root" in result["error"]

    def test_absolute_path_outside_root_rejected(self, root):
        outside = root.parent / "escape.txt"
        tool = WriteFileTool()
        result = tool.execute(path=str(outside), content="nope")
        assert result["success"] is False
        assert "outside project root" in result["error"]


# ── Integration-level checks ─────────────────────────────────────────────


class TestToolIntegration:
    def test_write_then_read(self, root):
        write_tool = WriteFileTool()
        read_tool = ReadFileTool()

        write_result = write_tool.execute(path="roundtrip.txt", content="ping")
        assert write_result["success"] is True

        read_result = read_tool.execute(path="roundtrip.txt")
        assert read_result["success"] is True
        assert read_result["result"] == "ping"

    def test_write_then_list(self, root):
        write_tool = WriteFileTool()
        list_tool = ListFilesTool()

        write_tool.execute(path="listed.txt", content="data")
        list_result = list_tool.execute(path=".")
        names = [e["name"] for e in list_result["result"]]
        assert "listed.txt" in names

    def test_write_then_read_subdirectory(self, root):
        write_tool = WriteFileTool()
        read_tool = ReadFileTool()

        write_tool.execute(path="sub/nested.txt", content="deep value")
        result = read_tool.execute(path="sub/nested.txt")
        assert result["result"] == "deep value"
