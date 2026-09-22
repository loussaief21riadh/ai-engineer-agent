"""Tests for V2.0-F Project Memory."""

from __future__ import annotations

import pytest

from app.agent.memory import MemoryEntry, ProjectMemory


class TestMemoryEntry:
    def test_entry_creation(self):
        entry = MemoryEntry(key="test", value="value", category="files")
        assert entry.key == "test"
        assert entry.value == "value"
        assert entry.category == "files"
        assert entry.confidence == 1.0

    def test_entry_default_confidence(self):
        entry = MemoryEntry(key="k", value="v")
        assert entry.confidence == 1.0


class TestProjectMemory:
    def test_default_memory(self):
        mem = ProjectMemory()
        assert mem.important_files == {}
        assert mem.architecture_notes == []
        assert mem.known_commands == {}
        assert mem.known_test_commands == []
        assert mem.important_decisions == []
        assert mem.previous_failures == []
        assert mem.successful_fixes == []
        assert mem.project_conventions == []
        assert mem.entries == []

    def test_add_important_file(self):
        mem = ProjectMemory()
        mem.add_important_file("app/main.py", "Entry point")
        assert mem.important_files["app/main.py"] == "Entry point"

    def test_add_architecture_note(self):
        mem = ProjectMemory()
        mem.add_architecture_note("Uses FastAPI")
        assert "Uses FastAPI" in mem.architecture_notes

    def test_add_architecture_note_limits(self):
        mem = ProjectMemory()
        for i in range(55):
            mem.add_architecture_note(f"Note {i}")
        assert len(mem.architecture_notes) == 50
        assert mem.architecture_notes[0] == "Note 5"

    def test_add_known_command(self):
        mem = ProjectMemory()
        mem.add_known_command("npm run dev", "Start dev server")
        assert mem.known_commands["npm run dev"] == "Start dev server"

    def test_add_test_command(self):
        mem = ProjectMemory()
        mem.add_test_command("pytest tests/")
        assert "pytest tests/" in mem.known_test_commands

    def test_add_test_command_no_duplicates(self):
        mem = ProjectMemory()
        mem.add_test_command("pytest tests/")
        mem.add_test_command("pytest tests/")
        assert len(mem.known_test_commands) == 1

    def test_add_decision(self):
        mem = ProjectMemory()
        mem.add_decision("Use pydantic for validation")
        assert "Use pydantic for validation" in mem.important_decisions

    def test_add_decision_limits(self):
        mem = ProjectMemory()
        for i in range(35):
            mem.add_decision(f"Decision {i}")
        assert len(mem.important_decisions) == 30
        assert mem.important_decisions[0] == "Decision 5"

    def test_add_failure(self):
        mem = ProjectMemory()
        mem.add_failure("test_x failed due to import error")
        assert "test_x failed due to import error" in mem.previous_failures

    def test_add_failure_limits(self):
        mem = ProjectMemory()
        for i in range(25):
            mem.add_failure(f"Failure {i}")
        assert len(mem.previous_failures) == 20

    def test_add_successful_fix(self):
        mem = ProjectMemory()
        mem.add_successful_fix("Added missing import")
        assert "Added missing import" in mem.successful_fixes

    def test_add_successful_fix_limits(self):
        mem = ProjectMemory()
        for i in range(25):
            mem.add_successful_fix(f"Fix {i}")
        assert len(mem.successful_fixes) == 20

    def test_add_convention(self):
        mem = ProjectMemory()
        mem.add_convention("Use snake_case for functions")
        assert "Use snake_case for functions" in mem.project_conventions

    def test_add_entry(self):
        mem = ProjectMemory()
        mem.add_entry("python_version", "3.14", category="environment")
        assert len(mem.entries) == 1
        assert mem.entries[0].key == "python_version"

    def test_get_entries_by_category(self):
        mem = ProjectMemory()
        mem.add_entry("k1", "v1", category="files")
        mem.add_entry("k2", "v2", category="commands")
        mem.add_entry("k3", "v3", category="files")
        files = mem.get_entries_by_category("files")
        assert len(files) == 2

    def test_to_context_string(self):
        mem = ProjectMemory()
        mem.add_important_file("app/main.py", "Entry point")
        mem.add_architecture_note("FastAPI app")
        mem.add_known_command("pytest", "Run tests")
        ctx = mem.to_context_string()
        assert "app/main.py" in ctx
        assert "FastAPI" in ctx
        assert "pytest" in ctx

    def test_to_context_string_empty(self):
        mem = ProjectMemory()
        ctx = mem.to_context_string()
        assert "No project memory" in ctx

    def test_to_context_string_limits_architecture(self):
        mem = ProjectMemory()
        for i in range(10):
            mem.add_architecture_note(f"Note {i}")
        ctx = mem.to_context_string()
        assert "Note 9" in ctx
        assert "Note 4" not in ctx

    def test_to_dict(self):
        mem = ProjectMemory()
        mem.add_important_file("a.py", "test")
        mem.add_entry("k", "v", category="cat")
        d = mem.to_dict()
        assert "a.py" in d["important_files"]
        assert len(d["entries"]) == 1

    def test_from_dict(self):
        data = {
            "important_files": {"a.py": "test"},
            "architecture_notes": ["note1"],
            "known_commands": {"ls": "list"},
            "known_test_commands": ["pytest"],
            "important_decisions": ["dec1"],
            "previous_failures": ["fail1"],
            "successful_fixes": ["fix1"],
            "project_conventions": ["conv1"],
            "entries": [{"key": "k", "value": "v", "category": "cat", "confidence": 0.9}],
        }
        mem = ProjectMemory.from_dict(data)
        assert mem.important_files == {"a.py": "test"}
        assert mem.architecture_notes == ["note1"]
        assert len(mem.entries) == 1
        assert mem.entries[0].confidence == 0.9

    def test_roundtrip(self):
        mem = ProjectMemory()
        mem.add_important_file("x.py", "desc")
        mem.add_convention("use type hints")
        mem.add_entry("key", "val")
        d = mem.to_dict()
        mem2 = ProjectMemory.from_dict(d)
        assert mem2.important_files == mem.important_files
        assert mem2.project_conventions == mem.project_conventions
        assert len(mem2.entries) == len(mem.entries)

    def test_memory_hierarchy(self):
        mem = ProjectMemory()
        mem.add_important_file("app/main.py", "verified via tool")
        mem.add_decision("model proposed: use class instead of function")
        ctx = mem.to_context_string()
        assert "app/main.py" in ctx
        assert "model proposed" in ctx
