"""Project memory for V2.0 — structured local knowledge retention."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    key: str
    value: str
    category: str = "general"
    confidence: float = 1.0


class ProjectMemory(BaseModel):
    important_files: dict[str, str] = Field(default_factory=dict)
    architecture_notes: list[str] = Field(default_factory=list)
    known_commands: dict[str, str] = Field(default_factory=dict)
    known_test_commands: list[str] = Field(default_factory=list)
    important_decisions: list[str] = Field(default_factory=list)
    previous_failures: list[str] = Field(default_factory=list)
    successful_fixes: list[str] = Field(default_factory=list)
    project_conventions: list[str] = Field(default_factory=list)
    entries: list[MemoryEntry] = Field(default_factory=list)

    def add_important_file(self, path: str, description: str) -> None:
        self.important_files[path] = description

    def add_architecture_note(self, note: str) -> None:
        self.architecture_notes.append(note)
        if len(self.architecture_notes) > 50:
            self.architecture_notes = self.architecture_notes[-50:]

    def add_known_command(self, command: str, description: str) -> None:
        self.known_commands[command] = description

    def add_test_command(self, command: str) -> None:
        if command not in self.known_test_commands:
            self.known_test_commands.append(command)

    def add_decision(self, decision: str) -> None:
        self.important_decisions.append(decision)
        if len(self.important_decisions) > 30:
            self.important_decisions = self.important_decisions[-30:]

    def add_failure(self, failure: str) -> None:
        self.previous_failures.append(failure)
        if len(self.previous_failures) > 20:
            self.previous_failures = self.previous_failures[-20:]

    def add_successful_fix(self, fix: str) -> None:
        self.successful_fixes.append(fix)
        if len(self.successful_fixes) > 20:
            self.successful_fixes = self.successful_fixes[-20:]

    def add_convention(self, convention: str) -> None:
        self.project_conventions.append(convention)
        if len(self.project_conventions) > 30:
            self.project_conventions = self.project_conventions[-30:]

    def add_entry(self, key: str, value: str, category: str = "general", confidence: float = 1.0) -> None:
        self.entries.append(MemoryEntry(key=key, value=value, category=category, confidence=confidence))

    def get_entries_by_category(self, category: str) -> list[MemoryEntry]:
        return [e for e in self.entries if e.category == category]

    def to_context_string(self) -> str:
        parts: list[str] = []

        if self.important_files:
            parts.append("Important files:")
            for path, desc in self.important_files.items():
                parts.append(f"  {path}: {desc}")

        if self.architecture_notes:
            parts.append("Architecture notes:")
            for note in self.architecture_notes[-5:]:
                parts.append(f"  - {note}")

        if self.known_commands:
            parts.append("Known commands:")
            for cmd, desc in self.known_commands.items():
                parts.append(f"  {cmd}: {desc}")

        if self.known_test_commands:
            parts.append("Test commands:")
            for cmd in self.known_test_commands:
                parts.append(f"  {cmd}")

        if self.important_decisions:
            parts.append("Decisions:")
            for dec in self.important_decisions[-5:]:
                parts.append(f"  - {dec}")

        if self.successful_fixes:
            parts.append("Successful fixes:")
            for fix in self.successful_fixes[-3:]:
                parts.append(f"  - {fix}")

        if self.project_conventions:
            parts.append("Conventions:")
            for conv in self.project_conventions[-5:]:
                parts.append(f"  - {conv}")

        return "\n".join(parts) if parts else "No project memory available."

    def to_dict(self) -> dict[str, Any]:
        return {
            "important_files": self.important_files,
            "architecture_notes": self.architecture_notes,
            "known_commands": self.known_commands,
            "known_test_commands": self.known_test_commands,
            "important_decisions": self.important_decisions,
            "previous_failures": self.previous_failures,
            "successful_fixes": self.successful_fixes,
            "project_conventions": self.project_conventions,
            "entries": [e.model_dump() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectMemory:
        entries = [MemoryEntry(**e) for e in data.get("entries", [])]
        return cls(
            important_files=data.get("important_files", {}),
            architecture_notes=data.get("architecture_notes", []),
            known_commands=data.get("known_commands", {}),
            known_test_commands=data.get("known_test_commands", []),
            important_decisions=data.get("important_decisions", []),
            previous_failures=data.get("previous_failures", []),
            successful_fixes=data.get("successful_fixes", []),
            project_conventions=data.get("project_conventions", []),
            entries=entries,
        )
