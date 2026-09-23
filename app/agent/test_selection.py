"""Test Selection for V4 — selects relevant tests from changed files."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent.codebase_graph import CodebaseGraph


class TestSelectionResult(BaseModel):
    """Result of test selection."""
    selected_tests: list[str] = Field(default_factory=list)
    all_tests: list[str] = Field(default_factory=list)
    selection_reason: str = ""
    coverage_estimate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_tests": self.selected_tests,
            "total_tests": len(self.all_tests),
            "selected_count": len(self.selected_tests),
            "selection_reason": self.selection_reason,
            "coverage_estimate": self.coverage_estimate,
        }


class TestSelector:
    """Selects relevant tests based on changed files and impact graph."""

    def __init__(self, graph: CodebaseGraph) -> None:
        self._graph = graph

    def select(
        self,
        changed_files: list[str],
        include_all: bool = False,
    ) -> TestSelectionResult:
        """Select relevant tests for the given changed files."""
        all_test_modules = [
            m.path for m in self._graph.get_all_modules()
            if "test" in m.path.lower()
        ]

        if include_all or not changed_files:
            return TestSelectionResult(
                selected_tests=all_test_modules,
                all_tests=all_test_modules,
                selection_reason="Full suite" if include_all else "No changes specified",
                coverage_estimate=1.0,
            )

        selected: set[str] = set()

        for changed_file in changed_files:
            direct_tests = self._graph.get_test_modules_for(changed_file)
            selected.update(direct_tests)

            dependents = self._graph.get_dependents(changed_file)
            for dep in dependents:
                dep_tests = self._graph.get_test_modules_for(dep)
                selected.update(dep_tests)

            for test_module in all_test_modules:
                if self._has_path_overlap(changed_file, test_module):
                    selected.add(test_module)

        selected_list = sorted(selected)
        coverage = len(selected_list) / len(all_test_modules) if all_test_modules else 0.0

        reason_parts = [f"{len(changed_files)} files changed"]
        if len(selected_list) < len(all_test_modules):
            reason_parts.append(f"selected {len(selected_list)}/{len(all_test_modules)} tests")
        else:
            reason_parts.append("full suite needed")

        return TestSelectionResult(
            selected_tests=selected_list,
            all_tests=all_test_modules,
            selection_reason=", ".join(reason_parts),
            coverage_estimate=coverage,
        )

    def _has_path_overlap(self, file1: str, file2: str) -> bool:
        parts1 = file1.split("/")
        parts2 = file2.split("/")

        if len(parts1) >= 2 and len(parts2) >= 2:
            if parts1[0] == parts2[0] and parts1[1] == parts2[1]:
                return True

        return False

    def get_test_count(self) -> int:
        return len([
            m for m in self._graph.get_all_modules()
            if "test" in m.path.lower()
        ])
