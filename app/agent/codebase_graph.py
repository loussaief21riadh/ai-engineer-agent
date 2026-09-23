"""Codebase Graph for V4 — lightweight module/function/test relationship graph."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.tools.security import is_secret_path


class ModuleNode(BaseModel):
    """A module (file) in the codebase graph."""
    path: str
    name: str = ""
    imports: list[str] = Field(default_factory=list)
    exports: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    test_targets: list[str] = Field(default_factory=list)


class ImportEdge(BaseModel):
    """An import relationship between modules."""
    source: str
    target: str
    import_type: str = "import"  # import, from_import


class TestTargetEdge(BaseModel):
    """A test-to-target relationship."""
    test_module: str
    target_module: str
    test_functions: list[str] = Field(default_factory=list)


class CodebaseGraph:
    """Lightweight graph of module relationships."""

    def __init__(self, root_path: str | Path = "") -> None:
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._modules: dict[str, ModuleNode] = {}
        self._import_edges: list[ImportEdge] = []
        self._test_target_edges: list[TestTargetEdge] = []

    def build(self) -> None:
        """Build the codebase graph."""
        self._scan_modules()
        self._build_import_graph()
        self._build_test_mapping()

    def _scan_modules(self) -> None:
        for root, dirs, files in os.walk(self.root_path):
            if ".git" in root or "__pycache__" in root or ".venv" in root:
                continue
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules"}]

            for f in files:
                if not f.endswith(".py"):
                    continue

                filepath = Path(root) / f
                if is_secret_path(filepath, self.root_path):
                    continue

                rel_path = str(filepath.relative_to(self.root_path))

                node = ModuleNode(path=rel_path, name=f[:-3])

                try:
                    source = filepath.read_text()
                    tree = ast.parse(source, filename=rel_path)

                    for item in ast.walk(tree):
                        if isinstance(item, ast.Import):
                            for alias in item.names:
                                node.imports.append(alias.name)
                        elif isinstance(item, ast.ImportFrom):
                            if item.module:
                                node.imports.append(item.module)
                        elif isinstance(item, ast.ClassDef):
                            node.classes.append(item.name)
                        elif isinstance(item, ast.FunctionDef):
                            node.functions.append(item.name)

                except (SyntaxError, OSError, UnicodeDecodeError):
                    pass

                self._modules[rel_path] = node

    def _build_import_graph(self) -> None:
        module_names = {node.name: path for path, node in self._modules.items()}

        for path, node in self._modules.items():
            for imp in node.imports:
                parts = imp.split(".")
                for i in range(len(parts), 0, -1):
                    candidate = ".".join(parts[:i])
                    if candidate in module_names:
                        target = module_names[candidate]
                        self._import_edges.append(ImportEdge(
                            source=path,
                            target=target,
                            import_type="from_import" if i < len(parts) else "import",
                        ))
                        break

    def _build_test_mapping(self) -> None:
        for path, node in self._modules.items():
            if "test" not in path.lower():
                continue

            for other_path, other_node in self._modules.items():
                if other_path == path:
                    continue

                for func_name in other_node.functions:
                    if any(t in func_name.lower() for t in ["test", "check", "verify"]):
                        if func_name in node.functions or func_name in node.classes:
                            self._test_target_edges.append(TestTargetEdge(
                                test_module=path,
                                target_module=other_path,
                                test_functions=[func_name],
                            ))

    def get_module(self, path: str) -> ModuleNode | None:
        return self._modules.get(path)

    def get_dependents(self, path: str) -> list[str]:
        """Get modules that import from the given module."""
        return [e.source for e in self._import_edges if e.target == path]

    def get_dependencies(self, path: str) -> list[str]:
        """Get modules that the given module imports."""
        return [e.target for e in self._import_edges if e.source == path]

    def get_test_modules_for(self, path: str) -> list[str]:
        """Get test modules that test the given module."""
        return [e.test_module for e in self._test_target_edges if e.target_module == path]

    def get_all_modules(self) -> list[ModuleNode]:
        return list(self._modules.values())

    def get_import_edges(self) -> list[ImportEdge]:
        return list(self._import_edges)

    def get_test_target_edges(self) -> list[TestTargetEdge]:
        return list(self._test_target_edges)

    def summary(self) -> dict[str, Any]:
        return {
            "total_modules": len(self._modules),
            "import_edges": len(self._import_edges),
            "test_target_edges": len(self._test_target_edges),
            "modules_with_imports": sum(1 for n in self._modules.values() if n.imports),
        }
