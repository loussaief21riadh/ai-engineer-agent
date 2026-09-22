"""Safe project environment discovery.

Discovers Python and pytest executables within PROJECT_ROOT without
exposing secrets, escaping the project root, or allowing arbitrary paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT


VENV_NAMES: list[str] = [".venv", "venv"]
PYTHON_NAMES: list[str] = ["python3", "python"]
PYTEST_NAMES: list[str] = ["pytest"]


@dataclass
class ProjectEnvironment:
    """Discovered project environment information."""

    python: str | None
    pytest: str | None
    venv_name: str | None
    working_directory: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "python": self.python,
            "pytest": self.pytest,
            "venv_name": self.venv_name,
            "working_directory": self.working_directory,
        }


def _validate_executable(path: Path, project_root: Path) -> str | None:
    """Validate that an executable is safe to use.

    Returns the resolved absolute path as a string if valid, None otherwise.

    Checks:
    - Must exist
    - Must be a regular file (not directory, not symlink to directory)
    - Must be executable
    - Must resolve (after symlink resolution) within PROJECT_ROOT
    """
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return None

    if not resolved.is_file():
        return None

    if not os.access(resolved, os.X_OK):
        return None

    try:
        resolved.relative_to(project_root.resolve())
    except ValueError:
        return None

    return str(resolved)


def discover_project_environment(project_root: Path | None = None) -> ProjectEnvironment:
    """Discover the project's Python and pytest executables.

    Searches only within project_root/.venv/bin/ and project_root/venv/bin/.
    Never searches parent directories, user home, or system paths.
    """
    root = project_root or PROJECT_ROOT
    python_path: str | None = None
    pytest_path: str | None = None
    venv_name: str | None = None

    for venv in VENV_NAMES:
        bin_dir = root / venv / "bin"
        if not bin_dir.is_dir():
            continue

        venv_name = venv

        if python_path is None:
            for name in PYTHON_NAMES:
                candidate = bin_dir / name
                validated = _validate_executable(candidate, root)
                if validated:
                    python_path = validated
                    break

        if pytest_path is None:
            for name in PYTEST_NAMES:
                candidate = bin_dir / name
                validated = _validate_executable(candidate, root)
                if validated:
                    pytest_path = validated
                    break

        if python_path and pytest_path:
            break

    return ProjectEnvironment(
        python=python_path,
        pytest=pytest_path,
        venv_name=venv_name,
        working_directory=str(root),
    )


def resolve_executable(
    command: str,
    project_root: Path | None = None,
) -> str | None:
    """Resolve a command name to a project-local executable if available.

    For 'python', 'python3', 'pytest' — checks project venv first.
    Returns the validated path if found, None to use system default.
    """
    root = project_root or PROJECT_ROOT

    if command in ("python", "python3"):
        env = discover_project_environment(root)
        return env.python

    if command == "pytest":
        env = discover_project_environment(root)
        return env.pytest

    return None
