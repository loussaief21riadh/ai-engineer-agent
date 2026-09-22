"""Tests for project environment discovery."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.tools.env import (
    ProjectEnvironment,
    _validate_executable,
    discover_project_environment,
    resolve_executable,
)


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    """Create a minimal fake project with a venv structure."""
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "app").mkdir()
    (project / "tests").mkdir()

    venv_bin = project / ".venv" / "bin"
    venv_bin.mkdir(parents=True)

    python = venv_bin / "python3"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)

    pytest_exe = venv_bin / "pytest"
    pytest_exe.write_text("#!/bin/sh\n")
    pytest_exe.chmod(0o755)

    return project


@pytest.fixture
def no_venv_project(tmp_path: Path) -> Path:
    """Create a project with no venv."""
    project = tmp_path / "novenv"
    project.mkdir()
    (project / "app").mkdir()
    return project


@pytest.fixture
def alt_venv_project(tmp_path: Path) -> Path:
    """Create a project with 'venv' (not '.venv') directory."""
    project = tmp_path / "altproject"
    project.mkdir()

    venv_bin = project / "venv" / "bin"
    venv_bin.mkdir(parents=True)

    python = venv_bin / "python3"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)

    pytest_exe = venv_bin / "pytest"
    pytest_exe.write_text("#!/bin/sh\n")
    pytest_exe.chmod(0o755)

    return project


# ---------------------------------------------------------------------------
# _validate_executable
# ---------------------------------------------------------------------------

class TestValidateExecutable:
    def test_valid_executable(self, fake_project: Path) -> None:
        exe = fake_project / ".venv" / "bin" / "python3"
        result = _validate_executable(exe, fake_project)
        assert result is not None
        assert result == str(exe.resolve())

    def test_nonexistent_file(self, fake_project: Path) -> None:
        exe = fake_project / ".venv" / "bin" / "nonexistent"
        result = _validate_executable(exe, fake_project)
        assert result is None

    def test_directory_rejected(self, fake_project: Path) -> None:
        result = _validate_executable(fake_project / ".venv" / "bin", fake_project)
        assert result is None

    def test_non_executable_rejected(self, fake_project: Path) -> None:
        not_exec = fake_project / ".venv" / "bin" / "notexec"
        not_exec.write_text("not executable")
        result = _validate_executable(not_exec, fake_project)
        assert result is None

    def test_outside_project_root_rejected(self, fake_project: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.py"
        outside.write_text("#!/bin/sh\n")
        outside.chmod(0o755)
        result = _validate_executable(outside, fake_project)
        assert result is None

    def test_symlink_outside_project_rejected(self, fake_project: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.py"
        outside.write_text("#!/bin/sh\n")
        outside.chmod(0o755)
        link = fake_project / ".venv" / "bin" / "link"
        link.symlink_to(outside)
        result = _validate_executable(link, fake_project)
        assert result is None


# ---------------------------------------------------------------------------
# discover_project_environment
# ---------------------------------------------------------------------------

class TestDiscoverProjectEnvironment:
    def test_discovers_dotvenv(self, fake_project: Path) -> None:
        env = discover_project_environment(fake_project)
        assert env.python is not None
        assert "python3" in env.python
        assert env.pytest is not None
        assert "pytest" in env.pytest
        assert env.venv_name == ".venv"
        assert env.working_directory == str(fake_project)

    def test_discovers_venv(self, alt_venv_project: Path) -> None:
        env = discover_project_environment(alt_venv_project)
        assert env.python is not None
        assert "python3" in env.python
        assert env.pytest is not None
        assert env.venv_name == "venv"

    def test_no_venv_returns_none(self, no_venv_project: Path) -> None:
        env = discover_project_environment(no_venv_project)
        assert env.python is None
        assert env.pytest is None
        assert env.venv_name is None

    def test_dotvenv_preferred_over_venv(self, tmp_path: Path) -> None:
        project = tmp_path / "both"
        project.mkdir()

        dotvenv_bin = project / ".venv" / "bin"
        dotvenv_bin.mkdir(parents=True)
        py1 = dotvenv_bin / "python3"
        py1.write_text("#!/bin/sh\n")
        py1.chmod(0o755)
        pt1 = dotvenv_bin / "pytest"
        pt1.write_text("#!/bin/sh\n")
        pt1.chmod(0o755)

        venv_bin = project / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        py2 = venv_bin / "python3"
        py2.write_text("#!/bin/sh\n")
        py2.chmod(0o755)

        env = discover_project_environment(project)
        assert env.venv_name == ".venv"
        assert env.python is not None
        assert ".venv" in env.python

    def test_to_dict(self, fake_project: Path) -> None:
        env = discover_project_environment(fake_project)
        d = env.to_dict()
        assert "python" in d
        assert "pytest" in d
        assert "venv_name" in d
        assert "working_directory" in d


# ---------------------------------------------------------------------------
# resolve_executable
# ---------------------------------------------------------------------------

class TestResolveExecutable:
    def test_python_resolves_to_venv(self, fake_project: Path) -> None:
        result = resolve_executable("python", fake_project)
        assert result is not None
        assert "python3" in result

    def test_python3_resolves_to_venv(self, fake_project: Path) -> None:
        result = resolve_executable("python3", fake_project)
        assert result is not None
        assert "python3" in result

    def test_pytest_resolves_to_venv(self, fake_project: Path) -> None:
        result = resolve_executable("pytest", fake_project)
        assert result is not None
        assert "pytest" in result

    def test_no_venv_returns_none(self, no_venv_project: Path) -> None:
        result = resolve_executable("python", no_venv_project)
        assert result is None

    def test_unknown_command_returns_none(self, fake_project: Path) -> None:
        result = resolve_executable("ls", fake_project)
        assert result is None

    def test_ls_returns_none(self, fake_project: Path) -> None:
        result = resolve_executable("ls", fake_project)
        assert result is None
