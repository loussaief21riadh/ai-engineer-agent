"""Shared test fixtures for environment discovery tests."""

from __future__ import annotations

from pathlib import Path

import pytest


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
