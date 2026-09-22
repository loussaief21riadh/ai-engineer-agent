import os
from pathlib import Path

import pytest


def test_agent_mode_enum_values():
    from app.config import AgentMode

    assert AgentMode.READ_ONLY.value == "READ_ONLY"
    assert AgentMode.ALLOW_EDITS.value == "ALLOW_EDITS"
    assert AgentMode.FULL_AUTONOMOUS.value == "FULL_AUTONOMOUS"


def test_project_root_is_path():
    from app.config import PROJECT_ROOT

    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.exists()


def test_defaults_when_env_vars_missing(monkeypatch):
    import importlib

    original_getenv = os.getenv

    def patched_getenv(key, default=None):
        defaults = {
            "OPENROUTER_API_KEY": "",
            "PRIMARY_MODEL": "openrouter/free",
            "REVIEWER_MODEL": "openrouter/free",
            "AGENT_MODE": "READ_ONLY",
            "MAX_AGENT_STEPS": "15",
            "COMMAND_TIMEOUT": "30",
        }
        if key in defaults:
            return defaults[key]
        return original_getenv(key, default)

    monkeypatch.setattr("os.getenv", patched_getenv)

    import app.config

    importlib.reload(app.config)

    assert app.config.OPENROUTER_API_KEY == ""
    assert app.config.PRIMARY_MODEL == "openrouter/free"
    assert app.config.REVIEWER_MODEL == "openrouter/free"
    assert app.config.AGENT_MODE.value == "READ_ONLY"
    assert app.config.MAX_AGENT_STEPS == 15
    assert app.config.COMMAND_TIMEOUT == 30


def test_config_imports_without_error():
    import app.config

    assert hasattr(app.config, "OPENROUTER_API_KEY")
    assert hasattr(app.config, "OPENROUTER_URL")
    assert hasattr(app.config, "PRIMARY_MODEL")
    assert hasattr(app.config, "REVIEWER_MODEL")
    assert hasattr(app.config, "PROJECT_ROOT")
    assert hasattr(app.config, "AGENT_MODE")
    assert hasattr(app.config, "MAX_AGENT_STEPS")
    assert hasattr(app.config, "COMMAND_TIMEOUT")
