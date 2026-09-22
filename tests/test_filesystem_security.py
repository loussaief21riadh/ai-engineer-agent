"""Security regression tests for filesystem access controls.

Covers: secret file protection, path traversal, and normal operation.
All tests monkeypatch PROJECT_ROOT into a tmp_path so nothing touches the real FS.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import app.config as config_mod
import app.tools.filesystem as fs_mod
from app.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool


@pytest.fixture(autouse=True)
def _patch_project_root(tmp_path, monkeypatch):
    """Redirect every reference to PROJECT_ROOT into the temp directory."""
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fs_mod, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def root(_patch_project_root):
    return _patch_project_root


# ── Secret file protection ────────────────────────────────────────────────


class TestSecretFileProtection:
    def test_env_file_cannot_be_read(self, root):
        (root / ".env").write_text("SECRET=x")
        result = ReadFileTool().execute(path=".env")
        assert result["success"] is False
        assert "secret" in result["error"].lower() or "denied" in result["error"].lower()

    def test_env_file_hidden_from_list(self, root):
        (root / ".env").write_text("SECRET=x")
        (root / "normal.txt").write_text("ok")
        result = ListFilesTool().execute(path=".")
        assert result["success"] is True
        names = [e["name"] for e in result["result"]]
        assert ".env" not in names
        assert "normal.txt" in names

    def test_nested_env_file_blocked(self, root):
        d = root / "config"
        d.mkdir()
        (d / ".env").write_text("DB_PASS=abc")
        result = ReadFileTool().execute(path="config/.env")
        assert result["success"] is False

    def test_key_file_blocked(self, root):
        d = root / "secrets"
        d.mkdir()
        (d / "api.key").write_text("sk-live-xxx")
        result = ReadFileTool().execute(path="secrets/api.key")
        assert result["success"] is False

    def test_pem_file_blocked(self, root):
        d = root / "keys"
        d.mkdir()
        (d / "private.pem").write_text("-----BEGIN RSA PRIVATE KEY-----")
        result = ReadFileTool().execute(path="keys/private.pem")
        assert result["success"] is False

    def test_crt_file_blocked(self, root):
        d = root / "certs"
        d.mkdir()
        (d / "ca.crt").write_text("-----BEGIN CERTIFICATE-----")
        result = ReadFileTool().execute(path="certs/ca.crt")
        assert result["success"] is False

    def test_env_star_pattern_blocked(self, root):
        (root / ".env.production").write_text("PROD_SECRET=1")
        result = ReadFileTool().execute(path=".env.production")
        assert result["success"] is False

    def test_id_rsa_blocked(self, root):
        (root / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----")
        result = ReadFileTool().execute(path="id_rsa")
        assert result["success"] is False

    def test_credentials_file_blocked(self, root):
        (root / "credentials").write_text("user:pass")
        result = ReadFileTool().execute(path="credentials")
        assert result["success"] is False

    def test_list_files_hides_pem_files(self, root):
        (root / "test.pem").write_text("cert data")
        (root / "readme.md").write_text("docs")
        result = ListFilesTool().execute(path=".")
        names = [e["name"] for e in result["result"]]
        assert "test.pem" not in names
        assert "readme.md" in names

    def test_list_files_hides_ssh_dir(self, root):
        ssh = root / ".ssh"
        ssh.mkdir()
        (ssh / "authorized_keys").write_text("ssh-rsa AAAA...")
        (ssh / "config").write_text("Host *")
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("print()")
        result = ListFilesTool().execute(path=".")
        names = [e["name"] for e in result["result"]]
        assert ".ssh" not in names
        assert "src" in names

    def test_write_file_rejects_env(self, root):
        result = WriteFileTool().execute(path=".env", content="SECRET=leaked")
        assert result["success"] is False

    def test_write_file_rejects_pem(self, root):
        result = WriteFileTool().execute(path="key.pem", content="-----BEGIN KEY-----")
        assert result["success"] is False


# ── Path traversal ─────────────────────────────────────────────────────────


class TestPathTraversal:
    def test_read_file_path_traversal(self, root):
        result = ReadFileTool().execute(path="../../etc/passwd")
        assert result["success"] is False
        assert "outside project root" in result["error"]

    def test_write_file_path_traversal(self, root):
        result = WriteFileTool().execute(path="../../etc/cron", content="evil")
        assert result["success"] is False
        assert "outside project root" in result["error"]

    def test_absolute_path_outside_root(self, root):
        result = ReadFileTool().execute(path="/etc/passwd")
        assert result["success"] is False
        assert "outside project root" in result["error"]

    def test_symlink_to_outside_blocked(self, root):
        outside = root.parent / "outside_secret.txt"
        outside.write_text("leaked data")
        link = root / "sneaky_link"
        os.symlink(outside, link)
        result = ReadFileTool().execute(path="sneaky_link")
        assert result["success"] is False
        assert "secret" in result["error"].lower() or "denied" in result["error"].lower()


# ── Normal operation ───────────────────────────────────────────────────────


class TestNormalOperation:
    def test_normal_file_read_works(self, root):
        (root / "data.txt").write_text("hello world", encoding="utf-8")
        result = ReadFileTool().execute(path="data.txt")
        assert result["success"] is True
        assert result["result"] == "hello world"

    def test_normal_file_write_works(self, root):
        result = WriteFileTool().execute(path="output.json", content='{"key": 1}')
        assert result["success"] is True
        assert (root / "output.json").read_text(encoding="utf-8") == '{"key": 1}'

    def test_normal_list_works(self, root):
        (root / "a.txt").write_text("a")
        (root / "b.txt").write_text("b")
        (root / "subdir").mkdir()
        result = ListFilesTool().execute(path=".")
        assert result["success"] is True
        names = sorted(e["name"] for e in result["result"])
        assert names == ["a.txt", "b.txt", "subdir"]
