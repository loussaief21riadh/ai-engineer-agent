"""Centralized security policies for path, secret, and command protection."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path


SECRET_PATH_PATTERNS: list[str] = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.crt",
    "*.p12",
    "*.pfx",
    "*.jks",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "credentials",
    "credentials.json",
    "service-account*.json",
    "*.keystore",
]

SECRET_DIR_NAMES: set[str] = {
    ".ssh",
    ".gnupg",
    ".aws",
}

_SHELL_META_RE = re.compile(r";|&&|\|\||`|\$\(|>>|>|<|\||\n|\r")


def is_secret_path(target: Path, project_root: Path) -> bool:
    try:
        rel = target.relative_to(project_root)
    except ValueError:
        return True

    parts = rel.parts

    for part in parts:
        if part in SECRET_DIR_NAMES:
            return True

    name = rel.name

    for pattern in SECRET_PATH_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True

    return False


def contains_shell_metacharacters(command: str) -> bool:
    return bool(_SHELL_META_RE.search(command))


def safe_path(path: str, project_root: Path | None = None) -> Path:
    """Resolve a path and ensure it stays within project root and is not a secret."""
    from app.config import PROJECT_ROOT
    root = project_root if project_root is not None else PROJECT_ROOT
    requested = Path(path)

    if requested.is_absolute():
        target = requested.resolve()
    else:
        target = (root / requested).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise PermissionError(
            f"Access denied: '{path}' resolves outside project root."
        )

    if is_secret_path(target, root):
        raise PermissionError(
            "Access denied: file is a protected secret."
        )

    return target
