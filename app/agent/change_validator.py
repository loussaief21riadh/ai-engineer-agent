"""Code Change Validator for V4 — validates changes before acceptance."""

from __future__ import annotations

import ast
from typing import Any

from pydantic import BaseModel, Field

from app.tools.security import SECRET_PATH_PATTERNS, is_secret_path
from pathlib import Path


class ChangeViolation(BaseModel):
    """A violation detected in code changes."""
    violation_type: str
    severity: str
    description: str
    file: str = ""
    line: int | None = None


class ChangeValidationResult(BaseModel):
    """Result of change validation."""
    passed: bool = True
    violations: list[ChangeViolation] = Field(default_factory=list)
    files_checked: int = 0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [v.model_dump() for v in self.violations],
            "files_checked": self.files_checked,
            "summary": self.summary,
        }


class ChangeValidator:
    """Validates code changes for safety and correctness."""

    DANGEROUS_IMPORTS = {
        "subprocess", "os", "shutil", "socket", "http",
        "urllib", "requests", "ctypes", "importlib",
    }

    DANGEROUS_PATTERNS = [
        "eval(", "exec(", "compile(", "__import__(",
        "os.system(", "os.popen(", "subprocess.run(",
        "subprocess.Popen(", "subprocess.call(",
    ]

    def __init__(self, project_root: str | Path = "") -> None:
        self.project_root = Path(project_root) if project_root else Path.cwd()

    def validate(
        self,
        files_modified: list[str],
        allowed_paths: list[str] | None = None,
        check_syntax: bool = True,
        check_security: bool = True,
        check_scope: bool = True,
    ) -> ChangeValidationResult:
        """Validate a set of file changes."""
        violations: list[ChangeViolation] = []

        for filepath in files_modified:
            if check_scope:
                violations.extend(self._check_scope(filepath, allowed_paths))

            if check_security:
                violations.extend(self._check_secret_path(filepath))

            if check_syntax and filepath.endswith(".py"):
                violations.extend(self._check_syntax(filepath))

            if check_security and filepath.endswith(".py"):
                violations.extend(self._check_dangerous_code(filepath))

        passed = all(v.severity != "CRITICAL" for v in violations)
        summary = self._build_summary(files_modified, violations, passed)

        return ChangeValidationResult(
            passed=passed,
            violations=violations,
            files_checked=len(files_modified),
            summary=summary,
        )

    def _check_scope(
        self, filepath: str, allowed_paths: list[str] | None,
    ) -> list[ChangeViolation]:
        violations: list[ChangeViolation] = []

        if allowed_paths:
            in_scope = any(filepath.startswith(p) for p in allowed_paths)
            if not in_scope:
                violations.append(ChangeViolation(
                    violation_type="SCOPE_VIOLATION",
                    severity="HIGH",
                    description=f"File {filepath} is outside allowed scope",
                    file=filepath,
                ))

        try:
            full_path = (self.project_root / filepath).resolve()
            if not str(full_path).startswith(str(self.project_root.resolve())):
                violations.append(ChangeViolation(
                    violation_type="PATH_TRAVERSAL",
                    severity="CRITICAL",
                    description=f"File {filepath} resolves outside project root",
                    file=filepath,
                ))
        except (OSError, ValueError):
            pass

        return violations

    def _check_secret_path(self, filepath: str) -> list[ChangeViolation]:
        violations: list[ChangeViolation] = []
        try:
            target = (self.project_root / filepath).resolve()
            if is_secret_path(target, self.project_root):
                violations.append(ChangeViolation(
                    violation_type="SECRET_PATH",
                    severity="CRITICAL",
                    description=f"File {filepath} is a protected secret path",
                    file=filepath,
                ))
        except (OSError, ValueError):
            pass
        return violations

    def _check_syntax(self, filepath: str) -> list[ChangeViolation]:
        violations: list[ChangeViolation] = []
        try:
            full_path = self.project_root / filepath
            source = full_path.read_text()
            ast.parse(source, filename=filepath)
        except SyntaxError as e:
            violations.append(ChangeViolation(
                violation_type="SYNTAX_ERROR",
                severity="CRITICAL",
                description=f"Syntax error in {filepath}: {e.msg}",
                file=filepath,
                line=e.lineno,
            ))
        except (OSError, UnicodeDecodeError):
            pass
        return violations

    def _check_dangerous_code(self, filepath: str) -> list[ChangeViolation]:
        violations: list[ChangeViolation] = []
        try:
            full_path = self.project_root / filepath
            source = full_path.read_text()
            tree = ast.parse(source, filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module in self.DANGEROUS_IMPORTS:
                            violations.append(ChangeViolation(
                                violation_type="DANGEROUS_IMPORT",
                                severity="MEDIUM",
                                description=f"Potentially dangerous import: {alias.name}",
                                file=filepath,
                                line=getattr(node, "lineno", None),
                            ))

                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split(".")[0]
                    if module in self.DANGEROUS_IMPORTS:
                        violations.append(ChangeViolation(
                            violation_type="DANGEROUS_IMPORT",
                            severity="MEDIUM",
                            description=f"Potentially dangerous import from: {node.module}",
                            file=filepath,
                            line=getattr(node, "lineno", None),
                        ))

        except (SyntaxError, OSError, UnicodeDecodeError):
            pass
        return violations

    def _build_summary(
        self,
        files: list[str],
        violations: list[ChangeViolation],
        passed: bool,
    ) -> str:
        parts = [f"Checked {len(files)} files"]
        if violations:
            critical = sum(1 for v in violations if v.severity == "CRITICAL")
            high = sum(1 for v in violations if v.severity == "HIGH")
            medium = sum(1 for v in violations if v.severity == "MEDIUM")
            parts.append(f"Found {len(violations)} violations (CRITICAL={critical}, HIGH={high}, MEDIUM={medium})")
        else:
            parts.append("No violations found")
        parts.append(f"Result: {'PASS' if passed else 'FAIL'}")
        return "\n".join(parts)
