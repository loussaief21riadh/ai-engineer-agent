from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ASTSecurityFinding:
    severity: str
    category: str
    description: str
    line: int | None = None
    node_type: str = ""


class ASTSecurityAnalyzer:
    """AST-based security analysis for Python files."""

    NETWORK_MODULES = {"requests", "urllib", "httpx", "aiohttp", "httplib2"}
    DANGEROUS_FUNCTIONS = {"eval", "exec", "compile"}
    OS_EXEC_FUNCTIONS = {"system", "popen", "execle", "execl", "execvp", "execvpe"}
    SUBPROCESS_CALLS = {"run", "Popen", "call", "check_call", "check_output"}

    def analyze(self, source: str, filename: str = "<string>") -> list[ASTSecurityFinding]:
        findings: list[ASTSecurityFinding] = []
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            return findings

        for node in ast.walk(tree):
            findings.extend(self._check_import(node))
            findings.extend(self._check_function_call(node))
            findings.extend(self._check_dynamic_import(node))
            findings.extend(self._check_open_call(node))

        return findings

    def _check_import(self, node: ast.AST) -> list[ASTSecurityFinding]:
        findings: list[ASTSecurityFinding] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in self.NETWORK_MODULES:
                    findings.append(ASTSecurityFinding(
                        severity="CRITICAL",
                        category="network_import",
                        description=f"Network module imported: {alias.name}",
                        line=node.lineno,
                        node_type="Import",
                    ))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module in self.NETWORK_MODULES:
                    findings.append(ASTSecurityFinding(
                        severity="CRITICAL",
                        category="network_import",
                        description=f"Network module imported from: {node.module}",
                        line=node.lineno,
                        node_type="ImportFrom",
                    ))
        return findings

    def _check_function_call(self, node: ast.AST) -> list[ASTSecurityFinding]:
        findings: list[ASTSecurityFinding] = []
        if not isinstance(node, ast.Call):
            return findings

        func_name = self._get_call_name(node)
        if not func_name:
            return findings

        if func_name in self.DANGEROUS_FUNCTIONS:
            findings.append(ASTSecurityFinding(
                severity="CRITICAL",
                category="dangerous_function",
                description=f"Dangerous function called: {func_name}()",
                line=node.lineno,
                node_type="Call",
            ))

        if func_name in ("os.system", "os.popen"):
            findings.append(ASTSecurityFinding(
                severity="CRITICAL",
                category="os_execution",
                description=f"OS command execution: {func_name}()",
                line=node.lineno,
                node_type="Call",
            ))

        if func_name.startswith("subprocess."):
            sub_method = func_name.split(".")[-1]
            if sub_method in self.SUBPROCESS_CALLS:
                findings.append(ASTSecurityFinding(
                    severity="HIGH",
                    category="subprocess_execution",
                    description=f"Subprocess call: {func_name}()",
                    line=node.lineno,
                    node_type="Call",
                ))

        return findings

    def _check_dynamic_import(self, node: ast.AST) -> list[ASTSecurityFinding]:
        findings: list[ASTSecurityFinding] = []
        if not isinstance(node, ast.Call):
            return findings

        func_name = self._get_call_name(node)
        if func_name == "__import__":
            findings.append(ASTSecurityFinding(
                severity="CRITICAL",
                category="dynamic_import",
                description="Dynamic import via __import__()",
                line=node.lineno,
                node_type="Call",
            ))
        return findings

    def _check_open_call(self, node: ast.AST) -> list[ASTSecurityFinding]:
        findings: list[ASTSecurityFinding] = []
        if not isinstance(node, ast.Call):
            return findings

        func_name = self._get_call_name(node)
        if func_name == "open" and node.args:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    val = arg.value.lower()
                    if any(sus in val for sus in ["/etc/passwd", "/etc/shadow", "/proc/", "/.ssh/"]):
                        findings.append(ASTSecurityFinding(
                            severity="CRITICAL",
                            category="sensitive_file_access",
                            description=f"Sensitive file access: open('{arg.value}')",
                            line=node.lineno,
                            node_type="Call",
                        ))
        return findings

    def _get_call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts = []
            current: ast.expr = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""


def analyze_python_file(filepath: str) -> list[ASTSecurityFinding]:
    try:
        with open(filepath) as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []
    return ASTSecurityAnalyzer().analyze(source, filename=filepath)
