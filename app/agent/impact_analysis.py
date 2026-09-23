"""Impact Analysis for V4 — predicts affected files before changes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent.codebase_graph import CodebaseGraph


class ImpactReport(BaseModel):
    """Report of potential impact from modifying a file."""
    target_file: str
    direct_dependents: list[str] = Field(default_factory=list)
    transitive_dependents: list[str] = Field(default_factory=list)
    related_tests: list[str] = Field(default_factory=list)
    security_sensitive: bool = False
    risk_level: str = "LOW"
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_file": self.target_file,
            "direct_dependents": self.direct_dependents,
            "transitive_dependents": self.transitive_dependents,
            "related_tests": self.related_tests,
            "security_sensitive": self.security_sensitive,
            "risk_level": self.risk_level,
            "summary": self.summary,
        }


class ImpactAnalyzer:
    """Analyzes impact of file changes using the codebase graph."""

    SECURITY_SENSITIVE_PATTERNS = [
        "security", "auth", "crypto", "password", "token",
        "secret", "credential", "permission", "access",
    ]

    def __init__(self, graph: CodebaseGraph) -> None:
        self._graph = graph

    def analyze(self, target_file: str) -> ImpactReport:
        """Analyze impact of modifying target_file."""
        direct_deps = self._graph.get_dependents(target_file)
        transitive_deps = self._find_transitive_dependents(target_file, max_depth=3)
        related_tests = self._graph.get_test_modules_for(target_file)

        all_affected = set(direct_deps) | set(transitive_deps) | set(related_tests)
        security_sensitive = any(
            p in target_file.lower() for p in self.SECURITY_SENSITIVE_PATTERNS
        )

        risk_level = self._assess_risk(
            len(direct_deps), len(transitive_deps), len(related_tests),
            security_sensitive, target_file,
        )

        summary = self._build_summary(
            target_file, direct_deps, transitive_deps, related_tests,
            security_sensitive, risk_level,
        )

        return ImpactReport(
            target_file=target_file,
            direct_dependents=direct_deps,
            transitive_dependents=transitive_deps,
            related_tests=related_tests,
            security_sensitive=security_sensitive,
            risk_level=risk_level,
            summary=summary,
        )

    def _find_transitive_dependents(
        self, target: str, max_depth: int = 3, visited: set[str] | None = None,
    ) -> list[str]:
        if visited is None:
            visited = set()

        if max_depth <= 0 or target in visited:
            return []

        visited.add(target)
        result: list[str] = []

        for dep in self._graph.get_dependents(target):
            if dep not in visited:
                result.append(dep)
                result.extend(self._find_transitive_dependents(dep, max_depth - 1, visited))

        return list(dict.fromkeys(result))

    def _assess_risk(
        self,
        direct_count: int,
        transitive_count: int,
        test_count: int,
        security_sensitive: bool,
        target_file: str,
    ) -> str:
        if security_sensitive:
            return "HIGH"
        if direct_count > 5 or transitive_count > 10:
            return "HIGH"
        if direct_count > 2 or transitive_count > 5:
            return "MEDIUM"
        if test_count == 0:
            return "MEDIUM"
        return "LOW"

    def _build_summary(
        self,
        target: str,
        direct_deps: list[str],
        transitive_deps: list[str],
        related_tests: list[str],
        security_sensitive: bool,
        risk_level: str,
    ) -> str:
        parts = [f"Modifying {target}:"]
        parts.append(f"  Risk: {risk_level}")

        if direct_deps:
            parts.append(f"  Direct dependents: {len(direct_deps)}")
            for d in direct_deps[:5]:
                parts.append(f"    - {d}")

        if transitive_deps:
            parts.append(f"  Transitive dependents: {len(transitive_deps)}")

        if related_tests:
            parts.append(f"  Related tests: {len(related_tests)}")
            for t in related_tests[:3]:
                parts.append(f"    - {t}")
        else:
            parts.append("  WARNING: No related tests found")

        if security_sensitive:
            parts.append("  WARNING: Security-sensitive file")

        return "\n".join(parts)

    def analyze_batch(self, files: list[str]) -> list[ImpactReport]:
        """Analyze impact for multiple files."""
        return [self.analyze(f) for f in files]
