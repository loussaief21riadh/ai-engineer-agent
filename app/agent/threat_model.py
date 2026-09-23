"""Security Threat Model for V5 — structured threat documentation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ThreatSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ThreatCategory(str, Enum):
    PROMPT_INJECTION = "PROMPT_INJECTION"
    MALICIOUS_REPOSITORY = "MALICIOUS_REPOSITORY"
    DANGEROUS_CODE = "DANGEROUS_CODE"
    CREDENTIAL_LEAKAGE = "CREDENTIAL_LEAKAGE"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    COMMAND_INJECTION = "COMMAND_INJECTION"
    NETWORK_EXFILTRATION = "NETWORK_EXFILTRATION"
    TOOL_ABUSE = "TOOL_ABUSE"
    MEMORY_POISONING = "MEMORY_POISONING"
    CHECKPOINT_TAMPERING = "CHECKPOINT_TAMPERING"
    SCOPE_ESCALATION = "SCOPE_ESCALATION"


class Threat(BaseModel):
    """A documented threat."""
    threat_id: str
    category: ThreatCategory
    severity: ThreatSeverity
    description: str
    attack_vector: str = ""
    current_defense: str = ""
    remaining_risk: str = ""
    mitigation: str = ""
    tested: bool = False


class SecurityThreatModel:
    """Structured security threat documentation."""

    def __init__(self) -> None:
        self._threats: list[Threat] = []
        self._register_default_threats()

    def _register_default_threats(self) -> None:
        self.register(Threat(
            threat_id="T001",
            category=ThreatCategory.PROMPT_INJECTION,
            severity=ThreatSeverity.HIGH,
            description="Adversarial instructions in repository files or user input",
            attack_vector="Malicious file contents, user prompt manipulation",
            current_defense="System prompt hardening, trust levels, injection pattern detection",
            remaining_risk="Sophisticated multi-turn injection may bypass defenses",
            mitigation="Treat all file contents as untrusted data",
        ))

        self.register(Threat(
            threat_id="T002",
            category=ThreatCategory.COMMAND_INJECTION,
            severity=ThreatSeverity.CRITICAL,
            description="Arbitrary command execution via terminal tool",
            attack_vector="Shell metacharacters, command chaining",
            current_defense="Command allowlist, shell=False, metacharacter blocking",
            remaining_risk="Novel bypass techniques may exist",
            mitigation="Defense-in-depth: allowlist + shell=False + metachar blocking",
        ))

        self.register(Threat(
            threat_id="T003",
            category=ThreatCategory.PATH_TRAVERSAL,
            severity=ThreatSeverity.HIGH,
            description="Access files outside project root",
            attack_vector="../ traversal, symlinks, absolute paths",
            current_defense="safe_path() with resolve() and relative_to() check",
            remaining_risk="Race conditions with symlinks",
            mitigation="Resolve paths before comparison",
        ))

        self.register(Threat(
            threat_id="T004",
            category=ThreatCategory.CREDENTIAL_LEAKAGE,
            severity=ThreatSeverity.CRITICAL,
            description="Exposure of API keys, secrets, or credentials",
            attack_vector="Repository contents, error messages, logs",
            current_defense="Secret path protection, redaction patterns, API key never exposed",
            remaining_risk="Indirect leakage through LLM responses",
            mitigation="Redact secrets in memory and output",
        ))

        self.register(Threat(
            threat_id="T005",
            category=ThreatCategory.DANGEROUS_CODE,
            severity=ThreatSeverity.HIGH,
            description="Execution of dangerous Python code (eval, exec, subprocess)",
            attack_vector="LLM-generated code with dangerous patterns",
            current_defense="AST security analysis, CRITICAL findings block execution",
            remaining_risk="Novel dangerous patterns may not be detected",
            mitigation="AST analysis + pattern matching + execution blocking",
        ))

        self.register(Threat(
            threat_id="T006",
            category=ThreatCategory.TOOL_ABUSE,
            severity=ThreatSeverity.MEDIUM,
            description="Abuse of available tools for unintended purposes",
            attack_vector="LLM using tools in unexpected ways",
            current_defense="Tool argument validation, mode gating",
            remaining_risk="Valid arguments may still cause unintended effects",
            mitigation="Mode restrictions + argument validation",
        ))

        self.register(Threat(
            threat_id="T007",
            category=ThreatCategory.MEMORY_POISONING,
            severity=ThreatSeverity.MEDIUM,
            description="Injection of malicious content into project memory",
            attack_vector="Adaptive content stored in memory files",
            current_defense="Sanitization patterns, trust levels",
            remaining_risk="Subtle poisoning may not trigger patterns",
            mitigation="Memory entries treated as untrusted",
        ))

        self.register(Threat(
            threat_id="T008",
            category=ThreatCategory.CHECKPOINT_TAMPERING,
            severity=ThreatSeverity.HIGH,
            description="Modification of checkpoint files for privilege escalation",
            attack_vector="Direct file modification of checkpoint JSON",
            current_defense="SHA-256 checksum verification",
            remaining_risk="Checksum collision (extremely unlikely)",
            mitigation="Integrity verification on load",
        ))

        self.register(Threat(
            threat_id="T009",
            category=ThreatCategory.SCOPE_ESCALATION,
            severity=ThreatSeverity.HIGH,
            description="Agent modifying files outside allowed scope",
            attack_vector="LLM proposing changes to unrelated files",
            current_defense="Mode gating, path validation",
            remaining_risk="Subtle scope creep in multi-file changes",
            mitigation="File-level scope enforcement",
        ))

    def register(self, threat: Threat) -> None:
        self._threats.append(threat)

    def get_threat(self, threat_id: str) -> Threat | None:
        for t in self._threats:
            if t.threat_id == threat_id:
                return t
        return None

    def get_by_category(self, category: ThreatCategory) -> list[Threat]:
        return [t for t in self._threats if t.category == category]

    def get_by_severity(self, severity: ThreatSeverity) -> list[Threat]:
        return [t for t in self._threats if t.severity == severity]

    def get_untested(self) -> list[Threat]:
        return [t for t in self._threats if not t.tested]

    def mark_tested(self, threat_id: str) -> bool:
        for t in self._threats:
            if t.threat_id == threat_id:
                t.tested = True
                return True
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "total_threats": len(self._threats),
            "tested": sum(1 for t in self._threats if t.tested),
            "untested": sum(1 for t in self._threats if not t.tested),
            "by_severity": {
                s.value: sum(1 for t in self._threats if t.severity == s)
                for s in ThreatSeverity
            },
        }

    def to_list(self) -> list[dict[str, Any]]:
        return [t.model_dump() for t in self._threats]
