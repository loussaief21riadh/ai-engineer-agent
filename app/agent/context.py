"""Task context engine for V2.0 — persistent context across phases."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TrustLevel(str, Enum):
    USER_ASSERTED = "USER_ASSERTED"
    TOOL_VERIFIED = "TOOL_VERIFIED"
    SYSTEM_DERIVED = "SYSTEM_DERIVED"
    MODEL_PROPOSED = "MODEL_PROPOSED"
    MODEL_INFERRED = "MODEL_INFERRED"
    FILE_CONTENT = "FILE_CONTENT"


class ContextEntry(BaseModel):
    content: str
    trust: TrustLevel
    phase: str = ""
    timestamp: float = 0.0


class TaskContext(BaseModel):
    task: str = ""
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    current_phase: str = ""
    phase_history: list[str] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    observations: list[ContextEntry] = Field(default_factory=list)
    inspected_files: dict[str, str] = Field(default_factory=dict)
    recent_executions: list[dict[str, Any]] = Field(default_factory=list)
    test_results: dict[str, Any] | None = None
    failures: list[str] = Field(default_factory=list)
    diagnoses: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)
    review_feedback: str = ""
    decisions: list[str] = Field(default_factory=list)
    iteration_count: int = 0
    retry_count: int = 0
    change_decision: str = ""
    assessment_retry_count: int = 0

    MAX_INSPECTED_FILES: int = 30
    MAX_OBSERVATIONS: int = 50
    MAX_DIAGNOSES: int = 15
    MAX_FIXES: int = 15
    MAX_DECISIONS: int = 20

    def add_observation(self, content: str, trust: TrustLevel, phase: str = "") -> None:
        self.observations.append(ContextEntry(
            content=content,
            trust=trust,
            phase=phase or self.current_phase,
        ))
        if len(self.observations) > self.MAX_OBSERVATIONS:
            self.observations = self.observations[-self.MAX_OBSERVATIONS:]

    def record_inspected_file(self, path: str, summary: str) -> None:
        self.inspected_files[path] = summary
        while len(self.inspected_files) > self.MAX_INSPECTED_FILES:
            oldest_key = next(iter(self.inspected_files))
            del self.inspected_files[oldest_key]

    def record_execution(self, execution: dict[str, Any]) -> None:
        self.recent_executions.append(execution)
        if len(self.recent_executions) > 20:
            self.recent_executions = self.recent_executions[-20:]

    def record_failure(self, failure: str) -> None:
        self.failures.append(failure)
        if len(self.failures) > 10:
            self.failures = self.failures[-10:]

    def record_diagnosis(self, diagnosis: str) -> None:
        self.diagnoses.append(diagnosis)
        if len(self.diagnoses) > self.MAX_DIAGNOSES:
            self.diagnoses = self.diagnoses[-self.MAX_DIAGNOSES:]

    def record_fix(self, fix: str) -> None:
        self.fixes.append(fix)
        if len(self.fixes) > self.MAX_FIXES:
            self.fixes = self.fixes[-self.MAX_FIXES:]

    def set_plan(self, plan: dict[str, Any]) -> None:
        self.plan = plan

    def record_decision(self, decision: str) -> None:
        self.decisions.append(decision)
        if len(self.decisions) > self.MAX_DECISIONS:
            self.decisions = self.decisions[-self.MAX_DECISIONS:]

    def transition_to(self, phase: str) -> None:
        if self.current_phase:
            self.phase_history.append(self.current_phase)
        self.current_phase = phase


class ContextBuilder:
    MAX_OBSERVATIONS = 30
    MAX_INSPECTED_FILES = 15
    MAX_RECENT_EXECUTIONS = 10
    MAX_FAILURES = 5
    MAX_DIAGNOSES = 5

    def build_phase_prompt(self, ctx: TaskContext, phase: str, task: str, memory_context: str = "") -> str:
        parts: list[str] = []

        if memory_context and memory_context != "No project memory available.":
            parts.append(memory_context)
            parts.append("")

        parts.append(f"Task: {task}")
        parts.append(f"Phase: {phase}")

        if ctx.plan:
            plan_summary = self._summarize_plan(ctx.plan)
            parts.append(f"\nCurrent plan:\n{plan_summary}")

        if ctx.requirements:
            parts.append(f"\nRequirements:\n" + "\n".join(f"- {r}" for r in ctx.requirements))

        if ctx.constraints:
            parts.append(f"\nConstraints:\n" + "\n".join(f"- {c}" for c in ctx.constraints))

        if ctx.inspected_files:
            files_section = self._summarize_inspected_files(ctx.inspected_files)
            parts.append(f"\nInspected files:\n{files_section}")

        recent = ctx.recent_executions[-self.MAX_RECENT_EXECUTIONS:]
        if recent:
            exec_lines = []
            for ex in recent:
                status = "OK" if ex.get("success") else "FAIL"
                name = ex.get("tool_name", "?")
                exec_lines.append(f"  {name} -> {status}")
            parts.append(f"\nRecent actions:\n" + "\n".join(exec_lines))

        if ctx.test_results:
            parts.append(f"\nLast test results:\n{self._summarize_test_results(ctx.test_results)}")

        if ctx.failures:
            recent_failures = ctx.failures[-self.MAX_FAILURES:]
            parts.append(f"\nRecent failures:\n" + "\n".join(f"- {f}" for f in recent_failures))

        if ctx.observations:
            recent_obs = ctx.observations[-self.MAX_OBSERVATIONS:]
            obs_lines = []
            for obs in recent_obs:
                trust_tag = f"[{obs.trust.value}]" if obs.trust else ""
                obs_lines.append(f"  {trust_tag} {obs.content}")
            parts.append(f"\nKey observations:\n" + "\n".join(obs_lines))

        if ctx.diagnoses:
            recent_diagnoses = ctx.diagnoses[-self.MAX_DIAGNOSES:]
            parts.append(f"\nPrevious diagnoses:\n" + "\n".join(f"- {d}" for d in recent_diagnoses))

        if ctx.fixes:
            parts.append(f"\nPrevious fixes:\n" + "\n".join(f"- {f}" for f in ctx.fixes[-3:]))

        if ctx.review_feedback:
            parts.append(f"\nReview feedback:\n{ctx.review_feedback}")

        if ctx.decisions:
            parts.append(f"\nDecisions made:\n" + "\n".join(f"- {d}" for d in ctx.decisions[-5:]))

        parts.append(f"\nIteration: {ctx.iteration_count}, Retries: {ctx.retry_count}")

        return "\n".join(parts)

    def build_review_context(self, ctx: TaskContext, task: str, changes: str, diff: str, test_results: str) -> str:
        parts: list[str] = []
        parts.append(f"Original task: {task}")

        if ctx.plan:
            parts.append(f"\nPlan:\n{self._summarize_plan(ctx.plan)}")

        if ctx.requirements:
            parts.append(f"\nRequirements:\n" + "\n".join(f"- {r}" for r in ctx.requirements))

        if changes:
            parts.append(f"\nChanges made:\n{changes}")

        if diff:
            parts.append(f"\nGit diff:\n{diff}")

        if test_results:
            parts.append(f"\nTest results:\n{test_results}")

        if ctx.observations:
            evidence_lines = []
            for obs in ctx.observations[-15:]:
                trust_tag = f"[{obs.trust.value}]" if obs.trust else "[UNKNOWN]"
                evidence_lines.append(f"  {trust_tag} {obs.content}")
            parts.append(f"\nExecution evidence (with provenance):\n" + "\n".join(evidence_lines))

        if ctx.diagnoses:
            parts.append(f"\nDiagnoses:\n" + "\n".join(f"- {d}" for d in ctx.diagnoses[-3:]))

        if ctx.fixes:
            parts.append(f"\nFixes applied:\n" + "\n".join(f"- {f}" for f in ctx.fixes[-3:]))

        return "\n".join(parts)

    def _summarize_plan(self, plan: dict[str, Any]) -> str:
        lines: list[str] = []
        objective = plan.get("objective", "")
        if objective:
            lines.append(f"Objective: {objective}")

        subtasks = plan.get("subtasks", [])
        if subtasks:
            lines.append("Subtasks:")
            for st in subtasks:
                st_id = st.get("id", "?")
                desc = st.get("description", "")
                status = st.get("status", "PENDING")
                lines.append(f"  [{status}] {st_id}: {desc}")

        return "\n".join(lines) if lines else "No plan available."

    def _summarize_inspected_files(self, files: dict[str, str]) -> str:
        entries = list(files.items())[-self.MAX_INSPECTED_FILES:]
        lines: list[str] = []
        for path, summary in entries:
            preview = summary[:200] + "..." if len(summary) > 200 else summary
            lines.append(f"  {path}: {preview}")
        return "\n".join(lines)

    def _summarize_test_results(self, results: dict[str, Any]) -> str:
        exit_code = results.get("exit_code", "?")
        success = results.get("success", False)
        stdout = results.get("stdout", "")
        status = "PASS" if success else "FAIL"
        lines = [f"Status: {status} (exit code {exit_code})"]
        if stdout:
            last_lines = stdout.strip().split("\n")[-5:]
            lines.append("Last output:")
            for line in last_lines:
                lines.append(f"  {line}")
        return "\n".join(lines)
