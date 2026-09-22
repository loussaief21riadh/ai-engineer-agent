"""Tests for V2.0-A Context Engine."""

from __future__ import annotations

import pytest

from app.agent.context import ContextBuilder, ContextEntry, TaskContext, TrustLevel


class TestTrustLevels:
    def test_all_trust_levels_defined(self):
        levels = [t.value for t in TrustLevel]
        assert "USER_ASSERTED" in levels
        assert "TOOL_VERIFIED" in levels
        assert "SYSTEM_DERIVED" in levels
        assert "MODEL_PROPOSED" in levels
        assert "MODEL_INFERRED" in levels

    def test_trust_level_count(self):
        assert len(TrustLevel) == 5


class TestContextEntry:
    def test_entry_creation(self):
        entry = ContextEntry(content="test content", trust=TrustLevel.TOOL_VERIFIED)
        assert entry.content == "test content"
        assert entry.trust == TrustLevel.TOOL_VERIFIED

    def test_entry_with_phase(self):
        entry = ContextEntry(content="obs", trust=TrustLevel.MODEL_PROPOSED, phase="INSPECT")
        assert entry.phase == "INSPECT"


class TestTaskContext:
    def test_default_context(self):
        ctx = TaskContext()
        assert ctx.task == ""
        assert ctx.requirements == []
        assert ctx.current_phase == ""
        assert ctx.observations == []
        assert ctx.inspected_files == {}
        assert ctx.recent_executions == []
        assert ctx.test_results is None
        assert ctx.failures == []
        assert ctx.diagnoses == []
        assert ctx.fixes == []
        assert ctx.review_feedback == ""
        assert ctx.decisions == []
        assert ctx.iteration_count == 0
        assert ctx.retry_count == 0

    def test_add_observation(self):
        ctx = TaskContext(task="test task")
        ctx.add_observation("found bug in main.py", trust=TrustLevel.TOOL_VERIFIED, phase="INSPECT")
        assert len(ctx.observations) == 1
        assert ctx.observations[0].content == "found bug in main.py"
        assert ctx.observations[0].trust == TrustLevel.TOOL_VERIFIED
        assert ctx.observations[0].phase == "INSPECT"

    def test_add_observation_uses_current_phase(self):
        ctx = TaskContext(task="test", current_phase="PLAN")
        ctx.add_observation("planning done", trust=TrustLevel.MODEL_PROPOSED)
        assert ctx.observations[0].phase == "PLAN"

    def test_record_inspected_file(self):
        ctx = TaskContext(task="test")
        ctx.record_inspected_file("app/main.py", "Main entry point")
        assert "app/main.py" in ctx.inspected_files
        assert ctx.inspected_files["app/main.py"] == "Main entry point"

    def test_record_execution(self):
        ctx = TaskContext(task="test")
        ex = {"tool_name": "read_file", "success": True, "path": "a.py"}
        ctx.record_execution(ex)
        assert len(ctx.recent_executions) == 1
        assert ctx.recent_executions[0]["tool_name"] == "read_file"

    def test_record_execution_limits_to_20(self):
        ctx = TaskContext(task="test")
        for i in range(25):
            ctx.record_execution({"tool_name": f"tool_{i}", "success": True})
        assert len(ctx.recent_executions) == 20
        assert ctx.recent_executions[0]["tool_name"] == "tool_5"

    def test_record_failure(self):
        ctx = TaskContext(task="test")
        ctx.record_failure("test_x failed")
        assert "test_x failed" in ctx.failures

    def test_record_failure_limits_to_10(self):
        ctx = TaskContext(task="test")
        for i in range(15):
            ctx.record_failure(f"failure_{i}")
        assert len(ctx.failures) == 10
        assert ctx.failures[0] == "failure_5"

    def test_record_diagnosis(self):
        ctx = TaskContext(task="test")
        ctx.record_diagnosis("root cause: missing import")
        assert "root cause: missing import" in ctx.diagnoses

    def test_record_fix(self):
        ctx = TaskContext(task="test")
        ctx.record_fix("added import statement")
        assert "added import statement" in ctx.fixes

    def test_set_plan(self):
        ctx = TaskContext(task="test")
        plan = {"objective": "fix bug", "subtasks": []}
        ctx.set_plan(plan)
        assert ctx.plan == plan

    def test_transition_to(self):
        ctx = TaskContext(task="test")
        ctx.transition_to("UNDERSTAND")
        assert ctx.current_phase == "UNDERSTAND"
        assert ctx.phase_history == []

    def test_transition_to_records_history(self):
        ctx = TaskContext(task="test", current_phase="UNDERSTAND")
        ctx.transition_to("PLAN")
        assert ctx.current_phase == "PLAN"
        assert "UNDERSTAND" in ctx.phase_history

    def test_multiple_transitions(self):
        ctx = TaskContext(task="test")
        for phase in ["UNDERSTAND", "PLAN", "INSPECT", "IMPLEMENT"]:
            ctx.transition_to(phase)
        assert ctx.current_phase == "IMPLEMENT"
        assert ctx.phase_history == ["UNDERSTAND", "PLAN", "INSPECT"]


class TestContextBuilder:
    def setup_method(self):
        self.builder = ContextBuilder()

    def test_build_phase_prompt_minimal(self):
        ctx = TaskContext(task="fix bug")
        prompt = self.builder.build_phase_prompt(ctx, "UNDERSTAND", "fix bug")
        assert "Task: fix bug" in prompt
        assert "Phase: UNDERSTAND" in prompt

    def test_build_phase_prompt_with_plan(self):
        ctx = TaskContext(task="fix bug")
        ctx.set_plan({
            "objective": "fix bug",
            "subtasks": [
                {"id": "1", "description": "inspect files", "status": "COMPLETED"},
                {"id": "2", "description": "fix code", "status": "PENDING"},
            ],
        })
        prompt = self.builder.build_phase_prompt(ctx, "IMPLEMENT", "fix bug")
        assert "fix bug" in prompt
        assert "COMPLETED" in prompt
        assert "PENDING" in prompt

    def test_build_phase_prompt_with_requirements(self):
        ctx = TaskContext(task="add feature")
        ctx.requirements = ["must be backward compatible", "must have tests"]
        prompt = self.builder.build_phase_prompt(ctx, "PLAN", "add feature")
        assert "must be backward compatible" in prompt
        assert "must have tests" in prompt

    def test_build_phase_prompt_with_constraints(self):
        ctx = TaskContext(task="add feature")
        ctx.constraints = ["no new dependencies", "max 100 lines"]
        prompt = self.builder.build_phase_prompt(ctx, "PLAN", "add feature")
        assert "no new dependencies" in prompt

    def test_build_phase_prompt_with_inspected_files(self):
        ctx = TaskContext(task="fix bug")
        ctx.record_inspected_file("app/main.py", "main entry point")
        ctx.record_inspected_file("app/config.py", "config file")
        prompt = self.builder.build_phase_prompt(ctx, "IMPLEMENT", "fix bug")
        assert "app/main.py" in prompt
        assert "main entry point" in prompt

    def test_build_phase_prompt_with_executions(self):
        ctx = TaskContext(task="fix bug")
        ctx.record_execution({"tool_name": "read_file", "success": True})
        ctx.record_execution({"tool_name": "run_tests", "success": False})
        prompt = self.builder.build_phase_prompt(ctx, "TEST", "fix bug")
        assert "read_file" in prompt
        assert "run_tests" in prompt
        assert "OK" in prompt
        assert "FAIL" in prompt

    def test_build_phase_prompt_with_test_results(self):
        ctx = TaskContext(task="fix bug")
        ctx.test_results = {"exit_code": 1, "success": False, "stdout": "FAIL test_x"}
        prompt = self.builder.build_phase_prompt(ctx, "DIAGNOSE", "fix bug")
        assert "FAIL" in prompt
        assert "test_x" in prompt

    def test_build_phase_prompt_with_failures(self):
        ctx = TaskContext(task="fix bug")
        ctx.record_failure("test_calc failed: assertion error")
        prompt = self.builder.build_phase_prompt(ctx, "DIAGNOSE", "fix bug")
        assert "test_calc failed" in prompt

    def test_build_phase_prompt_with_diagnoses(self):
        ctx = TaskContext(task="fix bug")
        ctx.record_diagnosis("root cause: missing None check")
        prompt = self.builder.build_phase_prompt(ctx, "FIX", "fix bug")
        assert "missing None check" in prompt

    def test_build_phase_prompt_with_fixes(self):
        ctx = TaskContext(task="fix bug")
        ctx.record_fix("added None check")
        prompt = self.builder.build_phase_prompt(ctx, "RETEST", "fix bug")
        assert "added None check" in prompt

    def test_build_phase_prompt_with_review_feedback(self):
        ctx = TaskContext(task="fix bug")
        ctx.review_feedback = "Missing error handling"
        prompt = self.builder.build_phase_prompt(ctx, "FIX", "fix bug")
        assert "Missing error handling" in prompt

    def test_build_phase_prompt_with_decisions(self):
        ctx = TaskContext(task="fix bug")
        ctx.decisions.append("Use pandas for data processing")
        prompt = self.builder.build_phase_prompt(ctx, "IMPLEMENT", "fix bug")
        assert "Use pandas" in prompt

    def test_build_phase_prompt_with_iterations(self):
        ctx = TaskContext(task="fix bug", iteration_count=5, retry_count=2)
        prompt = self.builder.build_phase_prompt(ctx, "TEST", "fix bug")
        assert "Iteration: 5" in prompt
        assert "Retries: 2" in prompt

    def test_build_review_context(self):
        ctx = TaskContext(task="fix bug")
        ctx.set_plan({"objective": "fix bug"})
        ctx.requirements = ["must pass tests"]
        prompt = self.builder.build_review_context(ctx, "fix bug", "Wrote main.py", "diff content", "All passed")
        assert "fix bug" in prompt
        assert "Wrote main.py" in prompt
        assert "diff content" in prompt
        assert "All passed" in prompt

    def test_build_review_context_with_diagnoses(self):
        ctx = TaskContext(task="fix bug")
        ctx.record_diagnosis("missing import")
        prompt = self.builder.build_review_context(ctx, "fix bug", "", "", "")
        assert "missing import" in prompt

    def test_context_continuity(self):
        ctx = TaskContext(task="fix bug")
        ctx.transition_to("UNDERSTAND")
        ctx.add_observation("found bug location", trust=TrustLevel.TOOL_VERIFIED, phase="UNDERSTAND")
        ctx.transition_to("PLAN")
        ctx.set_plan({"objective": "fix bug", "subtasks": []})
        ctx.transition_to("INSPECT")
        ctx.record_inspected_file("app/main.py", "contains the bug")
        ctx.transition_to("IMPLEMENT")
        ctx.record_execution({"tool_name": "write_file", "success": True})
        ctx.transition_to("TEST")
        ctx.test_results = {"exit_code": 0, "success": True}

        builder = ContextBuilder()
        prompt = builder.build_phase_prompt(ctx, "REVIEW", "fix bug")
        assert "app/main.py" in prompt
        assert "contains the bug" in prompt
        assert "write_file" in prompt

    def test_test_result_truncation(self):
        ctx = TaskContext(task="test")
        long_stdout = "line\n" * 100
        ctx.test_results = {"exit_code": 1, "success": False, "stdout": long_stdout}
        prompt = self.builder.build_phase_prompt(ctx, "DIAGNOSE", "test")
        lines = [l for l in prompt.split("\n") if "line" in l]
        assert len(lines) <= 6


class TestTrustLevelHierarchy:
    def test_tool_verified_over_model_proposed(self):
        entries = [
            ContextEntry(content="model says x", trust=TrustLevel.MODEL_PROPOSED),
            ContextEntry(content="tool says y", trust=TrustLevel.TOOL_VERIFIED),
        ]
        trust_order = {
            TrustLevel.TOOL_VERIFIED: 4,
            TrustLevel.USER_ASSERTED: 3,
            TrustLevel.SYSTEM_DERIVED: 2,
            TrustLevel.MODEL_INFERRED: 1,
            TrustLevel.MODEL_PROPOSED: 0,
        }
        sorted_entries = sorted(entries, key=lambda e: trust_order[e.trust], reverse=True)
        assert sorted_entries[0].trust == TrustLevel.TOOL_VERIFIED

    def test_model_proposed_not_auto_verified(self):
        ctx = TaskContext(task="test")
        ctx.add_observation("I fixed the bug", trust=TrustLevel.MODEL_PROPOSED)
        entry = ctx.observations[0]
        assert entry.trust == TrustLevel.MODEL_PROPOSED
        assert entry.trust != TrustLevel.TOOL_VERIFIED


class TestContextBounds:
    def test_observations_bounded(self):
        ctx = TaskContext()
        for i in range(60):
            ctx.add_observation(f"obs {i}", trust=TrustLevel.MODEL_INFERRED)
        assert len(ctx.observations) == ctx.MAX_OBSERVATIONS
        assert ctx.observations[-1].content == "obs 59"
        assert ctx.observations[0].content == f"obs {60 - ctx.MAX_OBSERVATIONS}"

    def test_inspected_files_bounded(self):
        ctx = TaskContext()
        for i in range(35):
            ctx.record_inspected_file(f"file_{i}.py", f"content {i}")
        assert len(ctx.inspected_files) == ctx.MAX_INSPECTED_FILES
        assert f"file_{35 - 1}.py" in ctx.inspected_files
        assert "file_0.py" not in ctx.inspected_files

    def test_diagnoses_bounded(self):
        ctx = TaskContext()
        for i in range(20):
            ctx.record_diagnosis(f"diagnosis {i}")
        assert len(ctx.diagnoses) == ctx.MAX_DIAGNOSES
        assert ctx.diagnoses[-1] == "diagnosis 19"
        assert ctx.diagnoses[0] == f"diagnosis {20 - ctx.MAX_DIAGNOSES}"

    def test_fixes_bounded(self):
        ctx = TaskContext()
        for i in range(20):
            ctx.record_fix(f"fix {i}")
        assert len(ctx.fixes) == ctx.MAX_FIXES
        assert ctx.fixes[-1] == "fix 19"
        assert ctx.fixes[0] == f"fix {20 - ctx.MAX_FIXES}"

    def test_decisions_bounded(self):
        ctx = TaskContext()
        for i in range(25):
            ctx.record_decision(f"decision {i}")
        assert len(ctx.decisions) == ctx.MAX_DECISIONS
        assert ctx.decisions[-1] == "decision 24"
        assert ctx.decisions[0] == f"decision {25 - ctx.MAX_DECISIONS}"

    def test_recent_executions_bounded(self):
        ctx = TaskContext()
        for i in range(25):
            ctx.record_execution({"step": i, "tool_name": f"tool_{i}"})
        assert len(ctx.recent_executions) == 20
        assert ctx.recent_executions[-1]["step"] == 24

    def test_failures_bounded(self):
        ctx = TaskContext()
        for i in range(15):
            ctx.record_failure(f"failure {i}")
        assert len(ctx.failures) == 10
        assert ctx.failures[-1] == "failure 14"

    def test_context_builder_works_with_bounded_context(self):
        ctx = TaskContext(task="test task")
        for i in range(35):
            ctx.record_inspected_file(f"file_{i}.py", f"content {i}")
        for i in range(60):
            ctx.add_observation(f"obs {i}", trust=TrustLevel.MODEL_INFERRED)

        builder = ContextBuilder()
        prompt = builder.build_phase_prompt(ctx, "INSPECT", "test task")
        assert "test task" in prompt
        assert "file_" in prompt

    def test_trust_levels_preserved_with_bounds(self):
        ctx = TaskContext()
        for i in range(60):
            trust = TrustLevel.TOOL_VERIFIED if i % 2 == 0 else TrustLevel.MODEL_INFERRED
            ctx.add_observation(f"obs {i}", trust=trust)
        assert len(ctx.observations) == ctx.MAX_OBSERVATIONS
        tool_verified = [o for o in ctx.observations if o.trust == TrustLevel.TOOL_VERIFIED]
        model_inferred = [o for o in ctx.observations if o.trust == TrustLevel.MODEL_INFERRED]
        assert len(tool_verified) > 0
        assert len(model_inferred) > 0
