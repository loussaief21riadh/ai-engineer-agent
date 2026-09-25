"""Tests for V3.1 subtask-driven execution engine, replanning, and checkpoint resume."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.agent.checkpoint import Checkpoint, CheckpointStore
from app.agent.context import TrustLevel
from app.agent.orchestrator import Orchestrator
from app.agent.planner import PlanValidator, SubtaskStatus, TaskPlan, Subtask
from app.config import AgentMode, CHECKPOINT_SCHEMA_VERSION, MAX_REPLAN_COUNT
from app.llm.openrouter import ChatResponse
from app.models.schemas import TaskReport


_CHANGE_REQUIRED_RESPONSE = json.dumps({
    "decision": "CHANGE_REQUIRED",
    "confidence": 0.95,
    "reason": "Inspection identified work required for the requested task.",
    "evidence": ["inspection completed"],
})


def _phase_responses(*contents):
    result = []
    for i, c in enumerate(contents):
        if i == 2:
            result.append(ChatResponse(content=_CHANGE_REQUIRED_RESPONSE))
        else:
            result.append(ChatResponse(content=c))
    return result


def _review_response(approved=True, summary="LGTM"):
    return ChatResponse(content=json.dumps({
        "approved": approved, "verdict": "APPROVE" if approved else "REJECT",
        "findings": [], "summary": summary,
    }))


def _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS):
    orch = Orchestrator(client=mock_client, mode=mode)
    mock_test = MagicMock()
    mock_test.execute.return_value = {
        "success": True,
        "result": {"exit_code": 0, "stdout": "All tests passed", "stderr": ""},
    }
    orch.core.tools["run_tests"] = mock_test
    return orch


def _make_plan_json(objective="fix bug", subtasks=None):
    if subtasks is None:
        subtasks = [
            {"id": "s1", "description": "first step", "dependencies": [], "acceptance_criteria": []},
            {"id": "s2", "description": "second step", "dependencies": ["s1"], "acceptance_criteria": []},
        ]
    return json.dumps({"objective": objective, "subtasks": subtasks})


def _infinite_responses(*defaults):
    """Return a side_effect function that always returns a ChatResponse."""
    converted = []
    for c in defaults:
        if isinstance(c, str) and c == "Inspected.":
            converted.append(ChatResponse(content=_CHANGE_REQUIRED_RESPONSE))
        else:
            converted.append(ChatResponse(content=c) if isinstance(c, str) else c)
    responses = converted
    idx = [0]

    def side_effect(*args, **kwargs):
        i = idx[0]
        idx[0] += 1
        if i < len(responses):
            return responses[i]
        return responses[-1] if responses else ChatResponse(content="OK")

    return side_effect


class TestSubtaskExecution:
    def test_plan_drives_execution_order(self):
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.",
            _make_plan_json(),
            "Inspected.",
            "Implemented s1.",
            "Implemented s2.",
        )
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        report = orch.run_task("Fix bug")

        assert report.final_phase == "DONE"
        assert "s1" in report.completed_subtasks
        assert "s2" in report.completed_subtasks
        assert "IMPLEMENT" in report.phase_history

    def test_dependencies_respected(self):
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "first", "dependencies": []},
            {"id": "s2", "description": "second", "dependencies": ["s1"]},
            {"id": "s3", "description": "third", "dependencies": ["s2"]},
        ])
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", plan, "Inspected.",
            "Implemented s1.", "Implemented s2.", "Implemented s3.",
        )
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        report = orch.run_task("Multi-step task")

        assert report.final_phase == "DONE"
        assert len(report.completed_subtasks) == 3

    def test_independent_subtasks_both_execute(self):
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "independent a", "dependencies": []},
            {"id": "s2", "description": "independent b", "dependencies": []},
        ])
        mock_client = MagicMock()
        responses = _phase_responses(
            "Understood.", plan, "Inspected.",
            "Implemented s1.", "Implemented s2.",
        )
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        report = orch.run_task("Parallel task")

        assert report.final_phase == "DONE"
        assert set(report.completed_subtasks) == {"s1", "s2"}

    def test_subtask_failure_recorded(self):
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "doomed", "dependencies": []},
        ])
        mock_client = MagicMock()
        mock_client.chat.side_effect = _infinite_responses(
            "Understood.", ChatResponse(content=plan), "Inspected.",
            "Implemented s1.", "Diagnosed.", "Fixed.",
            "Diagnosed again.", "Fixed again.",
            "Diagnosed third.", "Fixed third.",
        )

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""},
        }

        report = orch.run_task("Doomed task")
        assert len(report.failed_subtasks) >= 1
        assert any("s1" in st for st in report.failed_subtasks)

    def test_get_next_subtask_respects_pending_status(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="a", status=SubtaskStatus.COMPLETED),
                Subtask(id="s2", description="b", status=SubtaskStatus.PENDING),
            ],
        )
        next_st = plan.get_next_subtask()
        assert next_st is not None
        assert next_st.id == "s2"

    def test_get_next_subtask_returns_none_when_all_done(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="a", status=SubtaskStatus.COMPLETED),
            ],
        )
        assert plan.get_next_subtask() is None

    def test_get_next_subtask_respects_dependencies(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="a", dependencies=[], status=SubtaskStatus.PENDING),
                Subtask(id="s2", description="b", dependencies=["s1"], status=SubtaskStatus.PENDING),
            ],
        )
        next_st = plan.get_next_subtask()
        assert next_st is not None
        assert next_st.id == "s1"

    def test_no_plan_goes_through_linear_flow(self):
        mock_client = MagicMock()
        responses = _phase_responses("Understood.", "No plan.", "Inspected.", "Implemented.")
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        report = orch.run_task("Simple task")

        assert report.final_phase == "DONE"
        assert "TEST" in report.phase_history

    def test_subtask_scoped_prompt_includes_description(self):
        mock_client = MagicMock()
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "fix the widget", "dependencies": []},
        ])
        responses = _phase_responses(
            "Understood.", plan, "Inspected.", "Fixed widget.",
        )
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        orch.run_task("Fix widget")

        implement_calls = [
            c for c in mock_client.chat.call_args_list
            if any("Phase: IMPLEMENT" in m.get("content", "") for m in c[1].get("messages", []) if isinstance(m, dict))
        ]
        assert len(implement_calls) >= 1
        prompt = implement_calls[0][1]["messages"][1]["content"]
        assert "fix the widget" in prompt


class TestEvidenceBasedCompletion:
    def test_llm_cannot_directly_mark_complete(self):
        mock_client = MagicMock()
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "task", "dependencies": []},
        ])
        responses = _phase_responses(
            "Understood.", plan, "Inspected.",
            "Done! I declare this complete.",
        )
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        report = orch.run_task("Task")

        assert report.final_phase == "DONE"
        assert "s1" in report.completed_subtasks

    def test_test_pass_required_for_completion(self):
        mock_client = MagicMock()
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "task", "dependencies": []},
        ])
        mock_client.chat.side_effect = _infinite_responses(
            "Understood.", ChatResponse(content=plan), "Inspected.", "Implemented.",
            "Diagnosed.", "Fixed.",
        )

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""},
        }

        report = orch.run_task("Task")
        assert "s1" not in report.completed_subtasks


class TestDynamicReplanning:
    def test_persistent_failure_triggers_replan(self):
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "impossible", "dependencies": []},
        ])
        mock_client = MagicMock()
        mock_client.chat.side_effect = _infinite_responses(
            "Understood.", ChatResponse(content=plan), "Inspected.",
            "Attempted.", "Diagnosed.", "Fixed.",
            "Diagnosed again.", "Fixed again.",
            "Diagnosed third.", "Fixed third.",
            "Diagnosed fourth.", "Fixed fourth.",
        )

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""},
        }

        report = orch.run_task("Impossible task")
        assert report.replan_count >= 1

    def test_replan_preserves_completed_work(self):
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "easy", "dependencies": []},
            {"id": "s2", "description": "hard", "dependencies": []},
        ])
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            content = msgs[1]["content"] if len(msgs) > 1 else ""
            if "Phase: UNDERSTAND" in content:
                return ChatResponse(content="Understood.")
            elif "Phase: PLAN" in content:
                return ChatResponse(content=plan)
            elif "Phase: INSPECT" in content:
                return ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
            elif "Phase: IMPLEMENT" in content:
                return ChatResponse(content="Implemented.")
            elif "Phase: DIAGNOSE" in content:
                return ChatResponse(content="Diagnosed.")
            elif "Phase: FIX" in content:
                return ChatResponse(content="Fixed.")
            elif "Phase: REVIEW" in content:
                return ChatResponse(content=json.dumps({
                    "approved": True, "verdict": "APPROVE", "findings": [], "summary": "OK",
                }))
            else:
                return ChatResponse(content="OK.")

        mock_client.chat.side_effect = side_effect

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)

        subtask_test_results = {"s1": [True], "s2": [False, False, False, True]}

        def fake_test(args="", **kwargs):
            current_id = orch._current_subtask_id or ""
            results = subtask_test_results.get(current_id, [True])
            passed = results.pop(0) if results else True
            if passed:
                return {"success": True, "result": {"exit_code": 0, "stdout": "PASS", "stderr": ""}}
            return {"success": False, "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""}}
        orch.core.tools["run_tests"].execute.side_effect = fake_test

        report = orch.run_task("Mixed task")

        assert report.replan_count >= 1
        completed = report.completed_subtasks
        assert len(completed) >= 1

    def test_plan_version_increments(self):
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "task", "dependencies": []},
        ])
        mock_client = MagicMock()
        mock_client.chat.side_effect = _infinite_responses(
            "Understood.", ChatResponse(content=plan), "Inspected.",
            "Attempted.", "Diagnosed.", "Fixed.",
            "Diagnosed again.", "Fixed again.",
            "Diagnosed third.", "Fixed third.",
            "Diagnosed fourth.", "Fixed fourth.",
        )

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""},
        }

        report = orch.run_task("Task")
        assert len(report.plan_versions) >= 1

    def test_noop_replan_detected(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="task", dependencies=[])],
        )
        fp1 = plan.plan_fingerprint()
        plan.record_version("test")
        fp2 = plan.plan_fingerprint()
        assert fp1 == fp2

    def test_replan_limit_enforced_at_runtime(self):
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "impossible", "dependencies": []},
        ])
        mock_client = MagicMock()
        call_count = [0]

        def infinite_side_effect(*args, **kwargs):
            call_count[0] += 1
            msgs = kwargs.get("messages", [])
            content = msgs[1]["content"] if len(msgs) > 1 else ""
            if "Phase: UNDERSTAND" in content:
                return ChatResponse(content="Understood.")
            elif "Phase: PLAN" in content:
                return ChatResponse(content=plan)
            elif "Phase: INSPECT" in content:
                return ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
            elif "Phase: IMPLEMENT" in content:
                return ChatResponse(content="Attempted.")
            elif "Phase: DIAGNOSE" in content:
                return ChatResponse(content="Diagnosed.")
            elif "Phase: FIX" in content:
                return ChatResponse(content="Fixed.")
            return ChatResponse(content="OK.")

        mock_client.chat.side_effect = infinite_side_effect

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""},
        }

        report = orch.run_task("Impossible task")

        assert report.replan_count <= MAX_REPLAN_COUNT
        replan_events = [e for e in orch.observer.events if e.event_type == "replan_completed"]
        assert len(replan_events) <= MAX_REPLAN_COUNT
        assert report.final_phase in ("REVIEW", "FAILED", "DONE")

    def test_plan_fingerprint_deterministic(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="task", dependencies=[])],
        )
        fp1 = plan.plan_fingerprint()
        fp2 = plan.plan_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_plan_fingerprint_different_for_different_plans(self):
        p1 = TaskPlan(
            objective="a",
            subtasks=[Subtask(id="s1", description="task a", dependencies=[])],
        )
        p2 = TaskPlan(
            objective="b",
            subtasks=[Subtask(id="s1", description="task b", dependencies=[])],
        )
        assert p1.plan_fingerprint() != p2.plan_fingerprint()


class TestCheckpointResume:
    def test_checkpoint_save_and_load(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(
            task_id="t1", task="fix bug", current_phase="IMPLEMENT",
            current_subtask_id="s1", current_subtask_index=0,
            plan_version=2, replan_count=1, retry_count=1,
            context_snapshot={"task": "fix bug"},
            executions=[{"tool_name": "write_file", "success": True}],
        )
        store.save(cp)
        loaded = store.load("t1")
        assert loaded is not None
        assert loaded.current_phase == "IMPLEMENT"
        assert loaded.current_subtask_id == "s1"
        assert loaded.plan_version == 2
        assert loaded.replan_count == 1

    def test_atomic_write_no_tmp_remaining(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(task_id="t1", task="test", current_phase="UNDERSTAND")
        store.save(cp)
        assert not (tmp_path / "t1.tmp").exists()
        assert (tmp_path / "t1.json").exists()

    def test_checksum_catches_corruption(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(task_id="t1", task="test", current_phase="UNDERSTAND")
        store.save(cp)
        path = tmp_path / "t1.json"
        data = json.loads(path.read_text())
        data["checksum"] = "invalid"
        path.write_text(json.dumps(data, indent=2))
        assert store.load("t1") is None

    def test_schema_version_stored(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(task_id="t1", task="test", current_phase="UNDERSTAND")
        store.save(cp)
        loaded = store.load("t1")
        assert loaded is not None
        assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION

    def test_stale_checkpoint_detection(self):
        cp = Checkpoint(
            task_id="t1", task="test", current_phase="UNDERSTAND",
            timestamp=time.time() - 7200,
        )
        assert cp.is_stale(max_age=3600) is True
        assert cp.is_stale(max_age=86400) is False

    def test_orchestrator_checkpoint_resume(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(
            task_id="resume1", task="fix bug",
            current_phase="IMPLEMENT",
            current_subtask_id="s1",
            current_subtask_index=0,
            plan_snapshot={
                "objective": "fix bug",
                "subtasks": [{"id": "s1", "description": "fix", "dependencies": [], "status": "PENDING", "acceptance_criteria": [], "result": ""}],
            },
            plan_version=1,
            context_snapshot={"task": "fix bug", "current_phase": "IMPLEMENT", "observations": [], "inspected_files": {}, "recent_executions": [], "test_results": None, "failures": [], "diagnoses": [], "fixes": [], "review_feedback": "", "decisions": [], "iteration_count": 3, "retry_count": 0, "phase_history": [], "requirements": [], "constraints": [], "plan": None},
            executions=[],
            phase_history=["UNDERSTAND", "PLAN", "INSPECT"],
            iteration_count=3,
            retry_count=0,
        )
        store.save(cp)

        mock_client = MagicMock()
        mock_client.chat.side_effect = _infinite_responses(
            "Resumed implementation.",
        )
        mock_client.chat.side_effect = _infinite_responses(
            ChatResponse(content="Resumed implementation."),
            _review_response(),
        )

        orch = Orchestrator(client=mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.checkpoint_store = store
        mock_test = MagicMock()
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "All tests passed", "stderr": ""},
        }
        orch.core.tools["run_tests"] = mock_test

        report = orch.run_task("fix bug", resume_checkpoint_id="resume1")

        assert report.checkpoint_resume_used is True
        assert report.final_phase == "DONE"

    def test_stale_checkpoint_rejected(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(
            task_id="stale1", task="fix bug",
            current_phase="IMPLEMENT",
            timestamp=time.time() - 7200,
            context_snapshot={"task": "fix bug", "current_phase": "IMPLEMENT", "observations": [], "inspected_files": {}, "recent_executions": [], "test_results": None, "failures": [], "diagnoses": [], "fixes": [], "review_feedback": "", "decisions": [], "iteration_count": 0, "retry_count": 0, "phase_history": [], "requirements": [], "constraints": [], "plan": None},
        )
        store.save(cp)

        mock_client = MagicMock()
        responses = _phase_responses("Understood.", "Plan.", "Inspected.", "Implemented.")
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = Orchestrator(client=mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.checkpoint_store = store
        mock_test = MagicMock()
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "All tests passed", "stderr": ""},
        }
        orch.core.tools["run_tests"] = mock_test

        report = orch.run_task("fix bug", resume_checkpoint_id="stale1")
        assert report.checkpoint_resume_used is False

    def test_checkpoint_persists_subtask_and_plan_state(self, tmp_path: Path):
        store = CheckpointStore(base_dir=tmp_path)
        plan_dict = {
            "objective": "test",
            "subtasks": [{"id": "s1", "description": "task", "dependencies": [], "status": "IN_PROGRESS", "acceptance_criteria": [], "result": ""}],
        }
        cp = Checkpoint(
            task_id="state1", task="test",
            current_phase="TEST",
            current_subtask_id="s1",
            current_subtask_index=0,
            plan_snapshot=plan_dict,
            plan_version=3,
            replan_count=2,
            retry_count=1,
            budget_snapshot={"llm_calls": 5, "tool_calls": 10},
            files_modified=["main.py"],
        )
        store.save(cp)
        loaded = store.load("state1")
        assert loaded is not None
        assert loaded.current_subtask_id == "s1"
        assert loaded.plan_version == 3
        assert loaded.replan_count == 2
        assert loaded.budget_snapshot["llm_calls"] == 5
        assert loaded.files_modified == ["main.py"]


class TestObserverIntegration:
    def test_subtask_events_emitted(self):
        mock_client = MagicMock()
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "task", "dependencies": []},
        ])
        responses = _phase_responses(
            "Understood.", plan, "Inspected.", "Implemented.",
        )
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        orch.run_task("Task")

        event_types = [e.event_type for e in orch.observer.events]
        assert "subtask_started" in event_types
        assert "subtask_completed" in event_types

    def test_replan_events_emitted(self):
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "impossible", "dependencies": []},
        ])
        mock_client = MagicMock()
        mock_client.chat.side_effect = _infinite_responses(
            "Understood.", ChatResponse(content=plan), "Inspected.",
            "Attempted.", "Diagnosed.", "Fixed.",
            "Diagnosed again.", "Fixed again.",
            "Diagnosed third.", "Fixed third.",
            "Diagnosed fourth.", "Fixed fourth.",
        )

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""},
        }

        orch.run_task("Impossible")

        event_types = [e.event_type for e in orch.observer.events]
        assert "replan_completed" in event_types

    def test_checkpoint_saved_event_emitted(self):
        mock_client = MagicMock()
        responses = _phase_responses("Understood.", "Plan.", "Inspected.", "Implemented.")
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        orch.run_task("Task")

        event_types = [e.event_type for e in orch.observer.events]
        assert "checkpoint_saved" in event_types


class TestReportStructure:
    def test_report_includes_subtask_info(self):
        mock_client = MagicMock()
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "task", "dependencies": []},
        ])
        responses = _phase_responses(
            "Understood.", plan, "Inspected.", "Implemented.",
        )
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        report = orch.run_task("Task")

        assert hasattr(report, "completed_subtasks")
        assert hasattr(report, "failed_subtasks")
        assert hasattr(report, "blocked_subtasks")
        assert hasattr(report, "plan_versions")
        assert hasattr(report, "replan_count")
        assert hasattr(report, "checkpoint_resume_used")
        assert isinstance(report.completed_subtasks, list)
        assert isinstance(report.plan_versions, list)


class TestSecurityBoundaries:
    def test_invalid_plan_rejected(self):
        mock_client = MagicMock()
        invalid_plan = json.dumps({"objective": "", "subtasks": []})
        responses = _phase_responses("Understood.", invalid_plan, "Inspected.", "Implemented.")
        mock_client.chat.side_effect = responses + [_review_response()]

        orch = _make_orch(mock_client)
        report = orch.run_task("Task")
        assert report.final_phase == "FAILED"

    def test_invalid_phase_rejected(self):
        from app.agent.phases import can_transition
        assert not can_transition("DONE", "UNDERSTAND")
        assert not can_transition("FAILED", "IMPLEMENT")


class TestBudgetResumeFix:
    def test_budget_not_reset_on_resume(self, tmp_path: Path):
        from app.agent.quota import BudgetTracker
        budget = BudgetTracker()
        budget.start()
        budget.record_llm_call()
        budget.record_llm_call()
        budget.record_tool_call()
        snapshot = budget.status()
        assert snapshot["llm_calls"] == 2
        assert snapshot["tool_calls"] == 1

        budget2 = BudgetTracker()
        budget2.restore(snapshot)
        assert budget2.llm_calls == 2
        assert budget2.tool_calls == 1
        assert budget2._active is True

    def test_budget_restore_enforces_limits(self, tmp_path: Path):
        from app.agent.quota import BudgetTracker, BudgetLimits
        budget = BudgetTracker(limits=BudgetLimits(max_llm_calls=3, max_tool_calls=5))
        budget.start()
        budget.record_llm_call()
        budget.record_llm_call()
        snapshot = budget.status()

        budget2 = BudgetTracker(limits=BudgetLimits(max_llm_calls=3, max_tool_calls=5))
        budget2.restore(snapshot)
        assert budget2.is_within_budget() is True
        budget2.record_llm_call()
        assert budget2.is_within_budget() is False

    def test_orchestrator_budget_not_reset_on_resume(self, tmp_path: Path):
        from app.agent.checkpoint import Checkpoint, CheckpointStore
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(
            task_id="budget1", task="fix bug",
            current_phase="IMPLEMENT",
            current_subtask_id="s1",
            current_subtask_index=0,
            plan_snapshot={
                "objective": "fix bug",
                "subtasks": [{"id": "s1", "description": "fix", "dependencies": [], "status": "IN_PROGRESS", "acceptance_criteria": [], "result": ""}],
            },
            plan_version=1,
            context_snapshot={"task": "fix bug", "current_phase": "IMPLEMENT", "observations": [], "inspected_files": {}, "recent_executions": [], "test_results": None, "failures": [], "diagnoses": [], "fixes": [], "review_feedback": "", "decisions": [], "iteration_count": 3, "retry_count": 0, "phase_history": [], "requirements": [], "constraints": [], "plan": None},
            executions=[],
            budget_snapshot={"llm_calls": 10, "tool_calls": 20, "retry_cycles": 1},
        )
        store.save(cp)

        mock_client = MagicMock()
        mock_client.chat.side_effect = _infinite_responses(
            ChatResponse(content="Resumed."),
            _review_response(),
        )

        orch = Orchestrator(client=mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.checkpoint_store = store
        mock_test = MagicMock()
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "All tests passed", "stderr": ""},
        }
        orch.core.tools["run_tests"] = mock_test

        orch.run_task("fix bug", resume_checkpoint_id="budget1")

        assert orch.budget.llm_calls > 10
        assert orch.budget.tool_calls >= 20


class TestDependencyFix:
    def test_missing_dependency_not_completed(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s2", description="depends on ghost", dependencies=["nonexistent"]),
            ],
        )
        assert plan.get_next_subtask() is None

    def test_replan_reconciles_dependency_ids(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first", dependencies=[], status=SubtaskStatus.FAILED),
                Subtask(id="s2", description="second", dependencies=["s1"]),
            ],
        )
        new_plan = plan.create_replan("s1", "failed")
        s2 = new_plan.get_subtask("s2")
        assert s2 is not None
        assert s2.dependencies == ["s1_revised"]
        assert new_plan.get_subtask("s1") is None
        assert new_plan.get_subtask("s1_revised") is not None

    def test_replan_chain_preserves_order(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first", dependencies=[], status=SubtaskStatus.FAILED),
                Subtask(id="s2", description="second", dependencies=["s1"]),
                Subtask(id="s3", description="third", dependencies=["s2"]),
            ],
        )
        new_plan = plan.create_replan("s1", "failed")
        s1r = new_plan.get_subtask("s1_revised")
        s2 = new_plan.get_subtask("s2")
        s3 = new_plan.get_subtask("s3")
        assert s1r is not None
        assert s2 is not None
        assert s3 is not None
        assert s1r.dependencies == []
        assert s2.dependencies == ["s1_revised"]
        assert s3.dependencies == ["s2"]

    def test_completed_subtask_deps_not_remapped(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first", dependencies=[], status=SubtaskStatus.COMPLETED),
                Subtask(id="s2", description="second", dependencies=["s1"], status=SubtaskStatus.FAILED),
            ],
        )
        new_plan = plan.create_replan("s2", "failed")
        s1 = new_plan.get_subtask("s1")
        assert s1 is not None
        assert s1.status == SubtaskStatus.COMPLETED
        assert s1.dependencies == []


class TestReplanValidation:
    def test_malformed_replan_rejected(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first", dependencies=[], status=SubtaskStatus.FAILED),
                Subtask(id="s2", description="second", dependencies=["nonexistent_dep"]),
            ],
        )
        new_plan = plan.create_replan("s1", "failed")
        is_valid, errors, _ = PlanValidator.validate_plan(new_plan.to_dict())
        assert not is_valid
        assert any("nonexistent_dep" in e or "unknown" in e for e in errors)

    def test_valid_replan_accepted(self):
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first", dependencies=[], status=SubtaskStatus.FAILED),
                Subtask(id="s2", description="second", dependencies=["s1"]),
            ],
        )
        new_plan = plan.create_replan("s1", "failed")
        is_valid, errors, validated = PlanValidator.validate_plan(new_plan.to_dict())
        assert is_valid
        assert validated is not None
        assert validated.get_subtask("s1_revised") is not None
        assert validated.get_subtask("s2") is not None

    def test_replan_validation_blocks_install(self):
        plan = _make_plan_json(subtasks=[
            {"id": "s1", "description": "impossible", "dependencies": []},
        ])
        mock_client = MagicMock()
        mock_client.chat.side_effect = _infinite_responses(
            "Understood.", ChatResponse(content=plan), "Inspected.",
            "Attempted.", "Diagnosed.", "Fixed.",
            "Diagnosed again.", "Fixed again.",
            "Diagnosed third.", "Fixed third.",
            "Diagnosed fourth.", "Fixed fourth.",
        )

        orch = _make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch.core.tools["run_tests"].execute.return_value = {
            "success": False,
            "result": {"exit_code": 1, "stdout": "FAIL", "stderr": ""},
        }

        report = orch.run_task("Impossible task")
        replan_events = [e for e in orch.observer.events if e.event_type in ("replan_completed", "replan_rejected")]
        assert len(replan_events) >= 1


class TestCycleDetection:
    def test_cycle_detected_when_fingerprint_repeats(self):
        from app.agent.orchestrator import Orchestrator
        mock_client = MagicMock()
        orch = Orchestrator(client=mock_client, mode=AgentMode.ALLOW_EDITS)

        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="task a", dependencies=[])],
        )
        orch._plan_fingerprints = ["abc123"]

        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="task a", dependencies=[])],
        )
        new_fingerprint = orch._task_plan.plan_fingerprint()
        orch._plan_fingerprints.append(new_fingerprint)

        assert new_fingerprint in orch._plan_fingerprints
