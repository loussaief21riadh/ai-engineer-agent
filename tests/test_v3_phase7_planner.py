from __future__ import annotations

from unittest.mock import MagicMock

from app.agent.orchestrator import Orchestrator
from app.agent.planner import SubtaskStatus, TaskPlan, Subtask
from app.config import AgentMode


class TestSubtaskStatusTracking:
    def _make_orchestrator(self) -> Orchestrator:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(content="done")
        return Orchestrator(client=mock_client, mode=AgentMode.READ_ONLY)

    def test_advance_subtask_to_in_progress(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first"),
                Subtask(id="s2", description="second"),
            ],
        )
        orch._advance_subtask_toInProgress()
        assert orch._task_plan.subtasks[0].status == SubtaskStatus.IN_PROGRESS
        assert orch._current_subtask_index == 0

    def test_advance_picks_next_pending_with_deps_met(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first", status=SubtaskStatus.COMPLETED),
                Subtask(id="s2", description="second", dependencies=["s1"]),
            ],
        )
        orch._advance_subtask_toInProgress()
        assert orch._task_plan.subtasks[1].status == SubtaskStatus.IN_PROGRESS
        assert orch._current_subtask_index == 1

    def test_advance_skips_subtask_with_unmet_deps(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first"),
                Subtask(id="s2", description="second", dependencies=["s1"]),
            ],
        )
        orch._advance_subtask_toInProgress()
        assert orch._task_plan.subtasks[0].status == SubtaskStatus.IN_PROGRESS
        assert orch._task_plan.subtasks[1].status == SubtaskStatus.PENDING

    def test_complete_current_subtask(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="first")],
        )
        orch._current_subtask_index = 0
        orch._task_plan.subtasks[0].status = SubtaskStatus.IN_PROGRESS
        orch._complete_current_subtask(result="done")
        assert orch._task_plan.subtasks[0].status == SubtaskStatus.COMPLETED
        assert orch._task_plan.subtasks[0].result == "done"

    def test_fail_current_subtask(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="first")],
        )
        orch._current_subtask_index = 0
        orch._task_plan.subtasks[0].status = SubtaskStatus.IN_PROGRESS
        orch._fail_current_subtask(result="error")
        assert orch._task_plan.subtasks[0].status == SubtaskStatus.FAILED
        assert orch._task_plan.subtasks[0].result == "error"

    def test_no_plan_no_crash(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = None
        orch._advance_subtask_toInProgress()
        orch._complete_current_subtask()
        orch._fail_current_subtask()

    def test_empty_subtasks_no_crash(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(objective="test", subtasks=[])
        orch._advance_subtask_toInProgress()
        orch._complete_current_subtask()
        orch._fail_current_subtask()

    def test_complete_out_of_bounds_no_crash(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="first")],
        )
        orch._current_subtask_index = 99
        orch._complete_current_subtask()

    def test_progress_summary_updated(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first"),
                Subtask(id="s2", description="second"),
            ],
        )
        orch._current_subtask_index = 0
        orch._task_plan.subtasks[0].status = SubtaskStatus.IN_PROGRESS
        orch._complete_current_subtask()
        summary = orch._task_plan.progress_summary()
        assert summary["completed"] == 1
        assert summary["in_progress"] == 0
        assert summary["pending"] == 1

    def test_observer_emits_subtask_events(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="first")],
        )
        orch._advance_subtask_toInProgress()
        orch._complete_current_subtask(result="done")

        events = orch.observer.query(event_type="subtask_started")
        assert len(events) == 1
        assert events[0].subtask_id == "s1"

        events = orch.observer.query(event_type="subtask_completed")
        assert len(events) == 1
        assert events[0].subtask_id == "s1"

    def test_multi_subtask_lifecycle(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first"),
                Subtask(id="s2", description="second"),
            ],
        )
        orch._advance_subtask_toInProgress()
        assert orch._task_plan.subtasks[0].status == SubtaskStatus.IN_PROGRESS

        orch._complete_current_subtask()
        assert orch._task_plan.subtasks[0].status == SubtaskStatus.COMPLETED

        orch._advance_subtask_toInProgress()
        assert orch._task_plan.subtasks[1].status == SubtaskStatus.IN_PROGRESS

        orch._complete_current_subtask()
        assert orch._task_plan.all_completed()

    def test_failed_subtask_blocks_next(self) -> None:
        orch = self._make_orchestrator()
        orch._task_plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first"),
                Subtask(id="s2", description="second", dependencies=["s1"]),
            ],
        )
        orch._advance_subtask_toInProgress()
        orch._fail_current_subtask()

        next_st = orch._task_plan.get_next_subtask()
        assert next_st is None
        assert not orch._task_plan.all_completed()
