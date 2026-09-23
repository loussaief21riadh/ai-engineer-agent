from __future__ import annotations

from app.agent.planner import Subtask, SubtaskStatus, TaskPlan


class TestDynamicReplan:
    def test_replan_records_version_history(self) -> None:
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first"),
                Subtask(id="s2", description="second"),
            ],
        )
        plan.subtasks[0].status = SubtaskStatus.COMPLETED
        new_plan = plan.create_replan("s1", "persistent failure")
        assert new_plan.version == 2
        assert len(new_plan.history) == 1
        assert plan.history[0]["reason"] == "Replan after failure of s1: persistent failure"

    def test_replan_preserves_completed_subtasks(self) -> None:
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first", status=SubtaskStatus.COMPLETED, result="done"),
                Subtask(id="s2", description="second"),
            ],
        )
        new_plan = plan.create_replan("s2", "failed")
        s1 = new_plan.get_subtask("s1")
        assert s1 is not None
        assert s1.status == SubtaskStatus.COMPLETED
        assert s1.result == "done"

    def test_replan_creates_revised_subtask(self) -> None:
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="original task"),
            ],
        )
        new_plan = plan.create_replan("s1", "wrong approach")
        revised = new_plan.get_subtask("s1_revised")
        assert revised is not None
        assert "REPLANNED" in revised.description
        assert revised.status == SubtaskStatus.PENDING

    def test_replan_preserves_pending_subtasks(self) -> None:
        plan = TaskPlan(
            objective="test",
            subtasks=[
                Subtask(id="s1", description="first"),
                Subtask(id="s2", description="second", dependencies=["s1"]),
                Subtask(id="s3", description="third"),
            ],
        )
        new_plan = plan.create_replan("s1", "failed")
        s2 = new_plan.get_subtask("s2")
        s3 = new_plan.get_subtask("s3")
        assert s2 is not None
        assert s2.status == SubtaskStatus.PENDING
        assert s3 is not None
        assert s3.status == SubtaskStatus.PENDING

    def test_replan_version_increments(self) -> None:
        plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="first")],
        )
        new_plan = plan.create_replan("s1", "first failure")
        assert new_plan.version == 2
        newer_plan = new_plan.create_replan("s1_revised", "second failure")
        assert newer_plan.version == 3

    def test_replan_history_chain(self) -> None:
        plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="first")],
        )
        p2 = plan.create_replan("s1", "failure 1")
        p3 = p2.create_replan("s1_revised", "failure 2")
        assert len(p3.history) == 2
        assert p3.history[0]["version"] == 1
        assert p3.history[1]["version"] == 2

    def test_replan_preserves_objective(self) -> None:
        plan = TaskPlan(
            objective="build feature X",
            subtasks=[Subtask(id="s1", description="first")],
        )
        new_plan = plan.create_replan("s1", "failed")
        assert new_plan.objective == "build feature X"

    def test_replan_preserves_requirements(self) -> None:
        plan = TaskPlan(
            objective="test",
            requirements=["req1", "req2"],
            subtasks=[Subtask(id="s1", description="first")],
        )
        new_plan = plan.create_replan("s1", "failed")
        assert new_plan.requirements == ["req1", "req2"]

    def test_record_version_stores_snapshot(self) -> None:
        plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="first", status=SubtaskStatus.COMPLETED)],
        )
        plan.record_version("checkpoint")
        assert len(plan.history) == 1
        assert plan.history[0]["version"] == 1
        assert len(plan.history[0]["subtasks"]) == 1

    def test_to_dict_includes_version(self) -> None:
        plan = TaskPlan(
            objective="test",
            subtasks=[Subtask(id="s1", description="first")],
            version=3,
        )
        d = plan.to_dict()
        assert d["version"] == 3
