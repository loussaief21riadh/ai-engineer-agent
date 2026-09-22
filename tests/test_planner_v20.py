"""Tests for V2.0-B Structured Planning."""

from __future__ import annotations

import pytest

from app.agent.planner import PlanValidator, Subtask, SubtaskStatus, TaskPlan


class TestSubtask:
    def test_default_subtask(self):
        st = Subtask(id="1", description="do something")
        assert st.id == "1"
        assert st.description == "do something"
        assert st.dependencies == []
        assert st.status == SubtaskStatus.PENDING
        assert st.acceptance_criteria == []
        assert st.result == ""

    def test_subtask_with_deps(self):
        st = Subtask(id="2", description="second", dependencies=["1"])
        assert st.dependencies == ["1"]


class TestTaskPlan:
    def test_add_subtask(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="inspect"))
        assert len(plan.subtasks) == 1

    def test_get_subtask(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="inspect"))
        plan.add_subtask(Subtask(id="2", description="fix"))
        assert plan.get_subtask("1") is not None
        assert plan.get_subtask("2") is not None
        assert plan.get_subtask("3") is None

    def test_get_pending_subtasks(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a", status=SubtaskStatus.COMPLETED))
        plan.add_subtask(Subtask(id="2", description="b", status=SubtaskStatus.PENDING))
        plan.add_subtask(Subtask(id="3", description="c", status=SubtaskStatus.IN_PROGRESS))
        pending = plan.get_pending_subtasks()
        assert len(pending) == 1
        assert pending[0].id == "2"

    def test_get_completed_subtasks(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a", status=SubtaskStatus.COMPLETED))
        plan.add_subtask(Subtask(id="2", description="b", status=SubtaskStatus.PENDING))
        completed = plan.get_completed_subtasks()
        assert len(completed) == 1
        assert completed[0].id == "1"

    def test_get_next_subtask_no_deps(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a"))
        plan.add_subtask(Subtask(id="2", description="b"))
        next_st = plan.get_next_subtask()
        assert next_st is not None
        assert next_st.id == "1"

    def test_get_next_subtask_respects_deps(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a", status=SubtaskStatus.COMPLETED))
        plan.add_subtask(Subtask(id="2", description="b", dependencies=["1"]))
        next_st = plan.get_next_subtask()
        assert next_st is not None
        assert next_st.id == "2"

    def test_get_next_subtask_blocked_by_uncompleted_dep(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a", status=SubtaskStatus.IN_PROGRESS))
        plan.add_subtask(Subtask(id="2", description="b", dependencies=["1"]))
        next_st = plan.get_next_subtask()
        assert next_st is None

    def test_all_completed(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a", status=SubtaskStatus.COMPLETED))
        plan.add_subtask(Subtask(id="2", description="b", status=SubtaskStatus.COMPLETED))
        assert plan.all_completed() is True

    def test_all_completed_false(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a", status=SubtaskStatus.COMPLETED))
        plan.add_subtask(Subtask(id="2", description="b", status=SubtaskStatus.PENDING))
        assert plan.all_completed() is False

    def test_has_failures(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a", status=SubtaskStatus.FAILED))
        assert plan.has_failures() is True

    def test_has_failures_false(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a", status=SubtaskStatus.COMPLETED))
        assert plan.has_failures() is False

    def test_progress_summary(self):
        plan = TaskPlan(objective="fix bug")
        plan.add_subtask(Subtask(id="1", description="a", status=SubtaskStatus.COMPLETED))
        plan.add_subtask(Subtask(id="2", description="b", status=SubtaskStatus.FAILED))
        plan.add_subtask(Subtask(id="3", description="c", status=SubtaskStatus.IN_PROGRESS))
        plan.add_subtask(Subtask(id="4", description="d", status=SubtaskStatus.PENDING))
        summary = plan.progress_summary()
        assert summary["total"] == 4
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["in_progress"] == 1
        assert summary["pending"] == 1

    def test_to_dict(self):
        plan = TaskPlan(
            objective="fix bug",
            requirements=["must pass tests"],
            constraints=["no new deps"],
            subtasks=[Subtask(id="1", description="inspect")],
        )
        d = plan.to_dict()
        assert d["objective"] == "fix bug"
        assert d["requirements"] == ["must pass tests"]
        assert len(d["subtasks"]) == 1


class TestPlanValidator:
    def test_valid_plan(self):
        data = {
            "objective": "fix bug",
            "requirements": ["pass tests"],
            "constraints": ["minimal changes"],
            "subtasks": [
                {"id": "1", "description": "inspect files", "acceptance_criteria": ["read main.py"]},
                {"id": "2", "description": "fix code", "dependencies": ["1"]},
            ],
        }
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is True
        assert errors == []
        assert plan is not None
        assert plan.objective == "fix bug"
        assert len(plan.subtasks) == 2

    def test_invalid_plan_not_dict(self):
        valid, errors, plan = PlanValidator.validate_plan("not a dict")
        assert valid is False
        assert "JSON object" in errors[0]

    def test_invalid_plan_no_objective(self):
        data = {"subtasks": [{"id": "1", "description": "x"}]}
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is False
        assert any("objective" in e for e in errors)

    def test_invalid_plan_no_subtasks(self):
        data = {"objective": "fix bug", "subtasks": []}
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is False
        assert any("at least one" in e for e in errors)

    def test_invalid_plan_subtask_no_id(self):
        data = {"objective": "fix bug", "subtasks": [{"description": "x"}]}
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is False
        assert any("'id'" in e for e in errors)

    def test_invalid_plan_subtask_no_description(self):
        data = {"objective": "fix bug", "subtasks": [{"id": "1"}]}
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is False
        assert any("'description'" in e for e in errors)

    def test_invalid_plan_duplicate_ids(self):
        data = {
            "objective": "fix bug",
            "subtasks": [
                {"id": "1", "description": "a"},
                {"id": "1", "description": "b"},
            ],
        }
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is False
        assert any("Duplicate" in e for e in errors)

    def test_invalid_plan_unknown_dependency(self):
        data = {
            "objective": "fix bug",
            "subtasks": [
                {"id": "1", "description": "a", "dependencies": ["99"]},
            ],
        }
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is False
        assert any("unknown subtask" in e for e in errors)

    def test_invalid_plan_subtask_not_dict(self):
        data = {"objective": "fix bug", "subtasks": ["not a dict"]}
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is False
        assert any("JSON object" in e for e in errors)

    def test_valid_plan_empty_requirements(self):
        data = {
            "objective": "fix bug",
            "subtasks": [{"id": "1", "description": "x"}],
        }
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is True
        assert plan.requirements == []

    def test_valid_plan_validates_dependency_chain(self):
        data = {
            "objective": "fix bug",
            "subtasks": [
                {"id": "1", "description": "a"},
                {"id": "2", "description": "b", "dependencies": ["1"]},
                {"id": "3", "description": "c", "dependencies": ["2"]},
            ],
        }
        valid, errors, plan = PlanValidator.validate_plan(data)
        assert valid is True
        assert len(plan.subtasks) == 3


class TestSubtaskStatusTransitions:
    def test_status_values(self):
        assert SubtaskStatus.PENDING.value == "PENDING"
        assert SubtaskStatus.IN_PROGRESS.value == "IN_PROGRESS"
        assert SubtaskStatus.COMPLETED.value == "COMPLETED"
        assert SubtaskStatus.FAILED.value == "FAILED"
        assert SubtaskStatus.BLOCKED.value == "BLOCKED"

    def test_lifecycle(self):
        plan = TaskPlan(objective="test")
        st = Subtask(id="1", description="test task")
        plan.add_subtask(st)

        assert plan.get_next_subtask().id == "1"
        st.status = SubtaskStatus.IN_PROGRESS
        assert plan.get_next_subtask() is None
        st.status = SubtaskStatus.COMPLETED
        assert plan.all_completed()
