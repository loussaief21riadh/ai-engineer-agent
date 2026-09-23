"""Structured planning for V2.0 — task decomposition into subtasks."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SubtaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class Subtask(BaseModel):
    id: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.PENDING
    acceptance_criteria: list[str] = Field(default_factory=list)
    result: str = ""


class TaskPlan(BaseModel):
    objective: str
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    subtasks: list[Subtask] = Field(default_factory=list)
    version: int = 1
    history: list[dict[str, Any]] = Field(default_factory=list)

    def add_subtask(self, subtask: Subtask) -> None:
        self.subtasks.append(subtask)

    def get_subtask(self, subtask_id: str) -> Subtask | None:
        for st in self.subtasks:
            if st.id == subtask_id:
                return st
        return None

    def get_pending_subtasks(self) -> list[Subtask]:
        return [st for st in self.subtasks if st.status == SubtaskStatus.PENDING]

    def get_completed_subtasks(self) -> list[Subtask]:
        return [st for st in self.subtasks if st.status == SubtaskStatus.COMPLETED]

    def get_next_subtask(self) -> Subtask | None:
        for st in self.subtasks:
            if st.status == SubtaskStatus.PENDING:
                deps_met = all(
                    self._dep_completed(dep) for dep in st.dependencies
                )
                if deps_met:
                    return st
        return None

    def _dep_completed(self, dep_id: str) -> bool:
        dep = self.get_subtask(dep_id)
        if dep is None:
            return True
        return dep.status == SubtaskStatus.COMPLETED

    def all_completed(self) -> bool:
        return all(
            st.status == SubtaskStatus.COMPLETED
            for st in self.subtasks
        )

    def has_failures(self) -> bool:
        return any(
            st.status == SubtaskStatus.FAILED
            for st in self.subtasks
        )

    def progress_summary(self) -> dict[str, Any]:
        total = len(self.subtasks)
        completed = len([st for st in self.subtasks if st.status == SubtaskStatus.COMPLETED])
        failed = len([st for st in self.subtasks if st.status == SubtaskStatus.FAILED])
        in_progress = len([st for st in self.subtasks if st.status == SubtaskStatus.IN_PROGRESS])
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": total - completed - failed - in_progress,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "requirements": self.requirements,
            "constraints": self.constraints,
            "subtasks": [st.model_dump() for st in self.subtasks],
            "version": self.version,
        }

    def record_version(self, reason: str) -> None:
        self.history.append({
            "version": self.version,
            "subtasks": [st.model_dump() for st in self.subtasks],
            "reason": reason,
        })

    def create_replan(self, failed_subtask_id: str, failure_reason: str) -> TaskPlan:
        self.record_version(f"Replan after failure of {failed_subtask_id}: {failure_reason}")
        new_plan = TaskPlan(
            objective=self.objective,
            requirements=self.requirements,
            constraints=self.constraints,
            version=self.version + 1,
            history=list(self.history),
        )
        for st in self.subtasks:
            if st.id == failed_subtask_id:
                new_subtask = Subtask(
                    id=f"{st.id}_revised",
                    description=f"[REPLANNED] {st.description}",
                    dependencies=st.dependencies,
                    acceptance_criteria=st.acceptance_criteria,
                )
                new_plan.add_subtask(new_subtask)
            elif st.status == SubtaskStatus.COMPLETED:
                completed_copy = Subtask(
                    id=st.id,
                    description=st.description,
                    dependencies=st.dependencies,
                    status=SubtaskStatus.COMPLETED,
                    acceptance_criteria=st.acceptance_criteria,
                    result=st.result,
                )
                new_plan.add_subtask(completed_copy)
            else:
                new_plan.add_subtask(Subtask(
                    id=st.id,
                    description=st.description,
                    dependencies=st.dependencies,
                    acceptance_criteria=st.acceptance_criteria,
                ))
        return new_plan


class PlanValidator:
    @staticmethod
    def validate_plan(data: dict[str, Any]) -> tuple[bool, list[str], TaskPlan | None]:
        errors: list[str] = []

        if not isinstance(data, dict):
            return False, ["Plan must be a JSON object"], None

        objective = data.get("objective", "")
        if not objective:
            errors.append("Plan must have an 'objective' field")

        subtasks_raw = data.get("subtasks", [])
        if not isinstance(subtasks_raw, list):
            errors.append("'subtasks' must be a list")
            return False, errors, None

        if len(subtasks_raw) == 0:
            errors.append("Plan must have at least one subtask")

        seen_ids: set[str] = set()
        subtasks: list[Subtask] = []
        for i, st_data in enumerate(subtasks_raw):
            if not isinstance(st_data, dict):
                errors.append(f"Subtask {i} must be a JSON object")
                continue

            st_id = st_data.get("id", "")
            if not st_id:
                errors.append(f"Subtask {i} must have an 'id'")
                continue

            if st_id in seen_ids:
                errors.append(f"Duplicate subtask id: '{st_id}'")
            seen_ids.add(st_id)

            desc = st_data.get("description", "")
            if not desc:
                errors.append(f"Subtask '{st_id}' must have a 'description'")

            deps = st_data.get("dependencies", [])
            if not isinstance(deps, list):
                errors.append(f"Subtask '{st_id}' dependencies must be a list")
                deps = []

            criteria = st_data.get("acceptance_criteria", [])
            if not isinstance(criteria, list):
                criteria = []

            subtasks.append(Subtask(
                id=st_id,
                description=desc,
                dependencies=[d for d in deps if isinstance(d, str)],
                acceptance_criteria=[c for c in criteria if isinstance(c, str)],
            ))

        for st in subtasks:
            for dep in st.dependencies:
                if dep not in seen_ids:
                    errors.append(f"Subtask '{st.id}' depends on unknown subtask '{dep}'")

        if errors:
            return False, errors, None

        plan = TaskPlan(
            objective=objective,
            requirements=data.get("requirements", []) if isinstance(data.get("requirements"), list) else [],
            constraints=data.get("constraints", []) if isinstance(data.get("constraints"), list) else [],
            subtasks=subtasks,
        )
        return True, [], plan
