"""Task phases and valid transitions for the autonomous engineering loop."""

from __future__ import annotations

from enum import Enum


class TaskPhase(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    INSPECT = "INSPECT"
    IMPLEMENT = "IMPLEMENT"
    TEST = "TEST"
    DIAGNOSE = "DIAGNOSE"
    FIX = "FIX"
    RETEST = "RETEST"
    REVIEW = "REVIEW"
    VALIDATE = "VALIDATE"
    REPORT = "REPORT"
    DONE = "DONE"
    FAILED = "FAILED"


VALID_TRANSITIONS: dict[TaskPhase, list[TaskPhase]] = {
    TaskPhase.UNDERSTAND: [TaskPhase.PLAN, TaskPhase.FAILED],
    TaskPhase.PLAN: [TaskPhase.INSPECT, TaskPhase.FAILED],
    TaskPhase.INSPECT: [TaskPhase.IMPLEMENT, TaskPhase.TEST, TaskPhase.FAILED],
    TaskPhase.IMPLEMENT: [TaskPhase.TEST, TaskPhase.FAILED],
    TaskPhase.TEST: [TaskPhase.REVIEW, TaskPhase.DIAGNOSE, TaskPhase.DONE, TaskPhase.FAILED],
    TaskPhase.DIAGNOSE: [TaskPhase.FIX, TaskPhase.FAILED],
    TaskPhase.FIX: [TaskPhase.RETEST, TaskPhase.FAILED],
    TaskPhase.RETEST: [TaskPhase.TEST, TaskPhase.DIAGNOSE, TaskPhase.REVIEW, TaskPhase.FAILED],
    TaskPhase.REVIEW: [TaskPhase.VALIDATE, TaskPhase.FIX, TaskPhase.FAILED],
    TaskPhase.VALIDATE: [TaskPhase.REPORT, TaskPhase.FAILED],
    TaskPhase.REPORT: [TaskPhase.DONE, TaskPhase.FAILED],
    TaskPhase.DONE: [],
    TaskPhase.FAILED: [],
}


def can_transition(current: TaskPhase, target: TaskPhase) -> bool:
    """Check if a transition from current to target phase is valid."""
    return target in VALID_TRANSITIONS.get(current, [])


def next_phases(current: TaskPhase) -> list[TaskPhase]:
    """Return valid next phases from the current phase."""
    return VALID_TRANSITIONS.get(current, [])
