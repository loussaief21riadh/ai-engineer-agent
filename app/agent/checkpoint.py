from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT


class Checkpoint(BaseModel):
    """Serialized task execution state for resume."""
    task_id: str
    task: str
    phase: str
    subtask_id: str | None = None
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    plan_snapshot: dict[str, Any] | None = None
    executions: list[dict[str, Any]] = Field(default_factory=list)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    phase_history: list[str] = Field(default_factory=list)
    diagnoses: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)
    iteration_count: int = 0
    retry_count: int = 0
    timestamp: float = Field(default_factory=time.time)
    version: int = 1


class CheckpointStore:
    """Persists task checkpoints to disk for resume capability."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or (PROJECT_ROOT / ".opencode" / "checkpoints")

    def _ensure_dir(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, task_id: str) -> Path:
        return self._base_dir / f"{task_id}.json"

    def save(self, checkpoint: Checkpoint) -> Path:
        self._ensure_dir()
        path = self._checkpoint_path(checkpoint.task_id)
        data = checkpoint.model_dump()
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def load(self, task_id: str) -> Checkpoint | None:
        path = self._checkpoint_path(task_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            return Checkpoint(**raw)
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def exists(self, task_id: str) -> bool:
        return self._checkpoint_path(task_id).exists()

    def delete(self, task_id: str) -> bool:
        path = self._checkpoint_path(task_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_task_ids(self) -> list[str]:
        self._ensure_dir()
        return [
            p.stem for p in self._base_dir.glob("*.json")
        ]

    def list_checkpoints(self) -> list[Checkpoint]:
        checkpoints: list[Checkpoint] = []
        for task_id in self.list_task_ids():
            cp = self.load(task_id)
            if cp is not None:
                checkpoints.append(cp)
        return checkpoints

    def prune(self, max_age_seconds: float = 86400 * 7) -> int:
        self._ensure_dir()
        now = time.time()
        removed = 0
        for path in self._base_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                ts = data.get("timestamp", 0)
                if now - ts > max_age_seconds:
                    path.unlink()
                    removed += 1
            except (json.JSONDecodeError, ValueError):
                path.unlink()
                removed += 1
        return removed
