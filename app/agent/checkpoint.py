from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_STALE_SECONDS, PROJECT_ROOT


class Checkpoint(BaseModel):
    """Serialized task execution state for resume."""
    task_id: str
    task: str

    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    current_phase: str = ""
    phase: str = ""
    current_subtask_id: str | None = None
    current_subtask_index: int = 0

    plan_snapshot: dict[str, Any] | None = None
    plan_version: int = 1

    retry_count: int = 0
    max_retries: int = 3
    replan_count: int = 0

    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    executions: list[dict[str, Any]] = Field(default_factory=list)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    phase_history: list[str] = Field(default_factory=list)
    diagnoses: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)

    iteration_count: int = 0
    budget_snapshot: dict[str, Any] | None = None

    files_modified: list[str] = Field(default_factory=list)
    test_results_snapshot: dict[str, Any] | None = None

    timestamp: float = Field(default_factory=time.time)
    task_completed: bool = False
    final_phase: str | None = None

    checksum: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.current_phase and self.phase:
            self.current_phase = self.phase
        elif not self.phase and self.current_phase:
            self.phase = self.current_phase

    def compute_checksum(self) -> str:
        data = self.model_dump()
        data.pop("checksum", None)
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def is_stale(self, max_age: float = CHECKPOINT_STALE_SECONDS) -> bool:
        return (time.time() - self.timestamp) > max_age


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
        tmp_path = path.with_suffix(".tmp")

        checkpoint.checksum = checkpoint.compute_checksum()
        data = checkpoint.model_dump()

        tmp_path.write_text(json.dumps(data, indent=2, default=str))
        os.replace(str(tmp_path), str(path))
        return path

    def load(self, task_id: str) -> Checkpoint | None:
        path = self._checkpoint_path(task_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            stored_checksum = raw.pop("checksum", "")
            cp = Checkpoint(**raw)
            if stored_checksum and cp.compute_checksum() != stored_checksum:
                return None
            return cp
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
