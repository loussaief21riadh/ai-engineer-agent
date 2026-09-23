from __future__ import annotations

import json
import time
from pathlib import Path

from app.agent.checkpoint import Checkpoint, CheckpointStore


class TestCheckpoint:
    def test_default_values(self) -> None:
        cp = Checkpoint(task_id="t1", task="fix bug", current_phase="UNDERSTAND")
        assert cp.task_id == "t1"
        assert cp.task == "fix bug"
        assert cp.current_phase == "UNDERSTAND"
        assert cp.plan_version == 1
        assert cp.current_subtask_id is None
        assert cp.context_snapshot == {}
        assert cp.executions == []
        assert cp.trace_events == []
        assert cp.phase_history == []
        assert cp.diagnoses == []
        assert cp.fixes == []
        assert cp.iteration_count == 0
        assert cp.retry_count == 0
        assert cp.timestamp > 0
        assert cp.schema_version > 0

    def test_full_checkpoint(self) -> None:
        cp = Checkpoint(
            task_id="t2",
            task="add feature",
            current_phase="IMPLEMENT",
            current_subtask_id="sub-1",
            current_subtask_index=0,
            context_snapshot={"key": "value"},
            plan_snapshot={"objective": "test"},
            plan_version=3,
            executions=[{"tool": "write_file"}],
            trace_events=[{"event": "phase_start"}],
            phase_history=["UNDERSTAND", "PLAN", "IMPLEMENT"],
            diagnoses=["logic error"],
            fixes=["fixed variable"],
            iteration_count=5,
            retry_count=2,
            replan_count=1,
        )
        assert cp.current_subtask_id == "sub-1"
        assert cp.context_snapshot == {"key": "value"}
        assert cp.plan_snapshot == {"objective": "test"}
        assert len(cp.executions) == 1
        assert len(cp.trace_events) == 1
        assert cp.phase_history == ["UNDERSTAND", "PLAN", "IMPLEMENT"]
        assert cp.plan_version == 3

    def test_backward_compat_phase_alias(self) -> None:
        cp = Checkpoint(task_id="t1", task="test", phase="PLAN")
        assert cp.current_phase == "PLAN"
        assert cp.phase == "PLAN"

    def test_checksum_integrity(self) -> None:
        cp = Checkpoint(task_id="t1", task="test", current_phase="UNDERSTAND")
        checksum = cp.compute_checksum()
        assert len(checksum) == 32
        assert cp.compute_checksum() == checksum


class TestCheckpointStore:
    def test_save_and_load(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(task_id="t1", task="fix bug", current_phase="PLAN")
        store.save(cp)
        loaded = store.load("t1")
        assert loaded is not None
        assert loaded.task_id == "t1"
        assert loaded.task == "fix bug"
        assert loaded.current_phase == "PLAN"

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        assert store.load("nonexistent") is None

    def test_exists(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        assert not store.exists("t1")
        store.save(Checkpoint(task_id="t1", task="test", current_phase="UNDERSTAND"))
        assert store.exists("t1")

    def test_delete(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        store.save(Checkpoint(task_id="t1", task="test", current_phase="UNDERSTAND"))
        assert store.delete("t1")
        assert not store.exists("t1")
        assert not store.delete("t1")

    def test_list_task_ids(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        store.save(Checkpoint(task_id="t1", task="a", current_phase="UNDERSTAND"))
        store.save(Checkpoint(task_id="t2", task="b", current_phase="PLAN"))
        ids = store.list_task_ids()
        assert set(ids) == {"t1", "t2"}

    def test_list_checkpoints(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        store.save(Checkpoint(task_id="t1", task="a", current_phase="UNDERSTAND"))
        store.save(Checkpoint(task_id="t2", task="b", current_phase="PLAN"))
        checkpoints = store.list_checkpoints()
        assert len(checkpoints) == 2
        tasks = {cp.task for cp in checkpoints}
        assert tasks == {"a", "b"}

    def test_overwrite_checkpoint(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        store.save(Checkpoint(task_id="t1", task="v1", current_phase="UNDERSTAND"))
        store.save(Checkpoint(task_id="t1", task="v2", current_phase="PLAN"))
        loaded = store.load("t1")
        assert loaded is not None
        assert loaded.task == "v2"
        assert loaded.current_phase == "PLAN"

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        path = tmp_path / "bad.json"
        path.write_text("not valid json{{{")
        assert store.load("bad") is None

    def test_corrupt_checksum_returns_none(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(task_id="t1", task="test", current_phase="UNDERSTAND")
        store.save(cp)
        path = tmp_path / "t1.json"
        data = json.loads(path.read_text())
        data["checksum"] = "badchecksum"
        path.write_text(json.dumps(data, indent=2))
        assert store.load("t1") is None

    def test_create_directories(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        store = CheckpointStore(base_dir=deep)
        store.save(Checkpoint(task_id="t1", task="test", current_phase="UNDERSTAND"))
        assert store.load("t1") is not None

    def test_prune_old_checkpoints(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        store.save(Checkpoint(task_id="t1", task="old", current_phase="UNDERSTAND", timestamp=time.time() - 86400 * 10))
        store.save(Checkpoint(task_id="t2", task="new", current_phase="UNDERSTAND", timestamp=time.time()))
        removed = store.prune(max_age_seconds=86400 * 7)
        assert removed == 1
        assert store.load("t1") is None
        assert store.load("t2") is not None

    def test_prune_removes_corrupt_files(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        (tmp_path / "corrupt.json").write_text("not json")
        removed = store.prune()
        assert removed == 1
        assert not (tmp_path / "corrupt.json").exists()

    def test_checkpoint_roundtrip_complex_data(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(
            task_id="complex",
            task="build feature",
            current_phase="SECURITY_CHECK",
            current_subtask_id="sub-3",
            current_subtask_index=2,
            context_snapshot={"observations": [{"content": "test", "trust": "TOOL_VERIFIED"}]},
            plan_snapshot={"objective": "test", "subtasks": [{"id": "s1", "description": "do stuff"}]},
            plan_version=5,
            executions=[{"step": 1, "tool_name": "write_file", "success": True}],
            trace_events=[{"event_id": "e1", "event_type": "phase_start", "phase": "IMPLEMENT"}],
            phase_history=["UNDERSTAND", "PLAN", "INSPECT", "IMPLEMENT", "TEST", "SECURITY_CHECK"],
            diagnoses=["found issue"],
            fixes=["applied fix"],
            iteration_count=12,
            retry_count=1,
            replan_count=0,
        )
        store.save(cp)
        loaded = store.load("complex")
        assert loaded is not None
        assert loaded.context_snapshot["observations"][0]["trust"] == "TOOL_VERIFIED"
        assert loaded.plan_snapshot["subtasks"][0]["id"] == "s1"
        assert loaded.executions[0]["tool_name"] == "write_file"
        assert loaded.trace_events[0]["event_type"] == "phase_start"
        assert loaded.plan_version == 5

    def test_atomic_write_no_partial(self, tmp_path: Path) -> None:
        store = CheckpointStore(base_dir=tmp_path)
        cp = Checkpoint(task_id="t1", task="test", current_phase="UNDERSTAND")
        store.save(cp)
        assert not (tmp_path / "t1.tmp").exists()
        loaded = store.load("t1")
        assert loaded is not None
