from __future__ import annotations

import time

from app.models.schemas import TokenUsage, TraceEvent
from app.agent.observer import ObserverEngine


class TestTokenUsage:
    def test_default_values(self) -> None:
        tu = TokenUsage()
        assert tu.prompt_tokens == 0
        assert tu.completion_tokens == 0
        assert tu.total_tokens == 0

    def test_addition(self) -> None:
        a = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        b = TokenUsage(prompt_tokens=200, completion_tokens=30, total_tokens=230)
        result = a.add(b)
        assert result.prompt_tokens == 300
        assert result.completion_tokens == 80
        assert result.total_tokens == 380

    def test_addition_does_not_mutate(self) -> None:
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        b = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        a.add(b)
        assert a.prompt_tokens == 10
        assert a.completion_tokens == 5
        assert a.total_tokens == 15

    def test_addition_with_zeros(self) -> None:
        a = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        b = TokenUsage()
        result = a.add(b)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150

    def test_addition_chain(self) -> None:
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        b = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        c = TokenUsage(prompt_tokens=30, completion_tokens=15, total_tokens=45)
        result = a.add(b).add(c)
        assert result.prompt_tokens == 60
        assert result.completion_tokens == 30
        assert result.total_tokens == 90


class TestTraceEvent:
    def test_default_values(self) -> None:
        event = TraceEvent()
        assert event.event_id == ""
        assert event.task_id == ""
        assert event.timestamp > 0
        assert event.event_type == ""
        assert event.success is True
        assert event.details == {}

    def test_custom_values(self) -> None:
        tokens = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        event = TraceEvent(
            event_id="evt-001",
            task_id="task-001",
            subtask_id="sub-001",
            timestamp=1234567890.0,
            event_type="tool_call",
            phase="IMPLEMENT",
            tool="write_file",
            model="openrouter/free",
            duration_ms=150.0,
            tokens=tokens,
            cost_estimate=0.001,
            success=True,
            details={"path": "src/main.py"},
        )
        assert event.event_id == "evt-001"
        assert event.task_id == "task-001"
        assert event.subtask_id == "sub-001"
        assert event.event_type == "tool_call"
        assert event.phase == "IMPLEMENT"
        assert event.tool == "write_file"
        assert event.model == "openrouter/free"
        assert event.duration_ms == 150.0
        assert event.tokens == tokens
        assert event.cost_estimate == 0.001
        assert event.success is True
        assert event.details == {"path": "src/main.py"}


class TestObserverEngine:
    def test_init_default_task_id(self) -> None:
        obs = ObserverEngine()
        assert obs.task_id != ""
        assert len(obs.task_id) == 8

    def test_init_custom_task_id(self) -> None:
        obs = ObserverEngine(task_id="my-task")
        assert obs.task_id == "my-task"

    def test_emit_returns_event(self) -> None:
        obs = ObserverEngine()
        event = obs.emit("test_event", phase="TEST")
        assert isinstance(event, TraceEvent)
        assert event.event_type == "test_event"
        assert event.phase == "TEST"
        assert event.task_id == obs.task_id

    def test_emit_stores_event(self) -> None:
        obs = ObserverEngine()
        obs.emit("event_1")
        obs.emit("event_2")
        assert len(obs.events) == 2

    def test_emit_with_tokens_accumulates(self) -> None:
        obs = ObserverEngine()
        t1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        t2 = TokenUsage(prompt_tokens=200, completion_tokens=30, total_tokens=230)
        obs.emit("llm_call", tokens=t1)
        obs.emit("llm_call", tokens=t2)
        assert obs.total_tokens.prompt_tokens == 300
        assert obs.total_tokens.completion_tokens == 80
        assert obs.total_tokens.total_tokens == 380

    def test_emit_with_cost_accumulates(self) -> None:
        obs = ObserverEngine()
        obs.emit("llm_call", cost_estimate=0.001)
        obs.emit("llm_call", cost_estimate=0.002)
        assert abs(obs.total_cost - 0.003) < 1e-10

    def test_phase_start_and_end(self) -> None:
        obs = ObserverEngine()
        obs.phase_start("IMPLEMENT")
        assert len(obs.events) == 1
        assert obs.events[0].event_type == "phase_start"
        assert obs.events[0].phase == "IMPLEMENT"

        obs.phase_end("IMPLEMENT")
        assert len(obs.events) == 2
        assert obs.events[1].event_type == "phase_end"
        assert obs.events[1].phase == "IMPLEMENT"
        assert obs.events[1].duration_ms is not None
        assert obs.events[1].duration_ms >= 0

    def test_phase_end_without_start(self) -> None:
        obs = ObserverEngine()
        event = obs.phase_end("UNKNOWN_PHASE")
        assert event.duration_ms is None

    def test_tool_call_event(self) -> None:
        obs = ObserverEngine()
        event = obs.tool_call("write_file", phase="IMPLEMENT", duration_ms=50.0, success=True)
        assert event.event_type == "tool_call"
        assert event.tool == "write_file"
        assert event.phase == "IMPLEMENT"
        assert event.duration_ms == 50.0
        assert event.success is True

    def test_llm_call_event(self) -> None:
        obs = ObserverEngine()
        tokens = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        event = obs.llm_call(
            model="openrouter/free",
            phase="UNDERSTAND",
            duration_ms=2000.0,
            tokens=tokens,
            cost_estimate=0.005,
        )
        assert event.event_type == "llm_call"
        assert event.model == "openrouter/free"
        assert event.tokens == tokens
        assert event.cost_estimate == 0.005

    def test_security_check_event(self) -> None:
        obs = ObserverEngine()
        event = obs.security_check(findings_count=2, critical_count=1, success=False)
        assert event.event_type == "security_check"
        assert event.details["findings_count"] == 2
        assert event.details["critical_count"] == 1
        assert event.success is False

    def test_review_event(self) -> None:
        obs = ObserverEngine()
        event = obs.review(verdict="APPROVE")
        assert event.event_type == "review"
        assert event.details["verdict"] == "APPROVE"

    def test_validation_event(self) -> None:
        obs = ObserverEngine()
        event = obs.validation(passed=True)
        assert event.event_type == "validation"
        assert event.details["passed"] is True

    def test_query_by_event_type(self) -> None:
        obs = ObserverEngine()
        obs.emit("tool_call")
        obs.emit("llm_call")
        obs.emit("tool_call")
        result = obs.query(event_type="tool_call")
        assert len(result) == 2

    def test_query_by_phase(self) -> None:
        obs = ObserverEngine()
        obs.emit("event", phase="UNDERSTAND")
        obs.emit("event", phase="IMPLEMENT")
        obs.emit("event", phase="UNDERSTAND")
        result = obs.query(phase="UNDERSTAND")
        assert len(result) == 2

    def test_query_by_tool(self) -> None:
        obs = ObserverEngine()
        obs.emit("tool_call", tool="write_file")
        obs.emit("tool_call", tool="read_file")
        obs.emit("tool_call", tool="write_file")
        result = obs.query(tool="write_file")
        assert len(result) == 2

    def test_query_by_model(self) -> None:
        obs = ObserverEngine()
        obs.emit("llm_call", model="openrouter/free")
        obs.emit("llm_call", model="openrouter/premium")
        result = obs.query(model="openrouter/free")
        assert len(result) == 1

    def test_query_success_only(self) -> None:
        obs = ObserverEngine()
        obs.emit("tool_call", success=True)
        obs.emit("tool_call", success=False)
        obs.emit("tool_call", success=True)
        result = obs.query(success_only=True)
        assert len(result) == 2

    def test_query_combined_filters(self) -> None:
        obs = ObserverEngine()
        obs.emit("tool_call", phase="IMPLEMENT", tool="write_file", success=True)
        obs.emit("tool_call", phase="IMPLEMENT", tool="read_file", success=True)
        obs.emit("tool_call", phase="TEST", tool="write_file", success=True)
        result = obs.query(event_type="tool_call", phase="IMPLEMENT", tool="write_file")
        assert len(result) == 1

    def test_summarize(self) -> None:
        obs = ObserverEngine(task_id="sum-test")
        t1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        obs.emit("phase_start", phase="UNDERSTAND")
        obs.emit("llm_call", model="openrouter/free", tokens=t1, cost_estimate=0.001)
        obs.emit("tool_call", tool="read_file", phase="INSPECT")
        obs.emit("tool_call", tool="write_file", phase="IMPLEMENT", success=False)
        obs.phase_end("UNDERSTAND")

        summary = obs.summarize()
        assert summary["task_id"] == "sum-test"
        assert summary["total_events"] == 5
        assert "UNDERSTAND" in summary["phases_seen"]
        assert "read_file" in summary["tools_used"]
        assert "write_file" in summary["tools_used"]
        assert "openrouter/free" in summary["models_used"]
        assert summary["total_tokens"]["prompt_tokens"] == 100
        assert abs(summary["total_cost"] - 0.001) < 1e-10
        assert summary["failed_events"] == 1

    def test_events_returns_copy(self) -> None:
        obs = ObserverEngine()
        obs.emit("event")
        events = obs.events
        events.clear()
        assert len(obs.events) == 1

    def test_subtask_id_propagation(self) -> None:
        obs = ObserverEngine()
        event = obs.emit("tool_call", subtask_id="sub-001")
        assert event.subtask_id == "sub-001"

    def test_details_propagation(self) -> None:
        obs = ObserverEngine()
        event = obs.emit("tool_call", details={"custom": "value"})
        assert event.details == {"custom": "value"}

    def test_multiple_phases_timing(self) -> None:
        obs = ObserverEngine()
        obs.phase_start("UNDERSTAND")
        obs.phase_start("PLAN")
        obs.phase_end("UNDERSTAND")
        obs.phase_end("PLAN")

        understand_events = [e for e in obs.events if e.phase == "UNDERSTAND"]
        plan_events = [e for e in obs.events if e.phase == "PLAN"]
        assert len(understand_events) == 2
        assert len(plan_events) == 2

        understand_end = [e for e in understand_events if e.event_type == "phase_end"][0]
        plan_end = [e for e in plan_events if e.event_type == "phase_end"][0]
        assert understand_end.duration_ms is not None
        assert plan_end.duration_ms is not None
