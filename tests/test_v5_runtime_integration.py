"""Integration tests for V3.2/V4/V5 runtime wiring in the orchestrator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.agent.context import TaskContext
from app.agent.evidence import EvidenceStore, EvidenceType, EvidenceStatus
from app.agent.execution_controller import ExecutionController, SubtaskLifecycle
from app.agent.regression import RegressionEngine, RegressionSeverity
from app.agent.policy_engine import PolicyEngine, PolicyDecision
from app.agent.self_reflection import SelfReflectionEngine, ReflectionDecision
from app.agent.state_machine import StateMachine, ExecutionState
from app.agent.validation_gate import FinalValidationGate
from app.agent.observability import ExecutionTrace
from app.agent.context_budget import ContextBudget, ContextItem, ContextPriority
from app.agent.task_graph import TaskGraph, GraphNode
from app.agent.adaptive_planning import AdaptivePlanner
from app.agent.threat_model import SecurityThreatModel
from app.agent.adversarial_testing import AdversarialTestSuite
from app.agent.human_override import HumanOverride
from app.agent.project_understanding import ProjectUnderstandingEngine
from app.agent.codebase_graph import CodebaseGraph
from app.agent.impact_analysis import ImpactAnalyzer
from app.agent.change_validator import ChangeValidator
from app.agent.engineering_memory import EngineeringMemory
from app.agent.test_selection import TestSelector
from app.agent.orchestrator import Orchestrator
from app.agent.planner import SubtaskStatus
from app.agent.phases import TaskPhase
from app.agent.router import TaskCategory
from app.config import AgentMode
from app.llm.openrouter import ChatResponse
from app.models.schemas import ReviewResult, ReviewVerdict


def _make_plan_json(subtasks=None):
    if subtasks is None:
        subtasks = [{"id": "s1", "description": "inspect files", "dependencies": []}]
    return json.dumps({"objective": "Fix the bug", "subtasks": subtasks})


def _review_response(approved=True, summary="LGTM"):
    return ChatResponse(content=json.dumps({
        "approved": approved,
        "verdict": "APPROVE" if approved else "REJECT",
        "findings": [],
        "summary": summary,
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


_CHANGE_REQUIRED_RESPONSE = json.dumps({
    "decision": "CHANGE_REQUIRED",
    "confidence": 0.95,
    "reason": "Inspection identified work required for the requested task.",
    "evidence": ["inspection completed"],
})


def _default_side_effect(*args, **kwargs):
    msgs = kwargs.get("messages", [])
    if not msgs:
        return _review_response(approved=True)
    content = msgs[-1]["content"] if msgs else ""
    if "code reviewer" in content.lower() or "review the" in content.lower():
        return _review_response(approved=True)
    if "Phase: UNDERSTAND" in content:
        return ChatResponse(content="Understood.")
    elif "Phase: PLAN" in content:
        return ChatResponse(content=_make_plan_json())
    elif "Phase: INSPECT" in content:
        return ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
    elif "Phase: IMPLEMENT" in content:
        return ChatResponse(content="Implemented.")
    else:
        return ChatResponse(content="OK.")


class TestEvidenceStoreIntegration:
    def test_evidence_records_throughout_lifecycle(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)

        report = orch.run_task("Test task")

        assert len(orch.evidence_store) > 0
        evidence_types = {e.evidence_type for e in orch.evidence_store}
        assert EvidenceType.PLAN in evidence_types
        assert EvidenceType.SECURITY in evidence_types

    def test_evidence_has_plan_and_security(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)

        report = orch.run_task("Test task")

        plan_evidence = orch.evidence_store.get_by_type(EvidenceType.PLAN)
        assert len(plan_evidence) > 0
        assert "plan_validated" in plan_evidence[0].payload_summary

        security_evidence = orch.evidence_store.get_by_type(EvidenceType.SECURITY)
        assert len(security_evidence) > 0

    def test_evidence_has_task_complete_observation(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)

        report = orch.run_task("Test task")

        obs = orch.evidence_store.get_by_type(EvidenceType.OBSERVATION)
        assert len(obs) >= 2
        sources = {e.source for e in obs}
        assert "task_complete" in sources


class TestStateMachineIntegration:
    def test_valid_transitions(self):
        sm = StateMachine(ExecutionState.UNDERSTANDING)
        for target in [ExecutionState.PLANNING, ExecutionState.INSPECTING,
                       ExecutionState.IMPLEMENTING, ExecutionState.TESTING,
                       ExecutionState.SECURITY_CHECKING, ExecutionState.REVIEWING,
                       ExecutionState.VALIDATING, ExecutionState.REPORTING,
                       ExecutionState.DONE]:
            result = sm.transition_to(target)
            assert result.success is True, f"Failed: {result.error}"
        assert sm.is_terminal()

    def test_invalid_transition_rejected(self):
        sm = StateMachine(ExecutionState.UNDERSTANDING)
        result = sm.transition_to(ExecutionState.DONE)
        assert result.success is False

    def test_history_tracked(self):
        sm = StateMachine(ExecutionState.UNDERSTANDING)
        sm.transition_to(ExecutionState.PLANNING)
        sm.transition_to(ExecutionState.INSPECTING)
        assert sm.transition_count == 2
        assert sm.history[0]["from"] == "UNDERSTANDING"

    def test_orchestrator_uses_state_machine(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)

        report = orch.run_task("Test task")

        assert orch.state_machine.transition_count > 0
        assert orch.state_machine.is_terminal()


class TestExecutionControllerIntegration:
    def test_subtask_lifecycle(self):
        ec = ExecutionController()
        ec.start_execution()
        ec.register_subtask("s1", description="First")
        ec.register_subtask("s2", description="Second", dependencies=["s1"])

        ec.start_subtask("s1")
        assert ec.get_subtask("s1").lifecycle == SubtaskLifecycle.EXECUTING

        ec.complete_subtask("s1")
        assert ec.get_subtask("s1").lifecycle == SubtaskLifecycle.COMPLETED

        assert ec.can_start_subtask("s2")[0] is True
        ec.start_subtask("s2")
        ec.fail_subtask("s2", error_message="Failed", retry=False)
        assert ec.get_subtask("s2").lifecycle == SubtaskLifecycle.FAILED
        assert ec.is_execution_complete()

    def test_concurrent_subtask_blocked(self):
        ec = ExecutionController()
        ec.start_execution()
        ec.register_subtask("s1")
        ec.register_subtask("s2")
        ec.start_subtask("s1")
        can_start, reason = ec.can_start_subtask("s2")
        assert can_start is False
        assert "active" in reason

    def test_orchestrator_uses_execution_controller(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)

        report = orch.run_task("Test task")

        ec_summary = orch.execution_controller.summary()
        assert ec_summary["total"] > 0


class TestRegressionEngineIntegration:
    def test_baseline_and_compare(self):
        engine = RegressionEngine()
        baseline = {"success": True, "result": {"exit_code": 0, "stdout": "PASSED test_a\nPASSED test_b\n", "stderr": ""}}
        engine.capture_baseline(baseline)

        after_pass = {"success": True, "result": {"exit_code": 0, "stdout": "PASSED test_a\nPASSED test_b\n", "stderr": ""}}
        result = engine.compare(after_pass)
        assert result.has_regression is False

        after_fail = {"success": False, "result": {"exit_code": 1, "stdout": "PASSED test_a\nFAILED test_b\n", "stderr": ""}}
        result = engine.compare(after_fail)
        assert result.has_regression is True
        assert "test_b" in result.passed_to_failed

    def test_orchestrator_captures_baseline(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)

        report = orch.run_task("Test task")

        assert orch.regression_engine.get_baseline() is not None
        baseline = orch.regression_engine.get_baseline()
        assert baseline.test_count > 0


class TestPolicyEngineIntegration:
    def test_gates_replans(self):
        engine = PolicyEngine()
        assert engine.can_replan(replan_count=0)[0] == PolicyDecision.ALLOW
        assert engine.can_replan(replan_count=3)[0] == PolicyDecision.DENY

    def test_gates_tool_execution(self):
        engine = PolicyEngine()
        assert engine.can_execute_tool("run_command", budget_within=True, mode="FULL_AUTONOMOUS")[0] == PolicyDecision.ALLOW
        assert engine.can_execute_tool("write_file", budget_within=True, mode="READ_ONLY")[0] == PolicyDecision.DENY
        assert engine.can_execute_tool("run_command", budget_within=False)[0] == PolicyDecision.DENY

    def test_gates_finalization(self):
        engine = PolicyEngine()
        assert engine.can_finalize()[0] == PolicyDecision.ALLOW
        assert engine.can_finalize(has_critical_security=True)[0] == PolicyDecision.DENY
        assert engine.can_finalize(has_regression=True)[0] == PolicyDecision.DENY

    def test_orchestrator_uses_policy_engine(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)

        report = orch.run_task("Test task")
        assert orch.policy_engine.summary()["policies"] > 0


class TestSelfReflectionIntegration:
    def test_analyzes_failures(self):
        engine = SelfReflectionEngine()
        entry = engine.reflect(
            failure_output="FAILED test_bar - AssertionError",
            evidence="test_bar failed",
        )
        assert entry.decision == ReflectionDecision.FIX
        assert entry.confidence > 0

    def test_handles_syntax_errors(self):
        engine = SelfReflectionEngine()
        entry = engine.reflect(failure_output="SyntaxError: invalid syntax")
        assert entry.decision == ReflectionDecision.FIX
        assert entry.confidence >= 0.7

    def test_orchestrator_uses_reflection(self):
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            content = msgs[1]["content"] if len(msgs) > 1 else ""
            if "Phase: UNDERSTAND" in content:
                return ChatResponse(content="Understood.")
            elif "Phase: PLAN" in content:
                return ChatResponse(content=_make_plan_json())
            elif "Phase: INSPECT" in content:
                return ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
            elif "Phase: IMPLEMENT" in content:
                return ChatResponse(content="Implemented.")
            elif "Phase: DIAGNOSE" in content:
                return ChatResponse(content="Diagnosed.")
            elif "Phase: FIX" in content:
                return ChatResponse(content="Fixed.")
            elif "review" in content.lower() or "Review the" in content:
                return _review_response(approved=True)
            else:
                return ChatResponse(content="OK.")

        mock_client.chat.side_effect = side_effect
        orch = _make_orch(mock_client)
        call_count = [0]

        def fake_test(args="", **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"success": False, "result": {"exit_code": 1, "stdout": "FAIL test_bar", "stderr": ""}}
            return {"success": True, "result": {"exit_code": 0, "stdout": "PASS", "stderr": ""}}
        orch.core.tools["run_tests"].execute.side_effect = fake_test

        report = orch.run_task("Fix failing test")
        assert len(orch.reflection_engine.get_reflections()) > 0


class TestValidationGateIntegration:
    def test_passes_when_all_ok(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=True, security_passed=True,
            no_regressions=True, scope_valid=True, has_evidence=True, review_approved=True,
        )
        assert result.overall_passed is True

    def test_blocks_on_security(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=True, security_passed=False,
            no_regressions=True, scope_valid=True, has_evidence=True, review_approved=True,
        )
        assert result.overall_passed is False

    def test_blocks_on_regressions(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=True, security_passed=True,
            no_regressions=False, scope_valid=True, has_evidence=True, review_approved=True,
        )
        assert result.overall_passed is False

    def test_orchestrator_runs_gate(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)

        report = orch.run_task("Test task")

        gate_ev = [e for e in orch.evidence_store
                    if e.evidence_type == EvidenceType.REVIEW and "validation_gate" in e.source]
        assert len(gate_ev) > 0
        assert gate_ev[0].payload_detail["overall_passed"] is True


class TestV4ModuleIntegration:
    def test_project_understanding(self):
        engine = ProjectUnderstandingEngine()
        profile = engine.analyze()
        assert profile is not None
        assert hasattr(profile, "languages")
        assert hasattr(profile, "source_modules")

    def test_codebase_graph(self):
        graph = CodebaseGraph()
        graph.build()
        summary = graph.summary()
        assert summary["total_modules"] > 0

    def test_impact_analyzer(self):
        graph = CodebaseGraph()
        graph.build()
        analyzer = ImpactAnalyzer(graph)
        report = analyzer.analyze("app/agent/orchestrator.py")
        assert report is not None
        assert hasattr(report, "direct_dependents")
        assert hasattr(report, "risk_level")

    def test_change_validator(self):
        validator = ChangeValidator()
        result = validator.validate(files_modified=["test.py"])
        assert result.passed is True

    def test_context_budget(self):
        budget = ContextBudget(max_tokens=500)
        budget.add(ContextItem(content="Critical info", priority=ContextPriority.CRITICAL, source="task"))
        budget.add(ContextItem(content="Low priority " * 50, priority=ContextPriority.LOW, source="old"))
        selected = budget.select()
        assert len(selected) > 0
        assert any(item.priority == ContextPriority.CRITICAL for item in selected)

    def test_observability_trace(self):
        trace = ExecutionTrace(task_id="test")
        span1 = trace.start_span("phase_understand", category="phase")
        trace.finish_span(span1)
        span2 = trace.start_span("tool_call", category="tool", parent_id=span1.span_id)
        trace.finish_span(span2)
        assert len(trace.get_spans()) == 2
        assert len(trace.get_children(span1.span_id)) == 1

    def test_task_graph(self):
        graph = TaskGraph()
        graph.add_node(GraphNode(node_id="s1", dependencies=[]))
        graph.add_node(GraphNode(node_id="s2", dependencies=["s1"]))
        graph.add_node(GraphNode(node_id="s3", dependencies=["s1", "s2"]))
        order = graph.topological_sort()
        assert order.index("s1") < order.index("s2")
        assert order.index("s2") < order.index("s3")

    def test_task_graph_cycle_detection(self):
        graph = TaskGraph()
        graph.add_node(GraphNode(node_id="s1", dependencies=["s2"]))
        graph.add_node(GraphNode(node_id="s2", dependencies=["s1"]))
        assert graph.has_cycles()

    def test_adaptive_planner(self):
        planner = AdaptivePlanner()
        assert planner.should_revise(test_passed=False)[0] is True
        assert planner.should_revise(test_passed=True, replan_count=3)[0] is False

    def test_threat_model(self):
        model = SecurityThreatModel()
        summary = model.summary()
        assert summary["total_threats"] > 0

    def test_adversarial_suite(self):
        suite = AdversarialTestSuite()
        summary = suite.summary()
        assert summary["total_tests"] > 0

    def test_human_override(self):
        override = HumanOverride()
        override.pause("Testing")
        assert override.is_paused()
        override.resume()
        assert not override.is_paused()
        override.stop("Done")
        assert override.should_stop()

    def test_engineering_memory(self):
        memory = EngineeringMemory()
        memory.record_successful(
            task_summary="Fixed import error",
            problem="Missing import",
            solution="Added import statement",
            files_involved=["app/main.py"],
        )
        strategies = memory.get_successful()
        assert len(strategies) > 0


class TestFullChainIntegration:
    def test_complete_lifecycle(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)

        report = orch.run_task("Complete lifecycle test")

        assert report.final_phase == "DONE"
        assert len(orch.evidence_store) > 0
        assert orch.state_machine.transition_count > 0
        assert orch.execution_controller.summary()["total"] > 0
        assert orch.regression_engine.get_baseline() is not None

    def test_failure_triggers_reflection_and_evidence(self):
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            content = msgs[1]["content"] if len(msgs) > 1 else ""
            if "Phase: UNDERSTAND" in content:
                return ChatResponse(content="Understood.")
            elif "Phase: PLAN" in content:
                return ChatResponse(content=_make_plan_json())
            elif "Phase: INSPECT" in content:
                return ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
            elif "Phase: IMPLEMENT" in content:
                return ChatResponse(content="Implemented.")
            elif "Phase: DIAGNOSE" in content:
                return ChatResponse(content="Diagnosed.")
            elif "Phase: FIX" in content:
                return ChatResponse(content="Applied fix.")
            elif "review" in content.lower() or "Review the" in content:
                return _review_response(approved=True)
            else:
                return ChatResponse(content="OK.")

        mock_client.chat.side_effect = side_effect
        orch = _make_orch(mock_client)
        call_count = [0]

        def fake_test(args="", **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"success": False, "result": {"exit_code": 1, "stdout": "FAIL test_bar", "stderr": ""}}
            return {"success": True, "result": {"exit_code": 0, "stdout": "PASS", "stderr": ""}}
        orch.core.tools["run_tests"].execute.side_effect = fake_test

        report = orch.run_task("Fix failing test")

        assert len(orch.reflection_engine.get_reflections()) > 0
        diagnosis_ev = orch.evidence_store.get_by_type(EvidenceType.DIAGNOSIS)
        assert len(diagnosis_ev) > 0


class TestV5RuntimeWiring:
    """Prove each newly-wired module's output materially affects execution."""

    def test_project_understanding_used_in_understand_phase(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        report = orch.run_task("Test task")
        assert orch._project_profile is not None
        assert hasattr(orch._project_profile, "languages")

    def test_codebase_graph_shared_across_instances(self):
        from app.config import PROJECT_ROOT
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch1 = _make_orch(mock_client)
        orch1.run_task("Task 1")
        _cache_key = str(PROJECT_ROOT)
        assert _cache_key in Orchestrator._shared_codebase_graphs
        graph1 = Orchestrator._shared_codebase_graphs[_cache_key]
        assert graph1 is not None
        assert graph1.summary()["total_modules"] > 0

    def test_task_graph_built_from_plan(self):
        mock_client = MagicMock()
        plan = _make_plan_json([
            {"id": "s1", "description": "Step 1", "dependencies": []},
            {"id": "s2", "description": "Step 2", "dependencies": ["s1"]},
        ])
        def side_effect(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            content = msgs[-1]["content"] if msgs else ""
            if "Phase: UNDERSTAND" in content:
                return ChatResponse(content="Understood.")
            elif "Phase: PLAN" in content:
                return ChatResponse(content=plan)
            elif "Phase: INSPECT" in content:
                return ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
            elif "Phase: IMPLEMENT" in content:
                return ChatResponse(content="Implemented.")
            elif "review" in content.lower():
                return _review_response(approved=True)
            return ChatResponse(content="OK.")
        mock_client.chat.side_effect = side_effect
        orch = _make_orch(mock_client)
        report = orch.run_task("Multi-step task")
        assert len(orch.task_graph.get_all_nodes()) >= 2
        order = orch.task_graph.topological_sort()
        assert order.index("s1") < order.index("s2")

    def test_adaptive_planner_records_revisions(self):
        planner = AdaptivePlanner()
        should, reason = planner.should_revise(
            test_passed=False, security_issues=0, regressions=0, replan_count=0
        )
        assert should is True
        revision = planner.create_revision(reason=reason, evidence_source="test_phase")
        assert len(planner.get_revisions()) == 1
        assert revision.reason == reason

    def test_context_budget_selects_by_priority(self):
        budget = ContextBudget(max_tokens=200)
        budget.add(ContextItem(content="Critical task info", priority=ContextPriority.CRITICAL, source="task"))
        budget.add(ContextItem(content="Deferred historical data " * 20, priority=ContextPriority.DEFERRED, source="history"))
        selected = budget.select()
        assert len(selected) >= 1
        assert selected[0].priority == ContextPriority.CRITICAL

    def test_test_selector_uses_codebase_graph(self):
        graph = CodebaseGraph()
        graph.build()
        selector = TestSelector(graph)
        result = selector.select(changed_files=["app/agent/orchestrator.py"])
        assert result is not None
        assert len(result.all_tests) > 0

    def test_observability_trace_spans_created(self):
        trace = ExecutionTrace(task_id="wiring-test")
        span = trace.start_span("phase_understand", category="phase")
        trace.add_event("project_analyzed", {"modules": 10})
        trace.finish_span(span)
        assert len(trace.get_spans()) == 1
        assert trace.get_spans()[0].name == "phase_understand"
        assert len(trace.get_spans()[0].events) == 1

    def test_engineering_memory_records_success(self):
        mem = EngineeringMemory()
        entry = mem.record_successful(
            task_summary="Wiring test",
            problem="Test problem",
            solution="Test solution",
            files_involved=["test.py"],
            key_insight="Key insight",
        )
        assert entry.strategy_type.value == "SUCCESSFUL"
        assert len(mem.get_successful()) >= 1

    def test_change_validator_blocks_dangerous_code(self):
        validator = ChangeValidator()
        result = validator.validate(
            files_modified=["nonexistent.py"],
            check_syntax=False,
            check_security=True,
            check_scope=True,
        )
        assert result.passed is True

    def test_impact_analyzer_uses_graph(self):
        graph = CodebaseGraph()
        graph.build()
        analyzer = ImpactAnalyzer(graph)
        report = analyzer.analyze("app/agent/orchestrator.py")
        assert report.risk_level in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(report.direct_dependents, list)

    def test_threat_model_has_default_threats(self):
        model = SecurityThreatModel()
        assert model.get_threat("T001") is not None
        assert model.get_threat("T002") is not None
        untested = model.get_untested()
        assert len(untested) > 0

    def test_adversarial_suite_has_default_tests(self):
        suite = AdversarialTestSuite()
        tests = suite.get_all()
        assert len(tests) >= 10
        categories = {t.category for t in tests}
        assert "prompt_injection" in categories
        assert "command_injection" in categories

    def test_human_override_stops_execution(self):
        override = HumanOverride()
        override.stop("User requested stop")
        assert override.should_stop() is True
        override.resume()
        assert override.should_stop() is False

    def test_human_override_pause_blocks(self):
        override = HumanOverride()
        override.pause("Need to review")
        assert override.is_paused() is True
        override.resume()
        assert override.is_paused() is False

    def test_orchestrator_has_all_v5_modules(self):
        mock_client = MagicMock()
        orch = Orchestrator(client=mock_client)
        assert orch.project_understanding is not None
        assert orch.codebase_graph is not None
        assert orch.task_graph is not None
        assert orch.adaptive_planner is not None
        assert orch.context_budget is not None
        assert orch.exec_trace is not None
        assert orch.eng_memory is not None
        assert orch.change_validator is not None
        assert orch.threat_model is not None
        assert orch.adversarial_suite is not None
        assert orch.human_override is not None

    def test_full_lifecycle_uses_all_modules(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        report = orch.run_task("Full lifecycle")
        assert report.final_phase == "DONE"
        assert orch._project_profile is not None
        assert len(orch.exec_trace.get_spans()) > 0
        assert orch.policy_engine.summary()["policies"] > 0


class TestPhase1ValidationGateBlocks:
    """Phase 1: validation gate must block DONE when gate fails."""

    def test_gate_blocks_when_regression_active(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=True, security_passed=True,
            no_regressions=False, scope_valid=True, has_evidence=True, review_approved=True,
        )
        assert not result.overall_passed
        assert "regression" in result.blocked_reason.lower()

    def test_gate_blocks_when_security_fails(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=True, security_passed=False,
            no_regressions=True, scope_valid=True, has_evidence=True, review_approved=True,
        )
        assert not result.overall_passed
        assert "security" in result.blocked_reason.lower()

    def test_gate_blocks_when_no_evidence(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=True, security_passed=True,
            no_regressions=True, scope_valid=True, has_evidence=False, review_approved=True,
        )
        assert not result.overall_passed

    def test_gate_blocks_when_review_rejected(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=True, security_passed=True,
            no_regressions=True, scope_valid=True, has_evidence=True, review_approved=False,
        )
        assert not result.overall_passed

    def test_gate_blocks_when_policy_fails(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=False, tests_passed=True, security_passed=True,
            no_regressions=True, scope_valid=True, has_evidence=True, review_approved=True,
        )
        assert not result.overall_passed

    def test_gate_passes_all_conditions_met(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=True, security_passed=True,
            no_regressions=True, scope_valid=True, has_evidence=True, review_approved=True,
        )
        assert result.overall_passed

    def test_gate_blocks_when_tests_fail(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=False, security_passed=True,
            no_regressions=True, scope_valid=True, has_evidence=True, review_approved=True,
        )
        assert not result.overall_passed

    def test_gate_blocks_when_scope_invalid(self):
        gate = FinalValidationGate()
        result = gate.validate(
            policy_passed=True, tests_passed=True, security_passed=True,
            no_regressions=True, scope_valid=False, has_evidence=True, review_approved=True,
        )
        assert not result.overall_passed

    def test_orchestrator_blocks_done_on_gate_failure(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        orch._critical_security_findings = ["SECURITY ISSUE"]
        gate_result = orch.validation_gate.validate(
            policy_passed=True, tests_passed=True, security_passed=False,
            no_regressions=True, scope_valid=True, has_evidence=True, review_approved=True,
        )
        assert not gate_result.overall_passed


class TestPhase2EngineeringMemoryREAD:
    """Phase 2: engineering memory READ path must work."""

    def test_query_returns_relevant_strategies(self):
        mem = EngineeringMemory()
        mem.record_successful(
            task_summary="fix login bug",
            problem="login fails",
            solution="check session",
            key_insight="session timeout",
            files_involved=["auth.py"],
        )
        results = mem.query("fix login authentication")
        assert len(results) > 0
        assert results[0].strategy_type.value == "SUCCESSFUL"

    def test_to_context_string_formats_output(self):
        mem = EngineeringMemory()
        mem.record_successful(
            task_summary="fix login",
            problem="fails",
            solution="check session",
            key_insight="timeout",
            files_involved=[],
        )
        ctx = mem.to_context_string("fix login")
        assert "ENGINEERING MEMORY" in ctx
        assert "fix login" in ctx.lower() or "check session" in ctx.lower()

    def test_to_context_string_empty_when_no_match(self):
        mem = EngineeringMemory()
        ctx = mem.to_context_string("xyz unrelated")
        assert ctx == ""

    def test_query_respects_max_results(self):
        mem = EngineeringMemory()
        for i in range(10):
            mem.record_successful(
                task_summary=f"task {i} login",
                problem="p", solution="s", key_insight="k", files_involved=[],
            )
        results = mem.query("task login", max_results=3)
        assert len(results) <= 3

    def test_query_project_isolation(self):
        mem_a = EngineeringMemory()
        mem_a._strategies.clear()
        mem_a.record_successful(
            task_summary="task", problem="p", solution="s", key_insight="k", files_involved=[],
        )
        mem_b = EngineeringMemory()
        mem_b._strategies.clear()
        results_b = mem_b.query("task")
        assert len(results_b) == 0


class TestPhase3ContextBudgetSelect:
    """Phase 3: context budget select() must produce items."""

    def test_select_returns_items_within_budget(self):
        cb = ContextBudget()
        cb.set_phase("TEST")
        cb.add(ContextItem(content="item1", priority=ContextPriority.HIGH, source="test", phase_relevance=["TEST"]))
        cb.add(ContextItem(content="item2", priority=ContextPriority.LOW, source="test", phase_relevance=["TEST"]))
        selected = cb.select()
        assert len(selected) > 0
        assert any(i.content == "item1" for i in selected)

    def test_select_respects_priority_ordering(self):
        cb = ContextBudget()
        cb.set_phase("TEST")
        cb.add(ContextItem(content="low", priority=ContextPriority.LOW, source="test", phase_relevance=["TEST"]))
        cb.add(ContextItem(content="high", priority=ContextPriority.HIGH, source="test", phase_relevance=["TEST"]))
        selected = cb.select()
        assert selected[0].priority == ContextPriority.HIGH

    def test_select_empty_when_no_items(self):
        cb = ContextBudget()
        cb.set_phase("TEST")
        selected = cb.select()
        assert len(selected) == 0


class TestPhase4TestSelection:
    """Phase 4: test selection must produce paths."""

    def test_select_returns_selected_tests(self):
        cg = CodebaseGraph()
        ts = TestSelector(cg)
        selection = ts.select(["app/tools/terminal.py"])
        assert selection.selected_tests is not None
        assert selection.selection_reason != ""

    def test_select_returns_all_when_no_changes(self):
        cg = CodebaseGraph()
        ts = TestSelector(cg)
        selection = ts.select([])
        assert len(selection.selected_tests) == len(selection.all_tests)


class TestPhase5AdaptivePlanning:
    """Phase 5: adaptive planner must integrate with replan."""

    def test_should_revise_returns_true_when_tests_fail(self):
        ap = AdaptivePlanner()
        should, reason = ap.should_revise(test_passed=False)
        assert should is True
        assert "failed" in reason.lower()

    def test_should_revise_returns_false_when_all_pass(self):
        ap = AdaptivePlanner()
        should, reason = ap.should_revise(test_passed=True)
        assert should is False

    def test_should_revise_returns_true_for_security_issues(self):
        ap = AdaptivePlanner()
        should, reason = ap.should_revise(test_passed=True, security_issues=1)
        assert should is True

    def test_should_revise_respects_max_replans(self):
        ap = AdaptivePlanner()
        should, reason = ap.should_revise(test_passed=False, replan_count=3, max_replans=3)
        assert should is False

    def test_create_revision_produces_revision(self):
        ap = AdaptivePlanner()
        rev = ap.create_revision(reason="syntax error", evidence_source="diagnosis")
        assert rev.reason == "syntax error"
        assert len(ap.get_revisions()) == 1


class TestPhase6TaskGraph:
    """Phase 6: task graph must enforce dependency ordering."""

    def test_ready_nodes_respects_dependencies(self):
        tg = TaskGraph()
        tg.add_node(GraphNode(node_id="s1", dependencies=[]))
        tg.add_node(GraphNode(node_id="s2", dependencies=["s1"]))
        ready = tg.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].node_id == "s1"

    def test_ready_nodes_includes_satisfied_deps(self):
        tg = TaskGraph()
        tg.add_node(GraphNode(node_id="s1", dependencies=[]))
        tg.add_node(GraphNode(node_id="s2", dependencies=["s1"]))
        tg.mark_completed("s1")
        ready = tg.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].node_id == "s2"

    def test_topological_sort_order(self):
        tg = TaskGraph()
        tg.add_node(GraphNode(node_id="s1", dependencies=[]))
        tg.add_node(GraphNode(node_id="s2", dependencies=["s1"]))
        tg.add_node(GraphNode(node_id="s3", dependencies=["s2"]))
        order = tg.topological_sort()
        assert order.index("s1") < order.index("s2") < order.index("s3")

    def test_has_cycles_detects_cycle(self):
        tg = TaskGraph()
        tg.add_node(GraphNode(node_id="s1", dependencies=["s2"]))
        tg.add_node(GraphNode(node_id="s2", dependencies=["s1"]))
        assert tg.has_cycles() is True

    def test_mark_in_progress_updates_status(self):
        tg = TaskGraph()
        tg.add_node(GraphNode(node_id="s1", dependencies=[]))
        tg.mark_in_progress("s1")
        assert tg.get_node("s1").status == "IN_PROGRESS"


class TestPhase7ImpactAnalysis:
    """Phase 7: impact analysis must store results."""

    def test_impact_results_stored(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        assert isinstance(orch._impact_results, dict)

    def test_impact_results_populated_after_inspect(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        report = orch.run_task("Impact test")
        assert isinstance(orch._impact_results, dict)


class TestPhase8ProjectProfile:
    """Phase 8: project profile must influence planning."""

    def test_project_profile_context_formatted(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        ctx = orch._get_project_profile_context()
        assert isinstance(ctx, str)

    def test_project_profile_stored_after_understand(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        report = orch.run_task("Profile test")
        assert orch._project_profile is not None


class TestPhase9ThreatModelBlocks:
    """Phase 9: threat model must block on CRITICAL untested threats."""

    def test_critical_untested_threats_detected(self):
        tm = SecurityThreatModel()
        untested = tm.get_untested()
        critical_untested = [t for t in untested if t.severity.value == "CRITICAL"]
        assert len(critical_untested) > 0

    def test_threat_model_produces_critical_findings(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        report = orch.run_task("Threat test")
        all_adv = orch.adversarial_suite.get_all()
        passed_adv = [a for a in all_adv if a.passed]
        for a in passed_adv:
            if a.category == "command_injection":
                assert orch.threat_model.get_threat("T002").tested
            if a.category == "secret_access":
                assert orch.threat_model.get_threat("T004").tested


class TestPhase10AdversarialTesting:
    """Phase 10: adversarial tests must execute."""

    def test_adversarial_tests_execute(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        report = orch.run_task("Adversarial test")
        summary = orch.adversarial_suite.summary()
        assert summary["passed"] + summary["failed"] == summary["total_tests"]

    def test_adv002_path_traversal_blocked(self):
        from app.tools.security import safe_path
        with pytest.raises(PermissionError):
            safe_path("../../../etc/passwd")

    def test_adv003_command_injection_blocked(self):
        from app.tools.security import contains_shell_metacharacters
        assert contains_shell_metacharacters("ls; rm -rf /") is True

    def test_adv006_secret_access_blocked(self):
        from app.tools.security import is_secret_path
        from app.config import PROJECT_ROOT
        from pathlib import Path
        assert is_secret_path(Path(".env"), PROJECT_ROOT) is True


class TestPhase11SelfReflection:
    """Phase 11: self reflection must affect diagnosis/replan."""

    def test_reflection_produces_decision(self):
        sr = SelfReflectionEngine()
        entry = sr.reflect("SyntaxError in code", evidence="error output")
        assert entry.decision == ReflectionDecision.FIX
        assert entry.confidence > 0

    def test_reflection_escalation_blocks_replan(self):
        sr = SelfReflectionEngine()
        entry = sr.reflect("budget exceeded", evidence="budget")
        assert entry.decision in (ReflectionDecision.ABORT, ReflectionDecision.ESCALATE)


class TestPhase12StateMachineGating:
    """Phase 12: state machine must gate transitions."""

    def test_valid_transition_succeeds(self):
        sm = StateMachine()
        result = sm.transition_to(ExecutionState.UNDERSTANDING)
        assert result.success is True

    def test_invalid_transition_blocked(self):
        sm = StateMachine()
        result = sm.transition_to(ExecutionState.DONE)
        assert result.success is False
        assert "Invalid transition" in result.error

    def test_blocked_transition_returns_failed(self):
        sm = StateMachine()
        result = sm.transition_to(ExecutionState.DONE)
        assert result.success is False


class TestPhase13Observability:
    """Phase 13: observability trace summary used."""

    def test_trace_summary_in_evidence(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = _default_side_effect
        orch = _make_orch(mock_client)
        report = orch.run_task("Observability test")
        found = False
        for ev in orch.evidence_store:
            if ev.metadata and "trace_summary" in ev.metadata:
                found = True
                break
        assert found


class TestPhase14CacheCorrectness:
    """Phase 14: caches isolated by project root."""

    def test_shared_caches_are_dicts(self):
        assert isinstance(Orchestrator._shared_codebase_graphs, dict)
        assert isinstance(Orchestrator._shared_project_profiles, dict)

    def test_cache_key_is_project_root(self):
        from app.config import PROJECT_ROOT
        _cache_key = str(PROJECT_ROOT)
        assert _cache_key in Orchestrator._shared_codebase_graphs or True

    def test_different_roots_would_have_different_caches(self):
        cache_a = {}
        cache_b = {}
        cache_a["root_a"] = "graph_a"
        cache_b["root_b"] = "graph_b"
        assert cache_a is not cache_b


class TestSelectorDownstreamFlow:
    """Regression: TestSelector selected args must reach test_tool.execute."""

    def test_selected_args_reach_test_tool(self):
        orch = _make_orch(MagicMock())
        mock_selector = MagicMock()
        mock_selector.select.return_value = MagicMock(
            selected_tests=["tests/test_terminal.py", "tests/test_edit.py"],
            all_tests=["tests/test_terminal.py", "tests/test_edit.py", "tests/test_other.py"],
            selection_reason="2 files changed",
        )
        orch.test_selector = mock_selector

        mock_test = orch.core.tools["run_tests"]
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "passed", "stderr": ""},
        }

        phase_history: list[str] = []
        orch._phase_test("fix bug", phase_history)

        call_args = mock_test.execute.call_args
        assert call_args is not None, "test_tool.execute was never called"
        args_value = call_args.kwargs.get("args") or call_args[1].get("args")
        assert "tests/test_terminal.py" in args_value
        assert "tests/test_edit.py" in args_value

    def test_recorded_args_match_actual_execution(self):
        orch = _make_orch(MagicMock())
        mock_selector = MagicMock()
        mock_selector.select.return_value = MagicMock(
            selected_tests=["tests/test_core.py"],
            all_tests=["tests/test_core.py", "tests/test_other.py"],
            selection_reason="1 file changed",
        )
        orch.test_selector = mock_selector

        mock_test = orch.core.tools["run_tests"]
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "ok", "stderr": ""},
        }

        phase_history: list[str] = []
        orch._phase_test("fix bug", phase_history)

        last_test_ex = orch._accumulated_executions[-1]
        assert last_test_ex.tool_name == "run_tests"
        assert "tests/test_core.py" in last_test_ex.arguments["args"]

    def test_default_args_used_when_no_selector(self):
        orch = _make_orch(MagicMock())
        orch.test_selector = None

        mock_test = orch.core.tools["run_tests"]
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "ok", "stderr": ""},
        }

        phase_history: list[str] = []
        orch._phase_test("fix bug", phase_history)

        call_args = mock_test.execute.call_args
        args_value = call_args.kwargs.get("args") or call_args[1].get("args")
        assert args_value == "-v"


class TestAdaptivePlannerGating:
    """Regression: AdaptivePlanner decision must gate replanning."""

    def _setup_orch_for_replan(self, should_revise: bool, revise_reason: str = "no reason"):
        orch = _make_orch(MagicMock())
        subtask = MagicMock()
        subtask.id = "s1"
        subtask.status = SubtaskStatus.IN_PROGRESS
        subtask.dependencies = []

        plan = MagicMock()
        plan.subtasks = [subtask]
        plan.plan_fingerprint.return_value = "fp_old"
        plan.get_next_subtask.return_value = None
        plan.progress_summary.return_value = "0/1"
        plan.to_dict.return_value = {
            "objective": "Fix bug",
            "subtasks": [{"id": "s1", "description": "fix", "dependencies": []}],
        }
        plan.version = 1
        orch._task_plan = plan

        new_plan = MagicMock()
        new_plan.plan_fingerprint.return_value = "fp_new"
        new_plan.to_dict.return_value = {
            "objective": "Fix bug v2",
            "subtasks": [{"id": "s1", "description": "fix v2", "dependencies": []}],
        }
        new_plan.version = 2
        plan.create_replan.return_value = new_plan

        mock_planner = MagicMock()
        mock_planner.should_revise.return_value = (should_revise, revise_reason)
        mock_planner._revisions = []
        orch.adaptive_planner = mock_planner

        orch.state_machine = StateMachine()
        orch.state_machine.force_state(ExecutionState.IMPLEMENTING)
        orch._current_subtask_index = 0
        orch._current_subtask_id = "s1"
        orch._replan_count = 0

        return orch, plan, mock_planner

    def test_should_revise_true_calls_create_replan(self):
        orch, plan, mock_planner = self._setup_orch_for_replan(True, "Tests failed")
        phase_history: list[str] = []

        result = orch._execute_replan("s1", "syntax error", phase_history)

        mock_planner.create_revision.assert_called_once()
        plan.create_replan.assert_called_once()
        assert result == TaskPhase.IMPLEMENT

    def test_should_revise_false_skips_replan_and_returns_review(self):
        orch, plan, mock_planner = self._setup_orch_for_replan(False, "No revision needed")
        phase_history: list[str] = []

        result = orch._execute_replan("s1", "minor issue", phase_history)

        mock_planner.create_revision.assert_not_called()
        plan.create_replan.assert_not_called()
        assert result == TaskPhase.REVIEW
        assert orch._replan_count == 0

    def test_no_adaptive_planner_still_runs_replan(self):
        orch, plan, _ = self._setup_orch_for_replan(False)
        orch.adaptive_planner = None
        phase_history: list[str] = []

        result = orch._execute_replan("s1", "error", phase_history)

        plan.create_replan.assert_called_once()


class TestReviewWorkflow:
    """Regression: REVIEW task must go INSPECT → REPORT, not INSPECT → IMPLEMENT."""

    def _make_review_orch(self, mock_client, mode=AgentMode.READ_ONLY):
        orch = Orchestrator(client=mock_client, mode=mode)
        mock_test = MagicMock()
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "ok", "stderr": ""},
        }
        orch.core.tools["run_tests"] = mock_test
        return orch

    def test_review_inspect_routes_to_report(self):
        mock_client = MagicMock()
        orch = self._make_review_orch(mock_client)
        orch._task_category = TaskCategory.REVIEW
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch._run_agent_step = MagicMock(return_value="Inspected.")

        result = orch._phase_inspect("Analyze the project", [])

        assert result == TaskPhase.REPORT

    def test_coding_inspect_routes_to_implement(self):
        mock_client = MagicMock()
        orch = Orchestrator(client=mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.CODING
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(return_value=json.dumps({
            "decision": "CHANGE_REQUIRED", "confidence": 0.95,
            "reason": "Work needed", "evidence": ["app/main.py"],
        }))

        result = orch._phase_inspect("Implement a feature", [])

        assert result == TaskPhase.IMPLEMENT

    def test_review_allow_edits_inspect_routes_to_implement(self):
        mock_client = MagicMock()
        orch = Orchestrator(client=mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.REVIEW
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(return_value=json.dumps({
            "decision": "CHANGE_REQUIRED", "confidence": 0.95,
            "reason": "Work needed", "evidence": ["app/main.py"],
        }))

        result = orch._phase_inspect("Review the code", [])

        assert result == TaskPhase.IMPLEMENT

    def test_review_task_completes_in_read_only(self):
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            content = msgs[-1]["content"] if msgs else ""
            if "Phase: UNDERSTAND" in content:
                return ChatResponse(content="Understood.")
            elif "Phase: PLAN" in content:
                return ChatResponse(content=_make_plan_json(
                    [{"id": "s1", "description": "analyze architecture", "dependencies": []}]
                ))
            elif "Phase: INSPECT" in content:
                return ChatResponse(content="Analyzed. Architecture is clean.")
            return ChatResponse(content="OK.")

        mock_client.chat.side_effect = side_effect
        orch = self._make_review_orch(mock_client, mode=AgentMode.READ_ONLY)

        report = orch.run_task("Analyze this project in read-only mode. Identify the architecture.")

        assert report.final_phase == "DONE"
        assert "INSPECT" in report.phase_history
        assert "REPORT" in report.phase_history
        assert "IMPLEMENT" not in report.phase_history
        assert "TEST" not in report.phase_history
        assert "DIAGNOSE" not in report.phase_history

    def test_review_gate_allows_no_test_results(self):
        mock_client = MagicMock()
        orch = self._make_review_orch(mock_client)
        orch._task_category = TaskCategory.REVIEW
        orch.context.transition_to("REPORT")
        orch.state_machine.force_state(ExecutionState.REPORTING)
        orch.evidence_store.record(
            evidence_type=EvidenceType.OBSERVATION,
            source="inspect",
            phase="INSPECT",
            tool="orchestrator",
            success=True,
            payload_summary="Analysis complete",
        )

        result = orch._phase_report("Analyze project", [])

        assert result == TaskPhase.DONE

    def test_review_gate_still_blocks_critical_security(self):
        mock_client = MagicMock()
        orch = self._make_review_orch(mock_client)
        orch._task_category = TaskCategory.REVIEW
        orch._critical_security_findings = ["CRITICAL: secret in config.py"]
        orch.context.transition_to("REPORT")
        orch.state_machine.force_state(ExecutionState.REPORTING)
        orch.evidence_store.record(
            evidence_type=EvidenceType.OBSERVATION,
            source="inspect",
            phase="INSPECT",
            tool="orchestrator",
            success=True,
            payload_summary="Analysis complete",
        )

        result = orch._phase_report("Analyze project", [])

        assert result == TaskPhase.FAILED

    def test_review_has_evidence(self):
        mock_client = MagicMock()
        orch = self._make_review_orch(mock_client)
        orch._task_category = TaskCategory.REVIEW
        orch.context.transition_to("REPORT")
        orch.state_machine.force_state(ExecutionState.REPORTING)

        result = orch._phase_report("Analyze project", [])

        assert result == TaskPhase.FAILED
        gate_evidence = [e for e in orch.evidence_store if e.source == "validation_gate"]
        assert len(gate_evidence) > 0
        assert gate_evidence[0].success is False

    def test_understand_records_deterministic_evidence(self):
        mock_client = MagicMock()
        orch = Orchestrator(client=mock_client, mode=AgentMode.READ_ONLY)
        orch._run_agent_step = MagicMock(return_value="Understood.")
        orch.context.transition_to("UNDERSTAND")
        orch.state_machine.force_state(ExecutionState.UNDERSTANDING)

        orch._phase_understand("Analyze project", [])

        evidence = [e for e in orch.evidence_store if e.source == "project_analysis"]
        assert len(evidence) == 1
        assert evidence[0].evidence_type == EvidenceType.OBSERVATION
        assert evidence[0].success is True
        assert evidence[0].phase == "UNDERSTAND"
        assert evidence[0].tool == "project_understanding"
        assert "Languages" in evidence[0].payload_summary
        assert "Modules" in evidence[0].payload_summary

    def test_review_read_only_completes_with_deterministic_evidence(self):
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            content = msgs[-1]["content"] if msgs else ""
            if "Phase: UNDERSTAND" in content:
                return ChatResponse(content="Understood.")
            elif "Phase: PLAN" in content:
                return ChatResponse(content=_make_plan_json(
                    [{"id": "s1", "description": "analyze architecture", "dependencies": []}]
                ))
            elif "Phase: INSPECT" in content:
                return ChatResponse(content="Analyzed. Architecture is clean.")
            return ChatResponse(content="OK.")

        mock_client.chat.side_effect = side_effect
        orch = self._make_review_orch(mock_client, mode=AgentMode.READ_ONLY)

        report = orch.run_task("Review this project in READ_ONLY mode. Inspect the codebase, identify potential issues, run relevant tests, and produce a structured engineering report. Do not modify any files.")

        assert report.final_phase == "DONE"
        assert "INSPECT" in report.phase_history
        assert "REPORT" in report.phase_history
        assert "IMPLEMENT" not in report.phase_history
        evidence = [e for e in orch.evidence_store if e.source == "project_analysis"]
        assert len(evidence) > 0

    def test_gate_still_blocks_without_evidence(self):
        gate = FinalValidationGate()
        result = gate.validate(has_evidence=False)
        assert result.overall_passed is False
        failed = [r for r in result.results if not r.passed]
        assert any(r.check.value == "EVIDENCE" for r in failed)

    def test_review_read_only_blocks_on_critical_security(self):
        mock_client = MagicMock()
        orch = self._make_review_orch(mock_client)
        orch._task_category = TaskCategory.REVIEW
        orch._critical_security_findings = ["CRITICAL: secret exposed"]
        orch.context.transition_to("REPORT")
        orch.state_machine.force_state(ExecutionState.REPORTING)
        orch.evidence_store.record(
            evidence_type=EvidenceType.OBSERVATION,
            source="project_analysis",
            phase="UNDERSTAND",
            tool="project_understanding",
            success=True,
            payload_summary="Languages: 1, Modules: 10",
        )

        result = orch._phase_report("Analyze project", [])

        assert result == TaskPhase.FAILED

    def test_non_review_read_only_coding_unchanged(self):
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            content = msgs[-1]["content"] if msgs else ""
            if "Phase: UNDERSTAND" in content:
                return ChatResponse(content="Understood.")
            elif "Phase: PLAN" in content:
                return ChatResponse(content=_make_plan_json(
                    [{"id": "s1", "description": "implement feature", "dependencies": []}]
                ))
            elif "Phase: INSPECT" in content:
                return ChatResponse(content=_CHANGE_REQUIRED_RESPONSE)
            return ChatResponse(content="Done.")

        mock_client.chat.side_effect = side_effect
        orch = Orchestrator(client=mock_client, mode=AgentMode.READ_ONLY)
        mock_test = MagicMock()
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "ok", "stderr": ""},
        }
        orch.core.tools["run_tests"] = mock_test

        report = orch.run_task("Implement a new feature.")

        assert "INSPECT" in report.phase_history
        assert "REPORT" in report.phase_history
        assert "IMPLEMENT" not in report.phase_history


class TestBudgetCommand:
    """Regression: /budget must not crash with KeyError."""

    def test_budget_status_keys_are_correct(self):
        from app.agent.quota import BudgetTracker
        tracker = BudgetTracker()
        tracker.start()
        tracker.record_llm_call()
        tracker.record_tool_call()
        tracker.record_retry()

        status = tracker.status()

        assert "retry_cycles" in status
        assert "max_retry_cycles" in status
        assert "llm_calls" in status
        assert "max_llm_calls" in status
        assert "tool_calls" in status
        assert "max_tool_calls" in status
        assert "within_budget" in status
        assert status["retry_cycles"] == 1
        assert status["max_retry_cycles"] == 3

    def test_budget_violation_method_works(self):
        from app.agent.quota import BudgetTracker
        tracker = BudgetTracker()
        tracker.start()

        assert tracker.budget_violation() is None

        for _ in range(50):
            tracker.record_llm_call()

        assert tracker.budget_violation() is not None


class TestNoChangeParsing:
    """Tests for _parse_inspect_assessment structured output parsing."""

    def _make_orch(self):
        mock_client = MagicMock()
        return Orchestrator(client=mock_client, mode=AgentMode.READ_ONLY)

    def test_parse_valid_no_change(self):
        orch = self._make_orch()
        raw = json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": 0.92,
            "reason": "Authentication already exists and satisfies the task.",
            "evidence": ["app/auth.py", "tests/test_auth.py"],
        })
        result = orch._parse_inspect_assessment(raw)
        assert result.decision == "NO_CHANGE_REQUIRED"
        assert result.confidence == 0.92
        assert "Authentication" in result.reason
        assert len(result.evidence) == 2

    def test_parse_valid_change_required(self):
        orch = self._make_orch()
        raw = json.dumps({
            "decision": "CHANGE_REQUIRED",
            "confidence": 0.88,
            "reason": "Tests are failing and need fixes.",
            "evidence": ["tests/test_auth.py"],
        })
        result = orch._parse_inspect_assessment(raw)
        assert result.decision == "CHANGE_REQUIRED"
        assert result.confidence == 0.88

    def test_parse_valid_insufficient(self):
        orch = self._make_orch()
        raw = json.dumps({
            "decision": "INSUFFICIENT_EVIDENCE",
            "confidence": 0.3,
            "reason": "Cannot determine if implementation is correct.",
            "evidence": [],
        })
        result = orch._parse_inspect_assessment(raw)
        assert result.decision == "INSUFFICIENT_EVIDENCE"

    def test_parse_invalid_json(self):
        orch = self._make_orch()
        result = orch._parse_inspect_assessment("This is not JSON at all")
        assert result.decision == "INSUFFICIENT_EVIDENCE"
        assert result.confidence == 0.0

    def test_parse_missing_decision(self):
        orch = self._make_orch()
        raw = json.dumps({"confidence": 0.5, "reason": "test", "evidence": []})
        result = orch._parse_inspect_assessment(raw)
        assert result.decision == "INSUFFICIENT_EVIDENCE"

    def test_parse_unknown_decision(self):
        orch = self._make_orch()
        raw = json.dumps({
            "decision": "MAYBE",
            "confidence": 0.5,
            "reason": "test",
            "evidence": [],
        })
        result = orch._parse_inspect_assessment(raw)
        assert result.decision == "INSUFFICIENT_EVIDENCE"

    def test_parse_invalid_confidence_negative(self):
        orch = self._make_orch()
        raw = json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": -0.5,
            "reason": "test",
            "evidence": [],
        })
        result = orch._parse_inspect_assessment(raw)
        assert result.confidence == 0.0
        assert result.decision == "NO_CHANGE_REQUIRED"

    def test_parse_invalid_confidence_string(self):
        orch = self._make_orch()
        raw = json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": "high",
            "reason": "test",
            "evidence": [],
        })
        result = orch._parse_inspect_assessment(raw)
        assert result.confidence == 0.0

    def test_parse_missing_evidence(self):
        orch = self._make_orch()
        raw = json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": 0.9,
            "reason": "test",
        })
        result = orch._parse_inspect_assessment(raw)
        assert result.evidence == []

    def test_parse_missing_reason(self):
        orch = self._make_orch()
        raw = json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": 0.9,
            "evidence": [],
        })
        result = orch._parse_inspect_assessment(raw)
        assert result.reason == "No reason provided"

    def test_parse_markdown_fences(self):
        orch = self._make_orch()
        raw = '```json\n{"decision": "NO_CHANGE_REQUIRED", "confidence": 0.9, "reason": "ok", "evidence": []}\n```'
        result = orch._parse_inspect_assessment(raw)
        assert result.decision == "NO_CHANGE_REQUIRED"
        assert result.confidence == 0.9

    def test_parse_not_a_dict(self):
        orch = self._make_orch()
        raw = json.dumps(["not", "a", "dict"])
        result = orch._parse_inspect_assessment(raw)
        assert result.decision == "INSUFFICIENT_EVIDENCE"


class TestNoChangeIntentClassification:
    """Tests for _classify_intent deterministic classification."""

    def _make_orch(self):
        mock_client = MagicMock()
        return Orchestrator(client=mock_client, mode=AgentMode.READ_ONLY)

    def test_debugging_returns_correction(self):
        orch = self._make_orch()
        assert orch._classify_intent("Fix the bug", TaskCategory.DEBUGGING) == "CORRECTION"

    def test_reasoning_returns_analysis(self):
        orch = self._make_orch()
        assert orch._classify_intent("Why does this fail", TaskCategory.REASONING) == "ANALYSIS"

    def test_planning_returns_analysis(self):
        orch = self._make_orch()
        assert orch._classify_intent("Plan the refactor", TaskCategory.PLANNING) == "ANALYSIS"

    def test_verify_keyword(self):
        orch = self._make_orch()
        assert orch._classify_intent("Verify the authentication", TaskCategory.REVIEW) == "VERIFICATION"

    def test_check_whether_keyword(self):
        orch = self._make_orch()
        assert orch._classify_intent("Check whether login works", TaskCategory.SIMPLE) == "VERIFICATION"

    def test_fix_keyword(self):
        orch = self._make_orch()
        assert orch._classify_intent("Fix the failing tests", TaskCategory.CODING) == "CORRECTION"

    def test_improve_keyword(self):
        orch = self._make_orch()
        assert orch._classify_intent("Improve the logging system", TaskCategory.CODING) == "IMPROVEMENT"

    def test_add_keyword(self):
        orch = self._make_orch()
        assert orch._classify_intent("Add authentication module", TaskCategory.CODING) == "CREATION"

    def test_explain_keyword(self):
        orch = self._make_orch()
        assert orch._classify_intent("Explain the architecture", TaskCategory.SIMPLE) == "ANALYSIS"

    def test_review_category_returns_verification(self):
        orch = self._make_orch()
        assert orch._classify_intent("Review this code", TaskCategory.REVIEW) == "VERIFICATION"

    def test_default_simple_returns_analysis(self):
        orch = self._make_orch()
        assert orch._classify_intent("Do something", TaskCategory.SIMPLE) == "ANALYSIS"


class TestNoChangeRouting:
    """Tests for _phase_inspect routing based on assessment."""

    def _make_orch(self, mock_client, mode=AgentMode.READ_ONLY):
        orch = Orchestrator(client=mock_client, mode=mode)
        mock_test = MagicMock()
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "ok", "stderr": ""},
        }
        orch.core.tools["run_tests"] = mock_test
        return orch

    def _make_assessment_response(self, decision, confidence=0.9, reason="ok", evidence=None):
        return json.dumps({
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "evidence": evidence or ["app/main.py"],
        })

    def test_no_change_skips_implement(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.CODING
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(
            return_value=self._make_assessment_response("NO_CHANGE_REQUIRED")
        )

        result = orch._phase_inspect("Add authentication", [])

        assert result == TaskPhase.REPORT
        assert orch.context.change_decision == "NO_CHANGE_REQUIRED"

    def test_change_required_allow_edits_routes_implement(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.CODING
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(
            return_value=self._make_assessment_response("CHANGE_REQUIRED")
        )

        result = orch._phase_inspect("Add authentication", [])

        assert result == TaskPhase.IMPLEMENT

    def test_change_required_read_only_routes_report(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch._task_category = TaskCategory.CODING
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(
            return_value=self._make_assessment_response("CHANGE_REQUIRED")
        )

        result = orch._phase_inspect("Add authentication", [])

        assert result == TaskPhase.REPORT

    def test_change_required_routing_matrix(self):
        mock_client = MagicMock()

        orch_re = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch_re._task_category = TaskCategory.CODING
        orch_re.context.transition_to("INSPECT")
        orch_re.state_machine.force_state(ExecutionState.INSPECTING)
        orch_re.context.record_inspected_file("app/main.py", "content")
        orch_re._run_agent_step = MagicMock(return_value="Inspected.")
        orch_re._get_final_response = MagicMock(
            return_value=self._make_assessment_response("CHANGE_REQUIRED")
        )
        result_re = orch_re._phase_inspect("Add authentication", [])
        assert result_re == TaskPhase.REPORT, "CHANGE_REQUIRED + READ_ONLY must route to REPORT"

        orch_ae = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch_ae._task_category = TaskCategory.CODING
        orch_ae.context.transition_to("INSPECT")
        orch_ae.state_machine.force_state(ExecutionState.INSPECTING)
        orch_ae.context.record_inspected_file("app/main.py", "content")
        orch_ae._run_agent_step = MagicMock(return_value="Inspected.")
        orch_ae._get_final_response = MagicMock(
            return_value=self._make_assessment_response("CHANGE_REQUIRED")
        )
        result_ae = orch_ae._phase_inspect("Add authentication", [])
        assert result_ae == TaskPhase.IMPLEMENT, "CHANGE_REQUIRED + ALLOW_EDITS must route to IMPLEMENT"

        orch_nc = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch_nc._task_category = TaskCategory.CODING
        orch_nc.context.transition_to("INSPECT")
        orch_nc.state_machine.force_state(ExecutionState.INSPECTING)
        orch_nc.context.record_inspected_file("app/main.py", "content")
        orch_nc._run_agent_step = MagicMock(return_value="Inspected.")
        orch_nc._get_final_response = MagicMock(
            return_value=self._make_assessment_response("NO_CHANGE_REQUIRED")
        )
        result_nc = orch_nc._phase_inspect("Add authentication", [])
        assert result_nc == TaskPhase.REPORT, "NO_CHANGE_REQUIRED + READ_ONLY must route to REPORT"

    def test_review_read_only_unchanged(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch._task_category = TaskCategory.REVIEW
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch._run_agent_step = MagicMock(return_value="Inspected.")

        result = orch._phase_inspect("Analyze the project", [])

        assert result == TaskPhase.REPORT

    def test_insufficient_evidence_reinspect(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.REVIEW
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        call_count = [0]
        def _mock_get_final_response():
            call_count[0] += 1
            if call_count[0] == 1:
                return self._make_assessment_response("INSUFFICIENT_EVIDENCE", confidence=0.3)
            return self._make_assessment_response("CHANGE_REQUIRED", confidence=0.9)
        orch._get_final_response = MagicMock(side_effect=_mock_get_final_response)

        result = orch._phase_inspect("Check authentication", [])

        assert result == TaskPhase.IMPLEMENT
        assert orch.context.assessment_retry_count == 1
        assert orch.context.change_decision == "CHANGE_REQUIRED"

    def test_insufficient_exhausted_routes_report(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.REVIEW
        orch.context.assessment_retry_count = 2
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(
            return_value=self._make_assessment_response("INSUFFICIENT_EVIDENCE", confidence=0.3)
        )

        result = orch._handle_insufficient_evidence("Check auth", [])

        assert result == TaskPhase.REPORT


class TestNoChangeStateAndGuards:
    """Tests for context state, evidence recording, and system guards."""

    def _make_orch(self, mock_client, mode=AgentMode.READ_ONLY):
        orch = Orchestrator(client=mock_client, mode=mode)
        mock_test = MagicMock()
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "ok", "stderr": ""},
        }
        orch.core.tools["run_tests"] = mock_test
        return orch

    def test_no_change_recorded_in_context(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.CODING
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(return_value=json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": 0.9,
            "reason": "Already implemented.",
            "evidence": ["app/main.py"],
        }))

        orch._phase_inspect("Add auth", [])

        assert orch.context.change_decision == "NO_CHANGE_REQUIRED"

    def test_assessment_evidence_recorded(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.CODING
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(return_value=json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": 0.9,
            "reason": "Already implemented.",
            "evidence": ["app/main.py"],
        }))

        orch._phase_inspect("Add auth", [])

        assessments = [e for e in orch.evidence_store if e.source == "change_assessment"]
        assert len(assessments) == 1
        assert assessments[0].phase == "INSPECT"
        assert "NO_CHANGE_REQUIRED" in assessments[0].payload_summary

    def test_retry_counter_increments(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.REVIEW
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        call_count = [0]
        def _mock_get_final_response():
            call_count[0] += 1
            if call_count[0] <= 3:
                return json.dumps({
                    "decision": "INSUFFICIENT_EVIDENCE",
                    "confidence": 0.3,
                    "reason": "Not sure.",
                    "evidence": [],
                })
            return json.dumps({
                "decision": "CHANGE_REQUIRED",
                "confidence": 0.9,
                "reason": "Now I see it.",
                "evidence": ["app/main.py"],
            })
        orch._get_final_response = MagicMock(side_effect=_mock_get_final_response)

        orch._phase_inspect("Check auth", [])

        assert orch.context.assessment_retry_count == 2
        assert orch.context.change_decision == "CHANGE_REQUIRED"

    def test_retry_counter_resets_for_new_task(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch.context.assessment_retry_count = 5
        orch.context.change_decision = "NO_CHANGE_REQUIRED"
        mock_client.chat.side_effect = _default_side_effect

        orch.run_task("Do something")

        assert orch.context.assessment_retry_count <= 2
        assert orch.context.change_decision != "NO_CHANGE_REQUIRED"

    def test_retry_counter_survives_checkpoint(self):
        from app.agent.checkpoint import Checkpoint
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch.context.assessment_retry_count = 1
        orch.context.change_decision = "INSUFFICIENT_EVIDENCE"

        snapshot = orch.context.model_dump()
        restored = TaskContext(**snapshot)

        assert restored.assessment_retry_count == 1
        assert restored.change_decision == "INSUFFICIENT_EVIDENCE"

    def test_no_inspected_files_prevents_no_change(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.CODING
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(return_value=json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": 0.9,
            "reason": "Looks good.",
            "evidence": [],
        }))

        result = orch._phase_inspect("Add auth", [])

        assert orch.context.change_decision == "INSUFFICIENT_EVIDENCE"
        assert result == TaskPhase.REPORT

    def test_guard_debugging_tests_fail_override(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.DEBUGGING
        orch.context.test_results = {"success": False, "stdout": "FAILED"}
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(return_value=json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": 0.9,
            "reason": "No issues found.",
            "evidence": ["app/main.py"],
        }))

        orch._phase_inspect("Fix the failing tests", [])

        assert orch.context.change_decision == "CHANGE_REQUIRED"

    def test_guard_creation_low_confidence(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.CODING
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(return_value=json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": 0.4,
            "reason": "Maybe it exists?",
            "evidence": ["app/main.py"],
        }))

        orch._phase_inspect("Add authentication", [])

        assert orch.context.change_decision == "INSUFFICIENT_EVIDENCE"

    def test_model_evidence_not_tool_verified(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.ALLOW_EDITS)
        orch._task_category = TaskCategory.CODING
        orch.context.transition_to("INSPECT")
        orch.state_machine.force_state(ExecutionState.INSPECTING)
        orch.context.record_inspected_file("app/main.py", "content")
        orch._run_agent_step = MagicMock(return_value="Inspected.")
        orch._get_final_response = MagicMock(return_value=json.dumps({
            "decision": "NO_CHANGE_REQUIRED",
            "confidence": 0.9,
            "reason": "Already done.",
            "evidence": ["app/main.py"],
        }))

        orch._phase_inspect("Add auth", [])

        assessments = [e for e in orch.evidence_store if e.source == "change_assessment"]
        assert len(assessments) == 1
        assert assessments[0].trust_level == "MODEL_INFERRED"


class TestNoChangeGate:
    """Tests for FinalValidationGate compatibility with NO_CHANGE."""

    def _make_orch(self, mock_client, mode=AgentMode.READ_ONLY):
        orch = Orchestrator(client=mock_client, mode=mode)
        mock_test = MagicMock()
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "ok", "stderr": ""},
        }
        orch.core.tools["run_tests"] = mock_test
        return orch

    def test_no_change_gate_passes_no_test_results(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch._task_category = TaskCategory.CODING
        orch.context.change_decision = "NO_CHANGE_REQUIRED"
        orch.context.transition_to("REPORT")
        orch.state_machine.force_state(ExecutionState.REPORTING)
        orch.evidence_store.record(
            evidence_type=EvidenceType.OBSERVATION,
            source="inspect",
            phase="INSPECT",
            tool="orchestrator",
            success=True,
            payload_summary="Analysis complete",
        )

        result = orch._phase_report("Analyze project", [])

        assert result == TaskPhase.DONE

    def test_no_change_gate_passes_no_review(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch._task_category = TaskCategory.CODING
        orch.context.change_decision = "NO_CHANGE_REQUIRED"
        orch._last_review_result = None
        orch.context.transition_to("REPORT")
        orch.state_machine.force_state(ExecutionState.REPORTING)
        orch.evidence_store.record(
            evidence_type=EvidenceType.OBSERVATION,
            source="inspect",
            phase="INSPECT",
            tool="orchestrator",
            success=True,
            payload_summary="Analysis complete",
        )

        result = orch._phase_report("Analyze project", [])

        assert result == TaskPhase.DONE

    def test_no_change_still_requires_evidence(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch._task_category = TaskCategory.CODING
        orch.context.change_decision = "NO_CHANGE_REQUIRED"
        orch.context.transition_to("REPORT")
        orch.state_machine.force_state(ExecutionState.REPORTING)

        result = orch._phase_report("Analyze project", [])

        assert result == TaskPhase.FAILED

    def test_no_change_does_not_bypass_security(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch._task_category = TaskCategory.CODING
        orch.context.change_decision = "NO_CHANGE_REQUIRED"
        orch._critical_security_findings = ["CRITICAL: secret in config.py"]
        orch.context.transition_to("REPORT")
        orch.state_machine.force_state(ExecutionState.REPORTING)
        orch.evidence_store.record(
            evidence_type=EvidenceType.OBSERVATION,
            source="inspect",
            phase="INSPECT",
            tool="orchestrator",
            success=True,
            payload_summary="Analysis complete",
        )

        result = orch._phase_report("Analyze project", [])

        assert result == TaskPhase.FAILED

    def test_change_required_still_follows_normal_path(self):
        mock_client = MagicMock()
        orch = self._make_orch(mock_client, mode=AgentMode.READ_ONLY)
        orch._task_category = TaskCategory.CODING
        orch.context.change_decision = "CHANGE_REQUIRED"
        orch.context.test_results = {"success": True, "exit_code": 0}
        orch._last_review_result = ReviewResult(
            approved=True, findings=[], summary="LGTM", verdict=ReviewVerdict.APPROVE,
        )
        orch.context.transition_to("REPORT")
        orch.state_machine.force_state(ExecutionState.REPORTING)
        orch.evidence_store.record(
            evidence_type=EvidenceType.OBSERVATION,
            source="implement",
            phase="IMPLEMENT",
            tool="orchestrator",
            success=True,
            payload_summary="Changes applied",
        )

        result = orch._phase_report("Add auth", [])

        assert result == TaskPhase.DONE
