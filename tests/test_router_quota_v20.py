"""Tests for V2.0-E Model Router and Quota Tracker."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.agent.quota import BudgetLimits, BudgetTracker
from app.agent.router import ModelConfig, ModelRouter, TaskCategory


class TestTaskCategory:
    def test_all_categories(self):
        cats = [c.value for c in TaskCategory]
        assert "SIMPLE" in cats
        assert "PLANNING" in cats
        assert "CODING" in cats
        assert "DEBUGGING" in cats
        assert "REVIEW" in cats
        assert "REASONING" in cats

    def test_category_count(self):
        assert len(TaskCategory) == 6


class TestModelRouter:
    def test_route_default(self):
        router = ModelRouter(default_model="test-model")
        assert router.route(TaskCategory.SIMPLE) == "test-model"

    def test_route_custom_config(self):
        configs = {
            TaskCategory.CODING: ModelConfig(primary="coder-model", fallback="fallback-model"),
        }
        router = ModelRouter(configs=configs)
        assert router.route(TaskCategory.CODING) == "coder-model"

    def test_route_unconfigured_uses_default(self):
        router = ModelRouter(default_model="default")
        assert router.route(TaskCategory.REASONING) == "default"

    def test_get_fallback(self):
        configs = {
            TaskCategory.REVIEW: ModelConfig(primary="review-primary", fallback="review-fallback"),
        }
        router = ModelRouter(configs=configs, fallback_model="global-fallback")
        assert router.get_fallback(TaskCategory.REVIEW) == "review-fallback"

    def test_get_fallback_uses_global(self):
        router = ModelRouter(fallback_model="global-fallback")
        assert router.get_fallback(TaskCategory.SIMPLE) == "global-fallback"

    def test_classify_task_review(self):
        router = ModelRouter()
        assert router.classify_task("Review the code for bugs") == TaskCategory.REVIEW
        assert router.classify_task("Audit the security") == TaskCategory.REVIEW
        assert router.classify_task("Check the implementation") == TaskCategory.REVIEW

    def test_classify_task_debugging(self):
        router = ModelRouter()
        assert router.classify_task("Fix the bug in main.py") == TaskCategory.DEBUGGING
        assert router.classify_task("Debug the failing test") == TaskCategory.DEBUGGING
        assert router.classify_task("The error is in config") == TaskCategory.DEBUGGING

    def test_classify_task_planning(self):
        router = ModelRouter()
        assert router.classify_task("Plan the architecture") == TaskCategory.PLANNING
        assert router.classify_task("Design the API") == TaskCategory.PLANNING

    def test_classify_task_coding(self):
        router = ModelRouter()
        assert router.classify_task("Implement the feature") == TaskCategory.CODING
        assert router.classify_task("Write a new function") == TaskCategory.CODING
        assert router.classify_task("Create a test file") == TaskCategory.CODING

    def test_classify_task_reasoning(self):
        router = ModelRouter()
        assert router.classify_task("Explain how this works") == TaskCategory.REASONING
        assert router.classify_task("Compare the two approaches") == TaskCategory.REASONING

    def test_classify_task_simple(self):
        router = ModelRouter()
        assert router.classify_task("List files") == TaskCategory.SIMPLE
        assert router.classify_task("What is the time?") == TaskCategory.SIMPLE

    def test_classify_case_insensitive(self):
        router = ModelRouter()
        assert router.classify_task("FIX THE BUG") == TaskCategory.DEBUGGING

    def test_router_works_without_config(self):
        router = ModelRouter(default_model="fallback")
        for cat in TaskCategory:
            model = router.route(cat)
            assert model == "fallback"


class TestBudgetLimits:
    def test_default_limits(self):
        limits = BudgetLimits()
        assert limits.max_llm_calls == 50
        assert limits.max_tool_calls == 100
        assert limits.max_retry_cycles == 3
        assert limits.max_task_duration == 600.0
        assert limits.max_agent_steps == 15

    def test_custom_limits(self):
        limits = BudgetLimits(max_llm_calls=10, max_tool_calls=20)
        assert limits.max_llm_calls == 10
        assert limits.max_tool_calls == 20


class TestBudgetTracker:
    def test_initial_state(self):
        tracker = BudgetTracker()
        assert tracker.llm_calls == 0
        assert tracker.tool_calls == 0
        assert tracker.retry_cycles == 0

    def test_start_and_stop(self):
        tracker = BudgetTracker()
        tracker.start()
        assert tracker._active is True
        tracker.stop()
        assert tracker._active is False

    def test_record_llm_call(self):
        tracker = BudgetTracker()
        tracker.start()
        tracker.record_llm_call()
        tracker.record_llm_call()
        assert tracker.llm_calls == 2

    def test_record_tool_call(self):
        tracker = BudgetTracker()
        tracker.start()
        tracker.record_tool_call()
        assert tracker.tool_calls == 1

    def test_record_retry(self):
        tracker = BudgetTracker()
        tracker.start()
        tracker.record_retry()
        assert tracker.retry_cycles == 1

    def test_within_budget(self):
        tracker = BudgetTracker(limits=BudgetLimits(max_llm_calls=5))
        tracker.start()
        assert tracker.is_within_budget() is True
        for _ in range(4):
            tracker.record_llm_call()
        assert tracker.is_within_budget() is True
        tracker.record_llm_call()
        assert tracker.is_within_budget() is False

    def test_budget_violation_llm(self):
        tracker = BudgetTracker(limits=BudgetLimits(max_llm_calls=2))
        tracker.start()
        tracker.record_llm_call()
        tracker.record_llm_call()
        violation = tracker.budget_violation()
        assert violation is not None
        assert "LLM" in violation

    def test_budget_violation_tool(self):
        tracker = BudgetTracker(limits=BudgetLimits(max_tool_calls=1))
        tracker.start()
        tracker.record_tool_call()
        tracker.record_tool_call()
        violation = tracker.budget_violation()
        assert violation is not None
        assert "Tool" in violation

    def test_budget_violation_retry(self):
        tracker = BudgetTracker(limits=BudgetLimits(max_retry_cycles=1))
        tracker.start()
        tracker.record_retry()
        tracker.record_retry()
        violation = tracker.budget_violation()
        assert violation is not None
        assert "Retry" in violation

    def test_no_violation(self):
        tracker = BudgetTracker(limits=BudgetLimits(max_llm_calls=10))
        tracker.start()
        assert tracker.budget_violation() is None

    def test_status(self):
        tracker = BudgetTracker(limits=BudgetLimits(max_llm_calls=10))
        tracker.start()
        tracker.record_llm_call()
        status = tracker.status()
        assert status["llm_calls"] == 1
        assert status["max_llm_calls"] == 10
        assert status["within_budget"] is True

    def test_elapsed_time(self):
        tracker = BudgetTracker(limits=BudgetLimits(max_task_duration=10))
        tracker.start()
        time.sleep(0.01)
        assert tracker.elapsed() > 0

    def test_budget_violation_duration(self):
        tracker = BudgetTracker(limits=BudgetLimits(max_task_duration=0.001))
        tracker.start()
        time.sleep(0.01)
        violation = tracker.budget_violation()
        assert violation is not None
        assert "duration" in violation.lower()

    def test_stop_resets_elapsed(self):
        tracker = BudgetTracker()
        tracker.start()
        time.sleep(0.01)
        tracker.stop()
        assert tracker.elapsed() == 0.0

    def test_inactive_tracker_always_within_budget(self):
        tracker = BudgetTracker()
        assert tracker.is_within_budget() is True
        assert tracker.budget_violation() is None

    def test_start_resets_counters(self):
        tracker = BudgetTracker()
        tracker.start()
        tracker.record_llm_call()
        tracker.record_tool_call()
        tracker.start()
        assert tracker.llm_calls == 0
        assert tracker.tool_calls == 0
