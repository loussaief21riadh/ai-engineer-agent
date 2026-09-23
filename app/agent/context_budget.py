"""Context Budget for V4 — priority-based context management."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContextPriority(str, Enum):
    """Priority levels for context items."""
    CRITICAL = "CRITICAL"  # Current task, current subtask
    HIGH = "HIGH"          # Relevant files, test failures, security findings
    MEDIUM = "MEDIUM"      # Recent evidence, relevant memory
    LOW = "LOW"            # Historical context, old decisions
    DEFERRED = "DEFERRED"  # Can be dropped if budget exceeded


class ContextItem(BaseModel):
    """A piece of context with priority and size."""
    content: str
    priority: ContextPriority
    source: str = ""
    token_estimate: int = 0
    phase_relevance: list[str] = Field(default_factory=list)
    subtask_relevance: str = ""

    def estimate_tokens(self) -> int:
        if self.token_estimate > 0:
            return self.token_estimate
        return len(self.content) // 4


class ContextBudget:
    """Manages context budget with priority-based allocation."""

    DEFAULT_MAX_TOKENS = 8000

    PRIORITY_WEIGHTS = {
        ContextPriority.CRITICAL: 1.0,
        ContextPriority.HIGH: 0.8,
        ContextPriority.MEDIUM: 0.5,
        ContextPriority.LOW: 0.3,
        ContextPriority.DEFERRED: 0.1,
    }

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.max_tokens = max_tokens
        self._items: list[ContextItem] = []
        self._current_phase: str = ""

    def set_phase(self, phase: str) -> None:
        self._current_phase = phase

    def add(self, item: ContextItem) -> None:
        self._items.append(item)

    def add_items(self, items: list[ContextItem]) -> None:
        self._items.extend(items)

    def clear(self) -> None:
        self._items.clear()

    def select(self) -> list[ContextItem]:
        """Select items that fit within the budget, respecting priority."""
        scored_items: list[tuple[float, ContextItem]] = []

        for item in self._items:
            priority_weight = self.PRIORITY_WEIGHTS.get(item.priority, 0.5)
            phase_bonus = 1.2 if self._current_phase in item.phase_relevance else 1.0
            score = priority_weight * phase_bonus
            scored_items.append((score, item))

        scored_items.sort(key=lambda x: -x[0])

        selected: list[ContextItem] = []
        total_tokens = 0

        for score, item in scored_items:
            item_tokens = item.estimate_tokens()
            if total_tokens + item_tokens <= self.max_tokens:
                selected.append(item)
                total_tokens += item_tokens

        return selected

    def get_total_tokens(self) -> int:
        return sum(item.estimate_tokens() for item in self._items)

    def get_available_tokens(self) -> int:
        return max(0, self.max_tokens - self.get_total_tokens())

    def is_over_budget(self) -> bool:
        return self.get_total_tokens() > self.max_tokens

    def get_usage_summary(self) -> dict[str, Any]:
        by_priority: dict[str, int] = {}
        for item in self._items:
            key = item.priority.value
            by_priority[key] = by_priority.get(key, 0) + item.estimate_tokens()

        return {
            "total_items": len(self._items),
            "total_tokens": self.get_total_tokens(),
            "max_tokens": self.max_tokens,
            "available_tokens": self.get_available_tokens(),
            "over_budget": self.is_over_budget(),
            "by_priority": by_priority,
        }

    def summarize_selected(self, items: list[ContextItem]) -> str:
        """Summarize selected context for debugging."""
        parts = [f"Context Budget: {len(items)} items selected"]
        total = 0
        for item in items:
            tokens = item.estimate_tokens()
            total += tokens
            parts.append(f"  [{item.priority.value}] {item.source}: ~{tokens} tokens")
        parts.append(f"  Total: ~{total} tokens / {self.max_tokens} max")
        return "\n".join(parts)
