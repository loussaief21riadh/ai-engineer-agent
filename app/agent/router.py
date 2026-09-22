"""Model routing for V2.0 — task-complexity-based model selection."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel


class TaskCategory(str, Enum):
    SIMPLE = "SIMPLE"
    PLANNING = "PLANNING"
    CODING = "CODING"
    DEBUGGING = "DEBUGGING"
    REVIEW = "REVIEW"
    REASONING = "REASONING"


class ModelConfig(BaseModel):
    primary: str
    fallback: str = ""


DEFAULT_MODEL_CONFIGS: dict[TaskCategory, ModelConfig] = {}


class ModelRouter:
    def __init__(
        self,
        default_model: str = "",
        fallback_model: str = "",
        configs: dict[TaskCategory, ModelConfig] | None = None,
    ) -> None:
        self.default_model = default_model or os.getenv("PRIMARY_MODEL", "openrouter/free")
        self.fallback_model = fallback_model or os.getenv("FALLBACK_MODEL", "")
        self.configs = configs or dict(DEFAULT_MODEL_CONFIGS)

    def route(self, category: TaskCategory) -> str:
        config = self.configs.get(category)
        if config:
            return config.primary
        return self.default_model

    def get_fallback(self, category: TaskCategory) -> str:
        config = self.configs.get(category)
        if config and config.fallback:
            return config.fallback
        return self.fallback_model or self.default_model

    def classify_task(self, task_description: str) -> TaskCategory:
        lower = task_description.lower()

        review_keywords = ["review", "audit", "check", "inspect", "analyze", "assess"]
        if any(kw in lower for kw in review_keywords):
            return TaskCategory.REVIEW

        debug_keywords = ["debug", "fix", "error", "bug", "fail", "broken", "crash"]
        if any(kw in lower for kw in debug_keywords):
            return TaskCategory.DEBUGGING

        planning_keywords = ["plan", "design", "architect", "structure", "organize"]
        if any(kw in lower for kw in planning_keywords):
            return TaskCategory.PLANNING

        coding_keywords = ["implement", "write", "create", "add", "build", "develop", "code"]
        if any(kw in lower for kw in coding_keywords):
            return TaskCategory.CODING

        reasoning_keywords = ["explain", "why", "how", "reason", "compare", "evaluate"]
        if any(kw in lower for kw in reasoning_keywords):
            return TaskCategory.REASONING

        return TaskCategory.SIMPLE
