from __future__ import annotations

import json
from typing import Any

from app.agent.prompts import REVIEW_PROMPT
from app.config import REVIEWER_MODEL
from app.llm.openrouter import OpenRouterClient, OpenRouterError
from app.models.schemas import ReviewResult, ReviewVerdict


class Reviewer:
    def __init__(
        self,
        client: OpenRouterClient | None = None,
        on_llm_call: Any = None,
    ) -> None:
        self.client = client or OpenRouterClient()
        self._on_llm_call = on_llm_call

    def review(
        self,
        task: str,
        changes: str = "",
        diff: str = "",
        test_results: str = "",
        context: str = "",
        model: str = "",
    ) -> ReviewResult:
        prompt = REVIEW_PROMPT
        prompt += f"\n\nOriginal task:\n{task}"
        prompt += f"\n\nChanges made:\n{changes}"
        prompt += f"\n\nGit diff:\n{diff}"
        prompt += f"\n\nTest results:\n{test_results}"

        if context:
            prompt += f"\n\nAdditional context:\n{context}"

        messages = [{"role": "user", "content": prompt}]

        if self._on_llm_call is not None:
            self._on_llm_call()

        effective_model = model or REVIEWER_MODEL

        try:
            response = self.client.chat(messages=messages, model=effective_model)
        except OpenRouterError as exc:
            return ReviewResult(
                approved=False,
                verdict=ReviewVerdict.REJECT,
                summary=f"Reviewer LLM error: {exc}",
            )

        return self._parse_review(response.content)

    def _parse_review(self, response: str) -> ReviewResult:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    return ReviewResult(
                        approved=False,
                        verdict=ReviewVerdict.REJECT,
                        summary=f"Could not parse review response: {response[:500]}",
                    )
            else:
                return ReviewResult(
                    approved=False,
                    verdict=ReviewVerdict.REJECT,
                    summary=f"Could not parse review response: {response[:500]}",
                )

        verdict_raw = data.get("verdict", "").upper()
        try:
            verdict = ReviewVerdict(verdict_raw)
        except ValueError:
            verdict = ReviewVerdict.APPROVE if data.get("approved", False) else ReviewVerdict.REJECT

        return ReviewResult(
            approved=data.get("approved", False),
            verdict=verdict,
            findings=data.get("findings", []),
            summary=data.get("summary", ""),
        )
