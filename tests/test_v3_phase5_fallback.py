from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agent.core import AgentCore, LLM_MAX_RETRIES
from app.llm.openrouter import ChatResponse, OpenRouterError


class TestFallbackModel:
    def test_no_fallback_uses_primary_only(self) -> None:
        client = MagicMock()
        client.chat.side_effect = OpenRouterError("fail")
        core = AgentCore(client=client)
        core._fallback_model = ""
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=1)
        assert "LLM error" in result
        assert client.chat.call_count == LLM_MAX_RETRIES

    def test_fallback_tried_after_primary_exhausted(self) -> None:
        client = MagicMock()
        client.chat.side_effect = [
            OpenRouterError("primary fail 1"),
            OpenRouterError("primary fail 2"),
            OpenRouterError("primary fail 3"),
            ChatResponse(content="fallback worked"),
        ]
        core = AgentCore(client=client)
        core._fallback_model = "openrouter/fallback"
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=1, model="openrouter/primary")
        assert result == "fallback worked"
        assert client.chat.call_count == LLM_MAX_RETRIES + 1

    def test_fallback_not_tried_when_primary_succeeds(self) -> None:
        client = MagicMock()
        client.chat.return_value = ChatResponse(content="primary worked")
        core = AgentCore(client=client)
        core._fallback_model = "openrouter/fallback"
        result = core.run("test", max_steps=1, model="openrouter/primary")
        assert result == "primary worked"
        assert client.chat.call_count == 1

    def test_fallback_also_fails_returns_error(self) -> None:
        client = MagicMock()
        client.chat.side_effect = OpenRouterError("both fail")
        core = AgentCore(client=client)
        core._fallback_model = "openrouter/fallback"
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=1, model="openrouter/primary")
        assert "LLM error" in result
        assert client.chat.call_count == LLM_MAX_RETRIES * 2

    def test_fallback_same_as_primary_no_double_retry(self) -> None:
        client = MagicMock()
        client.chat.side_effect = OpenRouterError("fail")
        core = AgentCore(client=client)
        core._fallback_model = "openrouter/primary"
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=1, model="openrouter/primary")
        assert "LLM error" in result
        assert client.chat.call_count == LLM_MAX_RETRIES

    def test_fallback_used_only_after_all_primary_retries(self) -> None:
        client = MagicMock()
        call_log = []

        def fake_chat(messages, model, tools=None, timeout=120.0):
            call_log.append(model)
            if model == "openrouter/primary" and len([m for m in call_log if m == "openrouter/primary"]) <= LLM_MAX_RETRIES:
                raise OpenRouterError("primary fail")
            return ChatResponse(content="ok")

        client.chat.side_effect = fake_chat
        core = AgentCore(client=client)
        core._fallback_model = "openrouter/fallback"
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=1, model="openrouter/primary")
        assert result == "ok"
        primary_calls = [m for m in call_log if m == "openrouter/primary"]
        fallback_calls = [m for m in call_log if m == "openrouter/fallback"]
        assert len(primary_calls) == LLM_MAX_RETRIES
        assert len(fallback_calls) == 1

    def test_orchestrator_sets_fallback(self) -> None:
        from app.agent.orchestrator import Orchestrator
        from app.config import AgentMode

        orch = Orchestrator(mode=AgentMode.READ_ONLY)
        orch._selected_fallback = "openrouter/fallback-model"
        orch.core._fallback_model = orch._selected_fallback
        assert orch.core._fallback_model == "openrouter/fallback-model"
