from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from app.agent.core import AgentCore, LLM_MAX_RETRIES, LLM_RETRY_BASE_DELAY
from app.llm.openrouter import ChatResponse, OpenRouterError


class TestLLMRetry:
    def test_success_on_first_try_no_retry(self) -> None:
        client = MagicMock()
        client.chat.return_value = ChatResponse(content="done")
        core = AgentCore(client=client)
        result = core.run("test", max_steps=1)
        assert result == "done"
        assert client.chat.call_count == 1

    def test_retry_on_transient_failure(self) -> None:
        client = MagicMock()
        client.chat.side_effect = [
            OpenRouterError("timeout"),
            ChatResponse(content="recovered"),
        ]
        core = AgentCore(client=client)
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=1)
        assert result == "recovered"
        assert client.chat.call_count == 2

    def test_exhausted_retries_returns_error(self) -> None:
        client = MagicMock()
        client.chat.side_effect = OpenRouterError("persistent failure")
        core = AgentCore(client=client)
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=1)
        assert "LLM error after" in result
        assert "3 retries" in result
        assert client.chat.call_count == LLM_MAX_RETRIES

    def test_exponential_backoff_delays(self) -> None:
        client = MagicMock()
        client.chat.side_effect = [
            OpenRouterError("fail 1"),
            OpenRouterError("fail 2"),
            ChatResponse(content="ok"),
        ]
        core = AgentCore(client=client)
        sleep_calls = []
        with patch("app.agent.core.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            result = core.run("test", max_steps=1)
        assert result == "ok"
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == LLM_RETRY_BASE_DELAY
        assert sleep_calls[1] == LLM_RETRY_BASE_DELAY * 2

    def test_retry_respects_max_delay(self) -> None:
        client = MagicMock()
        client.chat.side_effect = [
            OpenRouterError("fail 1"),
            OpenRouterError("fail 2"),
            OpenRouterError("fail 3"),
            OpenRouterError("fail 4"),
            ChatResponse(content="ok"),
        ]
        core = AgentCore(client=client)
        sleep_calls = []
        with patch("app.agent.core.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            with patch("app.agent.core.LLM_MAX_RETRIES", 5):
                result = core.run("test", max_steps=1)
        assert result == "ok"
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0
        assert sleep_calls[2] == 4.0
        assert sleep_calls[3] == 8.0

    def test_retry_resets_on_new_step(self) -> None:
        client = MagicMock()
        call_count = 0

        def fake_chat(messages, model, tools=None, timeout=120.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OpenRouterError("timeout on first LLM call")
            return ChatResponse(content="done")

        client.chat.side_effect = fake_chat
        core = AgentCore(client=client)
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=3)
        assert result == "done"
        assert call_count == 2

    def test_non_retryable_error_immediate_fail(self) -> None:
        client = MagicMock()
        client.chat.side_effect = OpenRouterError("invalid api key")
        core = AgentCore(client=client)
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=1)
        assert "LLM error after" in result
        assert client.chat.call_count == LLM_MAX_RETRIES

    def test_retry_count_reported_in_error(self) -> None:
        client = MagicMock()
        client.chat.side_effect = OpenRouterError("fail")
        core = AgentCore(client=client)
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=1)
        assert f"after {LLM_MAX_RETRIES} retries" in result

    def test_retry_with_tool_calls(self) -> None:
        client = MagicMock()
        client.chat.side_effect = [
            OpenRouterError("timeout"),
            ChatResponse(
                content="",
                tool_calls=[{"name": "list_files", "arguments": {"path": "."}, "id": "tc1"}],
            ),
            ChatResponse(content="files listed"),
        ]
        core = AgentCore(client=client)
        from app.tools.filesystem import ListFilesTool
        core.register_tool(ListFilesTool())
        with patch("app.agent.core.time.sleep"):
            result = core.run("test", max_steps=5)
        assert result == "files listed"
        assert client.chat.call_count == 3
