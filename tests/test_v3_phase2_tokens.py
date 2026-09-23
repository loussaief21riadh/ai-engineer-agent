from __future__ import annotations

from app.llm.openrouter import ChatResponse


class TestChatResponseTokens:
    def test_default_tokens_are_zero(self) -> None:
        resp = ChatResponse(content="hello")
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.total_tokens == 0

    def test_tokens_from_kwargs(self) -> None:
        resp = ChatResponse(
            content="hello",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert resp.prompt_tokens == 100
        assert resp.completion_tokens == 50
        assert resp.total_tokens == 150

    def test_tokens_with_tool_calls(self) -> None:
        resp = ChatResponse(
            content="",
            tool_calls=[{"name": "read_file", "arguments": {"path": "a.py"}}],
            prompt_tokens=200,
            completion_tokens=30,
            total_tokens=230,
        )
        assert resp.has_tool_calls
        assert resp.prompt_tokens == 200
        assert resp.completion_tokens == 30
        assert resp.total_tokens == 230


class TestOpenRouterTokenExtraction:
    def test_chat_response_extracts_usage(self) -> None:
        resp = ChatResponse(
            content="test",
            raw={"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert resp.prompt_tokens == 100
        assert resp.completion_tokens == 50
        assert resp.total_tokens == 150

    def test_chat_response_missing_usage(self) -> None:
        resp = ChatResponse(content="test", raw={})
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.total_tokens == 0


class TestAgentCoreTokenCallback:
    def test_on_llm_response_called(self) -> None:
        from unittest.mock import MagicMock, patch
        from app.agent.core import AgentCore

        callback = MagicMock()
        client = MagicMock()
        mock_response = ChatResponse(
            content="done",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        client.chat.return_value = mock_response

        core = AgentCore(client=client, on_llm_response=callback)
        core.run("test task", max_steps=1)

        callback.assert_called_once_with(
            model="openrouter/free",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

    def test_token_usage_forwarded_to_observer(self) -> None:
        from unittest.mock import MagicMock, patch
        from app.agent.core import AgentCore
        from app.agent.observer import ObserverEngine
        from app.models.schemas import TokenUsage

        observer = ObserverEngine()
        client = MagicMock()
        mock_response = ChatResponse(
            content="done",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
        )
        client.chat.return_value = mock_response

        def on_llm_response(model, prompt_tokens, completion_tokens, total_tokens):
            tokens = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            observer.llm_call(model=model, tokens=tokens)

        core = AgentCore(client=client, on_llm_response=on_llm_response)
        core.run("test task", max_steps=1)

        assert observer.total_tokens.prompt_tokens == 200
        assert observer.total_tokens.completion_tokens == 100
        assert observer.total_tokens.total_tokens == 300

    def test_multiple_llm_calls_accumulate_tokens(self) -> None:
        from unittest.mock import MagicMock
        from app.agent.observer import ObserverEngine
        from app.models.schemas import TokenUsage

        observer = ObserverEngine()
        client = MagicMock()

        call_count = 0

        def fake_chat(messages, model, tools=None, timeout=120.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(content="", tool_calls=[{"name": "read_file", "arguments": {"path": "a.py"}, "id": "tc1"}], prompt_tokens=100, completion_tokens=20, total_tokens=120)
            return ChatResponse(content="done", prompt_tokens=50, completion_tokens=30, total_tokens=80)

        client.chat = fake_chat

        def on_llm_response(model, prompt_tokens, completion_tokens, total_tokens):
            tokens = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            observer.llm_call(model=model, tokens=tokens)

        from app.agent.core import AgentCore
        from app.tools.filesystem import ReadFileTool
        core = AgentCore(client=client, on_llm_response=on_llm_response)
        core.register_tool(ReadFileTool())
        core.run("test task", max_steps=5)

        assert observer.total_tokens.prompt_tokens == 150
        assert observer.total_tokens.completion_tokens == 50
        assert observer.total_tokens.total_tokens == 200
