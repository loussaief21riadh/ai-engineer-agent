from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.agent.core import AgentCore, build_tool_definitions
from app.llm.openrouter import ChatResponse
from app.tools.base import BaseTool, ToolSchema


class EchoTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="echo",
            description="Echoes input",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    def execute(self, text: str = "", **kwargs) -> dict:
        return {"success": True, "result": f"echo: {text}"}


class CountingTool(BaseTool):
    call_count = 0

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="counter",
            description="Increments counter",
            parameters={},
        )

    def execute(self, **kwargs) -> dict:
        CountingTool.call_count += 1
        return {"success": True, "result": f"count={CountingTool.call_count}"}


class FailingTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="failer",
            description="Always fails",
            parameters={},
        )

    def execute(self, **kwargs) -> dict:
        raise RuntimeError("boom")


# ---------- build_tool_definitions ----------


class TestBuildToolDefinitions:
    def test_empty_tools_returns_empty_list(self):
        assert build_tool_definitions({}) == []

    def test_single_tool_definition_format(self):
        tool = EchoTool()
        defs = build_tool_definitions({"echo": tool})
        assert len(defs) == 1
        d = defs[0]
        assert d["type"] == "function"
        assert d["function"]["name"] == "echo"
        assert d["function"]["description"] == "Echoes input"
        assert "properties" in d["function"]["parameters"]

    def test_multiple_tools(self):
        class EchoTool2(BaseTool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="echo2",
                    description="Another echo",
                    parameters={},
                )

            def execute(self, **kwargs) -> dict:
                return {"success": True, "result": "echo2"}

        tool1 = EchoTool()
        tool2 = EchoTool2()
        defs = build_tool_definitions({"echo": tool1, "echo2": tool2})
        assert len(defs) == 2
        names = {d["function"]["name"] for d in defs}
        assert names == {"echo", "echo2"}


# ---------- Native tool calling path ----------


class TestNativeToolCalling:
    def test_native_tool_call_executed(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[{"id": "tc_1", "name": "echo", "arguments": {"text": "hi"}}],
            ),
            ChatResponse(content="Done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        result = core.run("echo hi", max_steps=3)

        assert result == "Done."
        assert len(core.executions) == 1
        assert core.executions[0].tool_name == "echo"
        assert core.executions[0].success is True

    def test_native_tools_definition_sent_to_llm(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(content="No tools needed.")

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("hello", max_steps=1)

        call_kwargs = mock_client.chat.call_args
        tools_arg = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert tools_arg is not None
        assert len(tools_arg) == 1
        assert tools_arg[0]["function"]["name"] == "echo"

    def test_native_path_records_execution_with_duration(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[{"id": "tc_1", "name": "echo", "arguments": {"text": "test"}}],
            ),
            ChatResponse(content="OK"),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo test", max_steps=3)

        ex = core.executions[0]
        assert ex.duration_ms >= 0
        assert ex.arguments == {"text": "test"}

    def test_native_tool_call_stored_in_history(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="Calling tool.",
                tool_calls=[{"id": "tc_1", "name": "echo", "arguments": {"text": "x"}}],
            ),
            ChatResponse(content="Result is x."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo x", max_steps=3)

        roles = [m.role for m in core.history]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles


# ---------- Fallback path ----------


class TestFallbackToolCalling:
    def test_fallback_tool_call_parsed(self):
        mock_client = MagicMock()
        fallback_response = '```tool\n{"name": "echo", "arguments": {"text": "fallback"}}\n```'
        mock_client.chat.side_effect = [
            ChatResponse(content=fallback_response),
            ChatResponse(content="Done via fallback."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        result = core.run("echo fallback", max_steps=3)

        assert result == "Done via fallback."
        assert len(core.executions) == 1
        assert core.executions[0].tool_name == "echo"

    def test_fallback_does_not_send_tools_to_llm(self):
        mock_client = MagicMock()
        fallback_response = '```tool\n{"name": "echo", "arguments": {"text": "x"}}\n```'
        mock_client.chat.side_effect = [
            ChatResponse(content=fallback_response),
            ChatResponse(content="OK"),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo x", max_steps=3)

        second_call = mock_client.chat.call_args_list[1]
        tools_arg = second_call.kwargs.get("tools") or second_call[1].get("tools")
        assert tools_arg is None


# ---------- Unknown tool handling ----------


class TestUnknownToolHandling:
    def test_unknown_native_tool_returns_error(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[{"id": "tc_1", "name": "nonexistent", "arguments": {}}],
            ),
            ChatResponse(content="I see the error."),
        ]

        core = AgentCore(client=mock_client)
        result = core.run("call nonexistent", max_steps=3)

        assert result == "I see the error."
        messages = mock_client.chat.call_args_list[1][1]["messages"]
        assert "Unknown tool" in messages[-1]["content"]
        assert len(core.executions) == 0


# ---------- tool_call_id preservation ----------


class TestToolCallIdPreservation:
    def test_tool_call_id_stored_in_execution(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[{"id": "call_abc123", "name": "echo", "arguments": {"text": "test"}}],
            ),
            ChatResponse(content="Done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo test", max_steps=3)

        ex = core.executions[0]
        assert ex.tool_call_id == "call_abc123"

    def test_tool_call_id_used_in_tool_message(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[{"id": "call_xyz789", "name": "echo", "arguments": {"text": "hi"}}],
            ),
            ChatResponse(content="Done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo hi", max_steps=3)

        messages = mock_client.chat.call_args_list[1][1]["messages"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_xyz789"

    def test_multiple_tool_calls_preserve_individual_ids(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "echo", "arguments": {"text": "a"}},
                    {"id": "call_2", "name": "echo", "arguments": {"text": "b"}},
                ],
            ),
            ChatResponse(content="Done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo a and b", max_steps=3)

        assert core.executions[0].tool_call_id == "call_1"
        assert core.executions[1].tool_call_id == "call_2"

        messages = mock_client.chat.call_args_list[1][1]["messages"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[1]["tool_call_id"] == "call_2"


# ---------- Multiple tool calls ----------


class TestMultipleToolCalls:
    def test_two_tool_calls_execute_both(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "echo", "arguments": {"text": "first"}},
                    {"id": "c2", "name": "echo", "arguments": {"text": "second"}},
                ],
            ),
            ChatResponse(content="Both done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo first and second", max_steps=3)

        assert len(core.executions) == 2
        assert core.executions[0].arguments["text"] == "first"
        assert core.executions[1].arguments["text"] == "second"

    def test_two_tool_calls_generate_two_tool_messages(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "echo", "arguments": {"text": "a"}},
                    {"id": "c2", "name": "echo", "arguments": {"text": "b"}},
                ],
            ),
            ChatResponse(content="Done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo a and b", max_steps=3)

        messages = mock_client.chat.call_args_list[1][1]["messages"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

    def test_multiple_tool_calls_step_increments(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "echo", "arguments": {"text": "x"}},
                    {"id": "c2", "name": "echo", "arguments": {"text": "y"}},
                ],
            ),
            ChatResponse(content="Done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo x and y", max_steps=5)

        assert core.executions[0].step == 1
        assert core.executions[1].step == 1

    def test_mixed_known_and_unknown_tool_calls(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "echo", "arguments": {"text": "ok"}},
                    {"id": "c2", "name": "nonexistent", "arguments": {}},
                ],
            ),
            ChatResponse(content="Done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("do both", max_steps=3)

        assert len(core.executions) == 1
        assert core.executions[0].tool_name == "echo"

        messages = mock_client.chat.call_args_list[1][1]["messages"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

    def test_mixed_valid_and_invalid_args(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "echo", "arguments": {"text": "good"}},
                    {"id": "c2", "name": "echo", "arguments": {"bad_arg": "x"}},
                ],
            ),
            ChatResponse(content="Done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("do both", max_steps=3)

        assert len(core.executions) == 1
        assert core.executions[0].arguments["text"] == "good"


# ---------- MAX_AGENT_STEPS ----------


class TestMaxSteps:
    def test_max_steps_exact_boundary(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(
            content="",
            tool_calls=[{"id": "c1", "name": "echo", "arguments": {"text": "loop"}}],
        )

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        result = core.run("loop forever", max_steps=1)

        assert "Reached maximum steps" in result
        assert len(core.executions) == 1

    def test_zero_executions_message(self):
        from app.llm.openrouter import OpenRouterError

        mock_client = MagicMock()
        mock_client.chat.side_effect = OpenRouterError("API down")

        core = AgentCore(client=mock_client)
        result = core.run("do something", max_steps=1)

        assert "LLM error" in result
        assert len(core.executions) == 0

    def test_no_tool_call_after_limit(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(
            content="",
            tool_calls=[{"id": "c1", "name": "echo", "arguments": {"text": "x"}}],
        )

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("loop", max_steps=2)

        assert len(core.executions) == 2

    def test_max_steps_with_multiple_tool_calls(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(
            content="",
            tool_calls=[
                {"id": "c1", "name": "echo", "arguments": {"text": "a"}},
                {"id": "c2", "name": "echo", "arguments": {"text": "b"}},
            ],
        )

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        result = core.run("multi-loop", max_steps=2)

        assert "Reached maximum steps" in result
        assert len(core.executions) == 4
        assert core.executions[0].step == 1
        assert core.executions[2].step == 2


# ---------- Precedence ----------


class TestPrecedence:
    def test_native_takes_precedence_over_fallback(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(
                content='```tool\n{"name": "echo", "arguments": {"text": "fallback"}}\n```',
                tool_calls=[{"id": "c1", "name": "echo", "arguments": {"text": "native"}}],
            ),
            ChatResponse(content="Done."),
        ]

        core = AgentCore(client=mock_client)
        core.register_tool(EchoTool())
        core.run("echo", max_steps=3)

        assert core.executions[0].arguments["text"] == "native"
