from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from app.agent.core import AgentCore, _parse_tool_call_from_text
from app.agent.orchestrator import Orchestrator
from app.config import AgentMode
from app.llm.openrouter import ChatResponse
from app.models.schemas import ToolCall, ToolResult
from app.tools.base import BaseTool, ToolSchema


class DummyTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="dummy_tool",
            description="A dummy tool for testing",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": [],
            },
        )

    def execute(self, **kwargs) -> dict:
        query = kwargs.get("query", "")
        return {"success": True, "result": f"executed: {query}"}


class FailingTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="failing_tool",
            description="A tool that always fails",
            parameters={},
        )

    def execute(self, **kwargs) -> dict:
        raise RuntimeError("boom")


@pytest.fixture
def mock_client():
    client = MagicMock()
    return client


@pytest.fixture
def dummy_tool():
    return DummyTool()


@pytest.fixture
def failing_tool():
    return FailingTool()


class TestOrchestratorInit:
    @patch("app.agent.orchestrator.OpenRouterClient")
    def test_initializes_with_default_tools(self, MockClient):
        orch = Orchestrator()
        tool_names = set(orch.core.tools.keys())
        expected = {"list_files", "read_file", "run_command", "run_tests", "git_status", "git_diff"}
        assert expected == tool_names

    @patch("app.agent.orchestrator.OpenRouterClient")
    def test_initializes_with_write_tool_in_allow_edits_mode(self, MockClient):
        orch = Orchestrator(mode=AgentMode.ALLOW_EDITS)
        assert "write_file" in orch.core.tools

    @patch("app.agent.orchestrator.OpenRouterClient")
    def test_initializes_with_write_tool_in_full_autonomous_mode(self, MockClient):
        orch = Orchestrator(mode=AgentMode.FULL_AUTONOMOUS)
        assert "write_file" in orch.core.tools

    @patch("app.agent.orchestrator.OpenRouterClient")
    def test_no_write_tool_in_read_only_mode(self, MockClient):
        orch = Orchestrator(mode=AgentMode.READ_ONLY)
        assert "write_file" not in orch.core.tools


class TestOrchestratorRegisterTools:
    @patch("app.agent.orchestrator.OpenRouterClient")
    def test_register_tool_adds_to_core(self, MockClient):
        orch = Orchestrator()
        tool = DummyTool()
        orch.core.register_tool(tool)
        assert "dummy_tool" in orch.core.tools
        assert orch.core.tools["dummy_tool"] is tool

    @patch("app.agent.orchestrator.OpenRouterClient")
    def test_register_tool_replaces_existing(self, MockClient):
        orch = Orchestrator()
        tool1 = DummyTool()
        tool2 = DummyTool()
        orch.core.register_tool(tool1)
        orch.core.register_tool(tool2)
        assert orch.core.tools["dummy_tool"] is tool2


class TestOrchestratorSetMode:
    @patch("app.agent.orchestrator.OpenRouterClient")
    def test_set_mode_clears_and_reregisters_tools(self, MockClient):
        orch = Orchestrator(mode=AgentMode.READ_ONLY)
        assert "write_file" not in orch.core.tools
        orch.set_mode(AgentMode.FULL_AUTONOMOUS)
        assert "write_file" in orch.core.tools


class TestAgentCoreRun:
    def test_simple_response_no_tool_call(self, mock_client):
        mock_client.chat.return_value = ChatResponse(content="Here is my answer: 42.")
        core = AgentCore(client=mock_client)
        result = core.run("What is 6*7?", max_steps=3)
        assert result == "Here is my answer: 42."
        assert len(core.history) == 2
        assert core.history[0].role == "user"
        assert core.history[1].role == "assistant"

    def test_tool_call_and_result(self, mock_client, dummy_tool):
        tool_call_response = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "dummy_tool", "arguments": {"query": "hello"}}],
        )
        final_response = ChatResponse(content="The tool returned: executed: hello")
        mock_client.chat.side_effect = [tool_call_response, final_response]

        core = AgentCore(client=mock_client)
        core.register_tool(dummy_tool)
        result = core.run("Run dummy tool", max_steps=3)

        assert result == "The tool returned: executed: hello"
        assert mock_client.chat.call_count == 2

    def test_stops_after_max_steps(self, mock_client, dummy_tool):
        tool_call_response = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "dummy_tool", "arguments": {"query": "loop"}}],
        )
        mock_client.chat.return_value = tool_call_response

        core = AgentCore(client=mock_client)
        core.register_tool(dummy_tool)
        result = core.run("Loop forever", max_steps=2)

        assert "Reached maximum steps" in result
        assert "dummy_tool" in result
        assert mock_client.chat.call_count >= 1
        assert len(core.executions) >= 1

    def test_llm_error_returns_error_string(self, mock_client):
        from app.llm.openrouter import OpenRouterError
        mock_client.chat.side_effect = OpenRouterError("API down")
        core = AgentCore(client=mock_client)
        result = core.run("Fail", max_steps=1)
        assert "LLM error: API down" in result

    def test_tool_execution_exception_handled(self, mock_client, failing_tool):
        tool_call_response = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "failing_tool", "arguments": {}}],
        )
        final_response = ChatResponse(content="Done")
        mock_client.chat.side_effect = [tool_call_response, final_response]

        core = AgentCore(client=mock_client)
        core.register_tool(failing_tool)
        result = core.run("Fail tool", max_steps=3)

        assert mock_client.chat.call_count == 2
        messages = mock_client.chat.call_args_list[1][1]["messages"]
        tool_result_msg = messages[-1]["content"]
        assert "boom" in tool_result_msg


class TestToolDispatch:
    def test_unknown_tool_returns_error(self, mock_client):
        tool_call_response = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "nonexistent_tool", "arguments": {}}],
        )
        final_response = ChatResponse(content="I see an error.")
        mock_client.chat.side_effect = [tool_call_response, final_response]

        core = AgentCore(client=mock_client)
        result = core.run("Call unknown", max_steps=3)
        assert result == "I see an error."
        messages = mock_client.chat.call_args_list[1][1]["messages"]
        assert "Unknown tool" in messages[-1]["content"]

    def test_known_tool_is_dispatched(self, mock_client, dummy_tool):
        tool_call_response = ChatResponse(
            content="",
            tool_calls=[{"id": "1", "name": "dummy_tool", "arguments": {"query": "test"}}],
        )
        final_response = ChatResponse(content="Done")
        mock_client.chat.side_effect = [tool_call_response, final_response]

        core = AgentCore(client=mock_client)
        core.register_tool(dummy_tool)
        core.run("Do something", max_steps=3)

        messages = mock_client.chat.call_args_list[1][1]["messages"]
        tool_result_msg = messages[-1]["content"]
        assert "executed: test" in tool_result_msg


class TestParseToolCall:
    def test_extracts_tool_call_correctly(self):
        text = 'Some text\n```tool\n{"name": "read_file", "arguments": {"path": "app/main.py"}}\n```\nMore text'
        result = _parse_tool_call_from_text(text)
        assert result is not None
        assert result.name == "read_file"
        assert result.arguments == {"path": "app/main.py"}

    def test_returns_none_for_non_tool_text(self):
        result = _parse_tool_call_from_text("Just plain text with no tool call.")
        assert result is None

    def test_returns_none_for_invalid_json(self):
        text = '```tool\nnot valid json\n```'
        result = _parse_tool_call_from_text(text)
        assert result is None

    def test_returns_none_for_missing_name(self):
        text = '```tool\n{"arguments": {"x": 1}}\n```'
        result = _parse_tool_call_from_text(text)
        assert result is None

    def test_returns_none_for_empty_text(self):
        result = _parse_tool_call_from_text("")
        assert result is None

    def test_extracts_tool_call_without_arguments(self):
        text = '```tool\n{"name": "git_status"}\n```'
        result = _parse_tool_call_from_text(text)
        assert result is not None
        assert result.name == "git_status"
        assert result.arguments == {}
