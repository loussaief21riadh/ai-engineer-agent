from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agent.core import AgentCore
from app.agent.orchestrator import Orchestrator
from app.config import OPENROUTER_API_KEY
from app.llm.openrouter import ChatResponse
from app.models.schemas import ToolExecution
from app.tools.base import BaseTool, ToolSchema


class MockReadFileTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

    def execute(self, path: str = "", **kwargs) -> dict:
        if path == "missing.txt":
            return {"success": False, "error": "File not found: missing.txt"}
        return {"success": True, "result": f"content of {path}"}


class MockWriteFileTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Write a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        )

    def execute(self, path: str = "", content: str = "", **kwargs) -> dict:
        if path.startswith("/"):
            return {"success": False, "error": "Permission denied"}
        return {"success": True, "result": f"Wrote {len(content)} bytes to {path}"}


class MockRunCommandTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_command",
            description="Run a command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        )

    def execute(self, command: str = "", **kwargs) -> dict:
        if command == "rm -rf /":
            return {"success": False, "error": "Blocked command"}
        if command == "ls":
            return {
                "success": True,
                "result": {"command": "ls", "exit_code": 0, "stdout": "file1\nfile2\n", "stderr": ""},
            }
        if command == "ls -la":
            return {
                "success": True,
                "result": {"command": "ls -la", "exit_code": 0, "stdout": "total 0\n", "stderr": ""},
            }
        return {"success": True, "result": {"command": command, "exit_code": 0, "stdout": "", "stderr": ""}}


class MockRunTestsTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_tests",
            description="Run tests",
            parameters={
                "type": "object",
                "properties": {"args": {"type": "string"}},
                "required": [],
            },
        )

    def execute(self, args: str = "", **kwargs) -> dict:
        return {
            "success": True,
            "result": {
                "command": "pytest",
                "exit_code": 0,
                "stdout": "2 passed",
                "stderr": "",
            },
        }


class MockRunTestsFailingTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_tests",
            description="Run tests that fail",
            parameters={
                "type": "object",
                "properties": {"args": {"type": "string"}},
                "required": [],
            },
        )

    def execute(self, args: str = "", **kwargs) -> dict:
        return {
            "success": False,
            "result": {
                "command": "pytest",
                "exit_code": 1,
                "stdout": "1 failed",
                "stderr": "FAILED",
            },
        }


def _tool_response(name: str, arguments: dict) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[{"id": "1", "name": name, "arguments": arguments}],
    )


def _make_core_with_mocks(client, tools=None):
    core = AgentCore(client=client)
    if tools:
        for tool in tools:
            core.register_tool(tool)
    return core


# ---------- EXECUTION TRACKING ----------


class TestExecutionTracking:
    def test_successful_read_tracked(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("read_file", {"path": "app/main.py"})
        final_resp = ChatResponse(content="Here is the file content.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockReadFileTool()])
        core.run("read app/main.py", max_steps=5)

        assert len(core.executions) == 1
        ex = core.executions[0]
        assert ex.tool_name == "read_file"
        assert ex.success is True
        assert ex.arguments["path"] == "app/main.py"

    def test_failed_read_not_tracked_as_success(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("read_file", {"path": "missing.txt"})
        final_resp = ChatResponse(content="The file was not found.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockReadFileTool()])
        core.run("read missing.txt", max_steps=5)

        assert len(core.executions) == 1
        ex = core.executions[0]
        assert ex.tool_name == "read_file"
        assert ex.success is False
        assert "not found" in ex.error.lower()

    def test_successful_write_tracked(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("write_file", {"path": "output.txt", "content": "hello"})
        final_resp = ChatResponse(content="File written.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockWriteFileTool()])
        core.run("write output.txt", max_steps=5)

        assert len(core.executions) == 1
        ex = core.executions[0]
        assert ex.tool_name == "write_file"
        assert ex.success is True
        assert ex.arguments["path"] == "output.txt"

    def test_failed_write_not_tracked_as_success(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("write_file", {"path": "/etc/passwd", "content": "bad"})
        final_resp = ChatResponse(content="Cannot write there.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockWriteFileTool()])
        core.run("write to /etc/passwd", max_steps=5)

        assert len(core.executions) == 1
        ex = core.executions[0]
        assert ex.tool_name == "write_file"
        assert ex.success is False

    def test_successful_command_tracked(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("run_command", {"command": "ls"})
        final_resp = ChatResponse(content="Here are the files.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockRunCommandTool()])
        core.run("run ls", max_steps=5)

        assert len(core.executions) == 1
        ex = core.executions[0]
        assert ex.tool_name == "run_command"
        assert ex.success is True
        assert ex.arguments["command"] == "ls"

    def test_rejected_command_not_tracked(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("run_command", {"command": "rm -rf /"})
        final_resp = ChatResponse(content="That command was blocked.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockRunCommandTool()])
        core.run("run rm -rf /", max_steps=5)

        assert len(core.executions) == 1
        ex = core.executions[0]
        assert ex.success is False

    def test_successful_pytest_tracked(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("run_tests", {"args": ""})
        final_resp = ChatResponse(content="All tests passed.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockRunTestsTool()])
        core.run("run tests", max_steps=5)

        assert len(core.executions) == 1
        ex = core.executions[0]
        assert ex.tool_name == "run_tests"
        assert ex.success is True
        assert isinstance(ex.result, dict)

    def test_failed_pytest_tracked_with_failure(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("run_tests", {"args": ""})
        final_resp = ChatResponse(content="Some tests failed.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockRunTestsFailingTool()])
        core.run("run tests", max_steps=5)

        assert len(core.executions) == 1
        ex = core.executions[0]
        assert ex.tool_name == "run_tests"
        assert ex.success is False
        assert "exit_code" in ex.result
        assert ex.result["exit_code"] == 1


# ---------- ANTI-FABRICATION ----------


class TestAntiFabrication:
    def _make_orch(self, mock_client):
        orch = Orchestrator(client=mock_client)
        mock_test = MagicMock()
        mock_test.execute.return_value = {
            "success": True,
            "result": {"exit_code": 0, "stdout": "All tests passed", "stderr": ""},
        }
        orch.core.tools["run_tests"] = mock_test
        return orch

    def test_fabricated_claim_not_in_report(self):
        mock_client = MagicMock()
        final_resp = ChatResponse(content="I read app/main.py and it looks fine.")
        mock_client.chat.return_value = final_resp

        orch = self._make_orch(mock_client)
        report = orch.run_task("review app/main.py")

        assert report.files_inspected == []

    def test_fabricated_write_not_in_report(self):
        mock_client = MagicMock()
        final_resp = ChatResponse(content="I wrote file.py with the new code.")
        mock_client.chat.return_value = final_resp

        orch = self._make_orch(mock_client)
        report = orch.run_task("write file.py")

        assert report.files_modified == []


# ---------- MAX STEPS ----------


class TestMaxSteps:
    def test_max_steps_includes_execution_summary(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("read_file", {"path": "a.txt"})
        mock_client.chat.return_value = tool_resp

        core = _make_core_with_mocks(mock_client, [MockReadFileTool()])
        result = core.run("read everything", max_steps=2)

        assert "Reached maximum steps" in result
        assert "read_file" in result
        assert "OK" in result

    def test_max_steps_zero_executions_message(self):
        from app.llm.openrouter import OpenRouterError

        mock_client = MagicMock()
        mock_client.chat.side_effect = OpenRouterError("API down")

        core = _make_core_with_mocks(mock_client, [])
        result = core.run("do something", max_steps=1)

        assert "LLM error" in result
        assert len(core.executions) == 0


# ---------- HISTORY STRUCTURE ----------


class TestHistoryStructure:
    def test_history_contains_user_message(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(content="Simple answer.")

        core = _make_core_with_mocks(mock_client, [])
        core.run("Hello", max_steps=3)

        roles = [m.role for m in core.history]
        assert "user" in roles
        user_msgs = [m for m in core.history if m.role == "user"]
        assert any("Hello" in m.content for m in user_msgs)

    def test_history_contains_assistant_message(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(content="Simple answer.")

        core = _make_core_with_mocks(mock_client, [])
        core.run("Hello", max_steps=3)

        roles = [m.role for m in core.history]
        assert "assistant" in roles
        assistant_msgs = [m for m in core.history if m.role == "assistant"]
        assert any("Simple answer" in m.content for m in assistant_msgs)

    def test_history_contains_tool_execution(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("read_file", {"path": "x.py"})
        final_resp = ChatResponse(content="Done.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockReadFileTool()])
        core.run("read x.py", max_steps=3)

        tool_msgs = [m for m in core.history if m.role == "tool"]
        assert len(tool_msgs) >= 1
        parsed = json.loads(tool_msgs[0].content)
        assert parsed["tool_name"] == "read_file"
        assert parsed["success"] is True

    def test_executions_list_populated(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("read_file", {"path": "x.py"})
        final_resp = ChatResponse(content="Done.")
        mock_client.chat.side_effect = [tool_resp, final_resp]

        core = _make_core_with_mocks(mock_client, [MockReadFileTool()])
        core.run("read x.py", max_steps=3)

        assert len(core.executions) == 1
        assert core.executions[0].tool_name == "read_file"


# ---------- TASK REPORT ----------


class TestTaskReport:
    def test_report_files_inspected_populated(self):
        mock_client = MagicMock()
        resp1 = _tool_response("read_file", {"path": "a.py"})
        resp2 = _tool_response("read_file", {"path": "b.py"})
        final = ChatResponse(content="Both files reviewed.")
        mock_client.chat.side_effect = [resp1, resp2, final]

        core = _make_core_with_mocks(mock_client, [MockReadFileTool()])
        core.run("read a.py and b.py", max_steps=5)

        from app.agent.orchestrator import _build_report_from_executions

        report = _build_report_from_executions(
            task="read a.py and b.py",
            mode="READ_ONLY",
            final_response=final.content,
            steps_taken=len(core.history),
            executions=list(core.executions),
        )
        assert "a.py" in report.files_inspected
        assert "b.py" in report.files_inspected

    def test_report_files_modified_populated(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("write_file", {"path": "out.py", "content": "x"})
        final = ChatResponse(content="Written.")
        mock_client.chat.side_effect = [tool_resp, final]

        core = _make_core_with_mocks(mock_client, [MockWriteFileTool()])
        core.run("write out.py", max_steps=3)

        from app.agent.orchestrator import _build_report_from_executions

        report = _build_report_from_executions(
            task="write out.py",
            mode="ALLOW_EDITS",
            final_response=final.content,
            steps_taken=len(core.history),
            executions=list(core.executions),
        )
        assert "out.py" in report.files_modified

    def test_report_commands_executed_populated(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("run_command", {"command": "ls -la"})
        final = ChatResponse(content="Done.")
        mock_client.chat.side_effect = [tool_resp, final]

        core = _make_core_with_mocks(mock_client, [MockRunCommandTool()])
        core.run("list files", max_steps=3)

        from app.agent.orchestrator import _build_report_from_executions

        report = _build_report_from_executions(
            task="list files",
            mode="READ_ONLY",
            final_response=final.content,
            steps_taken=len(core.history),
            executions=list(core.executions),
        )
        assert "ls -la" in report.commands_executed

    def test_report_test_results_populated(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("run_tests", {"args": ""})
        final = ChatResponse(content="Tests done.")
        mock_client.chat.side_effect = [tool_resp, final]

        core = _make_core_with_mocks(mock_client, [MockRunTestsTool()])
        core.run("run tests", max_steps=3)

        from app.agent.orchestrator import _build_report_from_executions

        report = _build_report_from_executions(
            task="run tests",
            mode="READ_ONLY",
            final_response=final.content,
            steps_taken=len(core.history),
            executions=list(core.executions),
        )
        assert report.test_results is not None
        assert report.test_results["exit_code"] == 0

    def test_report_no_duplicates(self):
        mock_client = MagicMock()
        resp1 = _tool_response("read_file", {"path": "a.py"})
        resp2 = _tool_response("read_file", {"path": "a.py"})
        final = ChatResponse(content="Read twice.")
        mock_client.chat.side_effect = [resp1, resp2, final]

        core = _make_core_with_mocks(mock_client, [MockReadFileTool()])
        core.run("read a.py twice", max_steps=5)

        from app.agent.orchestrator import _build_report_from_executions

        report = _build_report_from_executions(
            task="read a.py twice",
            mode="READ_ONLY",
            final_response=final.content,
            steps_taken=len(core.history),
            executions=list(core.executions),
        )
        assert report.files_inspected.count("a.py") == 1

    def test_report_executions_field_populated(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("read_file", {"path": "a.py"})
        final = ChatResponse(content="Done.")
        mock_client.chat.side_effect = [tool_resp, final]

        core = _make_core_with_mocks(mock_client, [MockReadFileTool()])
        core.run("read a.py", max_steps=3)

        from app.agent.orchestrator import _build_report_from_executions

        report = _build_report_from_executions(
            task="read a.py",
            mode="READ_ONLY",
            final_response=final.content,
            steps_taken=len(core.history),
            executions=list(core.executions),
        )
        assert len(report.executions) == len(core.executions)
        assert report.executions[0].tool_name == "read_file"


# ---------- SENSITIVE DATA ----------


class TestSensitiveData:
    def test_api_key_not_in_executions(self):
        mock_client = MagicMock()
        tool_resp = _tool_response("read_file", {"path": "a.py"})
        final = ChatResponse(content="Done.")
        mock_client.chat.side_effect = [tool_resp, final]

        core = _make_core_with_mocks(mock_client, [MockReadFileTool()])
        core.run("read a.py", max_steps=3)

        for ex in core.executions:
            args_str = json.dumps(ex.arguments)
            result_str = json.dumps(ex.result, default=str) if ex.result else ""
            error_str = ex.error or ""
            assert OPENROUTER_API_KEY not in args_str
            assert OPENROUTER_API_KEY not in result_str
            assert OPENROUTER_API_KEY not in error_str
