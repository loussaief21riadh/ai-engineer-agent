from __future__ import annotations

import json
from typing import Any

from app.agent.core import AgentCore
from app.agent.reviewer import Reviewer
from app.config import AGENT_MODE, AgentMode
from app.llm.openrouter import OpenRouterClient
from app.models.schemas import TaskReport, ToolExecution
from app.tools.base import BaseTool
from app.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool
from app.tools.terminal import RunCommandTool
from app.tools.testing import RunTestsTool
from app.tools.git import GitDiffTool, GitStatusTool


def _build_report_from_executions(
    task: str,
    mode: str,
    final_response: str,
    steps_taken: int,
    executions: list[ToolExecution],
    review: Any = None,
) -> TaskReport:
    files_inspected: list[str] = []
    files_modified: list[str] = []
    commands_executed: list[str] = []
    test_results: dict[str, Any] | None = None

    for ex in executions:
        if not ex.success:
            continue

        if ex.tool_name == "read_file":
            path = ex.arguments.get("path", "")
            if path and path not in files_inspected:
                files_inspected.append(path)

        elif ex.tool_name == "list_files":
            path = ex.arguments.get("path", ".")
            label = f"[dir] {path}"
            if label not in files_inspected:
                files_inspected.append(label)

        elif ex.tool_name == "write_file":
            path = ex.arguments.get("path", "")
            if path and path not in files_modified:
                files_modified.append(path)

        elif ex.tool_name == "run_command":
            cmd = ex.arguments.get("command", "")
            if cmd and cmd not in commands_executed:
                commands_executed.append(cmd)

        elif ex.tool_name == "run_tests":
            if isinstance(ex.result, dict):
                test_results = ex.result

        elif ex.tool_name in ("git_status", "git_diff"):
            label = f"{ex.tool_name}"
            if label not in commands_executed:
                commands_executed.append(label)

    return TaskReport(
        task=task,
        mode=mode,
        files_inspected=files_inspected,
        files_modified=files_modified,
        commands_executed=commands_executed,
        test_results=test_results,
        review=review,
        final_response=final_response,
        steps_taken=steps_taken,
        executions=executions,
    )


class Orchestrator:
    def __init__(
        self,
        client: OpenRouterClient | None = None,
        mode: AgentMode | None = None,
    ) -> None:
        self.client = client or OpenRouterClient()
        self.mode = mode or AGENT_MODE
        self.core = AgentCore(client=self.client)
        self.reviewer = Reviewer(client=self.client)
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.core.register_tool(ListFilesTool())
        self.core.register_tool(ReadFileTool())
        self.core.register_tool(RunCommandTool())
        self.core.register_tool(RunTestsTool())
        self.core.register_tool(GitStatusTool())
        self.core.register_tool(GitDiffTool())

        if self.mode in (AgentMode.ALLOW_EDITS, AgentMode.FULL_AUTONOMOUS):
            self.core.register_tool(WriteFileTool())

    def set_mode(self, mode: AgentMode) -> None:
        self.mode = mode
        self.core.tools.clear()
        self._register_default_tools()

    def run_task(self, user_message: str) -> TaskReport:
        response = self.core.run(user_message)
        executions = list(self.core.executions)

        review_result = None
        try:
            review_result = self._run_review(user_message, executions, response)
        except Exception:
            pass

        return _build_report_from_executions(
            task=user_message,
            mode=self.mode.value,
            final_response=response,
            steps_taken=len(self.core.history),
            executions=executions,
            review=review_result,
        )

    def _run_review(
        self,
        task: str,
        executions: list[ToolExecution],
        final_response: str,
    ) -> Any:
        changes_lines: list[str] = []
        diff_lines: list[str] = []
        test_results_str = ""

        for ex in executions:
            if ex.success and ex.tool_name == "read_file":
                path = ex.arguments.get("path", "")
                content_preview = str(ex.result)[:500] if ex.result else ""
                changes_lines.append(f"Read {path}:\n{content_preview}")

            elif ex.success and ex.tool_name == "write_file":
                path = ex.arguments.get("path", "")
                changes_lines.append(f"Wrote {path}")

            elif ex.success and ex.tool_name == "run_command":
                cmd = ex.arguments.get("command", "")
                result_data = ex.result if isinstance(ex.result, dict) else {}
                stdout = result_data.get("stdout", "")
                diff_lines.append(f"$ {cmd}\n{stdout[:1000]}")

            elif ex.success and ex.tool_name in ("git_status", "git_diff"):
                result_str = str(ex.result) if ex.result else ""
                diff_lines.append(f"[{ex.tool_name}]\n{result_str[:1000]}")

            elif ex.tool_name == "run_tests" and isinstance(ex.result, dict):
                test_results_str = json.dumps(ex.result, indent=2, default=str)[:2000]

        return self.reviewer.review(
            task=task,
            changes="\n".join(changes_lines) if changes_lines else "No file changes recorded.",
            diff="\n".join(diff_lines) if diff_lines else "No diff available.",
            test_results=test_results_str,
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_dict() for tool in self.core.tools.values()]
