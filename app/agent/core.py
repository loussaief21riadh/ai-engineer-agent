from __future__ import annotations

import json
import re
import time
from typing import Any

from app.agent.prompts import SYSTEM_PROMPT
from app.config import MAX_AGENT_STEPS, PRIMARY_MODEL
from app.llm.openrouter import ChatResponse, OpenRouterClient, OpenRouterError
from app.models.schemas import AgentMessage, ToolCall, ToolExecution, ToolResult
from app.tools.base import BaseTool, ToolValidationError, validate_tool_arguments

LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 1.0
LLM_RETRY_MAX_DELAY = 30.0


def _parse_tool_call_from_text(text: str) -> ToolCall | None:
    pattern = r"```tool\s*\n(.*?)\n\s*```"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or "name" not in data:
        return None

    return ToolCall(
        name=data["name"],
        arguments=data.get("arguments", {}),
    )


def build_tool_definitions(tools: dict[str, BaseTool]) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for tool in tools.values():
        s = tool.schema
        definitions.append({
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            },
        })
    return definitions


class AgentCore:
    def __init__(
        self,
        client: OpenRouterClient | None = None,
        tools: dict[str, BaseTool] | None = None,
        on_llm_call: Any = None,
        on_llm_response: Any = None,
    ) -> None:
        self.client = client or OpenRouterClient()
        self.tools = tools or {}
        self.history: list[AgentMessage] = []
        self.executions: list[ToolExecution] = []
        self._use_native_tools = True
        self._on_llm_call = on_llm_call
        self._on_llm_response = on_llm_response
        self._fallback_model: str = ""

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.schema.name] = tool

    def _build_tool_descriptions(self) -> str:
        lines = []
        for tool in self.tools.values():
            s = tool.schema
            lines.append(f"- {s.name}: {s.description}")
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        tool_descriptions = self._build_tool_descriptions()
        base = f"{SYSTEM_PROMPT}\n\nAVAILABLE TOOLS:\n{tool_descriptions}"

        if not self._use_native_tools:
            base += """

TOOL CALL FORMAT (fallback mode):
When you need to use a tool, respond with exactly one tool call in this format:

```tool
{"name": "tool_name", "arguments": {"param": "value"}}
```

Always use valid JSON in the tool call block. Wait for the tool result before continuing.

IMPORTANT: Only call one tool at a time. After receiving the tool result, continue reasoning and decide whether another tool is needed or if you can provide your final answer.
"""

        return base

    def _call_llm(self, messages: list[dict[str, str]], model: str = "") -> ChatResponse:
        effective_model = model or PRIMARY_MODEL
        tool_defs = build_tool_definitions(self.tools) if self._use_native_tools and self.tools else None
        if self._on_llm_call is not None:
            self._on_llm_call()
        return self.client.chat(messages=messages, model=effective_model, tools=tool_defs)

    def _extract_tool_calls(self, chat_response: ChatResponse) -> list[ToolCall]:
        if chat_response.has_tool_calls:
            self._use_native_tools = True
            result = []
            for tc in chat_response.tool_calls:
                result.append(ToolCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}),
                    tool_call_id=tc.get("id", ""),
                ))
            return result

        fallback = _parse_tool_call_from_text(chat_response.content)
        if fallback is not None:
            self._use_native_tools = False
            return [fallback]

        return []

    def _execute_single_tool(self, tool_call: ToolCall, step: int) -> ToolResult:
        tool = self.tools[tool_call.name]

        validate_tool_arguments(tool.schema, tool_call.arguments)

        start = time.monotonic()
        try:
            result = tool.execute(**tool_call.arguments)
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        elapsed_ms = (time.monotonic() - start) * 1000

        tool_result = ToolResult(
            tool_name=tool_call.name,
            success=result.get("success", False),
            result=result.get("result"),
            error=result.get("error"),
            tool_call_id=tool_call.tool_call_id,
        )

        execution = ToolExecution(
            step=step,
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            success=tool_result.success,
            result=tool_result.result,
            error=tool_result.error,
            duration_ms=round(elapsed_ms, 1),
            tool_call_id=tool_call.tool_call_id,
        )
        self.executions.append(execution)

        return tool_result

    def run(self, user_message: str, max_steps: int = 0, model: str = "") -> str:
        effective_max = max_steps or MAX_AGENT_STEPS
        self.history = []
        self.executions = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_message},
        ]

        self.history.append(AgentMessage(role="user", content=user_message))

        steps = 0

        while steps < effective_max:
            steps += 1

            chat_response = None
            last_error = None
            for attempt in range(LLM_MAX_RETRIES):
                try:
                    chat_response = self._call_llm(messages, model=model)
                    break
                except OpenRouterError as exc:
                    last_error = exc
                    if attempt < LLM_MAX_RETRIES - 1:
                        delay = min(LLM_RETRY_BASE_DELAY * (2 ** attempt), LLM_RETRY_MAX_DELAY)
                        time.sleep(delay)
                    continue

            if chat_response is None and self._fallback_model and self._fallback_model != model:
                for attempt in range(LLM_MAX_RETRIES):
                    try:
                        chat_response = self._call_llm(messages, model=self._fallback_model)
                        break
                    except OpenRouterError as exc:
                        last_error = exc
                        if attempt < LLM_MAX_RETRIES - 1:
                            delay = min(LLM_RETRY_BASE_DELAY * (2 ** attempt), LLM_RETRY_MAX_DELAY)
                            time.sleep(delay)
                        continue

            if chat_response is None:
                return f"LLM error after {LLM_MAX_RETRIES} retries: {last_error}"

            if self._on_llm_response is not None:
                self._on_llm_response(
                    model=model or PRIMARY_MODEL,
                    prompt_tokens=chat_response.prompt_tokens,
                    completion_tokens=chat_response.completion_tokens,
                    total_tokens=chat_response.total_tokens,
                )

            tool_calls = self._extract_tool_calls(chat_response)

            if not tool_calls:
                self.history.append(AgentMessage(role="assistant", content=chat_response.content))
                return chat_response.content

            messages.append({"role": "assistant", "content": chat_response.content, "tool_calls": [
                {"id": tc.tool_call_id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in tool_calls
            ] if self._use_native_tools and any(tc.tool_call_id for tc in tool_calls) else None})
            if messages[-1].get("tool_calls") is None:
                messages[-1].pop("tool_calls", None)

            self.history.append(AgentMessage(role="assistant", content=chat_response.content))

            for tc in tool_calls:
                if tc.name not in self.tools:
                    error_msg = f"Unknown tool: {tc.name}"
                    if self._use_native_tools and tc.tool_call_id:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.tool_call_id,
                            "content": json.dumps({"success": False, "error": error_msg}),
                        })
                    else:
                        messages.append({"role": "user", "content": f"Error: {error_msg}. Use a valid tool."})
                    continue

                try:
                    tool_result = self._execute_single_tool(tc, steps)
                except ToolValidationError as exc:
                    error_msg = f"Invalid arguments: {'; '.join(exc.errors)}"
                    if self._use_native_tools and tc.tool_call_id:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.tool_call_id,
                            "content": json.dumps({"success": False, "error": error_msg}),
                        })
                    else:
                        messages.append({"role": "user", "content": f"Error: {error_msg}. Fix the arguments and try again."})
                    continue

                self.history.append(AgentMessage(
                    role="tool",
                    content=json.dumps(tool_result.model_dump(), indent=2, default=str),
                ))

                tool_result_str = json.dumps(
                    tool_result.model_dump(),
                    indent=2,
                    default=str,
                )

                if self._use_native_tools and tc.tool_call_id:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.tool_call_id,
                        "content": tool_result_str,
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Tool result:\n{tool_result_str}\n\nContinue reasoning. Call another tool if needed, or provide your final answer.",
                    })

        if self.executions:
            summary_parts = []
            for ex in self.executions:
                status = "OK" if ex.success else "FAIL"
                summary_parts.append(f"  Step {ex.step}: {ex.tool_name} → {status}")
            execution_summary = "\n".join(summary_parts)
            return (
                f"Reached maximum steps ({effective_max}) without completing the task.\n\n"
                f"Actions completed before limit:\n{execution_summary}\n\n"
                f"Please provide a summary of what was accomplished so far."
            )

        return "Reached maximum steps without completing the task."
