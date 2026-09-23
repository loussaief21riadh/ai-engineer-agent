from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import OPENROUTER_API_KEY, OPENROUTER_URL


class OpenRouterError(Exception):
    pass


class ChatResponse:
    def __init__(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        raw: dict[str, Any] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.raw = raw or {}
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class OpenRouterClient:
    def __init__(self, api_key: str = "", base_url: str = "") -> None:
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = base_url or OPENROUTER_URL

        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        timeout: float = 120.0,
    ) -> ChatResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if tools:
            payload["tools"] = tools

        try:
            response = httpx.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise OpenRouterError(f"Request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"HTTP error: {exc}") from exc

        if response.status_code != 200:
            raise OpenRouterError(
                f"OpenRouter returned status {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()

        if "choices" not in data or not data["choices"]:
            raise OpenRouterError("No choices in response")

        message = data["choices"][0].get("message", {})

        content = message.get("content", "") or ""

        native_tool_calls: list[dict[str, Any]] = []
        raw_tool_calls = message.get("tool_calls", [])
        if raw_tool_calls:
            for tc in raw_tool_calls:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    args_raw = func.get("arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    else:
                        args = args_raw

                    native_tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": args,
                    })

        return ChatResponse(
            content=content,
            tool_calls=native_tool_calls,
            raw=data,
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
        )
