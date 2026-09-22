from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolValidationError(Exception):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_tool_arguments(
    schema: ToolSchema,
    arguments: dict[str, Any],
) -> None:
    params = schema.parameters
    if not params:
        if arguments:
            raise ToolValidationError(
                [f"Unexpected arguments: {', '.join(arguments.keys())}"]
            )
        return

    properties: dict[str, Any] = params.get("properties", {})
    required: list[str] = params.get("required", [])
    errors: list[str] = []

    for req in required:
        if req not in arguments:
            errors.append(f"Missing required parameter: '{req}'")

    for key in arguments:
        if key not in properties:
            errors.append(f"Unexpected parameter: '{key}'")

    for key, value in arguments.items():
        if key not in properties:
            continue

        prop = properties[key]
        expected_type = prop.get("type")

        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Parameter '{key}' must be a string, got {type(value).__name__}")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"Parameter '{key}' must be an integer, got {type(value).__name__}")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"Parameter '{key}' must be a number, got {type(value).__name__}")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Parameter '{key}' must be a boolean, got {type(value).__name__}")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Parameter '{key}' must be an array, got {type(value).__name__}")
        elif expected_type == "object" and not isinstance(value, dict):
            errors.append(f"Parameter '{key}' must be an object, got {type(value).__name__}")

    if errors:
        raise ToolValidationError(errors)


class BaseTool(ABC):
    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.schema.name,
            "description": self.schema.description,
            "parameters": self.schema.parameters,
        }
