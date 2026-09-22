import pytest

from app.models.schemas import ToolResult
from app.tools.base import BaseTool, ToolSchema


class TestToolSchema:
    def test_creates_with_valid_fields(self):
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        assert schema.name == "test_tool"
        assert schema.description == "A test tool"
        assert schema.parameters == {"type": "object", "properties": {}}

    def test_defaults_to_empty_dict_for_parameters(self):
        schema = ToolSchema(name="t", description="d", parameters={})
        assert schema.parameters == {}

    def test_rejects_missing_name(self):
        with pytest.raises(Exception):
            ToolSchema(description="d", parameters={})

    def test_rejects_missing_description(self):
        with pytest.raises(Exception):
            ToolSchema(name="n", parameters={})

    def test_allows_extra_fields(self):
        schema = ToolSchema(name="n", description="d", parameters={}, extra="ok")
        assert schema.name == "n"

    def test_is_json_serializable(self):
        schema = ToolSchema(
            name="n", description="d", parameters={"k": "v"}
        )
        data = schema.model_dump()
        assert data["name"] == "n"
        assert data["parameters"]["k"] == "v"


class TestBaseTool:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseTool()

    def test_subclass_must_implement_schema(self):
        class IncompleteTool(BaseTool):
            def execute(self, **kwargs):
                return {}

        with pytest.raises(TypeError):
            IncompleteTool()

    def test_subclass_must_implement_execute(self):
        class IncompleteTool(BaseTool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(name="t", description="d", parameters={})

        with pytest.raises(TypeError):
            IncompleteTool()

    def test_concrete_subclass_works(self):
        class ConcreteTool(BaseTool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="concrete",
                    description="works",
                    parameters={"type": "object"},
                )

            def execute(self, **kwargs):
                return {"ok": True}

        tool = ConcreteTool()
        assert tool.schema.name == "concrete"
        assert tool.execute() == {"ok": True}

    def test_to_dict(self):
        class DummyTool(BaseTool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="dummy",
                    description="A dummy",
                    parameters={"type": "object", "properties": {"x": {}}},
                )

            def execute(self, **kwargs):
                return None

        tool = DummyTool()
        result = tool.to_dict()
        assert result == {
            "name": "dummy",
            "description": "A dummy",
            "parameters": {"type": "object", "properties": {"x": {}}},
        }

    def test_to_dict_matches_schema(self):
        class DummyTool(BaseTool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="s", description="d", parameters={"p": 1}
                )

            def execute(self, **kwargs):
                return None

        tool = DummyTool()
        assert tool.to_dict()["name"] == tool.schema.name
        assert tool.to_dict()["description"] == tool.schema.description
        assert tool.to_dict()["parameters"] == tool.schema.parameters


class TestToolResult:
    def test_success_result(self):
        r = ToolResult(tool_name="t", success=True, result="all good")
        assert r.tool_name == "t"
        assert r.success is True
        assert r.result == "all good"

    def test_failure_result(self):
        r = ToolResult(tool_name="t", success=False, error="something broke")
        assert r.success is False
        assert r.error == "something broke"

    def test_optional_fields(self):
        r = ToolResult(tool_name="t", success=True)
        assert r.result is None
        assert r.error is None

    def test_tool_name_required(self):
        with pytest.raises(Exception):
            ToolResult(success=True)
