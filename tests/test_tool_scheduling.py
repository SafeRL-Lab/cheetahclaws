"""Tests for tool scheduling (depends_on, tool_call_alias) and param coercion."""
from tool_registry import (
    get_tool_schemas,
    execute_tool,
    register_tool,
    ToolDef,
    _coerce_params,
    _SCHEDULING_PROPS,
)

# Trigger builtin tool registration
import tools  # noqa: F401


class TestSchedulingPropsInjection:
    def test_schemas_contain_scheduling_fields(self):
        schemas = get_tool_schemas()
        assert len(schemas) > 0
        for s in schemas:
            props = s.get("properties", {})
            assert "tool_call_alias" in props, f"Missing tool_call_alias in {s.get('name')}"
            assert "depends_on" in props, f"Missing depends_on in {s.get('name')}"

    def test_scheduling_props_have_correct_types(self):
        schemas = get_tool_schemas()
        s = schemas[0]
        assert s["properties"]["tool_call_alias"]["type"] == "string"
        assert s["properties"]["depends_on"]["type"] == "array"

    def test_original_schema_not_mutated(self):
        """Verify deepcopy prevents mutation of registered schemas."""
        schemas1 = get_tool_schemas()
        schemas1[0]["properties"]["tool_call_alias"]["EXTRA"] = True
        schemas2 = get_tool_schemas()
        assert "EXTRA" not in schemas2[0]["properties"]["tool_call_alias"]


class TestCoerceParams:
    def test_int_coercion(self):
        schema = {"properties": {"limit": {"type": "integer"}}}
        assert _coerce_params({"limit": "42"}, schema) == {"limit": 42}

    def test_float_coercion(self):
        schema = {"properties": {"rate": {"type": "number"}}}
        assert _coerce_params({"rate": "3.14"}, schema) == {"rate": 3.14}

    def test_bool_true(self):
        schema = {"properties": {"flag": {"type": "boolean"}}}
        assert _coerce_params({"flag": "true"}, schema) == {"flag": True}

    def test_bool_false(self):
        schema = {"properties": {"flag": {"type": "boolean"}}}
        assert _coerce_params({"flag": "false"}, schema) == {"flag": False}

    def test_array_coercion(self):
        schema = {"properties": {"items": {"type": "array"}}}
        result = _coerce_params({"items": '["a","b"]'}, schema)
        assert result == {"items": ["a", "b"]}

    def test_object_coercion(self):
        schema = {"properties": {"meta": {"type": "object"}}}
        result = _coerce_params({"meta": '{"k": 1}'}, schema)
        assert result == {"meta": {"k": 1}}

    def test_passthrough_string(self):
        schema = {"properties": {"name": {"type": "string"}}}
        assert _coerce_params({"name": "hello"}, schema) == {"name": "hello"}

    def test_invalid_json_passthrough(self):
        schema = {"properties": {"items": {"type": "array"}}}
        assert _coerce_params({"items": "not-json"}, schema) == {"items": "not-json"}

    def test_unknown_prop_passthrough(self):
        schema = {"properties": {}}
        assert _coerce_params({"x": "y"}, schema) == {"x": "y"}


class TestExecuteToolStripsScheduling:
    def setup_method(self):
        self._received = {}

        def _handler(params, config=None):
            self._received = dict(params)
            return "ok"

        register_tool(ToolDef(
            name="test_sched_tool",
            schema={
                "name": "test_sched_tool",
                "description": "test tool",
                "properties": {"msg": {"type": "string"}},
            },
            func=_handler,
            read_only=True,
        ))

    def test_scheduling_params_stripped(self):
        execute_tool(
            "test_sched_tool",
            {"msg": "hi", "tool_call_alias": "t1", "depends_on": ["w1"]},
            config={},
        )
        assert "tool_call_alias" not in self._received
        assert "depends_on" not in self._received
        assert self._received.get("msg") == "hi"
