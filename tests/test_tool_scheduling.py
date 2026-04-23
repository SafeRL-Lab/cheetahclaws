"""Tests for tool scheduling (depends_on, tool_call_alias) and ID uniquification."""
import pytest

from tool_registry import (
    get_tool_schemas,
    execute_tool,
    register_tool,
    clear_registry,
    ToolDef,
    _SCHEDULING_PROPS,
)
from id_uniquify import uniquify_tool_call_ids, _collect_used_ids

import tools  # noqa: F401


@pytest.fixture(autouse=True)
def _ensure_builtins():
    """Guarantee builtins are registered even after another test cleared the registry."""
    clear_registry()
    tools._register_builtins()
    yield
    clear_registry()


class TestSchedulingPropsInjection:
    def test_schemas_contain_scheduling_fields(self):
        schemas = get_tool_schemas()
        assert len(schemas) > 0
        for s in schemas:
            # Handle both schema styles
            props = s.get("properties") or s.get("input_schema", {}).get("properties", {})
            assert "tool_call_alias" in props, f"Missing tool_call_alias in {s.get('name')}"
            assert "depends_on" in props, f"Missing depends_on in {s.get('name')}"

    def test_scheduling_props_have_correct_types(self):
        schemas = get_tool_schemas()
        s = schemas[0]
        props = s.get("properties") or s.get("input_schema", {}).get("properties", {})
        assert props["tool_call_alias"]["type"] == "string"
        assert props["depends_on"]["type"] == "array"

    def test_original_schema_not_mutated(self):
        """Verify deepcopy prevents mutation of registered schemas."""
        schemas1 = get_tool_schemas()
        s1 = schemas1[0]
        props1 = s1.get("properties") or s1.get("input_schema", {}).get("properties", {})
        props1["tool_call_alias"]["EXTRA"] = True
        schemas2 = get_tool_schemas()
        s2 = schemas2[0]
        props2 = s2.get("properties") or s2.get("input_schema", {}).get("properties", {})
        assert "EXTRA" not in props2["tool_call_alias"]

    def test_input_schema_style_gets_scheduling_in_right_place(self):
        """Built-in tools use input_schema; scheduling props must land there."""
        schemas = get_tool_schemas()
        read_schema = next(s for s in schemas if s["name"] == "Read")
        # Must be inside input_schema.properties, NOT top-level properties
        assert "tool_call_alias" in read_schema["input_schema"]["properties"]
        assert "depends_on" in read_schema["input_schema"]["properties"]
        assert "properties" not in read_schema or "tool_call_alias" not in read_schema.get("properties", {})


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
                "input_schema": {
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                },
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


class TestIdUniquify:
    """Tests for tool_call_id collision prevention."""

    def _make_state(self, messages=None):
        """Create a minimal state object with messages and turn_count."""
        class _State:
            pass
        s = _State()
        s.messages = messages or []
        s.turn_count = 2
        return s

    def test_fresh_ids_pass_through(self):
        state = self._make_state()
        tcs = [{"id": "r1", "name": "Read", "input": {}}]
        remap = uniquify_tool_call_ids(tcs, state)
        assert remap == {}
        assert tcs[0]["id"] == "r1"

    def test_colliding_id_remapped(self):
        state = self._make_state([
            {"role": "assistant", "content": "", "tool_calls": [{"id": "r1", "name": "Read", "input": {}}]},
            {"role": "tool", "tool_call_id": "r1", "name": "Read", "content": "data"},
        ])
        tcs = [{"id": "r1", "name": "Read", "input": {}}]
        remap = uniquify_tool_call_ids(tcs, state)
        assert remap == {"r1": "t2_r1"}
        assert tcs[0]["id"] == "t2_r1"

    def test_depends_on_rewritten(self):
        state = self._make_state([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "w1", "name": "Write", "input": {}},
                {"id": "r1", "name": "Read", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "w1", "name": "Write", "content": "ok"},
            {"role": "tool", "tool_call_id": "r1", "name": "Read", "content": "data"},
        ])
        tcs = [
            {"id": "w1", "name": "Write", "input": {}},
            {"id": "r1", "name": "Read", "input": {"depends_on": ["w1"]}},
        ]
        remap = uniquify_tool_call_ids(tcs, state)
        assert tcs[0]["id"] == "t2_w1"
        assert tcs[1]["id"] == "t2_r1"
        # depends_on must be rewritten to match the new w1 id
        assert tcs[1]["input"]["depends_on"] == ["t2_w1"]

    def test_multiple_collisions_get_numeric_suffix(self):
        state = self._make_state([
            {"role": "assistant", "content": "", "tool_calls": [{"id": "r1", "name": "Read", "input": {}}]},
            {"role": "tool", "tool_call_id": "r1", "name": "Read", "content": ""},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t2_r1", "name": "Read", "input": {}}]},
            {"role": "tool", "tool_call_id": "t2_r1", "name": "Read", "content": ""},
        ])
        tcs = [{"id": "r1", "name": "Read", "input": {}}]
        remap = uniquify_tool_call_ids(tcs, state)
        # t2_r1 is taken, so should get t2_r1_2
        assert tcs[0]["id"] == "t2_r1_2"

    def test_collect_used_ids(self):
        state = self._make_state([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a1", "name": "Read", "input": {}},
                {"id": "a2", "name": "Write", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "a1", "name": "Read", "content": "x"},
            {"role": "tool", "tool_call_id": "a2", "name": "Write", "content": "y"},
        ])
        used = _collect_used_ids(state)
        assert used == {"a1", "a2"}

    def test_empty_tool_calls(self):
        state = self._make_state()
        assert uniquify_tool_call_ids([], state) == {}

    def test_mixed_fresh_and_colliding(self):
        state = self._make_state([
            {"role": "assistant", "content": "", "tool_calls": [{"id": "r1", "name": "Read", "input": {}}]},
            {"role": "tool", "tool_call_id": "r1", "name": "Read", "content": ""},
        ])
        tcs = [
            {"id": "r1", "name": "Read", "input": {}},
            {"id": "w1", "name": "Write", "input": {"depends_on": ["r1"]}},
        ]
        remap = uniquify_tool_call_ids(tcs, state)
        assert tcs[0]["id"] == "t2_r1"  # remapped
        assert tcs[1]["id"] == "w1"     # fresh, untouched
        assert tcs[1]["input"]["depends_on"] == ["t2_r1"]  # ref rewritten
