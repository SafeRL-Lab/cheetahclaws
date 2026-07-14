from __future__ import annotations

import pytest

from cheetahclaws.tool_registry import (
    ToolDef,
    clear_tool_cache,
    clear_registry,
    execute_tool,
    get_active_tool_names,
    get_all_tools,
    get_tool,
    get_tool_schemas,
    register_tool,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate registry tests without leaving later integration tests empty."""
    original_tools = get_all_tools()
    clear_registry()
    clear_tool_cache()
    yield
    clear_registry()
    clear_tool_cache()
    for tool in original_tools:
        register_tool(tool)


def _make_echo_tool(name: str = "echo", read_only: bool = False) -> ToolDef:
    """Helper to build a simple echo tool."""
    schema = {
        "name": name,
        "description": f"Echo tool ({name})",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "text to echo"},
            },
            "required": ["text"],
        },
    }

    def func(params: dict, config: dict) -> str:
        return params["text"]

    return ToolDef(
        name=name,
        schema=schema,
        func=func,
        read_only=read_only,
        concurrent_safe=True,
    )


# ------------------------------------------------------------------
# register and get
# ------------------------------------------------------------------

def test_register_and_get():
    tool = _make_echo_tool()
    register_tool(tool)
    result = get_tool("echo")
    assert result is not None
    assert result.name == "echo"


def test_get_unknown_returns_none():
    assert get_tool("no_such_tool") is None


# ------------------------------------------------------------------
# get_all_tools
# ------------------------------------------------------------------

def test_get_all_tools_empty():
    assert get_all_tools() == []


def test_get_all_tools():
    register_tool(_make_echo_tool("a"))
    register_tool(_make_echo_tool("b"))
    names = [t.name for t in get_all_tools()]
    assert sorted(names) == ["a", "b"]


# ------------------------------------------------------------------
# get_tool_schemas
# ------------------------------------------------------------------

def test_get_tool_schemas():
    register_tool(_make_echo_tool("echo"))
    schemas = get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "echo"


def test_tool_profiles_filter_schemas_and_names():
    register_tool(ToolDef(
        name="core",
        schema={"name": "core", "input_schema": {}},
        func=lambda _p, _c: "core",
        profiles=frozenset({"standard"}),
    ))
    register_tool(ToolDef(
        name="research_only",
        schema={"name": "research_only", "input_schema": {}},
        func=lambda _p, _c: "research",
        profiles=frozenset({"research"}),
    ))
    register_tool(ToolDef(
        name="full_only",
        schema={"name": "full_only", "input_schema": {}},
        func=lambda _p, _c: "full",
        profiles=frozenset({"full"}),
    ))

    assert [s["name"] for s in get_tool_schemas("standard")] == ["core"]
    assert [s["name"] for s in get_tool_schemas("research")] == [
        "core", "research_only",
    ]
    assert get_active_tool_names("orchestration") == frozenset({"core"})
    assert {s["name"] for s in get_tool_schemas("full")} == {
        "core", "research_only", "full_only",
    }


def test_tool_profiles_honor_disabled_tools():
    register_tool(ToolDef(
        name="core",
        schema={"name": "core", "input_schema": {}},
        func=lambda _p, _c: "core",
        profiles=frozenset({"standard"}),
    ))
    assert get_tool_schemas("standard", disabled_tools=["core"]) == []


# ------------------------------------------------------------------
# execute_tool
# ------------------------------------------------------------------

def test_execute_tool():
    register_tool(_make_echo_tool())
    result = execute_tool("echo", {"text": "hello"}, config={})
    assert result == "hello"


def test_execute_unknown_tool():
    result = execute_tool("missing", {}, config={})
    assert "unknown" in result.lower() or "not found" in result.lower()


# ------------------------------------------------------------------
# output truncation
# ------------------------------------------------------------------

def test_output_truncation():
    def big_func(params: dict, config: dict) -> str:
        return "x" * 100

    tool = ToolDef(
        name="big",
        schema={"name": "big", "description": "big", "input_schema": {"type": "object", "properties": {}}},
        func=big_func,
        read_only=True,
        concurrent_safe=True,
    )
    register_tool(tool)

    result = execute_tool("big", {}, config={}, max_output=40)
    # first half = 20 chars, last quarter = 10 chars, marker in between.
    # The truncation marker now includes a model-context-safety message
    # which is ~80-150 chars depending on file_path hint.
    assert len(result) < 200
    assert "truncated" in result
    # The kept portion: first 20 + last 10 should be present
    assert result.startswith("x" * 20)
    assert result.endswith("x" * 10)


def test_no_truncation_when_within_limit():
    register_tool(_make_echo_tool())
    result = execute_tool("echo", {"text": "short"}, config={})
    assert result == "short"


def test_cache_stores_bounded_result_and_reapplies_smaller_cap():
    calls = 0

    def big_func(params: dict, config: dict) -> str:
        nonlocal calls
        calls += 1
        return "x" * 20_000

    register_tool(ToolDef(
        name="cached_big",
        schema={"name": "cached_big", "input_schema": {}},
        func=big_func,
        read_only=True,
    ))

    first = execute_tool(
        "cached_big", {}, {"max_tool_cache_output": 2_000}, max_output=10_000,
    )
    second = execute_tool(
        "cached_big", {}, {"max_tool_cache_output": 2_000}, max_output=1_500,
    )

    assert calls == 1
    assert "truncated" in first
    assert "truncated" in second
    assert len(second) < len(first)


def test_cache_key_includes_input_bound_settings():
    calls = 0

    def config_echo(_params: dict, config: dict) -> str:
        nonlocal calls
        calls += 1
        return str(config["tool_read_max_bytes"])

    register_tool(ToolDef(
        name="config_sensitive",
        schema={"name": "config_sensitive", "input_schema": {}},
        func=config_echo,
        read_only=True,
    ))

    assert execute_tool("config_sensitive", {}, {"tool_read_max_bytes": 10}) == "10"
    assert execute_tool("config_sensitive", {}, {"tool_read_max_bytes": 20}) == "20"
    assert calls == 2


# ------------------------------------------------------------------
# duplicate register overwrites
# ------------------------------------------------------------------

def test_duplicate_register_overwrites():
    register_tool(_make_echo_tool("dup"))

    def new_func(params: dict, config: dict) -> str:
        return "new"

    replacement = ToolDef(
        name="dup",
        schema={"name": "dup", "description": "new", "input_schema": {"type": "object", "properties": {}}},
        func=new_func,
        read_only=False,
        concurrent_safe=False,
    )
    register_tool(replacement)

    assert len(get_all_tools()) == 1
    result = execute_tool("dup", {}, config={})
    assert result == "new"
