"""Tool plugin registry for cheetahclaws.

Provides a central registry for tool definitions, lookup, schema export,
dispatch with output truncation, and result caching for read-only tools.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDef:
    """Definition of a single tool plugin.

    Attributes:
        name: unique tool identifier
        schema: JSON-schema dict sent to the API (name, description, input_schema)
        func: callable(params: dict, config: dict) -> str
        read_only: True if the tool never mutates state
        concurrent_safe: True if safe to run in parallel with other tools
    """
    name: str
    schema: Dict[str, Any]
    func: Callable[[Dict[str, Any], Dict[str, Any]], str]
    read_only: bool = False
    concurrent_safe: bool = False


# --------------- internal state ---------------

_registry: Dict[str, ToolDef] = {}

# --------------- result cache (read-only tools only) ---------------

_CACHE_MAX = 64  # max cached entries
_cache: Dict[str, str] = {}   # hash → result
_cache_order: list[str] = []  # LRU eviction order


def _cache_key(name: str, params: Dict[str, Any]) -> str:
    """Create a stable hash from tool name + params."""
    raw = json.dumps({"n": name, "p": params}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def clear_tool_cache() -> None:
    """Clear the tool result cache. Called on file writes to invalidate."""
    _cache.clear()
    _cache_order.clear()


# --------------- public API ---------------

def register_tool(tool_def: ToolDef) -> None:
    """Register a tool, overwriting any existing tool with the same name."""
    _registry[tool_def.name] = tool_def


def get_tool(name: str) -> Optional[ToolDef]:
    """Look up a tool by name. Returns None if not found."""
    return _registry.get(name)


def get_all_tools() -> List[ToolDef]:
    """Return all registered tools (insertion order)."""
    return list(_registry.values())


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return the schemas of all registered tools (for API tool parameter)."""
    return [t.schema for t in _registry.values()]


def execute_tool(
    name: str,
    params: Dict[str, Any],
    config: Dict[str, Any],
    max_output: int = 32000,
) -> str:
    """Dispatch a tool call by name.

    Args:
        name: tool name
        params: tool input parameters dict
        config: runtime configuration dict
        max_output: maximum allowed output length in characters

    Returns:
        Tool result string, possibly truncated.
    """
    tool = get_tool(name)
    if tool is None:
        return f"Error: tool '{name}' not found."

    # Cache hit for read-only tools (same name + same params = same result)
    use_cache = tool.read_only
    if use_cache:
        key = _cache_key(name, params)
        if key in _cache:
            return _cache[key]
    else:
        # Write tools invalidate cache (file content may have changed)
        if name in ("Write", "Edit", "Bash", "NotebookEdit"):
            clear_tool_cache()

    try:
        result = tool.func(params, config)
    except Exception as e:
        return f"Error executing {name}: {e}"

    # Store in cache for read-only tools
    if use_cache:
        _cache[key] = result
        _cache_order.append(key)
        # Evict oldest if over limit
        while len(_cache_order) > _CACHE_MAX:
            old = _cache_order.pop(0)
            _cache.pop(old, None)

    if len(result) > max_output:
        first_half = max_output // 2
        last_quarter = max_output // 4
        truncated = len(result) - first_half - last_quarter
        result = (
            result[:first_half]
            + f"\n[... {truncated} chars truncated ...]\n"
            + result[-last_quarter:]
        )

    return result


def clear_registry() -> None:
    """Remove all registered tools. Intended for testing."""
    _registry.clear()


# ── Tool scheduling support ────────────────────────────────────────────────

import copy as _copy
import json as _json

_SCHEDULING_PROPS = {
    "tool_call_alias": {
        "type": "string",
        "description": (
            "Optional alias for this tool call. "
            "Other tools can reference it in depends_on."
        ),
    },
    "depends_on": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "List of tool_call IDs or aliases that must complete before this tool runs."
        ),
    },
}


def _coerce_params(params: dict, schema: dict) -> dict:
    """Coerce string parameter values to their schema-declared types.

    Coercion failure is not a hard error: the original string is kept and
    passed to the tool handler, which will surface a clear type error to
    the model (e.g. `expected int, got 'abc'`) far more usefully than a
    ValueError from the registry wrapper.
    """
    props = schema.get("properties", {})
    return {k: _coerce_value_for(k, v, props) for k, v in params.items()}


def _coerce_value_for(key: str, value, props: dict):
    """Coerce a single value according to its declared type, else return as-is."""
    prop_schema = props.get(key)
    if not prop_schema or not isinstance(value, str):
        return value
    coercer = _COERCERS.get(prop_schema.get("type"))
    if coercer is None:
        return value
    return coercer(value)


def _coerce_int(value):
    try:
        return int(value)
    except ValueError:
        return value  # intentional: tool handler reports the real type mismatch


def _coerce_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def _coerce_bool(value):
    return value.lower() in ("true", "1", "yes")


def _coerce_json(value):
    try:
        return _json.loads(value)
    except (ValueError, _json.JSONDecodeError):
        return value


_COERCERS = {
    "integer": _coerce_int,
    "number":  _coerce_float,
    "boolean": _coerce_bool,
    "array":   _coerce_json,
    "object":  _coerce_json,
}


# Wrap get_tool_schemas to inject scheduling properties
_orig_get_tool_schemas = get_tool_schemas


def get_tool_schemas():
    """Return tool schemas with scheduling properties injected."""
    schemas = _orig_get_tool_schemas()
    result = []
    for s in schemas:
        s = _copy.deepcopy(s)
        props = s.setdefault("properties", {})
        for k, v in _SCHEDULING_PROPS.items():
            props.setdefault(k, _copy.deepcopy(v))
        result.append(s)
    return result


# Wrap execute_tool to strip scheduling params and coerce types
_orig_execute_tool = execute_tool


def execute_tool(name, params, *args, **kwargs):
    """Execute a tool after stripping scheduling params and coercing types."""
    clean = {k: v for k, v in params.items() if k not in _SCHEDULING_PROPS}
    tool = get_tool(name)
    if tool is not None:
        clean = _coerce_params(clean, tool.schema)
    return _orig_execute_tool(name, clean, *args, **kwargs)
