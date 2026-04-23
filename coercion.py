"""Parameter type coercion for LLM tool calls.

LLMs sometimes send typed values as strings (e.g. ``"42"`` instead of ``42``
for an integer property).  This module coerces string parameters to their
schema-declared types so tool handlers receive the expected Python types.

Coercion failure is intentionally *not* a hard error: the original string
is kept so the tool handler can surface a clear type mismatch to the model.
"""
from __future__ import annotations

import json as _json


def coerce_params(params: dict, schema: dict) -> dict:
    """Coerce string parameter values to their schema-declared types.

    Handles both schema styles:
    - Top-level ``properties`` (rare, e.g. test fixtures)
    - Anthropic-style ``input_schema.properties`` (all built-in tools)
    """
    props = (
        schema.get("properties")
        or schema.get("input_schema", {}).get("properties", {})
    )
    if not props:
        return dict(params)
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
        return value


def _coerce_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def _coerce_bool(value):
    """Coerce string to bool. Returns original string if unrecognised."""
    low = value.lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    return value  # tool handler reports the real type mismatch


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
