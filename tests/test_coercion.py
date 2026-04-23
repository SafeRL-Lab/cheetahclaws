"""Tests for parameter type coercion (coercion.py)."""
from coercion import coerce_params, _coerce_bool


class TestCoerceParams:
    def test_int_coercion(self):
        schema = {"properties": {"limit": {"type": "integer"}}}
        assert coerce_params({"limit": "42"}, schema) == {"limit": 42}

    def test_float_coercion(self):
        schema = {"properties": {"rate": {"type": "number"}}}
        assert coerce_params({"rate": "3.14"}, schema) == {"rate": 3.14}

    def test_bool_true_variants(self):
        schema = {"properties": {"flag": {"type": "boolean"}}}
        for val in ("true", "True", "TRUE", "1", "yes", "Yes"):
            assert coerce_params({"flag": val}, schema) == {"flag": True}, f"Failed for {val!r}"

    def test_bool_false_variants(self):
        schema = {"properties": {"flag": {"type": "boolean"}}}
        for val in ("false", "False", "FALSE", "0", "no", "No"):
            assert coerce_params({"flag": val}, schema) == {"flag": False}, f"Failed for {val!r}"

    def test_bool_garbage_returns_original(self):
        """Unknown boolean-like strings must pass through for tool handler to report."""
        schema = {"properties": {"flag": {"type": "boolean"}}}
        result = coerce_params({"flag": "banana"}, schema)
        assert result == {"flag": "banana"}

    def test_array_coercion(self):
        schema = {"properties": {"items": {"type": "array"}}}
        result = coerce_params({"items": '["a","b"]'}, schema)
        assert result == {"items": ["a", "b"]}

    def test_object_coercion(self):
        schema = {"properties": {"meta": {"type": "object"}}}
        result = coerce_params({"meta": '{"k": 1}'}, schema)
        assert result == {"meta": {"k": 1}}

    def test_passthrough_string(self):
        schema = {"properties": {"name": {"type": "string"}}}
        assert coerce_params({"name": "hello"}, schema) == {"name": "hello"}

    def test_invalid_json_passthrough(self):
        schema = {"properties": {"items": {"type": "array"}}}
        assert coerce_params({"items": "not-json"}, schema) == {"items": "not-json"}

    def test_invalid_int_passthrough(self):
        schema = {"properties": {"limit": {"type": "integer"}}}
        assert coerce_params({"limit": "abc"}, schema) == {"limit": "abc"}

    def test_unknown_prop_passthrough(self):
        schema = {"properties": {}}
        assert coerce_params({"x": "y"}, schema) == {"x": "y"}

    def test_non_string_passthrough(self):
        """Already-typed values must not be touched."""
        schema = {"properties": {"limit": {"type": "integer"}}}
        assert coerce_params({"limit": 42}, schema) == {"limit": 42}

    def test_input_schema_style(self):
        """Anthropic-style schemas with input_schema must be handled."""
        schema = {
            "name": "receiver",
            "input_schema": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "msg": {"type": "string"},
                },
            },
        }
        result = coerce_params({"count": "5", "msg": "hi"}, schema)
        assert result == {"count": 5, "msg": "hi"}

    def test_empty_schema(self):
        assert coerce_params({"x": "y"}, {}) == {"x": "y"}


class TestCoerceBool:
    def test_true_values(self):
        assert _coerce_bool("true") is True
        assert _coerce_bool("1") is True
        assert _coerce_bool("yes") is True

    def test_false_values(self):
        assert _coerce_bool("false") is False
        assert _coerce_bool("0") is False
        assert _coerce_bool("no") is False

    def test_garbage_returns_original(self):
        assert _coerce_bool("banana") == "banana"
        assert _coerce_bool("maybe") == "maybe"
        assert _coerce_bool("") == ""
