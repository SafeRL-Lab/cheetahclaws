"""
Example plugin tools for CheetahClaws.

This file demonstrates how to define tools that the AI can call.
Export your tools as a TOOL_DEFS list — do NOT call register_tool() directly.
"""
from cheetahclaws.tool_registry import ToolDef


def _example_search(params: dict, config: dict) -> str:
    """Example tool: search for something.

    Args:
        params: {"query": str, "limit": int}
        config: runtime config dict

    Returns:
        String result shown to the AI.
    """
    query = params["query"]
    limit = params.get("limit", 5)
    # Replace this with your actual logic
    return f"Found {limit} results for: {query}\n\n1. Example result 1\n2. Example result 2"


def _example_status(params: dict, config: dict) -> str:
    """Example read-only tool: return status information."""
    return "Example plugin is active and working."


# ── Export this list — the plugin loader reads it automatically ──────────
TOOL_DEFS = [
    ToolDef(
        name="ExampleSearch",
        schema={
            "name": "ExampleSearch",
            "description": (
                "Search using the example plugin. "
                "Returns matching results for the given query."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        func=_example_search,
        read_only=False,
        concurrent_safe=True,
    ),
    ToolDef(
        name="ExampleStatus",
        schema={
            "name": "ExampleStatus",
            "description": "Check the status of the example plugin.",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        func=_example_status,
        read_only=True,
        concurrent_safe=True,
    ),
]
