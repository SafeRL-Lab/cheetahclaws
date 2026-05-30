"""Self-registering GetFolderDescription tool."""
from __future__ import annotations

from tool_registry import ToolDef, register_tool
from folder_desc.tree import get_folder_description

_SCHEMA = {
    "name": "GetFolderDescription",
    "description": (
        "Return a recursive tree of code files in a folder with their [desc] one-line "
        "descriptions. If descriptions are missing, they are generated automatically "
        "(parallel LLM calls) before the tree is returned. Useful for understanding a "
        "codebase at a glance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "folder_path": {
                "type": "string",
                "description": "Absolute path to the folder to describe",
            },
        },
        "required": ["folder_path"],
    },
}


def _get_folder_description(params: dict, config: dict) -> str:
    folder_path = params.get("folder_path", "")
    if not folder_path:
        return "Error: missing required parameter 'folder_path'"
    return get_folder_description(folder_path, config)


register_tool(ToolDef(
    name="GetFolderDescription",
    schema=_SCHEMA,
    func=_get_folder_description,
    read_only=True,
    concurrent_safe=True,
))
