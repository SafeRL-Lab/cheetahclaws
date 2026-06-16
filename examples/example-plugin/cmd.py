"""
Example plugin commands for CheetahClaws.

This file demonstrates how to define slash commands that users type in the REPL.
Export your commands as a COMMAND_DEFS dict.
"""
from cheetahclaws.ui.render import info, ok, warn, err


def _cmd_example(args: str, state, config) -> bool:
    """Handle /example and its subcommands."""
    parts = args.split() if args.strip() else []
    sub = parts[0] if parts else ""
    rest = " ".join(parts[1:])

    if sub == "status":
        info("Example plugin is active.")
        info(f"  Messages in session: {len(state.messages)}")
        info(f"  Current model: {config.get('model', '?')}")
        return True

    elif sub == "greet":
        name = rest or "world"
        ok(f"Hello, {name}! This is the example plugin.")
        return True

    else:
        info("Example Plugin")
        info("  /example status   — show plugin status")
        info("  /example greet    — say hello")
        return True


# ── Export this dict — merged into COMMANDS at startup ───────────────────
COMMAND_DEFS = {
    "example": {
        "func": _cmd_example,
        "help": ("Example plugin commands", ["status", "greet"]),
        "aliases": ["ex"],
    },
}
