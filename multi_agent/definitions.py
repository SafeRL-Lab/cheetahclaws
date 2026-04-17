"""Agent type definitions and loading for cheetahclaws multi-agent system."""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """Definition of a sub-agent type."""

    name: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    model: Optional[str] = None
    description: str = ""


_BUILTIN_AGENTS: dict[str, AgentDefinition] = {
    "general-purpose": AgentDefinition(
        name="general-purpose",
        system_prompt="You are a helpful coding assistant.",
        description="General-purpose coding agent with all tools available.",
    ),
    "coder": AgentDefinition(
        name="coder",
        system_prompt=textwrap.dedent("""\
            You are an expert software engineer focused on writing clean,
            correct code. You have access to file and shell tools.
            Focus on implementation — write code, run tests, fix errors.
        """),
        tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        description="Focused coding agent with file and shell tools.",
    ),
    "reviewer": AgentDefinition(
        name="reviewer",
        system_prompt=textwrap.dedent("""\
            You are a senior code reviewer. Analyze code for bugs, style issues,
            security concerns, and suggest improvements.
            You can read files but should NOT modify them.
        """),
        tools=["Read", "Glob", "Grep", "Bash"],
        description="Code review agent — reads code and provides feedback.",
    ),
    "researcher": AgentDefinition(
        name="researcher",
        system_prompt=textwrap.dedent("""\
            You are a research assistant. Search the web, read documentation,
            and synthesize information to answer questions.
        """),
        tools=["WebSearch", "WebFetch", "Read", "Glob", "Grep"],
        description="Research agent with web search capabilities.",
    ),
    "tester": AgentDefinition(
        name="tester",
        system_prompt=textwrap.dedent("""\
            You are a testing specialist. Write and run tests to verify code
            correctness. Focus on edge cases and thorough coverage.
        """),
        tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        description="Testing agent focused on writing and running tests.",
    ),
}


def _parse_agent_md(path: Path) -> AgentDefinition:
    """Parse a .md agent definition file with optional YAML-like frontmatter."""
    text = path.read_text(encoding="utf-8")
    name = path.stem
    metadata: dict = {}
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()
            for line in frontmatter.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    if key == "tools":
                        metadata["tools"] = [
                            t.strip() for t in value.split(",") if t.strip()
                        ]
                    elif key == "model":
                        metadata["model"] = value or None
                    elif key == "description":
                        metadata["description"] = value

    return AgentDefinition(
        name=name,
        system_prompt=body,
        tools=metadata.get("tools", []),
        model=metadata.get("model"),
        description=metadata.get("description", ""),
    )


def load_agent_definitions(
    config_dir: str | Path | None = None,
) -> dict[str, AgentDefinition]:
    """Load built-in + custom agent definitions from config directory."""
    agents = dict(_BUILTIN_AGENTS)

    if config_dir is None:
        config_dir = Path(os.path.expanduser("~/.cheetahclaws"))
    else:
        config_dir = Path(config_dir)

    agents_dir = config_dir / "agents"
    if agents_dir.is_dir():
        for md_file in sorted(agents_dir.glob("*.md")):
            agent_def = _parse_agent_md(md_file)
            agents[agent_def.name] = agent_def
            log.debug("Loaded custom agent type: %s", agent_def.name)

    return agents


def get_agent_definition(
    name: str, config_dir: str | Path | None = None
) -> AgentDefinition:
    """Look up an agent definition by name."""
    agents = load_agent_definitions(config_dir)
    if name not in agents:
        available = ", ".join(sorted(agents.keys()))
        raise ValueError(
            f"Unknown agent type {name!r}. Available: {available}"
        )
    return agents[name]
