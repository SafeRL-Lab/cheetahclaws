"""System context: CLAUDE.md, git info, cwd injection.

Prompt assembly pipeline:

    build_system_prompt(config) -> str
        = pick_base_prompt(provider, model)      # default.md + matched overlay
        + _render_env_block(config)              # date / cwd / platform / git / CLAUDE.md
        + memory index (if any)
        + tmux fragment (if tmux available)      # prompts/fragments/tmux.md
        + plan mode fragment (if plan active)    # prompts/fragments/plan.md

Base + overlay design lives under ``prompts/`` — see ``prompts/README.md``.
Base/overlay files contain no placeholders and are loaded verbatim.
Dynamic per-run data (date, cwd, CLAUDE.md, plan file path) is rendered
separately and appended.

Callers outside this module should only touch ``build_system_prompt``.
The helper functions (``get_git_info``, ``get_claude_md``,
``get_platform_hints``) are exposed for tests and for REPL commands
that want to show individual context blocks (e.g. ``/doctor``).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from datetime import datetime

from memory import get_memory_context
from prompts import pick_base_prompt, load_fragment

_THINK_OUT_LOUD_PROMPT = """
# Think-Out-Loud Mode

Wrap your internal reasoning in `<""" + """thinking>...</""" + """thinking>` XML tags.
These are displayed to the user in italic but stripped from your context in future turns.

Use thinking for: planning, analyzing code, debugging, weighing options.
Keep it focused -- commit to an approach, avoid loops.
You can interleave thinking blocks with visible text freely.
Do NOT wrap simple responses in thinking.
"""

# ── Prompt injection detection ───────────────────────────────────────────
_THREAT_PATTERNS = [
    re.compile(r'ignore\s+(previous|all|above|prior)(\s+\w+)*\s+(instructions?|prompts?|rules?)', re.I),
    re.compile(r'system\s+prompt\s+(override|replace|change|modify|ignore)', re.I),
    re.compile(r'you\s+are\s+now\s+(a|an|no\s+longer)', re.I),
    re.compile(r'disregard\s+(all|any|your)\s+(previous|prior|above)', re.I),
    re.compile(r'new\s+instructions?\s*:', re.I),
    re.compile(r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)', re.I),
    re.compile(r'(cat|echo|print|export)\s+.*\$(ANTHROPIC|OPENAI|API|SECRET|TOKEN)', re.I),
    re.compile(r'base64\s+(encode|decode).*\b(key|token|secret|password)\b', re.I),
]


def _scan_for_threats(content: str, source: str) -> str | None:
    """Scan content for prompt injection patterns. Returns warning or None."""
    for pattern in _THREAT_PATTERNS:
        match = pattern.search(content)
        if match:
            return (
                f"[SECURITY WARNING] Potential prompt injection detected in {source}:\n"
                f"  Pattern: {match.group()!r}\n"
                f"  This content has been excluded from the system prompt."
            )
    return None


def get_git_info() -> str:
    """Return git branch/status summary if in a git repo."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL, text=True).strip()
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5"],
            stderr=subprocess.DEVNULL, text=True).strip()
        parts = [f"- Git branch: {branch}"]
        if status:
            lines = status.split('\n')[:10]
            parts.append("- Git status:\n" + "\n".join(f"  {l}" for l in lines))
        if log:
            parts.append("- Recent commits:\n" + "\n".join(f"  {l}" for l in log.split('\n')))
        return "\n".join(parts) + "\n"
    except Exception:
        return ""


def get_claude_md() -> str:
    """Load CLAUDE.md from cwd or parents, and ~/.claude/CLAUDE.md.

    Each file is scanned for prompt injection patterns before inclusion.
    """
    content_parts = []
    warnings = []

    # Global CLAUDE.md
    global_md = Path.home() / ".claude" / "CLAUDE.md"
    if global_md.exists():
        try:
            text = global_md.read_text()
            threat = _scan_for_threats(text, f"Global CLAUDE.md ({global_md})")
            if threat:
                warnings.append(threat)
            else:
                content_parts.append(f"[Global CLAUDE.md]\n{text}")
        except Exception:
            pass

    # Project CLAUDE.md (walk up from cwd)
    p = Path.cwd()
    for _ in range(10):
        candidate = p / "CLAUDE.md"
        if candidate.exists():
            try:
                text = candidate.read_text()
                threat = _scan_for_threats(text, f"Project CLAUDE.md ({candidate})")
                if threat:
                    warnings.append(threat)
                else:
                    content_parts.append(f"[Project CLAUDE.md: {candidate}]\n{text}")
            except Exception:
                pass
            break
        parent = p.parent
        if parent == p:
            break
        p = parent

    # Print warnings to stderr so user sees them
    if warnings:
        import sys
        for w in warnings:
            print(f"\033[33m{w}\033[0m", file=sys.stderr)

    if not content_parts:
        return ""
    return "\n# Memory / CLAUDE.md\n" + "\n\n".join(content_parts) + "\n"


def get_platform_hints() -> str:
    """Return shell hints tailored to the current OS."""
    import platform as _plat
    if _plat.system() == "Windows":
        return (
            "\n## Windows Shell Hints\n"
            "You are on Windows. Do NOT use Unix commands. Use these instead:\n"
            "- `type file.txt` instead of `cat file.txt`\n"
            "- `type file.txt | findstr /n /i \"pattern\"` instead of `grep`\n"
            "- `powershell -Command \"Get-Content file.txt -Tail 20\"` instead of `tail -n 20`\n"
            "- `powershell -Command \"Get-Content file.txt -Head 20\"` instead of `head -n 20`\n"
            "- `dir /s /b *.py` or `powershell -Command \"Get-ChildItem -Recurse -Filter *.py\"` instead of `find . -name '*.py'`\n"
            "- `del file.txt` instead of `rm file.txt`\n"
            "- `mkdir folder` works on both (no -p needed)\n"
            "- `copy` / `move` instead of `cp` / `mv`\n"
            "- Use `&&` to chain commands, not `;`\n"
            "- Paths use backslashes `\\` but forward slashes `/` also work in most cases\n"
            "- Python is available: `python -c \"...\"` works for complex text processing\n"
        )
    return ""


def _render_env_block(config: dict | None = None) -> str:
    """Render the per-run environment block (date / cwd / platform / git / CLAUDE.md).

    This used to be the ``# Environment`` section at the bottom of the
    monolithic SYSTEM_PROMPT_TEMPLATE.  It now renders fresh every call
    so the base prompt can remain pure static text.
    """
    import platform as _plat
    # Trailing \n on the Platform line is load-bearing: get_git_info()
    # returns content that starts with "- Git branch:" (no leading newline),
    # so without this \n it concatenates as "Platform: Linux- Git branch:".
    header = (
        "# Environment\n"
        f"- Current date: {datetime.now().strftime('%Y-%m-%d %A')}\n"
        f"- Working directory: {Path.cwd()}\n"
        f"- Platform: {_plat.system()}\n"
    )
    return header + get_platform_hints() + get_git_info() + get_claude_md()


def _render_plan_fragment(config: dict) -> str:
    """Load the plan-mode fragment and fill in {plan_file}."""
    import runtime
    plan_file = runtime.get_ctx(config).plan_file or ""
    template = load_fragment("plan")
    return template.format(plan_file=plan_file)


def _tmux_available() -> bool:
    try:
        from tmux_tools import tmux_available
        return tmux_available()
    except ImportError:
        return False


def build_system_prompt(config: dict | None = None) -> str:
    """Build the full system prompt for the current session.

    Structure (top → bottom):
        1. Provider-selected base prompt (``prompts/base/<provider>.md``)
        2. Per-run environment block (date, cwd, platform, git, CLAUDE.md)
        3. Memory index (if any memories exist)
        4. Tmux fragment (if tmux is installed)
        5. Plan-mode fragment (if ``permission_mode == "plan"``)
    """
    # Resolve provider lazily to avoid circular imports at module load.
    from providers import detect_provider

    cfg = config or {}
    model_id = cfg.get("model", "")
    # No model -> empty provider so pick_base_prompt falls through to
    # default.md.  The previous "anthropic" fallback silently gave Claude-
    # styled prompts (XML tags, minimal-scope guard) to whatever model
    # picked them up later, which is wrong for non-Claude families.
    provider = detect_provider(model_id) if model_id else ""

    parts: list[str] = [
        pick_base_prompt(provider, model_id),
        _render_env_block(cfg),
    ]

    memory_ctx = get_memory_context()
    if memory_ctx:
        parts.append(f"# Memory\nYour persistent memories:\n{memory_ctx}")

    if _tmux_available():
        parts.append(load_fragment("tmux"))

    if cfg.get("permission_mode") == "plan":
        parts.append(_render_plan_fragment(cfg))

    # Collapse any trailing whitespace on each part so the "\n\n"
    # separator produces a consistent two-newline gap regardless of how
    # each file/helper terminates.
    return "\n\n".join(p.rstrip() for p in parts if p)
