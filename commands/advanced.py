"""
commands/advanced.py — Advanced power commands for CheetahClaws.

Commands: /brainstorm, /worker, /ssj, /memory, /agents, /skills,
          /mcp, /plugin, /tasks
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Union

from ui.render import (
    clr, info, ok, warn, err,
    _start_tool_spinner, _stop_tool_spinner,
)
from tools import _is_in_tg_turn, _is_in_web_turn


# ── Brainstorm ─────────────────────────────────────────────────────────────

_TECH_PERSONAS = {
    "architect":   {"icon": "🏗️", "role": "Principal Software Architect",       "desc": "Focus on modularity, clear boundaries, patterns, and long-term maintainability."},
    "innovator":   {"icon": "💡", "role": "Pragmatic Product Innovator",          "desc": "Focus on bold, technically feasible ideas that add high user value and differentiation."},
    "security":    {"icon": "🛡️", "role": "Security & Risk Engineer",            "desc": "Focus on vulnerabilities, data integrity, secrets handling, and project robustness."},
    "refactor":    {"icon": "🔧", "role": "Senior Code Quality Lead",             "desc": "Focus on code smells, complexity reduction, DRY principles, and readability."},
    "performance": {"icon": "⚡", "role": "Performance & Optimization Specialist","desc": "Focus on I/O bottlenecks, resource efficiency, latency, and scalability."},
}


def _generate_personas(topic: str, curr_model: str, config: dict, count: int = 5) -> dict | None:
    from providers import stream, TextChunk
    import json

    example_entries = "\n".join(
        f'  "p{i+1}": {{"icon": "emoji", "role": "Expert Title", "desc": "One sentence describing their analytical angle."}}'
        for i in range(count)
    )
    user_msg = f"""Generate {count} expert personas for a multi-perspective brainstorming debate on: "{topic}"

Return ONLY a valid JSON object — no markdown fences, no extra text — like this:
{{
{example_entries}
}}

Choose experts whose domains are most relevant to analyzing "{topic}" from different angles."""

    internal_config = config.copy()
    internal_config["no_tools"] = True
    chunks = []
    try:
        for event in stream(curr_model, "You are a debate facilitator. Return only valid JSON.", [{"role": "user", "content": user_msg}], [], internal_config):
            if isinstance(event, TextChunk):
                chunks.append(event.text)
    except Exception:
        return None

    raw = "".join(chunks).strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    try:
        return json.loads(raw)
    except Exception:
        return None


def cmd_brainstorm(args: str, state, config) -> bool:
    """Run a multi-persona iterative brainstorming session on the project."""
    from providers import stream, TextChunk
    import time
    from tools import ask_input_interactive

    readme_path = Path("README.md")
    readme_content = readme_path.read_text("utf-8", errors="replace") if readme_path.exists() else ""
    claude_md = Path("CLAUDE.md")
    claude_content = claude_md.read_text("utf-8", errors="replace") if claude_md.exists() else ""
    project_files = "\n".join([f.name for f in Path(".").glob("*") if f.is_file() and not f.name.startswith(".")])

    user_topic = args.strip() or "general project improvement and architectural evolution"

    if _is_in_tg_turn(config) or _is_in_web_turn(config):
        agent_count = 5
    else:
        try:
            ans = ask_input_interactive(clr("  How many agents? (2-100, default 5) > ", "cyan"), config).strip()
            agent_count = int(ans) if ans else 5
            agent_count = max(2, min(agent_count, 100))
        except (ValueError, KeyboardInterrupt, EOFError):
            agent_count = 5

    snapshot = f"""PROJECT CONTEXT:
README:
{readme_content[:3000]}

CLAUDE.MD:
{claude_content[:1000]}

ROOT FILES:
{project_files}

USER FOCUS: {user_topic}
"""
    curr_model = config["model"]

    info(clr(f"Generating {agent_count} topic-appropriate expert personas...", "dim"))
    personas = _generate_personas(user_topic, curr_model, config, count=agent_count)
    if not personas:
        info(clr("(persona generation failed, using default tech personas)", "dim"))
        personas = dict(list(_TECH_PERSONAS.items())[:agent_count])

    def get_identity(letter):
        try:
            from faker import Faker
            fake = Faker()
            return f"{letter}", fake.name()
        except Exception:
            import random
            first = ["Alex", "Sam", "Taylor", "Jordan", "Casey", "Riley", "Drew", "Avery"]
            last = ["Garcia", "Martinez", "Lopez", "Hernandez", "Gonzalez", "Sanchez", "Ramirez", "Torres"]
            return f"{letter}", f"{random.choice(first)} {random.choice(last)}"

    outputs_dir = Path("brainstorm_outputs")
    outputs_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = outputs_dir / f"brainstorm_{ts}.md"

    brainstorm_history = []
    ok(f"Starting {agent_count}-Agent Brainstorming Session on: {clr(user_topic, 'bold')}")
    info(clr("Generating diverse perspectives...", "dim"))

    def call_persona(persona_name, p_data, history):
        letter, name = get_identity(persona_name[0].upper())
        system_prompt = f"""You are {name}, the {p_data['role']}. Identity: Agent {letter}.
{p_data['desc']}

TOPIC UNDER DISCUSSION: {user_topic}

PROJECT CONTEXT (if relevant to the topic):
{snapshot}

INSTRUCTIONS:
1. Provide 3-5 concrete, actionable insights or ideas from your expert perspective on the topic.
2. If there are prior ideas from other agents, briefly acknowledge them and build upon or challenge them.
3. Be specific, well-reasoned, and professional. Stay in character as your role.
4. Prefix each of your points with: [Agent {letter} — {name}]
5. Output your response in clean Markdown.
"""
        user_msg = f"TOPIC: {user_topic}\n\nPRIOR IDEAS FROM DEBATE:\n{history or 'No previous ideas yet. You are the first to speak.'}"
        full_response = []
        internal_config = config.copy()
        internal_config["no_tools"] = True
        try:
            for event in stream(curr_model, system_prompt, [{"role": "user", "content": user_msg}], [], internal_config):
                if isinstance(event, TextChunk):
                    full_response.append(event.text)
        except Exception as e:
            return f"Error from Agent {letter}: {e}"
        return "".join(full_response).strip()

    full_log = [f"# Brainstorming Session: {user_topic}", f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}", f"**Model:** {curr_model}", "---"]

    for p_name, p_data in personas.items():
        icon = p_data.get("icon", "🤖")
        info(f"{icon} {clr(p_data['role'], 'yellow')} is thinking...")
        _start_tool_spinner()
        hist_text = "\n\n".join(brainstorm_history) if brainstorm_history else ""
        content = call_persona(p_name, p_data, hist_text)
        _stop_tool_spinner()
        if content:
            brainstorm_history.append(content)
            full_log.append(f"## {icon} {p_data['role']}\n{content}")
            print(clr("  └─ Perspective captured.", "dim"))
        else:
            err(f"  └─ Failed to capture {p_name} perspective.")

    final_output = "\n\n".join(full_log)
    out_file.write_text(final_output, encoding="utf-8")
    ok(f"Brainstorming complete! Results saved to {clr(str(out_file), 'bold')}")

    info(clr("Injecting debate results into current session for final analysis...", "dim"))
    synthesis_prompt = f"""I have just completed a multi-agent brainstorming session regarding: '{user_topic}'.
The full debate results have been saved to the file: {out_file}

Please read that file, then analyze the diverse perspectives. Identify the strongest ideas, potential conflicts, and provide a synthesized 'Master Plan' with concrete phases. Be concise and actionable."""

    return ("__brainstorm__", synthesis_prompt, str(out_file))


def _save_synthesis(state, out_file: str) -> None:
    """Append the last assistant response as the synthesis section of the brainstorm file."""
    for msg in reversed(state.messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            return
        text = text.strip()
        if not text:
            return
        try:
            with Path(out_file).open("a", encoding="utf-8") as f:
                f.write("\n\n---\n\n## 🧠 Synthesis — Master Plan\n\n")
                f.write(text)
                f.write("\n")
            ok(f"Synthesis appended to {clr(out_file, 'bold')}")
        except Exception as e:
            err(f"Failed to save synthesis: {e}")
        return


# ── Worker ─────────────────────────────────────────────────────────────────

def cmd_worker(args: str, state, config) -> bool:
    """Auto-implement pending tasks from a todo_list.txt file."""
    raw = args.strip()
    todo_path_override = None
    task_nums_str      = None
    max_workers        = None

    tokens = raw.split() if raw else []
    remaining = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--path" and i + 1 < len(tokens):
            todo_path_override = tokens[i + 1]; i += 2
        elif tok.startswith("--path="):
            todo_path_override = tok[len("--path="):]; i += 1
        elif tok == "--tasks" and i + 1 < len(tokens):
            task_nums_str = tokens[i + 1]; i += 2
        elif tok.startswith("--tasks="):
            task_nums_str = tok[len("--tasks="):]; i += 1
        elif tok == "--workers" and i + 1 < len(tokens):
            max_workers = tokens[i + 1]; i += 2
        elif tok.startswith("--workers="):
            max_workers = tok[len("--workers="):]; i += 1
        else:
            remaining.append(tok); i += 1

    if remaining:
        leftover = " ".join(remaining)
        if todo_path_override is None and (
            "/" in leftover or "\\" in leftover
            or leftover.endswith(".txt") or leftover.endswith(".md")
        ):
            todo_path_override = leftover
        elif task_nums_str is None:
            task_nums_str = leftover

    todo_path = Path(todo_path_override) if todo_path_override else Path("brainstorm_outputs") / "todo_list.txt"

    if not todo_path.exists():
        err(f"No todo file found at {todo_path}.")
        if not todo_path_override:
            info("Run /brainstorm first, or specify a path with --path /your/todo.txt")
        return True

    content = todo_path.read_text(encoding="utf-8", errors="replace")
    lines   = content.splitlines()
    pending = [(i, ln) for i, ln in enumerate(lines) if ln.strip().startswith("- [ ]")]

    if not pending:
        any_tasks = any(ln.strip().startswith("- [") for ln in lines)
        if any_tasks:
            ok(f"All tasks completed! No pending items in {todo_path}.")
        else:
            err(f"No task lines found in {todo_path}.")
            info("Worker expects lines like:  - [ ] task description")
        return True

    if task_nums_str:
        try:
            nums = [int(x.strip()) for x in task_nums_str.split(",") if x.strip()]
            selected = []
            for n in nums:
                if 1 <= n <= len(pending):
                    selected.append(pending[n - 1])
                else:
                    err(f"Task #{n} out of range (1-{len(pending)}).")
                    return True
            pending = selected
        except ValueError:
            err(f"Invalid task number(s): '{task_nums_str}'. Use e.g. 1,4,6")
            return True

    worker_count = len(pending)
    if max_workers is not None:
        try:
            worker_count = max(1, int(max_workers))
        except ValueError:
            err(f"Invalid --workers value: '{max_workers}'. Must be a positive integer.")
            return True
    if worker_count < len(pending):
        info(f"Workers: {worker_count} — running first {worker_count} of {len(pending)} pending task(s) this session.")
        pending = pending[:worker_count]

    ok(f"Worker starting — {len(pending)} task(s) | file: {todo_path}")
    info("Pending tasks:")
    for n, (_, ln) in enumerate(pending, 1):
        print(f"  {n}. {ln.strip()}")

    worker_prompts = []
    for line_idx, task_line in pending:
        task_text = task_line.strip().replace("- [ ] ", "", 1)
        prompt = (
            f"You are the Worker. Your job is to implement this task:\n\n"
            f"  {task_text}\n\n"
            f"Instructions:\n"
            f"1. Read the relevant files, understand the codebase.\n"
            f"2. Implement the task — write code, edit files, run tests.\n"
            f"3. When DONE, use the Edit tool to mark this exact line in {todo_path}:\n"
            f'   Change "- [ ] {task_text}" to "- [x] {task_text}"\n'
            f"4. If you CANNOT complete it, leave it as - [ ] and explain why.\n"
            f"5. Be concise. Act, don't explain."
        )
        worker_prompts.append((line_idx, task_text, prompt))

    return ("__worker__", worker_prompts)


# ── SSJ Trading Sub-menu ───────────────────────────────────────────────────

def _ssj_trading_submenu(config, state):
    """Interactive trading sub-menu for SSJ mode."""
    from tools import ask_input_interactive

    _TRADING_SUBMENU = (
        clr("\n╭─ 📈 Trading Agent ", "dim") + clr("━━━━━━━━━━━━━━━━━━━━━━━━━", "dim")
        + "\n│"
        + "\n│  " + clr("a.", "bold") + " 🔍  Quick Analyze — Full multi-agent analysis (Bull/Bear + Risk + PM)"
        + "\n│  " + clr("b.", "bold") + " 📊  Backtest     — Test a strategy on historical data"
        + "\n│  " + clr("c.", "bold") + " 💰  Price Check  — Current price & key metrics"
        + "\n│  " + clr("d.", "bold") + " 📉  Indicators   — Technical indicators report"
        + "\n│  " + clr("e.", "bold") + " 🤖  Trading Bot  — Launch autonomous trading agent"
        + "\n│  " + clr("f.", "bold") + " 📜  History      — Past trading decisions"
        + "\n│  " + clr("g.", "bold") + " 🧠  Memory       — Trading memory status"
        + "\n│  " + clr("0.", "bold") + " ↩   Back to SSJ"
        + "\n│"
        + "\n" + clr("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "dim")
    )

    print(_TRADING_SUBMENU)
    try:
        choice = ask_input_interactive(clr("\n  📈 Trading » ", "cyan", "bold"), config, _TRADING_SUBMENU).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return True

    if choice in ("0", "q", ""):
        return True

    elif choice == "a":
        symbol = ask_input_interactive(
            clr("  Symbol (e.g. AAPL, BTC, NVDA): ", "cyan"), config
        ).strip().upper()
        if not symbol:
            err("Symbol required.")
            return True
        return ("__ssj_passthrough__", f"/trading analyze {symbol}")

    elif choice == "b":
        symbol = ask_input_interactive(
            clr("  Symbol (e.g. AAPL, BTC): ", "cyan"), config
        ).strip().upper()
        if not symbol:
            err("Symbol required.")
            return True

        _STRAT_MENU = (
            clr("\n  Strategies:", "cyan")
            + "\n    1. dual_ma          — SMA(20/50) crossover (trend following)"
            + "\n    2. rsi_mean_reversion — RSI 30/70 (mean reversion)"
            + "\n    3. bollinger_breakout — Bollinger Band breakout"
            + "\n    4. macd_crossover   — MACD histogram (momentum)"
            + "\n    5. all              — Run all & compare"
        )
        print(_STRAT_MENU)
        strat_choice = ask_input_interactive(
            clr("  Strategy [1-5, default=5]: ", "cyan"), config, _STRAT_MENU
        ).strip()

        strat_map = {"1": "dual_ma", "2": "rsi_mean_reversion",
                     "3": "bollinger_breakout", "4": "macd_crossover"}

        if strat_choice in ("5", ""):
            # Run all strategies and compare
            return ("__ssj_query__",
                    f"Run backtests on {symbol} using all 4 built-in strategies "
                    f"(dual_ma, rsi_mean_reversion, bollinger_breakout, macd_crossover). "
                    f"Use the RunBacktest tool for each. Present results in a comparison table "
                    f"and recommend the best strategy based on Sharpe ratio.")
        elif strat_choice in strat_map:
            strategy = strat_map[strat_choice]
            return ("__ssj_passthrough__", f"/trading backtest {symbol} {strategy}")
        else:
            err(f"Invalid choice: {strat_choice}")
            return True

    elif choice == "c":
        symbol = ask_input_interactive(
            clr("  Symbol (e.g. AAPL, BTC, ETH): ", "cyan"), config
        ).strip().upper()
        if symbol:
            return ("__ssj_passthrough__", f"/trading price {symbol}")
        err("Symbol required.")
        return True

    elif choice == "d":
        symbol = ask_input_interactive(
            clr("  Symbol (e.g. AAPL, NVDA): ", "cyan"), config
        ).strip().upper()
        if symbol:
            return ("__ssj_passthrough__", f"/trading indicators {symbol}")
        err("Symbol required.")
        return True

    elif choice == "e":
        watchlist = ask_input_interactive(
            clr("  Watchlist (comma-separated, default: AAPL,MSFT,GOOGL,NVDA,BTC,ETH): ", "cyan"), config
        ).strip()
        if not watchlist:
            watchlist = "AAPL,MSFT,GOOGL,NVDA,BTC,ETH"
        return ("__ssj_query__",
                f"You are the CheetahClaws Trading Agent. Analyze each symbol in this watchlist: {watchlist}. "
                f"For each symbol:\n"
                f"1. Use GetPrice and GetTechnicalIndicators to gather data\n"
                f"2. Run the full multi-agent analysis pipeline:\n"
                f"   - Bull Researcher: build bullish case with data\n"
                f"   - Bear Researcher: build bearish case with data\n"
                f"   - Research Judge: decisive BUY/SELL/HOLD recommendation\n"
                f"   - Risk Panel: aggressive/conservative/neutral perspectives\n"
                f"   - Portfolio Manager: final RATING (BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL)\n"
                f"3. Present a summary table at the end with all symbols and ratings.\n"
                f"Be specific — cite actual indicator values and fundamentals.")

    elif choice == "f":
        return ("__ssj_passthrough__", "/trading history")

    elif choice == "g":
        return ("__ssj_passthrough__", "/trading status")

    else:
        err(f"Invalid choice: {choice}")
    return True


# ── SSJ ────────────────────────────────────────────────────────────────────

def cmd_ssj(args: str, state, config) -> bool:
    """SSJ Developer Mode — Interactive power menu for project workflows."""
    try:
        import modular
        _all_cmds = modular.load_all_commands()
        _VIDEO_AVAILABLE    = "video"   in _all_cmds
        _VOICE_MODULAR      = "voice"   in _all_cmds
        _TRADING_AVAILABLE  = "trading" in _all_cmds
    except Exception:
        _VIDEO_AVAILABLE   = False
        _VOICE_MODULAR     = False
        _TRADING_AVAILABLE = False

    from tools import ask_input_interactive

    _SSJ_MENU = (
        clr("\n╭─ SSJ Developer Mode ", "dim") + clr("⚡", "yellow") + clr(" ─────────────────────────", "dim")
        + "\n│"
        + "\n│  " + clr(" 1.", "bold") + " 💡  Brainstorm — Multi-persona AI debate"
        + "\n│  " + clr(" 2.", "bold") + " 📋  Show TODO — View todo_list.txt"
        + "\n│  " + clr(" 3.", "bold") + " 👷  Worker — Auto-implement pending tasks"
        + "\n│  " + clr(" 4.", "bold") + " 🧠  Debate — Expert debate on a file"
        + "\n│  " + clr(" 5.", "bold") + " ✨  Propose — AI improvement for a file"
        + "\n│  " + clr(" 6.", "bold") + " 🔎  Review — Quick file analysis"
        + "\n│  " + clr(" 7.", "bold") + " 📘  Readme — Auto-generate README.md"
        + "\n│  " + clr(" 8.", "bold") + " 💬  Commit — AI-suggested commit message"
        + "\n│  " + clr(" 9.", "bold") + " 🧪  Scan — Analyze git diff"
        + "\n│  " + clr("10.", "bold") + " 📝  Promote — Idea to tasks"
        + ("\n│  " + clr("11.", "bold") + " 🎬  Video — AI video content factory" if _VIDEO_AVAILABLE else "")
        + ("\n│  " + clr("12.", "bold") + " 🎙  TTS   — AI voice generation (any style)" if _VOICE_MODULAR else "")
        + "\n│  " + clr("13.", "bold") + " 📡  Monitor — AI subscriptions & alerts"
        + ("\n│  " + clr("14.", "bold") + " 📈  Trading — Market analysis, backtest & trading agent" if _TRADING_AVAILABLE else "")
        + "\n│  " + clr("15.", "bold") + " 🤖  Agent  — Autonomous task agents (research / bug-fix / code / write)"
        + "\n│  " + clr("16.", "bold") + " 🔍  Research — 20-source topic search (arXiv · HN · GitHub · Zhihu · B站 · Weibo · 小红书 · …)"
        + "\n│  " + clr("17.", "bold") + " 📊  Trend Track — Auto-rerun /research weekly on a topic (/monitor hookup)"
        + "\n│  " + clr("18.", "bold") + " 📁  Reports   — Browse saved research briefs"
        + "\n│  " + clr(" 0.", "bold") + " 🚪  Exit SSJ Mode  (or type q)"
        + "\n│"
        + "\n" + clr("╰──────────────────────────────────────────────", "dim")
    )

    def _pick_file(prompt_text="  Select file #: ", exts=None):
        files = sorted([
            f for f in Path(".").iterdir()
            if f.is_file() and not f.name.startswith(".")
            and (exts is None or f.suffix in exts)
        ])
        if not files:
            err("No matching files found in current directory.")
            return None
        menu_text = clr(f"\n  📂 Files in {Path.cwd().name}/", "cyan")
        for i, f in enumerate(files, 1):
            menu_text += ("\n" + f"  {i:3d}. {f.name}")
        sel = ask_input_interactive(clr(prompt_text, "cyan"), config, menu_text).strip()
        if sel.isdigit() and 1 <= int(sel) <= len(files):
            return str(files[int(sel) - 1])
        elif sel:
            return sel
        err("Invalid selection.")
        return None

    print(_SSJ_MENU)

    while True:
        try:
            choice = ask_input_interactive(clr("\n  ⚡ SSJ » ", "yellow", "bold"), config, _SSJ_MENU).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice.startswith("/"):
            return ("__ssj_passthrough__", choice)

        if choice == "0" or choice.lower() in ("exit", "q"):
            ok("Exiting SSJ Mode.")
            break

        elif choice == "1":
            topic = ask_input_interactive(clr("  Topic (Enter for general): ", "cyan"), config).strip()
            return ("__ssj_cmd__", "brainstorm", topic)

        elif choice == "2":
            todo_path = Path("brainstorm_outputs") / "todo_list.txt"
            if todo_path.exists():
                content = todo_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                task_lines = [(i, l) for i, l in enumerate(lines) if l.strip().startswith("- [")]
                pending_lines = [(i, l) for i, l in task_lines if l.strip().startswith("- [ ]")]
                done_lines = [(i, l) for i, l in task_lines if l.strip().startswith("- [x]")]
                pending = len(pending_lines)
                done = len(done_lines)
                print(clr(f"\n  📋 TODO List ({done} done / {pending} pending):", "cyan"))
                print(clr("  " + "─" * 46, "dim"))
                for _, ln in done_lines:
                    label = ln.strip()[5:].strip()
                    print(clr(f"       ✓ {label}", "green"))
                for num, (_, ln) in enumerate(pending_lines, 1):
                    label = ln.strip()[5:].strip()
                    print(f"  {num:3d}. ○ {label}")
                print(clr("  " + "─" * 46, "dim"))
                print(clr("  Tip: use Worker (3) with pending task #s e.g. 1,4,6", "dim"))
            else:
                err("No todo_list.txt found. Run Brainstorm (1) first.")
            print(_SSJ_MENU)
            continue

        elif choice == "3":
            _default_todo = Path("brainstorm_outputs") / "todo_list.txt"
            pending_count = 0
            if _default_todo.exists():
                _lines = _default_todo.read_text(encoding="utf-8", errors="replace").splitlines()
                pending_count = sum(1 for l in _lines if l.strip().startswith("- [ ]"))
                if pending_count:
                    print(clr(f"\n  👷 Worker — {pending_count} pending task(s) in {_default_todo}", "cyan"))
                else:
                    print(clr(f"\n  ✓ All tasks completed in {_default_todo}", "green"))
            task_sel = ask_input_interactive(
                clr("  Task #s to run (e.g. 1,3,5), or Enter for all: ", "cyan"), config
            ).strip()
            return ("__ssj_cmd__", "worker", task_sel)

        elif choice == "4":
            fpath = _pick_file("  Select file for debate: ")
            if fpath:
                return ("__ssj_query__", f"Act as a panel of 3 expert engineers. Each gives 2-3 critical insights on this file: {fpath}. Be specific and constructive.")

        elif choice == "5":
            fpath = _pick_file("  Select file to improve: ")
            if fpath:
                return ("__ssj_query__", f"Analyze {fpath} and propose 3 high-impact improvements with code examples. Focus on correctness, performance, or maintainability.")

        elif choice == "6":
            fpath = _pick_file("  Select file to review: ")
            if fpath:
                return ("__ssj_query__", f"Give a quick code review of {fpath}: identify bugs, code smells, or missing edge cases. Be concise.")

        elif choice == "7":
            return ("__ssj_query__", "Generate a comprehensive README.md for this project. Include: project description, features, installation, usage examples, and contributing guidelines. Use the project files and CLAUDE.md for context.")

        elif choice == "8":
            return ("__ssj_query__", "Review the git diff (git diff HEAD) and suggest a concise, descriptive commit message following conventional commits format. Also list files changed.")

        elif choice == "9":
            return ("__ssj_query__", "Run git diff HEAD and analyze the changes. Summarize what was changed, why it might have been changed, and flag any potential issues or regressions.")

        elif choice == "10":
            idea = ask_input_interactive(clr("  Describe your idea or feature: ", "cyan"), config).strip()
            if idea:
                return ("__ssj_promote_worker__", idea)

        elif choice == "11" and _VIDEO_AVAILABLE:
            return ("__ssj_passthrough__", "/video")

        elif choice == "12" and _VOICE_MODULAR:
            return ("__ssj_passthrough__", "/tts")

        elif choice == "13":
            return ("__ssj_passthrough__", "/monitor")

        elif choice == "14" and _TRADING_AVAILABLE:
            return _ssj_trading_submenu(config, state)

        elif choice == "15":
            return ("__ssj_passthrough__", "/agent")

        elif choice == "16":
            # 傻瓜式研究向导:问 topic → 可选 range + citations → 直接跑 /research
            topic = ask_input_interactive(
                clr("  Topic to research: ", "cyan"), config
            ).strip()
            if not topic:
                err("No topic given.")
                print(_SSJ_MENU)
                continue
            range_menu = clr(
                "\n  Time range:"
                "\n    1. Last 7 days"
                "\n    2. Last 30 days  (default)"
                "\n    3. Last 6 months"
                "\n    4. Last 1 year"
                "\n    5. All time",
                "cyan",
            )
            range_choice = ask_input_interactive(
                clr("  Range [1-5, Enter for 2]: ", "cyan"), config, range_menu
            ).strip() or "2"
            range_map = {"1": "7d", "2": "30d", "3": "6m", "4": "1y", "5": "all"}
            r = range_map.get(range_choice, "30d")
            cite_choice = ask_input_interactive(
                clr("  Include notable-citer analysis? (y/N): ", "cyan"), config
            ).strip().lower()
            cmd_line = f"/research --range {r}"
            if cite_choice.startswith("y"):
                cmd_line += " --citations"
            cmd_line += f" {topic}"
            return ("__ssj_passthrough__", cmd_line)

        elif choice == "17":
            topic = ask_input_interactive(
                clr("  Topic to track weekly: ", "cyan"), config
            ).strip()
            if not topic:
                err("No topic given.")
                print(_SSJ_MENU)
                continue
            range_menu = clr(
                "\n  Tracking window (what each weekly run pulls):"
                "\n    1. Last 7 days    (default — matches the weekly schedule)"
                "\n    2. Last 30 days"
                "\n    3. Last 3 months",
                "cyan",
            )
            rc = ask_input_interactive(
                clr("  Window [1-3, Enter for 1]: ", "cyan"), config, range_menu
            ).strip() or "1"
            rmap = {"1": "7d", "2": "30d", "3": "90d"}
            rng = rmap.get(rc, "7d")
            freq_menu = clr(
                "\n  Frequency:"
                "\n    1. Daily"
                "\n    2. Weekly  (default — recommended for trend tracking)"
                "\n    3. Every 12 hours",
                "cyan",
            )
            fc = ask_input_interactive(
                clr("  Frequency [1-3, Enter for 2]: ", "cyan"), config, freq_menu
            ).strip() or "2"
            freq_map = {"1": "daily", "2": "weekly", "3": "12h"}
            freq = freq_map.get(fc, "weekly")
            topic_id = f"research:{rng}:{topic}"
            cmd_line = f"/subscribe {topic_id} {freq}"
            ok(f"  Subscribing to {topic_id}  ({freq})")
            return ("__ssj_passthrough__", cmd_line)

        elif choice == "18":
            return ("__ssj_passthrough__", "/reports")

        else:
            err(f"Invalid choice: {choice}")

        print(_SSJ_MENU)

    return True


# ── Memory ─────────────────────────────────────────────────────────────────

def cmd_memory(args: str, _state, config) -> bool:
    from memory import search_memory, load_index
    from memory.scan import scan_all_memories, format_memory_manifest, memory_freshness_text

    stripped = args.strip()

    if stripped == "consolidate":
        from memory import consolidate_session
        msgs = _state.get("messages", []) if hasattr(_state, 'get') else getattr(_state, 'messages', [])
        info("  Analyzing session for long-term memories…")
        saved = consolidate_session(msgs, config)
        if saved:
            info(f"  ✓ Consolidated {len(saved)} memory/memories: {', '.join(saved)}")
        else:
            info("  Nothing new worth saving (session too short, or nothing extractable).")
        return True

    if stripped:
        results = search_memory(stripped)
        if not results:
            info(f"No memories matching '{stripped}'")
            return True
        info(f"  {len(results)} result(s) for '{stripped}':")
        for m in results:
            conf_tag = f" conf:{m.confidence:.0%}" if m.confidence < 1.0 else ""
            src_tag = f" src:{m.source}" if m.source and m.source != "user" else ""
            info(f"  [{m.type:9s}|{m.scope:7s}] {m.name}{conf_tag}{src_tag}: {m.description}")
            info(f"    {m.content[:120]}{'...' if len(m.content) > 120 else ''}")
        return True

    headers = scan_all_memories()
    if not headers:
        info("No memories stored. The model saves memories via MemorySave.")
        return True
    info(f"  {len(headers)} memory/memories (newest first):")
    for h in headers:
        fresh_warn = "  ⚠ stale" if memory_freshness_text(h.mtime_s) else ""
        tag = f"[{h.type or '?':9s}|{h.scope:7s}]"
        info(f"  {tag} {h.filename}{fresh_warn}")
        if h.description:
            info(f"    {h.description}")
    return True


# ── Agents ─────────────────────────────────────────────────────────────────

def cmd_agents(_args: str, _state, config) -> bool:
    try:
        from multi_agent.tools import get_agent_manager
        mgr = get_agent_manager()
        tasks = mgr.list_tasks()
        if not tasks:
            info("No sub-agent tasks.")
            return True
        info(f"  {len(tasks)} sub-agent task(s):")
        for t in tasks:
            preview = t.prompt[:50] + ("..." if len(t.prompt) > 50 else "")
            wt_info = f"  branch:{t.worktree_branch}" if t.worktree_branch else ""
            info(f"  {t.id} [{t.status:9s}] name={t.name}{wt_info}  {preview}")
    except Exception:
        info("Sub-agent system not initialized.")
    return True


def _print_background_notifications():
    """Print notifications for newly completed background agent tasks."""
    try:
        from multi_agent.tools import get_agent_manager
        mgr = get_agent_manager()
    except Exception:
        return

    if not hasattr(_print_background_notifications, "_seen"):
        _print_background_notifications._seen = set()

    for task in mgr.list_tasks():
        if task.id in _print_background_notifications._seen:
            continue
        if task.status in ("completed", "failed", "cancelled"):
            _print_background_notifications._seen.add(task.id)
            icon = "✓" if task.status == "completed" else "✗"
            color = "green" if task.status == "completed" else "red"
            branch_info = f" [branch: {task.worktree_branch}]" if task.worktree_branch else ""
            print(clr(
                f"\n  {icon} Background agent '{task.name}' {task.status}{branch_info}",
                color, "bold"
            ))
            if task.result:
                preview = task.result[:200] + ("..." if len(task.result) > 200 else "")
                print(clr(f"    {preview}", "dim"))
            print()


# ── Skills ─────────────────────────────────────────────────────────────────

def cmd_skills(_args: str, _state, config) -> bool:
    from skill import load_skills
    skills = load_skills()
    if not skills:
        info("No skills found.")
        return True
    info(f"Available skills ({len(skills)}):")
    for s in skills:
        triggers = ", ".join(s.triggers)
        source_label = f"[{s.source}]" if s.source != "builtin" else ""
        hint = f"  args: {s.argument_hint}" if s.argument_hint else ""
        print(f"  {clr(s.name, 'cyan'):24s} {s.description}  {clr(triggers, 'dim')}{hint} {clr(source_label, 'yellow')}")
        if s.when_to_use:
            print(f"    {clr(s.when_to_use[:80], 'dim')}")
    return True


# ── MCP ────────────────────────────────────────────────────────────────────

def cmd_mcp(args: str, _state, config) -> bool:
    """Show MCP server status, or manage servers."""
    from cc_mcp.client import get_mcp_manager
    from cc_mcp.config import (load_mcp_configs, add_server_to_user_config,
                                remove_server_from_user_config, list_config_files)
    from cc_mcp.tools import initialize_mcp, reload_mcp, refresh_server

    parts = args.split() if args.strip() else []
    subcmd = parts[0].lower() if parts else ""

    if subcmd == "reload":
        target = parts[1] if len(parts) > 1 else ""
        if target:
            mcp_err = refresh_server(target)
            if mcp_err:
                err(f"Failed to reload '{target}': {mcp_err}")
            else:
                ok(f"Reloaded MCP server: {target}")
        else:
            errors = reload_mcp()
            for name, e in errors.items():
                if e:
                    print(f"  {clr('✗', 'red')} {name}: {e}")
                else:
                    print(f"  {clr('✓', 'green')} {name}: connected")
        return True

    if subcmd == "add":
        if len(parts) < 3:
            err("Usage: /mcp add <name> <command> [arg1 arg2 ...]")
            return True
        name = parts[1]
        command = parts[2]
        cmd_args = parts[3:]
        raw = {"type": "stdio", "command": command}
        if cmd_args:
            raw["args"] = cmd_args
        add_server_to_user_config(name, raw)
        ok(f"Added MCP server '{name}' → restart or /mcp reload to connect")
        return True

    if subcmd == "remove":
        if len(parts) < 2:
            err("Usage: /mcp remove <name>")
            return True
        name = parts[1]
        removed = remove_server_from_user_config(name)
        if removed:
            ok(f"Removed MCP server '{name}' from user config")
        else:
            err(f"Server '{name}' not found in user config")
        return True

    mgr = get_mcp_manager()
    servers = mgr.list_servers()
    config_files = list_config_files()
    if config_files:
        info(f"Config files: {', '.join(str(f) for f in config_files)}")

    if not servers:
        configs = load_mcp_configs()
        if not configs:
            info("No MCP servers configured.")
            info("Add servers in ~/.cheetahclaws/mcp.json or .mcp.json")
            info("Example: /mcp add my-git uvx mcp-server-git")
        else:
            info("MCP servers configured but not yet connected. Run /mcp reload")
        return True

    info(f"MCP servers ({len(servers)}):")
    total_tools = 0
    for client in servers:
        status_color = {
            "connected":    "green",
            "connecting":   "yellow",
            "disconnected": "dim",
            "error":        "red",
        }.get(client.state.value, "dim")
        print(f"  {clr(client.status_line(), status_color)}")
        for tool in client._tools:
            print(f"      {clr(tool.qualified_name, 'cyan')}  {tool.description}")
            total_tools += 1

    if total_tools:
        info(f"Total: {total_tools} MCP tool(s) available to Claude")
    return True


# ── Plugin ─────────────────────────────────────────────────────────────────

def cmd_plugin(args: str, _state, config) -> bool:
    """Manage plugins."""
    from plugin import (
        install_plugin, uninstall_plugin, enable_plugin, disable_plugin,
        disable_all_plugins, update_plugin, list_plugins, get_plugin,
        PluginScope, recommend_plugins, format_recommendations,
    )

    parts = args.split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    rest   = parts[1].strip() if len(parts) > 1 else ""

    if not subcmd:
        plugins = list_plugins()
        if not plugins:
            info("No plugins installed.")
            info("Install: /plugin install name@git_url")
            info("Recommend: /plugin recommend")
            return True
        info(f"Installed plugins ({len(plugins)}):")
        for p in plugins:
            state_color = "green" if p.enabled else "dim"
            state_str   = "enabled" if p.enabled else "disabled"
            desc = p.manifest.description if p.manifest else ""
            print(f"  {clr(p.name, state_color)} [{p.scope.value}] {state_str}  {desc[:60]}")
        return True

    if subcmd == "install":
        if not rest:
            err("Usage: /plugin install name@git_url")
            return True
        scope_str = "user"
        if " --project" in rest:
            scope_str = "project"
            rest = rest.replace("--project", "").strip()
        scope = PluginScope(scope_str)
        success, msg = install_plugin(rest, scope=scope)
        (ok if success else err)(msg)
        return True

    if subcmd == "uninstall":
        if not rest:
            err("Usage: /plugin uninstall name")
            return True
        success, msg = uninstall_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "enable":
        if not rest:
            err("Usage: /plugin enable name")
            return True
        success, msg = enable_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "disable":
        if not rest:
            err("Usage: /plugin disable name")
            return True
        success, msg = disable_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "disable-all":
        success, msg = disable_all_plugins()
        (ok if success else err)(msg)
        return True

    if subcmd == "update":
        if not rest:
            err("Usage: /plugin update name")
            return True
        success, msg = update_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "recommend":
        context = rest
        if not context:
            from plugin.recommend import recommend_from_files
            files = list(Path.cwd().glob("**/*"))[:200]
            recs = recommend_from_files(files)
        else:
            recs = recommend_plugins(context)
        print(format_recommendations(recs))
        return True

    if subcmd == "info":
        if not rest:
            err("Usage: /plugin info name")
            return True
        entry = get_plugin(rest)
        if entry is None:
            err(f"Plugin '{rest}' not found.")
            return True
        m = entry.manifest
        print(f"Name:    {entry.name}")
        print(f"Scope:   {entry.scope.value}")
        print(f"Source:  {entry.source}")
        print(f"Dir:     {entry.install_dir}")
        print(f"Enabled: {entry.enabled}")
        if m:
            print(f"Version: {m.version}")
            print(f"Author:  {m.author}")
            print(f"Desc:    {m.description}")
            if m.tags:
                print(f"Tags:    {', '.join(m.tags)}")
            if m.tools:
                print(f"Tools:   {', '.join(m.tools)}")
            if m.skills:
                print(f"Skills:  {', '.join(m.skills)}")
        return True

    err(f"Unknown plugin subcommand: {subcmd}  (try /plugin or /help)")
    return True


# ── Tasks ──────────────────────────────────────────────────────────────────

def cmd_tasks(args: str, _state, config) -> bool:
    """Show and manage tasks."""
    from task import list_tasks, get_task, create_task, update_task, delete_task, clear_all_tasks
    from task.types import TaskStatus

    parts = args.split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    rest   = parts[1].strip() if len(parts) > 1 else ""

    STATUS_MAP = {
        "done":   "completed",
        "start":  "in_progress",
        "cancel": "cancelled",
    }

    if not subcmd:
        tasks = list_tasks()
        if not tasks:
            info("No tasks. Use TaskCreate tool or /tasks create <subject>.")
            return True
        resolved = {t.id for t in tasks if t.status == TaskStatus.COMPLETED}
        total = len(tasks)
        done  = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        info(f"Tasks ({done}/{total} completed):")
        for t in tasks:
            pending_blockers = [b for b in t.blocked_by if b not in resolved]
            owner_str   = f" {clr(f'({t.owner})', 'dim')}" if t.owner else ""
            blocked_str = clr(f" [blocked by #{', #'.join(pending_blockers)}]", "yellow") if pending_blockers else ""
            status_color = {
                TaskStatus.PENDING:     "dim",
                TaskStatus.IN_PROGRESS: "cyan",
                TaskStatus.COMPLETED:   "green",
                TaskStatus.CANCELLED:   "red",
            }.get(t.status, "dim")
            icon = t.status_icon()
            print(f"  #{t.id} {clr(icon + ' ' + t.status.value, status_color)} {t.subject}{owner_str}{blocked_str}")
        return True

    if subcmd == "create":
        if not rest:
            err("Usage: /tasks create <subject>")
            return True
        t = create_task(rest, description="(created via REPL)")
        ok(f"Task #{t.id} created: {t.subject}")
        return True

    if subcmd in STATUS_MAP:
        new_status = STATUS_MAP[subcmd]
        if not rest:
            err(f"Usage: /tasks {subcmd} <task_id>")
            return True
        task, fields = update_task(rest, status=new_status)
        if task is None:
            err(f"Task #{rest} not found.")
        else:
            ok(f"Task #{task.id} → {new_status}: {task.subject}")
        return True

    if subcmd == "delete":
        if not rest:
            err("Usage: /tasks delete <task_id>")
            return True
        removed = delete_task(rest)
        if removed:
            ok(f"Task #{rest} deleted.")
        else:
            err(f"Task #{rest} not found.")
        return True

    if subcmd == "get":
        if not rest:
            err("Usage: /tasks get <task_id>")
            return True
        t = get_task(rest)
        if t is None:
            err(f"Task #{rest} not found.")
            return True
        print(f"  #{t.id} [{t.status.value}] {t.subject}")
        print(f"  Description: {t.description}")
        if t.owner:         print(f"  Owner:       {t.owner}")
        if t.active_form:   print(f"  Active form: {t.active_form}")
        if t.blocked_by:    print(f"  Blocked by:  #{', #'.join(t.blocked_by)}")
        if t.blocks:        print(f"  Blocks:      #{', #'.join(t.blocks)}")
        if t.metadata:      print(f"  Metadata:    {t.metadata}")
        print(f"  Created: {t.created_at[:19]}  Updated: {t.updated_at[:19]}")
        return True

    if subcmd == "clear":
        clear_all_tasks()
        ok("All tasks deleted.")
        return True

    err(f"Unknown tasks subcommand: {subcmd}  (try /tasks or /help)")
    return True
