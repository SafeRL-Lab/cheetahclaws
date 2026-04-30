"""
commands/core.py — Core utility commands for CheetahClaws.

Commands: /help, /clear, /context, /cost, /compact, /init, /export,
          /copy, /status, /doctor, /proactive, /image, /circuit
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union

from ui.render import clr, info, ok, warn, err

# VERSION is imported lazily from cheetahclaws to avoid circular imports
_VERSION_STR = ""

def _get_version() -> str:
    global _VERSION_STR
    if not _VERSION_STR:
        try:
            import importlib
            cc = importlib.import_module("cheetahclaws")
            _VERSION_STR = getattr(cc, "VERSION", "?")
        except Exception:
            _VERSION_STR = "?"
    return _VERSION_STR


def cmd_help(_args: str, _state, config) -> bool:
    try:
        import cheetahclaws
    except Exception:
        info("CheetahClaws — type /model, /save, /load, /history, /context, /exit for commands.")
        return True

    doc = cheetahclaws.__doc__ or ""
    print(doc)

    # Safety net: surface any registered command that the curated docstring
    # forgot to mention (e.g. modular/plugin additions, or newly added commands
    # whose author didn't update the docstring). Walks COMMANDS, groups by
    # handler so aliases share a row, skips anything already referenced.
    commands = getattr(cheetahclaws, "COMMANDS", {})
    meta     = getattr(cheetahclaws, "_CMD_META", {})

    aliases_by_func: dict[object, list[str]] = {}
    for name, func in commands.items():
        aliases_by_func.setdefault(func, []).append(name)

    missing: list[tuple[str, str]] = []
    seen: set[object] = set()
    for func, names in aliases_by_func.items():
        if func in seen:
            continue
        seen.add(func)
        if any(f"/{n}" in doc for n in names):
            continue
        primary = min(names, key=len)
        extra = [n for n in names if n != primary]
        label = f"/{primary}" + (f" (/{', /'.join(extra)})" if extra else "")
        desc = next((meta[n][0] for n in names if n in meta), "(no description)")
        missing.append((label, desc))

    if missing:
        print()
        print("Also available (auto-detected — not in curated list above):")
        w = max(len(m[0]) for m in missing)
        for label, desc in missing:
            print(f"  {label:<{w}}  {desc}")

    return True


def cmd_clear(_args: str, state, config) -> bool:
    state.messages.clear()
    state.turn_count = 0
    ok("Conversation cleared.")
    return True


def cmd_context(_args: str, state, config) -> bool:
    msg_chars = sum(len(str(m.get("content", ""))) for m in state.messages)
    est_tokens = msg_chars // 4
    info(f"Messages:         {len(state.messages)}")
    info(f"Estimated tokens: ~{est_tokens:,}")
    info(f"Model:            {config['model']}")
    info(f"Max tokens:       {config['max_tokens']:,}")
    return True


def cmd_cost(_args: str, state, config) -> bool:
    from cc_config import calc_cost
    cost = calc_cost(config["model"],
                     state.total_input_tokens,
                     state.total_output_tokens)
    info(f"Input tokens:  {state.total_input_tokens:,}")
    info(f"Output tokens: {state.total_output_tokens:,}")
    info(f"Est. cost:     ${cost:.4f} USD")
    return True


def cmd_compact(args: str, state, config) -> bool:
    """Manually compact conversation history."""
    from compaction import manual_compact
    from ui.render import _start_tool_spinner, _stop_tool_spinner, set_spinner_phrase
    focus = args.strip()
    set_spinner_phrase(f"compacting conversation{'  [' + focus + ']' if focus else ''}…")
    _start_tool_spinner()
    try:
        success, msg = manual_compact(state, config, focus=focus)
    finally:
        _stop_tool_spinner()
    if success:
        info(msg)
    else:
        err(msg)
    return True


def cmd_init(args: str, state, config) -> bool:
    """Initialize a CLAUDE.md file in the current directory."""
    target = Path.cwd() / "CLAUDE.md"
    if target.exists():
        err(f"CLAUDE.md already exists at {target}")
        info("Edit it directly or delete it first.")
        return True

    project_name = Path.cwd().name
    template = (
        f"# {project_name}\n\n"
        "## Project Overview\n"
        "<!-- Describe what this project does -->\n\n"
        "## Tech Stack\n"
        "<!-- Languages, frameworks, key dependencies -->\n\n"
        "## Conventions\n"
        "<!-- Coding style, naming conventions, patterns to follow -->\n\n"
        "## Important Files\n"
        "<!-- Key entry points, config files, etc. -->\n\n"
        "## Testing\n"
        "<!-- How to run tests, testing conventions -->\n\n"
    )
    target.write_text(template, encoding="utf-8")
    info(f"Created {target}")
    info("Edit it to give Claude context about your project.")
    return True


def cmd_export(args: str, state, config) -> bool:
    """Export conversation history to a file."""
    if not state.messages:
        err("No conversation to export.")
        return True

    arg = args.strip()
    if arg:
        out_path = Path(arg)
    else:
        export_dir = Path.cwd() / ".nano_claude" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = export_dir / f"conversation_{ts}.md"

    is_json = out_path.suffix.lower() == ".json"

    if is_json:
        out_path.write_text(
            json.dumps(state.messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        lines = []
        for m in state.messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "(structured content)"
            if role == "user":
                lines.append(f"## User\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"## Assistant\n\n{content}\n")
            elif role == "tool":
                name = m.get("name", "tool")
                lines.append(f"### Tool: {name}\n\n```\n{content[:2000]}\n```\n")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")

    info(f"Exported {len(state.messages)} messages to {out_path}")
    return True


def cmd_copy(args: str, state, config) -> bool:
    """Copy the last assistant response to clipboard."""
    last_reply = None
    for m in reversed(state.messages):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                last_reply = content
                break

    if not last_reply:
        err("No assistant response to copy.")
        return True

    try:
        import subprocess as _sp
        if sys.platform == "win32":
            proc = _sp.Popen(["clip"], stdin=_sp.PIPE)
            proc.communicate(last_reply.encode("utf-16le"))
        elif sys.platform == "darwin":
            proc = _sp.Popen(["pbcopy"], stdin=_sp.PIPE)
            proc.communicate(last_reply.encode("utf-8"))
        else:
            for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
                try:
                    proc = _sp.Popen(cmd, stdin=_sp.PIPE)
                    proc.communicate(last_reply.encode("utf-8"))
                    break
                except FileNotFoundError:
                    continue
            else:
                err("No clipboard tool found. Install xclip or xsel.")
                return True
        info(f"Copied {len(last_reply)} chars to clipboard.")
    except Exception as e:
        err(f"Failed to copy: {e}")
    return True


def cmd_status(args: str, state, config) -> bool:
    """Show current session status."""
    from providers import detect_provider
    from compaction import estimate_tokens, get_context_limit

    model = config.get("model", "unknown")
    provider = detect_provider(model)
    perm_mode = config.get("permission_mode", "auto")
    session_id = config.get("_session_id", "N/A")
    turn_count = getattr(state, "turn_count", 0)
    msg_count = len(getattr(state, "messages", []))
    tokens_in = getattr(state, "total_input_tokens", 0)
    tokens_out = getattr(state, "total_output_tokens", 0)
    est_ctx = estimate_tokens(getattr(state, "messages", []))
    ctx_limit = get_context_limit(model)
    ctx_pct = (est_ctx / ctx_limit * 100) if ctx_limit else 0
    plan_mode = config.get("permission_mode") == "plan"

    print(f"  Version:     {_get_version()}")
    print(f"  Model:       {model} ({provider})")
    print(f"  Permissions: {perm_mode}" + (" [PLAN MODE]" if plan_mode else ""))
    print(f"  Session:     {session_id}")
    print(f"  Turns:       {turn_count}")
    print(f"  Messages:    {msg_count}")
    print(f"  Tokens:      ~{tokens_in} in / ~{tokens_out} out")
    print(f"  Context:     ~{est_ctx} / {ctx_limit} ({ctx_pct:.0f}%)")
    return True


def cmd_doctor(args: str, state, config) -> bool:
    """Diagnose installation health and connectivity."""
    import subprocess as _sp
    from providers import PROVIDERS, detect_provider, get_api_key

    ok_n = warn_n = fail_n = 0

    def _print_safe(s):
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", errors="replace").decode())

    def _ok(msg):
        nonlocal ok_n; ok_n += 1
        _print_safe(clr("  [PASS] ", "green") + msg)

    def _warn(msg):
        nonlocal warn_n; warn_n += 1
        _print_safe(clr("  [WARN] ", "yellow") + msg)

    def _fail(msg):
        nonlocal fail_n; fail_n += 1
        _print_safe(clr("  [FAIL] ", "red") + msg)

    info("Running diagnostics...")
    print()

    v = sys.version_info
    if v >= (3, 10):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python {v.major}.{v.minor}.{v.micro} (need ≥3.10)")

    try:
        r = _sp.run(["git", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _ok(f"Git: {r.stdout.strip()}")
        else:
            _fail("Git: not working")
    except Exception:
        _fail("Git: not found")

    try:
        r = _sp.run(["git", "rev-parse", "--is-inside-work-tree"],
                    capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _ok("Inside a git repository")
        else:
            _warn("Not inside a git repository")
    except Exception:
        _warn("Could not check git repo status")

    model = config.get("model", "")
    provider = detect_provider(model)
    key = get_api_key(provider, config)

    if key:
        _ok(f"API key for {provider}: set ({key[:4]}...{key[-4:]})")
    elif provider in ("ollama", "lmstudio"):
        _ok(f"Provider {provider}: no key needed (local)")
    else:
        _fail(f"API key for {provider}: NOT SET")

    if key or provider in ("ollama", "lmstudio"):
        print(f"  ... testing {provider} API connectivity...")
        try:
            import urllib.request, urllib.error
            prov = PROVIDERS.get(provider, {})
            ptype = prov.get("type", "openai")

            if ptype == "anthropic":
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=json.dumps({
                        "model": model,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    }).encode(),
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                )
                try:
                    urllib.request.urlopen(req, timeout=10)
                    _ok(f"Anthropic API: reachable, model {model} works")
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        _fail("Anthropic API: invalid API key (401)")
                    elif e.code == 404:
                        _fail(f"Anthropic API: model {model} not found (404)")
                    elif e.code == 429:
                        _warn("Anthropic API: rate limited (429) — key is valid")
                    else:
                        _warn(f"Anthropic API: HTTP {e.code}")
                except Exception as e:
                    _fail(f"Anthropic API: connection error — {e}")
            elif ptype == "ollama":
                base = prov.get("base_url", "http://localhost:11434")
                try:
                    urllib.request.urlopen(f"{base}/api/tags", timeout=5)
                    _ok(f"Ollama: reachable at {base}")
                except Exception:
                    _fail(f"Ollama: cannot reach {base} — is Ollama running?")
            else:
                base = prov.get("base_url", "")
                if provider == "custom":
                    base = config.get("custom_base_url", base or "")
                if base:
                    models_url = base.rstrip("/") + "/models"
                    req = urllib.request.Request(
                        models_url,
                        headers={"Authorization": f"Bearer {key}"},
                    )
                    try:
                        urllib.request.urlopen(req, timeout=10)
                        _ok(f"{provider} API: reachable")
                    except urllib.error.HTTPError as e:
                        if e.code == 401:
                            _fail(f"{provider} API: invalid API key (401)")
                        elif e.code == 429:
                            _warn(f"{provider} API: rate limited (429) — key is valid")
                        else:
                            _warn(f"{provider} API: HTTP {e.code}")
                    except Exception as e:
                        _fail(f"{provider} API: connection error — {e}")
                else:
                    _warn(f"{provider}: no base_url configured")
        except Exception as e:
            _warn(f"API test skipped: {e}")

    print()
    for pname, pdata in PROVIDERS.items():
        if pname == provider:
            continue
        env_var = pdata.get("api_key_env")
        if env_var and os.environ.get(env_var, ""):
            _ok(f"{pname} key ({env_var}): set")

    # ── General network connectivity ──
    print()
    try:
        import urllib.request
        urllib.request.urlopen("https://httpbin.org/status/200", timeout=5)
        _ok("Internet connectivity: OK")
    except Exception:
        _fail("Internet connectivity: cannot reach external hosts")

    # ── Dependencies ──
    print()
    for mod, desc, required in [
        ("rich", "Rich (live markdown rendering)", True),
        ("pyte", "pyte (terminal emulator for bridges)", True),
        ("PIL", "Pillow (clipboard image /image)", False),
        ("sounddevice", "sounddevice (voice recording)", False),
        ("faster_whisper", "faster-whisper (local STT)", False),
    ]:
        try:
            __import__(mod)
            _ok(desc)
        except ImportError:
            if required:
                _fail(f"{desc}: not installed (required)")
            else:
                _warn(f"{desc}: not installed (optional)")

    print()
    claude_md = Path.cwd() / "CLAUDE.md"
    global_md = Path.home() / ".claude" / "CLAUDE.md"
    if claude_md.exists():
        _ok(f"Project CLAUDE.md: {claude_md}")
    else:
        _warn("No project CLAUDE.md (run /init to create)")
    if global_md.exists():
        _ok(f"Global CLAUDE.md: {global_md}")

    ckpt_root = Path.home() / ".nano_claude" / "checkpoints"
    if ckpt_root.exists():
        total = sum(f.stat().st_size for f in ckpt_root.rglob("*") if f.is_file())
        mb = total / (1024 * 1024)
        sessions = sum(1 for d in ckpt_root.iterdir() if d.is_dir())
        if mb > 100:
            _warn(f"Checkpoints: {mb:.1f} MB ({sessions} sessions)")
        else:
            _ok(f"Checkpoints: {mb:.1f} MB ({sessions} sessions)")

    perm = config.get("permission_mode", "auto")
    if perm == "accept-all":
        _warn(f"Permission mode: {perm} (all operations auto-approved)")
    else:
        _ok(f"Permission mode: {perm}")

    print()
    total = ok_n + warn_n + fail_n
    summary = f"  {ok_n} passed, {warn_n} warnings, {fail_n} failures ({total} checks)"
    if fail_n:
        _print_safe(clr(summary, "red"))
    elif warn_n:
        _print_safe(clr(summary, "yellow"))
    else:
        _print_safe(clr(summary, "green"))

    return True


# ── Setup wizard ──────────────────────────────────────────────────────────

def run_setup_wizard(config: dict) -> None:
    """Interactive first-run setup: pick provider, set API key, verify."""
    from cc_config import save_config
    from providers import PROVIDERS, detect_provider, get_api_key

    print()
    info("Welcome to CheetahClaws! Let's get you set up.\n")

    # ── Step 1: Pick provider ──
    providers_list = [
        ("ollama",    "Ollama (local, free, no API key)"),
        ("anthropic", "Anthropic Claude (cloud, API key required)"),
        ("openai",    "OpenAI GPT (cloud, API key required)"),
        ("gemini",    "Google Gemini (cloud, API key required)"),
        ("deepseek",  "DeepSeek (cloud, API key required)"),
        ("custom",    "Custom OpenAI-compatible endpoint"),
    ]

    info("Which provider would you like to use?\n")
    for i, (pname, desc) in enumerate(providers_list):
        prov = PROVIDERS.get(pname, {})
        env_var = prov.get("api_key_env", "")
        env_set = bool(env_var and os.environ.get(env_var))
        marker = clr(" (key detected)", "green") if env_set else ""
        print(f"  {clr(f'[{i+1}]', 'yellow')} {desc}{marker}")

    print()
    try:
        choice = input(clr("  Select [1-6] (default: 1 for Ollama): ", "cyan")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(providers_list) else 0
    chosen_pname, chosen_desc = providers_list[idx]
    prov = PROVIDERS.get(chosen_pname, {})

    # ── Step 2: Set model ──
    models = prov.get("models", [])
    if chosen_pname == "ollama":
        # Check if Ollama is running and list local models
        try:
            from providers import list_ollama_models
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            local_models = list_ollama_models(base_url)
            if local_models:
                info(f"\nLocal Ollama models found:")
                for i, m in enumerate(local_models[:10]):
                    print(f"  {clr(f'[{i+1}]', 'yellow')} {m}")
                print()
                try:
                    mc = input(clr("  Select a model number (default: 1): ", "cyan")).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                mi = int(mc) - 1 if mc.isdigit() and 1 <= int(mc) <= len(local_models) else 0
                config["model"] = f"ollama/{local_models[mi]}"
            else:
                warn("Ollama is running but no models found. Pull one with: ollama pull gemma4:e4b")
                config["model"] = "ollama/gemma4:e4b"
        except Exception:
            warn("Cannot reach Ollama. Make sure it's running: ollama serve")
            config["model"] = "ollama/gemma4:e4b"
    elif models:
        config["model"] = f"{chosen_pname}/{models[0]}" if chosen_pname != "anthropic" else models[0]
        info(f"\nDefault model: {config['model']}")
    else:
        config["model"] = chosen_pname + "/default"

    # ── Step 3: Set API key (if needed) ──
    # `or ""` (not just .get(..., "")) is load-bearing: ollama / lmstudio
    # entries in PROVIDERS have ``api_key_env: None``, and dict.get returns
    # the stored None — not the default — when the key is present.  Passing
    # None into os.environ.get raises TypeError ("str expected, not NoneType")
    # because os.environ fsencodes its keys.  See issue #59.
    env_var   = prov.get("api_key_env") or ""
    key_field = f"{chosen_pname}_api_key"
    existing_key = (os.environ.get(env_var, "") if env_var else "") \
                   or config.get(key_field, "")

    if chosen_pname not in ("ollama", "lmstudio"):
        if existing_key:
            ok(f"API key detected ({existing_key[:4]}...{existing_key[-4:]})")
        else:
            print()
            info(f"Enter your {chosen_desc.split('(')[0].strip()} API key")
            if env_var:
                info(f"(or set {env_var} env var and restart)")
            try:
                key_input = input(clr("  API key: ", "cyan")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if key_input:
                config[key_field] = key_input
                existing_key = key_input

    if chosen_pname == "custom":
        base = config.get("custom_base_url", "")
        if not base:
            print()
            try:
                base = input(clr("  Base URL (e.g. http://localhost:8000/v1): ", "cyan")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if base:
                config["custom_base_url"] = base

    # ── Step 4: Verify connection ──
    print()
    info("Verifying connection...")
    try:
        import urllib.request, urllib.error
        if chosen_pname == "ollama":
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
            ok("Ollama: connected!")
        elif chosen_pname == "anthropic" and existing_key:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps({"model": config["model"], "max_tokens": 1,
                                 "messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"x-api-key": existing_key, "anthropic-version": "2023-06-01",
                          "content-type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=10)
                ok("Anthropic API: connected!")
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    ok("Anthropic API: key valid (rate limited)")
                elif e.code == 401:
                    err("Invalid API key. You can fix it later with: /config anthropic_api_key=YOUR_KEY")
                else:
                    warn(f"Anthropic API: HTTP {e.code}")
        elif existing_key:
            base = prov.get("base_url", config.get("custom_base_url", ""))
            if base:
                req = urllib.request.Request(
                    base.rstrip("/") + "/models",
                    headers={"Authorization": f"Bearer {existing_key}"},
                )
                try:
                    urllib.request.urlopen(req, timeout=10)
                    ok(f"{chosen_pname} API: connected!")
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        err("Invalid API key. Fix later with: /config")
                    elif e.code == 429:
                        ok(f"{chosen_pname} API: key valid (rate limited)")
                    else:
                        warn(f"{chosen_pname} API: HTTP {e.code}")
    except Exception as e:
        warn(f"Connection test failed: {e}")

    # ── Save ──
    save_config(config)
    print()
    ok(f"Setup complete! Model: {config['model']}")
    info("Type a message to start, or /help for available commands.\n")


def cmd_proactive(args: str, state, config) -> bool:
    """Manage proactive background polling.

    /proactive            — show current status
    /proactive 5m         — enable, trigger after 5 min of inactivity
    /proactive 30s / 1h   — enable with custom interval
    /proactive off        — disable
    """
    args = args.strip().lower()

    import runtime
    sctx = runtime.get_ctx(config)

    if not args:
        if sctx.proactive_enabled:
            interval = sctx.proactive_interval
            info(f"Proactive background polling: ON  (triggering every {interval}s of inactivity)")
        else:
            info("Proactive background polling: OFF  (use /proactive 5m to enable)")
        return True

    if args == "off":
        sctx.proactive_enabled = False
        info("Proactive background polling: OFF")
        return True

    multiplier = 1
    val_str = args
    if args.endswith("m"):
        multiplier = 60
        val_str = args[:-1]
    elif args.endswith("h"):
        multiplier = 3600
        val_str = args[:-1]
    elif args.endswith("s"):
        val_str = args[:-1]

    try:
        val = int(val_str)
        sctx.proactive_interval = val * multiplier
    except ValueError:
        err(f"Invalid duration: '{args}'. Use '5m', '30s', '1h', or 'off'.")
        return True

    sctx.proactive_enabled = True
    sctx.last_interaction_time = time.time()
    info(f"Proactive background polling: ON  (triggering every {sctx.proactive_interval}s of inactivity)")
    return True


def cmd_image(args: str, state, config) -> Union[bool, tuple]:
    """Grab image from clipboard and send to vision model with optional prompt."""
    try:
        from PIL import ImageGrab
        import io, base64
    except ImportError:
        err("Pillow is required for /image. Install with: pip install cheetahclaws[vision]")
        if sys.platform == "linux":
            err("On Linux, clipboard support also requires xclip: sudo apt install xclip")
        return True

    img = ImageGrab.grabclipboard()
    if img is None:
        if sys.platform == "linux":
            err("No image found in clipboard. On Linux, xclip is required (sudo apt install xclip). "
                "Copy an image with Flameshot, GNOME Screenshot, or: xclip -selection clipboard -t image/png -i file.png")
        elif sys.platform == "darwin":
            err("No image found in clipboard. Copy an image first "
                "(Cmd+Ctrl+Shift+4 captures a screenshot region to clipboard).")
        else:
            err("No image found in clipboard. Copy an image first "
                "(Win+Shift+S captures a screenshot region to clipboard).")
        return True

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    size_kb = len(buf.getvalue()) / 1024

    info(f"📷 Clipboard image captured ({size_kb:.0f} KB, {img.size[0]}x{img.size[1]})")
    import runtime
    runtime.get_ctx(config).pending_image = b64

    prompt = args.strip() if args.strip() else "What do you see in this image? Describe it in detail."
    return ("__image__", prompt)


_web_thread = None  # daemon thread running start_web_server(), if any


def cmd_web(args: str, state, config) -> bool:
    """Start the web terminal / chat UI in a background thread.

    /web                          — start on 127.0.0.1:8080 (auto-picks free port)
    /web 9000                     — use port 9000
    /web --host 0.0.0.0           — bind to network
    /web --no-auth                — disable terminal password (local only)
    /web status                   — show whether it's running
    """
    global _web_thread
    import threading

    tokens = (args or "").strip().split()
    sub = tokens[0].lower() if tokens else ""

    if sub == "status":
        if _web_thread and _web_thread.is_alive():
            info("Web server: running (started via /web this session).")
        else:
            info("Web server: not running.")
        return True

    if _web_thread and _web_thread.is_alive():
        info("Web server already running in this session. Use /web status to check.")
        return True

    if os.environ.get("CHEETAHCLAWS_WEB_SERVER") == "1":
        warn("You're already inside a web-terminal session. Nested web launch refused.")
        return True

    port: int | None = None
    host = "127.0.0.1"
    no_auth = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.isdigit():
            port = int(t)
        elif t == "--no-auth":
            no_auth = True
        elif t == "--host" and i + 1 < len(tokens):
            host = tokens[i + 1]; i += 1
        elif t.startswith("--host="):
            host = t.split("=", 1)[1]
        elif t.startswith("--port="):
            try: port = int(t.split("=", 1)[1])
            except ValueError: pass
        else:
            warn(f"Unknown /web arg: {t}  (try: [port] [--host H] [--no-auth])")
            return True
        i += 1

    try:
        from web.server import start_web_server
    except ImportError as e:
        err(f"Web module unavailable: {e}")
        return True

    def _run():
        try:
            start_web_server(port=port, host=host, no_auth=no_auth)
        except SystemExit:
            pass
        except Exception as e:
            import logging_utils as _log
            _log.error("web_server_crashed", error=str(e)[:200])

    _web_thread = threading.Thread(target=_run, daemon=True, name="web-server")
    _web_thread.start()
    time.sleep(0.3)  # let the banner print before the REPL redraws its prompt
    info("Web server started in background. Continue typing — REPL is still live.")
    return True


def cmd_circuit(args: str, state, config) -> bool:
    """Inspect and manage per-provider circuit breakers.

    /circuit                    — list all breakers and their state
    /circuit status [provider]  — same as above, optionally filtered
    /circuit reset <provider>   — force-close a breaker (or 'all')
    """
    import circuit_breaker as _cb

    parts = args.strip().split()
    sub = parts[0].lower() if parts else "status"
    target = parts[1] if len(parts) > 1 else ""

    if sub in ("reset", "close", "clear"):
        if not target:
            err("Usage: /circuit reset <provider>  (or 'all')")
            return True
        if target.lower() == "all":
            names = list(_cb._registry.keys())
            if not names:
                info("No circuit breakers to reset.")
                return True
            for name in names:
                _cb.reset_breaker(name)
            ok(f"Reset {len(names)} circuit breaker(s): {', '.join(names)}")
            return True
        if target not in _cb._registry:
            warn(f"No circuit breaker registered for '{target}'. Nothing to reset.")
            return True
        _cb.reset_breaker(target)
        ok(f"Circuit breaker for '{target}' reset (force-closed).")
        return True

    if sub not in ("status", ""):
        err(f"Unknown /circuit subcommand: {sub}. Use: status | reset")
        return True

    breakers = _cb._registry
    if target:
        breakers = {k: v for k, v in breakers.items() if k == target}

    if not breakers:
        info("No circuit breakers active yet (none have been exercised this session).")
        return True

    for name, b in breakers.items():
        st = b.state.value
        color = {"closed": "green", "half_open": "yellow", "open": "red"}.get(st, "dim")
        line = f"  {name:<12} state={clr(st, color)}  failures={len(b._failure_times)}/{b.threshold}"
        if b._opened_at is not None and b.state.value == "open":
            remaining = max(0.0, b.cooldown - (time.monotonic() - b._opened_at))
            line += f"  cooldown_remaining={remaining:.0f}s"
        print(line)
    return True
