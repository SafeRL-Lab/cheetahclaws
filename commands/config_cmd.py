"""
commands/config_cmd.py — Configuration and model commands for CheetahClaws.

Commands: /model, /config, /verbose, /thinking, /permissions, /cwd
"""
from __future__ import annotations

import json
import os

from ui.render import clr, info, ok, warn, err


def cmd_model(args: str, _state, config) -> bool:
    from providers import PROVIDERS, detect_provider
    if not args:
        model = config["model"]
        pname = detect_provider(model)
        info(f"Current model:    {model}  (provider: {pname})")
        info("\nAvailable models by provider:")
        for pn, pdata in PROVIDERS.items():
            if pn == "ollama":
                # Show live local models instead of hardcoded list
                from providers import list_ollama_models
                base_url = (
                    os.environ.get("OLLAMA_BASE_URL")
                    or config.get("ollama_base_url")
                    or pdata.get("base_url", "http://localhost:11434")
                )
                local = list_ollama_models(base_url)
                if local:
                    info(f"  {'ollama':12s}  " + ", ".join(local[:6]) + ("..." if len(local) > 6 else ""))
                    info(f"  {'':12s}  " + clr(f"({len(local)} local models — /model ollama to pick)", "dim"))
                else:
                    info(f"  {'ollama':12s}  " + clr("(not running or no models pulled)", "dim"))
                continue
            ms = pdata.get("models", [])
            if ms:
                info(f"  {pn:12s}  " + ", ".join(ms[:4]) + ("..." if len(ms) > 4 else ""))
        info("\nFormat: 'provider/model' or just model name (auto-detected)")
        info("  e.g. /model gpt-4o")
        info("  e.g. /model ollama/qwen2.5-coder")
        info("  e.g. /model kimi:moonshot-v1-32k")
    else:
        m = args.strip()
        # "/model ollama" with no model name → interactive picker
        if m == "ollama":
            if _interactive_ollama_picker(config):
                return True
            return True
        if "/" not in m and ":" in m:
            left, right = m.split(":", 1)
            if left in PROVIDERS:
                m = f"{left}/{right}"
        config["model"] = m
        pname = detect_provider(m)
        ok(f"Model set to {m}  (provider: {pname})")
        from cc_config import save_config
        save_config(config)
    return True


def _interactive_ollama_picker(config: dict) -> bool:
    """Prompt the user to select from locally available Ollama models."""
    from providers import PROVIDERS, list_ollama_models
    from tools import ask_input_interactive
    prov = PROVIDERS.get("ollama", {})
    base_url = (
        os.environ.get("OLLAMA_BASE_URL")
        or config.get("ollama_base_url")
        or prov.get("base_url", "http://localhost:11434")
    )

    models = list_ollama_models(base_url)
    if not models:
        err(f"No local Ollama models found at {base_url}.")
        return False

    menu_buf = clr("\n  ── Local Ollama Models ──", "dim")
    for i, m in enumerate(models):
        menu_buf += "\n" + clr(f"  [{i+1:2d}] ", "yellow") + m
    print(menu_buf)
    print()

    try:
        ans = ask_input_interactive(clr("  Select a model number or Enter to cancel > ", "cyan"), config, menu_buf).strip()
        if not ans: return False
        idx = int(ans) - 1
        if 0 <= idx < len(models):
            new_model = f"ollama/{models[idx]}"
            config["model"] = new_model
            from cc_config import save_config
            save_config(config)
            ok(f"Model updated to {new_model}")
            return True
        else:
            err("Invalid selection.")
    except (ValueError, KeyboardInterrupt, EOFError):
        pass
    return False


def cmd_config(args: str, _state, config) -> bool:
    from cc_config import save_config
    if not args:
        _SECRETS = {"api_key", "anthropic_api_key", "telegram_token", "wechat_token"}
        display = {k: v for k, v in config.items()
                   if k not in _SECRETS and not k.startswith("_")
                   and not k.endswith(("_key", "_token", "_secret"))}
        print(json.dumps(display, indent=2))
    elif "=" in args:
        key, _, val = args.partition("=")
        key, val = key.strip(), val.strip()
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.isdigit():
            val = int(val)
        # JSON-style values: lists, objects, numbers with signs, quoted strings.
        # Without this branch, /config wechat_smart_reply_whitelist=["a","b"]
        # silently stored the literal string '["a","b"]'.
        elif val and val[0] in '[{"-' or (val.startswith("-") and val[1:].isdigit()):
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                pass  # leave as string
        config[key] = val
        save_config(config)
        ok(f"Set {key} = {val!r}")
        if key == "context_window" and isinstance(val, int) and not isinstance(val, bool) and val > 0:
            # The override drives the prompt %, /context, AND the compaction
            # trigger. Warn if it exceeds the model's real window, since that
            # disables compaction and the API may reject oversized prompts.
            from compaction import get_context_limit
            # Real window with the override forced off (keeps custom_base_url so
            # custom/vLLM endpoints still get their live lookup).
            real = get_context_limit(config.get("model", ""), {**config, "context_window": 0})
            if real and val > real:
                warn(f"context_window={val:,} exceeds the model's real window "
                     f"(~{real:,}); compaction won't fire before the real limit, "
                     "so the API may reject oversized prompts. Use this only to "
                     "correct a wrong default.")
            info("Takes effect on the next prompt (no restart needed).")
    else:
        k = args.strip()
        v = config.get(k, "(not set)")
        info(f"{k} = {v}")
    return True


def cmd_verbose(_args: str, _state, config) -> bool:
    from cc_config import save_config
    config["verbose"] = not config.get("verbose", False)
    state_str = "ON" if config["verbose"] else "OFF"
    ok(f"Verbose mode: {state_str}")
    save_config(config)
    return True


def cmd_quiet(_args: str, _state, config) -> bool:
    from cc_config import save_config
    config["quiet"] = not config.get("quiet", True)
    state_str = "ON" if config["quiet"] else "OFF"
    ok(f"Quiet mode: {state_str}  "
       + ("(hide tool execution, show a per-turn summary)" if config["quiet"]
          else "(show each tool call)"))
    save_config(config)
    return True


def cmd_thinking(_args: str, _state, config) -> bool:
    from cc_config import save_config
    config["thinking"] = not config.get("thinking", False)
    state_str = "ON" if config["thinking"] else "OFF"
    ok(f"Extended thinking: {state_str}")
    save_config(config)
    return True


def cmd_permissions(args: str, _state, config) -> bool:
    from cc_config import save_config
    from tools import ask_input_interactive
    modes = ["auto", "accept-all", "manual"]
    mode_desc = {
        "auto":       "Prompt for each tool call (default)",
        "accept-all": "Allow all tool calls silently",
        "manual":     "Prompt for each tool call (strict)",
    }
    if not args.strip():
        current = config.get("permission_mode", "auto")
        menu_buf = clr("\n  ── Permission Mode ──", "dim")
        for i, m in enumerate(modes):
            marker = clr("●", "green") if m == current else clr("○", "dim")
            menu_buf += f"\n  {marker} {clr(f'[{i+1}]', 'yellow')} {clr(m, 'cyan')}  {clr(mode_desc[m], 'dim')}"
        print(menu_buf)
        print()
        try:
            ans = ask_input_interactive(clr("  Select a mode number or Enter to cancel > ", "cyan"), config, menu_buf).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return True
        if not ans:
            return True
        if ans.isdigit() and 1 <= int(ans) <= len(modes):
            m = modes[int(ans) - 1]
            config["permission_mode"] = m
            save_config(config)
            ok(f"Permission mode set to: {m}")
        else:
            err("Invalid selection.")
    else:
        m = args.strip()
        if m not in modes:
            err(f"Unknown mode: {m}. Choose: {', '.join(modes)}")
        else:
            config["permission_mode"] = m
            save_config(config)
            ok(f"Permission mode set to: {m}")
    return True


def cmd_cwd(args: str, _state, config) -> bool:
    if not args.strip():
        info(f"Working directory: {os.getcwd()}")
    else:
        p = args.strip()
        try:
            os.chdir(p)
            ok(f"Changed directory to: {os.getcwd()}")
        except Exception as e:
            err(str(e))
    return True
