"""
commands/checkpoint_plan.py — Checkpoint and plan mode commands for CheetahClaws.

Commands: /checkpoint, /rewind (alias), /plan
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Union

from cheetahclaws.ui.render import clr, info, ok, warn, err


def cmd_checkpoint(args: str, state, config) -> bool:
    """List or restore checkpoints.

    /checkpoint          — list all checkpoints
    /checkpoint <id>     — restore to checkpoint #id
    /checkpoint clear    — delete all checkpoints for this session
    """
    from cheetahclaws import checkpoint as ckpt
    from cheetahclaws.tools import ask_input_interactive

    session_id = config.get("_session_id")
    if not session_id:
        err("No active session.")
        return True

    arg = args.strip()

    if arg == "clear":
        ckpt.delete_session_checkpoints(session_id)
        info("All checkpoints cleared.")
        return True

    if not arg:
        snaps = ckpt.list_snapshots(session_id)
        if not snaps:
            info("No checkpoints yet.")
            return True
        info(f"Checkpoints ({len(snaps)} total):")
        for s in snaps:
            ts = s["created_at"]
            try:
                t = datetime.fromisoformat(ts).strftime("%H:%M")
            except Exception:
                t = ts[:16]
            preview = s["user_prompt_preview"]
            if preview:
                preview = f'  "{preview[:40]}{"..." if len(preview) > 40 else ""}"'
            else:
                preview = "  (initial state)"
            print(f"  #{s['id']:<3} [turn {s['turn_count']}]  {t}{preview}")
        return True

    try:
        snap_id = int(arg)
    except ValueError:
        err(f"Unknown subcommand: {arg}")
        return True

    snap = ckpt.get_snapshot(session_id, snap_id)
    if snap is None:
        err(f"Checkpoint #{snap_id} not found.")
        return True

    changed = ckpt.files_changed_since(session_id, snap_id)
    ts = snap.created_at
    try:
        t = datetime.fromisoformat(ts).strftime("%H:%M")
    except Exception:
        t = ts[:16]

    info(f"Checkpoint #{snap_id} (turn {snap.turn_count}, {t})")
    if changed:
        shown = changed[:4]
        extra = f" (+{len(changed) - 4} files)" if len(changed) > 4 else ""
        info(f"Files changed since: {', '.join(Path(f).name for f in shown)}{extra}")
    print()
    menu_buf = "  1. Restore conversation + files\n  2. Restore conversation only\n  3. Restore files only\n  4. Cancel"
    print(menu_buf)
    print()

    try:
        choice = ask_input_interactive("Choice [1-4]: ", config, menu_buf).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return True

    restore_conversation = choice in ("1", "2")
    restore_files = choice in ("1", "3")

    if choice == "4" or choice not in ("1", "2", "3"):
        info("Cancelled.")
        return True

    results = []

    if restore_conversation:
        state.messages = state.messages[:snap.message_index]
        state.turn_count = snap.turn_count
        state.total_input_tokens = snap.token_snapshot.get("input", 0)
        state.total_output_tokens = snap.token_snapshot.get("output", 0)
        state.total_cache_read_tokens = snap.token_snapshot.get("cache_read", 0)
        state.total_cache_write_tokens = snap.token_snapshot.get("cache_write", 0)
        results.append("conversation restored")

    if restore_files:
        file_results = ckpt.rewind_files(session_id, snap_id)
        for r in file_results:
            print(f"  {r}")
        results.append(f"{len(file_results)} file(s) processed")

    ckpt.reset_tracked()
    ckpt.make_snapshot(
        session_id, state, config,
        f"[rewind to #{snap_id}]",
        tracked_edits=None,
    )

    info(f"Done: {', '.join(results)}. New checkpoint created.")
    return True


# /rewind is an alias for /checkpoint
cmd_rewind = cmd_checkpoint


def cmd_plan(args: str, state, config) -> Union[bool, tuple]:
    """Enter/exit plan mode or show current plan.

    /plan <description>  — enter plan mode and start planning
    /plan                — show current plan file contents
    /plan done           — exit plan mode, restore permissions
    /plan status         — show plan mode status
    """
    arg = args.strip()

    from cheetahclaws import runtime
    sctx = runtime.get_ctx(config)
    plan_file = sctx.plan_file or ""
    in_plan_mode = config.get("permission_mode") == "plan"

    if arg == "done":
        if not in_plan_mode:
            err("Not in plan mode.")
            return True
        prev = sctx.prev_permission_mode or "auto"
        sctx.prev_permission_mode = None
        config["permission_mode"] = prev
        info(f"Exited plan mode. Permission mode restored to: {prev}")
        if plan_file:
            info(f"Plan saved at: {plan_file}")
            info("You can now ask Claude to implement the plan.")
        return True

    if arg == "status":
        if in_plan_mode:
            info("Plan mode: ACTIVE")
            info(f"Plan file: {plan_file}")
            info("Only the plan file is writable. Use /plan done to exit.")
        else:
            info("Plan mode: inactive")
        return True

    if not arg:
        if not plan_file:
            info("Not in plan mode. Use /plan <description> to start planning.")
            return True
        p = Path(plan_file)
        if p.exists() and p.stat().st_size > 0:
            info(f"Plan file: {plan_file}")
            print(p.read_text(encoding="utf-8"))
        else:
            info(f"Plan file is empty: {plan_file}")
        return True

    if in_plan_mode:
        err("Already in plan mode. Use /plan done to exit first.")
        return True

    session_id = config.get("_session_id", "default")
    plans_dir = Path.cwd() / ".nano_claude" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / f"{session_id}.md"
    plan_path.write_text(f"# Plan: {arg}\n\n", encoding="utf-8")

    sctx.prev_permission_mode = config.get("permission_mode", "auto")
    config["permission_mode"] = "plan"
    sctx.plan_file = str(plan_path)

    info("Plan mode activated (read-only except plan file).")
    info(f"Plan file: {plan_path}")
    info("Use /plan done to exit and start implementation.")
    print()

    return ("__plan__", arg)
