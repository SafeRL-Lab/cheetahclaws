# Add autosave-on-exit and `/resume` for last-session continuation

## Summary

This change adds a lightweight autosave + resume flow so users can continue their latest REPL session without manually selecting a file.

## What was added

- New autosave function: `save_latest(args, state, _config)`
  - File: `cheetahclaws.py`
  - Saves to: `MR_SESSION_DIR / "session_latest.json"`
  - Ensures parent directory exists:
    - `path.parent.mkdir(parents=True, exist_ok=True)`
  - Persists:
    - `messages`
    - `turn_count`
    - `total_input_tokens`
    - `total_output_tokens`

- New resume function: `cmd_resume(args, state, _config)`
  - File: `cheetahclaws.py`
  - `/resume` with no args loads:
    - `MR_SESSION_DIR / "session_latest.json"`
  - `/resume <file>` loads:
    - `MR_SESSION_DIR / <file>` (or direct path if `/` present)
  - Restores state in same style as `/load`.

- Slash command registration:
  - Added `"resume": cmd_resume` in `COMMANDS`.

## REPL exit behavior

Autosave now runs on abrupt prompt exit in the main REPL loop:

- On `EOFError` / `KeyboardInterrupt` while waiting for input:
  - Calls `save_latest("", state, config)`
  - Then exits cleanly.

Also on explicit command exit:

- `/exit` and `/quit` (`cmd_exit`) call `save_latest("", _state, _config)` before `sys.exit(0)`.

## Slash dispatch flow

`handle_slash()` now treats known slash commands as handled and returns `True`, preventing `/resume` from falling through into normal `run_query(...)` chat execution.

## User flow

1. Use the agent normally.
2. Exit via `/exit`, `Ctrl+C`, or `Ctrl+D` at prompt.
3. Session autosaves to `mr_sessions/session_latest.json`.
4. Restart agent and run `/resume`.
5. Continue from restored conversation state.
