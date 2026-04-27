"""End-to-end: drive a real agent.run() conversation where the LLM calls Write,
and verify the checkpoint hook intercepts the call and files a backup to disk.

Only the LLM provider is mocked (via monkeypatching agent.stream). The Write
tool, checkpoint hooks and checkpoint store all run for real against tmp_path.
"""
from __future__ import annotations

import pytest

import tools as _tools_init  # noqa: F401 - force built-in tool registration
from agent import AgentState, run
from providers import AssistantTurn
from checkpoint import hooks as checkpoint_hooks
from checkpoint import store as checkpoint_store


def _scripted_stream(turns):
    cursor = iter(turns)

    def fake_stream(**_kwargs):
        spec = next(cursor)
        yield AssistantTurn(
            text=spec.get("text", ""),
            tool_calls=spec.get("tool_calls") or [],
            in_tokens=1, out_tokens=1,
        )

    return fake_stream


@pytest.fixture
def sandboxed_checkpoints(tmp_path, monkeypatch):
    """Run checkpoint store against tmp_path and install hooks on built-in tools."""
    monkeypatch.setattr(
        checkpoint_store, "_checkpoints_root", lambda: tmp_path / ".checkpoints"
    )
    checkpoint_store.reset_file_versions()
    checkpoint_hooks.set_session("e2e-session")
    checkpoint_hooks.reset_tracked()
    checkpoint_hooks.install_hooks()
    yield tmp_path
    checkpoint_hooks.reset_tracked()


def test_llm_write_triggers_checkpoint_backup(monkeypatch, sandboxed_checkpoints):
    """When the LLM calls Write, the checkpoint hook must back the pre-edit file up.

    Pre-populate a small file, then let the LLM overwrite it via the Write
    tool. The hook should copy the old content into checkpoints/.../backups/
    before the Write executes, so the backup holds the original bytes.
    """
    target = sandboxed_checkpoints / "hello.py"
    target.write_text("print('before')\n", encoding="utf-8")

    turns = [
        {"tool_calls": [{
            "id": "w1",
            "name": "Write",
            "input": {"file_path": str(target), "content": "print('after')\n"},
        }]},
        {"text": "done"},
    ]
    monkeypatch.setattr("agent.stream", _scripted_stream(turns))

    state = AgentState()
    config = {"model": "test", "permission_mode": "accept-all",
              "_session_id": "e2e-session"}
    list(run("overwrite the file", state, config, "system prompt"))

    # After the turn: Write applied the new content
    assert target.read_text(encoding="utf-8") == "print('after')\n"

    # And the checkpoint hook filed a backup with the pre-edit content
    backups_dir = sandboxed_checkpoints / ".checkpoints" / "e2e-session" / "backups"
    backups = list(backups_dir.iterdir())
    assert backups, "checkpoint hook did not create a backup file"
    assert any(b.read_text(encoding="utf-8") == "print('before')\n" for b in backups)


def test_oversized_write_logs_to_stderr_not_stdout(
    monkeypatch, sandboxed_checkpoints, capfd
):
    """Over the _MAX_FILE_SIZE threshold the hook skips + logs — to stderr only.

    This is the actual user-visible contract of PR #47: checkpoint skips must
    not pollute stdout (which carries the conversation transcript), they must
    land on stderr where operators look.
    """
    monkeypatch.setattr(checkpoint_store, "_MAX_FILE_SIZE", 20)
    big = sandboxed_checkpoints / "big.py"
    big.write_text("x" * 100, encoding="utf-8")

    turns = [
        {"tool_calls": [{
            "id": "w1",
            "name": "Write",
            "input": {"file_path": str(big), "content": "y" * 100},
        }]},
        {"text": "ok"},
    ]
    monkeypatch.setattr("agent.stream", _scripted_stream(turns))

    state = AgentState()
    list(run("rewrite", state, {"model": "test", "permission_mode": "accept-all",
                                 "_session_id": "e2e-session",
                                 "disabled_tools": ["Agent"]},
              "sys"))

    out, errtxt = capfd.readouterr()
    assert "[checkpoint] skipping large file" in errtxt
    assert "[checkpoint] skipping large file" not in out
