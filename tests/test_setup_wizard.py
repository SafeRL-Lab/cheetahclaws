"""Regression tests for ``commands.core.run_setup_wizard``.

These tests pin two related invariants:

1. **Ollama / LMStudio paths must not crash** even though their PROVIDERS
   entries declare ``api_key_env: None``.  Issue #59 tracked a
   ``TypeError: str expected, not NoneType`` on a fresh container where
   the wizard fired on first run, the user picked Ollama, and the code
   passed ``None`` into ``os.environ.get(...)`` (which fsencodes its key).

2. **The wizard must successfully write a config** for local providers
   without prompting for an API key.

The wizard is interactive, so we mock ``input``, ``urllib.request.urlopen``,
``providers.list_ollama_models``, and ``config.save_config`` to keep
the test fully offline.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import cheetahclaws.commands.core as _core


@pytest.fixture(autouse=True)
def _ensure_real_providers_module():
    """Defend against test pollution.

    ``tests/test_research.py`` historically replaced ``sys.modules["cheetahclaws.providers"]``
    with a stub and didn't restore it.  Even after that's been fixed,
    keep this fixture as a safety net so future stubbing accidents in
    other suites don't silently break the wizard tests.
    """
    import importlib
    saved = sys.modules.pop("cheetahclaws.providers", None)
    try:
        importlib.invalidate_caches()
        # Re-import from disk; raises a clear error if the import is broken.
        importlib.import_module("cheetahclaws.providers")
        yield
    finally:
        if saved is not None and getattr(saved, "PROVIDERS", None) is None:
            # The pre-existing entry was a stub — drop it.
            sys.modules.pop("cheetahclaws.providers", None)


def _run_wizard(monkeypatch, inputs: list[str], config: dict,
                ollama_models: list[str] | None = None) -> dict:
    """Drive the wizard end-to-end with canned input + offline mocks.

    Returns the (mutated) config dict for assertions.  Raises if the
    wizard does — exactly what we want to catch in the regression case.
    """
    queue = iter(inputs)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(queue)
        except StopIteration as exc:
            raise EOFError(
                "wizard asked for more input than the test provided "
                f"(canned answers: {inputs!r})"
            ) from exc

    monkeypatch.setattr("builtins.input", fake_input)

    # Block any real network the wizard might attempt during "verify".
    class _FakeResponse:
        def read(self): return b""
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(*_a, **_kw):
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    # Stub list_ollama_models — providers.py imports it from itself
    # inside the wizard via `from providers import list_ollama_models`.
    from cheetahclaws import providers
    if ollama_models is not None:
        monkeypatch.setattr(providers, "list_ollama_models",
                             lambda *_a, **_kw: list(ollama_models),
                             raising=False)

    # Don't write to ~/.cheetahclaws/config.json from a test.
    # NB: alias the module so it doesn't shadow the `config` dict param.
    from cheetahclaws import config as _config_mod
    monkeypatch.setattr(_config_mod, "save_config", lambda *_a, **_kw: None)

    _core.run_setup_wizard(config)
    return config


# ── The actual issue-#59 regression ──────────────────────────────────────


def test_ollama_wizard_does_not_crash_on_none_api_key_env(monkeypatch):
    """Issue #59: selecting Ollama + a model triggered TypeError because
    ``os.environ.get(None, "")`` is illegal — the PROVIDERS entry has
    ``api_key_env: None`` and ``dict.get(key, default)`` returns the
    stored ``None``, not the default.  This must complete without raising.
    """
    config: dict = {}
    result = _run_wizard(
        monkeypatch,
        inputs=["1", "1"],   # provider 1 = ollama, model 1
        config=config,
        ollama_models=["qwen3.5:9b", "gemma4:e4b"],
    )
    assert result["model"] == "ollama/qwen3.5:9b"
    # API key prompt MUST be skipped for ollama.
    assert "ollama_api_key" not in result


def test_lmstudio_provider_does_not_crash_on_none_api_key_env(monkeypatch):
    """LMStudio's PROVIDERS entry also has api_key_env: None.  We test
    the unsafe predicate directly rather than driving the full wizard
    (the wizard's provider menu doesn't list LMStudio, but other code
    paths in the same module read the same field).
    """
    from cheetahclaws.providers import PROVIDERS
    prov = PROVIDERS["lmstudio"]
    # The fixed predicate: `or ""` tolerates a None value.
    env_var = prov.get("api_key_env") or ""
    assert env_var == ""
    # And the resulting os.environ.get call must be legal.
    assert os.environ.get(env_var, "") == ""


def test_anthropic_path_still_prompts_for_api_key(monkeypatch):
    """The fix must NOT silently bypass the API-key step for cloud providers."""
    config: dict = {}
    # Inputs: provider 2 = anthropic, then API key value.
    # Default model is taken from PROVIDERS["anthropic"]["models"][0],
    # so no model number prompt fires for anthropic.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = _run_wizard(
        monkeypatch,
        inputs=["2", "sk-ant-test-key"],
        config=config,
    )
    assert "anthropic_api_key" in result
    assert result["anthropic_api_key"] == "sk-ant-test-key"


def test_anthropic_picks_up_existing_env_var_without_prompt(monkeypatch):
    """If ANTHROPIC_API_KEY is already set, the wizard must NOT ask again."""
    config: dict = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    # Only one input expected (provider choice).  If the wizard prompts
    # for a key, _run_wizard's fake_input raises EOFError.
    result = _run_wizard(
        monkeypatch,
        inputs=["2"],
        config=config,
    )
    # The env var is detected; we don't write it into config.
    assert "anthropic_api_key" not in result
    assert result.get("model")  # some model was set
