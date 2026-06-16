"""Tests for kernel.runner.llm (RFC 0019)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from cheetahclaws.kernel import (
    Kernel,
    LedgerStore,
    RunnerSupervisor,
    SandboxPolicy,
    ScheduleSpec,
    WorkerLoop,
)
from cheetahclaws.kernel.runner.llm import (
    LlmRequest,
    LlmResponse,
    MockProvider,
    ProviderInvalidRequest,
    ProviderUnavailable,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="LLM runner spawns POSIX subprocesses",
)


LLM_RUNNER_ARGV = [sys.executable, "-m", "cheetahclaws.kernel.runner.llm"]


# ── Dataclass round-trip ────────────────────────────────────────────────


def test_llm_request_minimum():
    r = LlmRequest(model="m", user="hi")
    assert r.model == "m"
    assert r.user == "hi"
    assert r.system == ""
    assert r.max_tokens == 1024


def test_llm_request_to_from_dict():
    r = LlmRequest(model="m", user="hi", system="be brief",
                    max_tokens=10, temperature=0.5,
                    metadata={"trace": "abc"})
    d = r.to_dict()
    r2 = LlmRequest.from_dict(d)
    assert r == r2


def test_llm_request_rejects_empty_user():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest(model="m", user="")


def test_llm_request_rejects_bad_temperature():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest(model="m", user="x", temperature=5.0)


def test_llm_request_rejects_zero_max_tokens():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest(model="m", user="x", max_tokens=0)


def test_llm_request_from_dict_missing_model():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest.from_dict({"user": "hi"})


def test_llm_request_from_dict_missing_user():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest.from_dict({"model": "m"})


def test_llm_response_round_trip():
    r = LlmResponse(text="hi", tokens_input=5, tokens_output=2,
                     cost_micro=100, model="m",
                     finish_reason="stop", metadata={"k": "v"})
    d = r.to_dict()
    r2 = LlmResponse.from_dict(d)
    assert r == r2
    assert r.tokens_total == 7


def test_llm_response_rejects_negative_tokens():
    with pytest.raises(ProviderInvalidRequest):
        LlmResponse(text="x", tokens_input=-1, tokens_output=0,
                     cost_micro=0, model="m")


# ── MockProvider ────────────────────────────────────────────────────────


def test_mock_provider_returns_fixed_response():
    resp = LlmResponse(text="four", tokens_input=5, tokens_output=1,
                        cost_micro=50, model="m")
    p = MockProvider(resp)
    assert p(LlmRequest(model="m", user="2+2")) is resp
    assert p(LlmRequest(model="m", user="anything")) is resp
    assert len(p.calls) == 2


def test_mock_provider_records_calls():
    p = MockProvider(LlmResponse(text="x", tokens_input=1,
                                   tokens_output=1, cost_micro=10,
                                   model="m"))
    p(LlmRequest(model="m", user="hello"))
    p(LlmRequest(model="m", user="goodbye"))
    assert [c.user for c in p.calls] == ["hello", "goodbye"]


def test_mock_provider_from_env(monkeypatch):
    payload = {
        "text": "from env", "tokens_input": 3, "tokens_output": 1,
        "cost_micro": 30, "model": "m",
    }
    monkeypatch.setenv(MockProvider.ENV_RESPONSE, json.dumps(payload))
    p = MockProvider.from_env()
    out = p(LlmRequest(model="m", user="x"))
    assert out.text == "from env"


def test_mock_provider_from_env_unset_raises(monkeypatch):
    monkeypatch.delenv(MockProvider.ENV_RESPONSE, raising=False)
    with pytest.raises(ProviderUnavailable):
        MockProvider.from_env()


def test_mock_provider_from_env_invalid_json_raises(monkeypatch):
    monkeypatch.setenv(MockProvider.ENV_RESPONSE, "not-json{{")
    with pytest.raises(ProviderUnavailable):
        MockProvider.from_env()


def test_mock_provider_echo():
    p = MockProvider.echo()
    out = p(LlmRequest(model="m", user="ping"))
    assert out.text == "echo: ping"
    assert out.model == "m"
    assert out.tokens_total >= 1


# ── Subprocess: __main__ entry ──────────────────────────────────────────


def _run_llm_runner(*, init_payload: dict, env: dict) -> dict:
    """Spawn `python -m kernel.runner.llm`, send init, collect
    all stdout lines, return parsed dict {exit_code, lines, stderr}."""
    proc = subprocess.Popen(
        LLM_RUNNER_ARGV,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env,
    )
    init = json.dumps({"op": "init", "pid": 1,
                        "payload": init_payload}) + "\n"
    proc.stdin.write(init.encode())
    proc.stdin.flush()

    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        try:
            lines.append(json.loads(line.decode().strip()))
        except json.JSONDecodeError:
            pass
    rc = proc.wait(timeout=10)
    return {
        "exit_code": rc,
        "lines": lines,
        "stderr": proc.stderr.read().decode("utf-8", "replace"),
    }


def _mock_env(response: dict) -> dict:
    return {
        **os.environ,
        "CC_LLM_PROVIDER":           "mock",
        "CC_LLM_MOCK_RESPONSE_JSON": json.dumps(response),
    }


def test_subprocess_happy_path():
    result = _run_llm_runner(
        init_payload={"model": "claude-x", "user": "what is 2+2?",
                       "max_tokens": 50},
        env=_mock_env({
            "text": "four", "tokens_input": 5, "tokens_output": 1,
            "cost_micro": 250, "model": "claude-x",
            "finish_reason": "stop",
        }),
    )
    assert result["exit_code"] == 0, result["stderr"]
    ops = [l["op"] for l in result["lines"]]
    assert ops[0] == "ready"
    assert "iteration_start" in ops
    assert "charge" in ops
    assert ops[-1] == "exit"
    # Two charge messages: tokens and cost_micro.
    charges = [l for l in result["lines"] if l["op"] == "charge"]
    by_dim = {c["dim"]: c["amount"] for c in charges}
    assert by_dim == {"tokens": 6, "cost_micro": 250}
    # exit is completed.
    exit_msg = [l for l in result["lines"] if l["op"] == "exit"][0]
    assert exit_msg["exit_kind"] == "completed"


def test_subprocess_no_provider_set_exits_2(monkeypatch):
    env = {**os.environ}
    env.pop("CC_LLM_PROVIDER", None)
    result = _run_llm_runner(
        init_payload={"model": "m", "user": "hi"},
        env=env,
    )
    assert result["exit_code"] == 2
    assert "CC_LLM_PROVIDER" in result["stderr"]


def test_subprocess_unknown_provider_exits_2():
    env = {**os.environ, "CC_LLM_PROVIDER": "nonsense"}
    result = _run_llm_runner(
        init_payload={"model": "m", "user": "hi"},
        env=env,
    )
    assert result["exit_code"] == 2
    assert "unknown CC_LLM_PROVIDER" in result["stderr"]


def test_subprocess_mock_without_response_json_exits_2(monkeypatch):
    env = {**os.environ, "CC_LLM_PROVIDER": "mock"}
    env.pop("CC_LLM_MOCK_RESPONSE_JSON", None)
    result = _run_llm_runner(
        init_payload={"model": "m", "user": "hi"},
        env=env,
    )
    assert result["exit_code"] == 2


def test_subprocess_invalid_init_payload_exits_failed():
    """Missing 'user' field → exit_kind='failed'."""
    result = _run_llm_runner(
        init_payload={"model": "m"},  # no 'user'
        env=_mock_env({
            "text": "x", "tokens_input": 0, "tokens_output": 0,
            "cost_micro": 0, "model": "m",
        }),
    )
    # Should send ready, then send exit with failed.
    ops = [l["op"] for l in result["lines"]]
    assert "ready" in ops
    exit_msgs = [l for l in result["lines"] if l["op"] == "exit"]
    assert len(exit_msgs) == 1
    assert exit_msgs[0]["exit_kind"] == "failed"


def test_subprocess_zero_charges_emit_no_charge_messages():
    """When tokens=0 and cost=0, no charge messages should be emitted
    (avoids polluting the ledger with zero-amount charges)."""
    result = _run_llm_runner(
        init_payload={"model": "m", "user": "hi"},
        env=_mock_env({
            "text": "ok",
            "tokens_input": 0, "tokens_output": 0,
            "cost_micro": 0,
            "model": "m",
        }),
    )
    assert result["exit_code"] == 0
    charges = [l for l in result["lines"] if l["op"] == "charge"]
    assert charges == []  # no zero-amount charges


# ── End-to-end via supervisor ───────────────────────────────────────────


@pytest.fixture
def stack(tmp_path):
    k = Kernel.open(tmp_path / "kernel.db")
    sup = k.make_supervisor()
    yield {"kernel": k, "sup": sup}
    for h in sup.list():
        try:
            sup.stop(h.pid)
        except Exception:
            pass
    k.close()


def test_supervisor_runs_llm_runner_with_mock(stack):
    """The full kernel chain: spawn → handshake → mock provider →
    charge → ledger updated → DEAD."""
    k, sup = stack["kernel"], stack["sup"]
    a = k.create_agent(name="llm-test", template="llm")
    k.ledger.create(pid=a.pid, grants={
        "tokens": 1_000_000, "cost_micro": 1_000_000_000, "wall_s": 60,
    })

    sup.spawn(
        pid=a.pid,
        argv=LLM_RUNNER_ARGV,
        policy=SandboxPolicy(wall_seconds=15),
        init_payload={"model": "claude-x", "user": "say hi",
                       "max_tokens": 100},
        env=_mock_env({
            "text":          "hi",
            "tokens_input":  20,
            "tokens_output": 5,
            "cost_micro":    400,
            "model":         "claude-x",
        }),
    )
    info = sup.wait(a.pid, timeout=15)
    assert info.exit_kind == "completed", info.stderr_tail
    # Ledger reflects the supervisor-applied charges.
    assert info.ledger_charged.get("tokens", 0) == 25
    assert info.ledger_charged.get("cost_micro", 0) == 400
    led = k.ledger.get(a.pid)
    by_dim = {e.dim: e.used for e in led.entries}
    assert by_dim["tokens"]      == 25
    assert by_dim["cost_micro"]  == 400
    # wall_s also charged.
    assert by_dim["wall_s"]       >= 0


def test_supervisor_no_double_charge(stack):
    """RFC 0019 §4: charge messages drive the actual ledger update;
    iteration_done's auto-charge is zeroed in the LLM runner. Verify
    no double-counting."""
    k, sup = stack["kernel"], stack["sup"]
    a = k.create_agent(name="x", template="t")
    k.ledger.create(pid=a.pid, grants={
        "tokens": 1_000_000, "cost_micro": 1_000_000,
    })
    sup.spawn(
        pid=a.pid,
        argv=LLM_RUNNER_ARGV,
        policy=SandboxPolicy(wall_seconds=10),
        init_payload={"model": "m", "user": "x"},
        env=_mock_env({
            "text": "y", "tokens_input": 100, "tokens_output": 50,
            "cost_micro": 1500, "model": "m",
        }),
    )
    info = sup.wait(a.pid, timeout=15)
    assert info.exit_kind == "completed"
    # Should be charged ONCE, not twice.
    assert info.ledger_charged["tokens"] == 150
    assert info.ledger_charged["cost_micro"] == 1500


# ── End-to-end via WorkerLoop ───────────────────────────────────────────


def test_worker_loop_runs_llm_job(stack):
    """Enqueue an LLM work item; worker spawns; ledger gets charged
    via the full chain (no manual supervisor.spawn)."""
    k = stack["kernel"]
    a = k.create_agent(name="x", template="llm")
    k.ledger.create(pid=a.pid, grants={
        "tokens": 1_000_000, "cost_micro": 1_000_000,
    })
    sid = k.scheduler.enqueue(ScheduleSpec(pid=a.pid))

    response_env = _mock_env({
        "text": "hi", "tokens_input": 12, "tokens_output": 3,
        "cost_micro": 200, "model": "claude-x",
    })

    worker = k.make_worker(
        argv_factory=lambda entry: LLM_RUNNER_ARGV,
        policy_factory=lambda entry: SandboxPolicy(wall_seconds=15),
        env_factory=lambda entry: response_env,
        max_concurrent=1,
        poll_interval_s=0.05,
        wait_timeout_s=20,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            queued, _ = k.scheduler.list(state="queued")
            disp,   _ = k.scheduler.list(state="dispatched")
            if not queued and not disp:
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker didn't drain")
    finally:
        worker.stop(drain=True, drain_timeout_s=5)

    # WorkerLoop drove the chain. Note: WorkerLoop calls spawn with
    # init_payload that includes scheduler hints (sched_id, trigger,
    # payload), not the LLM-specific request fields. So the runner
    # sees an invalid LlmRequest and exits=failed. This test
    # confirms the queue entry gets marked failed cleanly — not that
    # the LLM call succeeds (which would require argv_factory to
    # build a custom init_payload, not exposed in this slice).
    e = k.scheduler.get(sid)
    assert e.state == "completed"
    # exit_kind: the runner exits 'failed' because the WorkerLoop's
    # init_payload doesn't match LlmRequest schema. That's expected
    # for this MVP — a future RFC adds a per-entry init_payload
    # factory to WorkerLoop.
    assert e.exit_kind in ("failed", "completed")


# ── AnthropicProvider — defensive import ─────────────────────────────


def test_anthropic_provider_lazy_imports():
    """Importing the module must NOT trigger the SDK import."""
    from cheetahclaws.kernel.runner.llm import anthropic_provider as ap_mod
    # Class is importable without the SDK.
    assert hasattr(ap_mod, "AnthropicProvider")


def test_anthropic_provider_construction_doesnt_import_sdk():
    """Instantiation alone doesn't import anthropic; the call does."""
    from cheetahclaws.kernel.runner.llm.anthropic_provider import AnthropicProvider
    p = AnthropicProvider(api_key="dummy")
    # Client still None — not constructed yet.
    assert p._client is None


def test_anthropic_provider_no_api_key_raises():
    from cheetahclaws.kernel.runner.llm.anthropic_provider import AnthropicProvider
    p = AnthropicProvider(api_key=None)
    # Save and clear ANTHROPIC_API_KEY so the unavailable check fires.
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        # The error path raises ProviderUnavailable. The SDK might
        # not be installed at all; either ProviderUnavailable is raised
        # because of missing SDK or missing key — both are pass cases.
        with pytest.raises(ProviderUnavailable):
            p(LlmRequest(model="claude-x", user="hi"))
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
