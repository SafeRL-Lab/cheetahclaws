"""
Multi-provider support for CheetahClaws.

Supported providers:
  anthropic  — Claude (claude-opus-4-6, claude-sonnet-4-6, ...)
  openai     — GPT (gpt-4o, o3-mini, ...)
  gemini     — Google Gemini (gemini-2.0-flash, gemini-1.5-pro, ...)
  kimi       — Moonshot AI (moonshot-v1-8k/32k/128k)
  qwen       — Alibaba DashScope (qwen-max, qwen-plus, ...)
  zhipu      — Zhipu GLM (glm-4, glm-4-plus, ...)
  deepseek   — DeepSeek (deepseek-v4-flash, deepseek-v4-pro, deepseek-chat, deepseek-reasoner)
  minimax    — MiniMax (MiniMax-Text-01, abab6.5s-chat, ...)
  ollama     — Local Ollama (llama3.3, qwen2.5-coder, ...)
  lmstudio   — Local LM Studio (any loaded model)
  custom     — Any OpenAI-compatible endpoint

Model string formats:
  "claude-opus-4-6"          auto-detected → anthropic
  "gpt-4o"                   auto-detected → openai
  "ollama/qwen2.5-coder"     explicit provider prefix
  "custom/my-model"          uses CUSTOM_BASE_URL from config
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
from typing import Generator

# ── Provider registry ──────────────────────────────────────────────────────

PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "type":       "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "context_limit": 200000,
        "models": [
            "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
            "claude-opus-4-5", "claude-sonnet-4-5",
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        ],
    },
    "openai": {
        "type":       "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url":   "https://api.openai.com/v1",
        "context_limit": 128000,
        "max_completion_tokens": 16384,  # safe cap across gpt-4o/gpt-4.1 family
        "models": [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4.1", "gpt-4.1-mini",
            "gpt-5", "gpt-5-nano", "gpt-5-mini",
            "o4-mini", "o3", "o3-mini", "o1", "o1-mini",
        ],
    },
    "gemini": {
        "type":       "openai",
        "api_key_env": "GEMINI_API_KEY",
        "base_url":   "https://generativelanguage.googleapis.com/v1beta/openai/",
        "context_limit": 1000000,
        "models": [
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.0-flash", "gemini-2.0-flash-lite",
            "gemini-1.5-pro", "gemini-1.5-flash",
        ],
    },
    "kimi": {
        "type":       "openai",
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url":   "https://api.moonshot.cn/v1",
        "context_limit": 128000,
        "models": [
            "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
            "kimi-latest",
        ],
    },
    "qwen": {
        "type":       "openai",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url":   "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_limit": 1000000,
        "models": [
            "qwen-max", "qwen-plus", "qwen-turbo", "qwen-long",
            "qwen2.5-72b-instruct", "qwen2.5-coder-32b-instruct",
            "qwq-32b",
        ],
    },
    "zhipu": {
        "type":       "openai",
        "api_key_env": "ZHIPU_API_KEY",
        "base_url":   "https://open.bigmodel.cn/api/paas/v4/",
        "context_limit": 128000,
        "models": [
            "glm-4-plus", "glm-4", "glm-4-flash", "glm-4-air",
            "glm-z1-flash",
        ],
    },
    "deepseek": {
        "type":       "openai",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url":   "https://api.deepseek.com/v1",
        "context_limit": 128000,
        "models": [
            "deepseek-v4-pro", "deepseek-v4-flash",
            "deepseek-chat", "deepseek-coder", "deepseek-reasoner",
        ],
    },
    "minimax": {
        "type":       "openai",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url":   "https://api.minimaxi.chat/v1",
        "context_limit": 1000000,
        "models": [
            "MiniMax-Text-01", "MiniMax-VL-01",
            "abab6.5s-chat", "abab6.5-chat",
            "abab5.5s-chat", "abab5.5-chat",
        ],
    },
    "ollama": {
        "type":       "ollama",
        "api_key_env": None,
        "base_url":   "http://localhost:11434",
        "api_key":    "ollama",
        "context_limit": 128000,
        "models": [
            "llama3.3", "llama3.2", "phi4", "mistral", "mixtral",
            "qwen2.5-coder", "deepseek-r1", "gemma3",
        ],
    },
    "lmstudio": {
        "type":       "openai",
        "api_key_env": None,
        "base_url":   "http://localhost:1234/v1",
        "api_key":    "lm-studio",
        "context_limit": 128000,
        "models": [],   # dynamic, depends on loaded model
    },
    "custom": {
        "type":       "openai",
        "api_key_env": "CUSTOM_API_KEY",
        "base_url":   None,   # read from config["custom_base_url"]
        "context_limit": 128000,
        "models": [],
    },
    # NVIDIA NIM (build.nvidia.com) — free tier, no payment info required.
    # OpenAI-compatible. Get a key at https://build.nvidia.com (free signup).
    # Model IDs use the upstream <vendor>/<name> form, so callers must use
    # the double-prefixed `nim/<vendor>/<model>` invocation, e.g.
    #   cheetahclaws --model nim/meta/llama-3.3-70b-instruct
    # The catalog evolves — this list is a curated 2026-vintage starting set;
    # any model the catalog still serves works regardless of presence here.
    "nim": {
        "type":       "openai",
        "api_key_env": "NVIDIA_API_KEY",
        "base_url":   "https://integrate.api.nvidia.com/v1",
        "context_limit": 128000,
        "max_completion_tokens": 16384,
        "models": [
            "deepseek-ai/deepseek-r1",
            "deepseek-ai/deepseek-v3.1",
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-405b-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "mistralai/mixtral-8x22b-instruct-v0.1",
            "qwen/qwen2.5-72b-instruct",
            "qwen/qwen2.5-coder-32b-instruct",
            "microsoft/phi-3-medium-128k-instruct",
            "google/gemma-2-27b-it",
        ],
    },
}

# Cost per million tokens (approximate, fallback to 0 for unknown)
COSTS = {
    "claude-opus-4-6":          (15.0, 75.0),
    "claude-sonnet-4-6":        (3.0,  15.0),
    "claude-haiku-4-5-20251001": (0.8,  4.0),
    "gpt-4o":                   (2.5,  10.0),
    "gpt-4o-mini":              (0.15,  0.6),
    "o3-mini":                  (1.1,   4.4),
    "gemini-2.0-flash":         (0.075, 0.3),
    "gemini-1.5-pro":           (1.25,  5.0),
    "gemini-2.5-pro-preview-03-25": (1.25, 10.0),
    "moonshot-v1-8k":           (1.0,   3.0),
    "moonshot-v1-32k":          (2.4,   7.0),
    "moonshot-v1-128k":         (8.0,  24.0),
    "qwen-max":                 (2.4,   9.6),
    "qwen-plus":                (0.4,   1.2),
    "deepseek-chat":            (0.27,  1.1),
    "deepseek-reasoner":        (0.55,  2.19),
    # DeepSeek v4 — pricing placeholder (matches v3 tiers; verify before billing UX)
    "deepseek-v4-flash":        (0.27,  1.1),
    "deepseek-v4-pro":          (0.55,  2.19),
    "glm-4-plus":               (0.7,   0.7),
    "MiniMax-Text-01":          (0.7,   2.1),
    "abab6.5s-chat":            (0.1,   0.1),
    "abab6.5-chat":             (0.5,   0.5),
    # NVIDIA NIM — free tier (no per-token billing on build.nvidia.com).
    # Listed for completeness so cost displays show $0 instead of "unknown".
    # If NVIDIA later monetises a tier, update only the affected rows.
    "deepseek-ai/deepseek-r1":                   (0.0, 0.0),
    "deepseek-ai/deepseek-v3.1":                 (0.0, 0.0),
    "meta/llama-3.3-70b-instruct":               (0.0, 0.0),
    "meta/llama-3.1-405b-instruct":              (0.0, 0.0),
    "nvidia/llama-3.1-nemotron-70b-instruct":    (0.0, 0.0),
    "mistralai/mixtral-8x22b-instruct-v0.1":     (0.0, 0.0),
    "qwen/qwen2.5-72b-instruct":                 (0.0, 0.0),
    "qwen/qwen2.5-coder-32b-instruct":           (0.0, 0.0),
    "microsoft/phi-3-medium-128k-instruct":      (0.0, 0.0),
    "google/gemma-2-27b-it":                     (0.0, 0.0),
}

# Auto-detection: prefix → provider name
_PREFIXES = [
    ("claude-",       "anthropic"),
    ("gpt-",          "openai"),
    ("o1",            "openai"),
    ("o3",            "openai"),
    ("gemini-",       "gemini"),
    ("moonshot-",     "kimi"),
    ("kimi-",         "kimi"),
    ("qwen",          "qwen"),  # qwen-max, qwen2.5-...
    ("qwq-",          "qwen"),
    ("glm-",          "zhipu"),
    ("deepseek-",     "deepseek"),
    ("minimax-",      "minimax"),
    ("MiniMax-",      "minimax"),
    ("abab",          "minimax"),
    ("llama",         "ollama"),
    ("mistral",       "ollama"),
    ("phi",           "ollama"),
    ("gemma",         "ollama"),
]


def detect_provider(model: str) -> str:
    """Return provider name for a model string.
    Supports 'provider/model' explicit format, or auto-detect by prefix."""
    if "/" in model:
        return model.split("/", 1)[0]
    for prefix, pname in _PREFIXES:
        if model.lower().startswith(prefix):
            return pname
    return "openai"   # fallback


def bare_model(model: str) -> str:
    """Strip 'provider/' prefix if present."""
    return model.split("/", 1)[1] if "/" in model else model


def nim_next_model(current: str) -> str | None:
    """Return the next NIM model after `current` in the curated chain, or None.

    Used by the agent loop's 429 cascade: when one NIM model is rate-limited,
    try the next one. Wraps around to the first model after the last so a
    long-running session can keep cycling, but the agent's per-turn fallback
    counter caps total attempts to prevent a busy loop when the whole tier
    is throttled.

    Accepts either bare ('meta/llama-3.3-70b-instruct') or fully prefixed
    ('nim/meta/llama-3.3-70b-instruct') input; returns the same form as input.
    """
    had_nim_prefix = current.startswith("nim/")
    bare = current[4:] if had_nim_prefix else current
    chain = PROVIDERS["nim"]["models"]
    if bare not in chain:
        # Unknown model — start from the head so user-supplied IDs at least
        # cycle into a known-good chain on the first 429.
        nxt = chain[0]
    else:
        idx = chain.index(bare)
        nxt = chain[(idx + 1) % len(chain)]
    return f"nim/{nxt}" if had_nim_prefix else nxt


# ── Auto max_tokens cap ────────────────────────────────────────────────────

# Per-model output limits for well-known models (output tokens, not context)
_MODEL_OUTPUT_LIMITS: dict[str, int] = {
    # Anthropic
    "claude-opus-4-6":            16000,
    "claude-sonnet-4-6":          16000,
    "claude-haiku-4-5-20251001":  8192,
    "claude-opus-4-5":            16000,
    "claude-sonnet-4-5":          16000,
    "claude-3-5-sonnet-20241022": 8192,
    "claude-3-5-haiku-20241022":  8192,
    # OpenAI
    "gpt-4o":      16384,
    "gpt-4o-mini": 16384,
    "gpt-4.1":     32768,
    "gpt-4.1-mini":32768,
    "gpt-5":       32768,
    "o1":          32768,
    "o3":          100000,
    "o4-mini":     100000,
    # Gemini
    "gemini-2.5-pro-preview-03-25": 65536,
    "gemini-2.0-flash":             8192,
    "gemini-1.5-pro":               8192,
    # DeepSeek
    "deepseek-chat":       8192,
    "deepseek-reasoner":   32768,
    "deepseek-v4-flash":   32768,
    "deepseek-v4-pro":     32768,
}

# Per-model TOTAL context window (input+output), separate from output limit above.
# Used by get_model_context_window to drive both compaction trigger and dynamic
# max_tokens cap. Keys are bare model IDs; lookup also tries lowercase prefix
# match for variants like "qwen2.5-72b-instruct-vllm-build".
_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Qwen 2.5 family
    "qwen2.5-7b":                  32768,
    "qwen2.5-14b":                 32768,
    "qwen2.5-32b":                 32768,
    "qwen2.5-72b":                 32768,
    "qwen2.5-72b-instruct":        32768,
    "qwen2.5-coder":               32768,
    "qwen2.5-coder-7b":            32768,
    "qwen2.5-coder-32b":           32768,
    "qwen2.5-coder-32b-instruct":  32768,
    # Qwen 3 family (most variants 32k by default)
    "qwen3-7b":                    32768,
    "qwen3-32b":                   32768,
    "qwen3-72b":                   32768,
    # QwQ
    "qwq-32b":                     32768,
    # Llama family
    "llama3.3":                    131072,
    "llama3.2":                    131072,
    "llama3.1":                    131072,
    "llama-3.3-70b-instruct":      131072,
    "llama-3.1-405b-instruct":     131072,
    # Mistral / Mixtral
    "mistral":                     32768,
    "mistral-7b":                  32768,
    "mixtral":                     32768,
    "mixtral-8x7b":                32768,
    "mixtral-8x22b":               65536,
    # Phi
    "phi-3-medium-128k-instruct":  131072,
    "phi4":                        16384,
    # Gemma
    "gemma-2-27b-it":              8192,
    "gemma3":                      8192,
    "gemma4":                      8192,
    # DeepSeek local variants
    "deepseek-r1":                 65536,
    "deepseek-coder-v2":           128000,
    # CodeLlama
    "codellama":                   16384,
    # Llava (vision)
    "llava":                       4096,
}

# Cache: base_url → {model_id → max_model_len}
_custom_ctx_cache: dict[str, dict[str, int]] = {}


def _fetch_custom_model_limit(base_url: str, model: str, api_key: str) -> int | None:
    """Query /v1/models on a custom (vLLM/etc.) endpoint for the TOTAL context
    window (max_model_len). Returns None on any failure. Results cached per
    base_url. Also backfills PROVIDERS['custom']['context_limit'] in-memory the
    first time it succeeds, so compaction.get_context_limit() — which doesn't
    have direct access to base_url — sees the real limit instead of the stale
    128000 default.
    """
    cache = _custom_ctx_cache.setdefault(base_url, {})
    if model in cache:
        return cache[model]
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key or 'dummy'}"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        for entry in data.get("data", []):
            mid = entry.get("id", "")
            limit = entry.get("max_model_len") or entry.get("context_window")
            if limit:
                cache[mid] = int(limit)
        result = cache.get(model)
        # Backfill provider-level default with the most conservative value seen.
        # Compaction reads PROVIDERS[provider]['context_limit'] without knowing
        # base_url, so this makes the threshold see the real limit.
        if cache:
            smallest = min(v for v in cache.values() if v)
            if smallest:
                PROVIDERS.setdefault("custom", {})["context_limit"] = smallest
        return result
    except Exception:
        return None


def get_model_context_window(provider: str, model: str,
                              base_url: str = "", api_key: str = "") -> int:
    """Return the TOTAL context window (input + output) for a model.

    Single source of truth for compaction trigger and dynamic max_tokens cap.

    Priority:
      1. Per-model registry (_MODEL_CONTEXT_LIMITS)
      2. Custom provider with base_url: live /v1/models query (cached)
      3. Provider-level PROVIDERS[provider]['context_limit']
      4. Fallback 128000
    """
    bare = bare_model(model)
    if bare in _MODEL_CONTEXT_LIMITS:
        return _MODEL_CONTEXT_LIMITS[bare]
    bare_lc = bare.lower()
    for k, v in _MODEL_CONTEXT_LIMITS.items():
        if bare_lc.startswith(k.lower()):
            return v
    if provider == "custom" and base_url:
        live = _fetch_custom_model_limit(base_url, model, api_key)
        if live:
            return live
    prov_ctx = PROVIDERS.get(provider, {}).get("context_limit")
    if prov_ctx:
        return prov_ctx
    return 128000


def dynamic_cap_max_tokens(
    messages: list,
    system,
    tool_schemas,
    ctx_window: int,
    configured: int,
    safety_margin: int = 1024,
) -> int:
    """Cap max_tokens so input + output fits within the model's context window.

    Estimates current prompt size (messages + system + tool schemas) using the
    same chars/2.8 heuristic as compaction.estimate_tokens, then returns
    min(configured, ctx_window - input_estimate - safety_margin).

    Floors at 256 — if even that is infeasible, the caller's compaction layer
    should already have fired; returning a tiny floor lets the API call surface
    a clear error rather than silently sending an oversized request.
    """
    import compaction  # local import: compaction imports providers, avoid cycle
    msg_tok = compaction.estimate_tokens(messages or [])
    sys_tok = 0
    if isinstance(system, str):
        sys_tok = int(len(system) / 2.8 * 1.1)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                txt = block.get("text") or block.get("content") or ""
                if isinstance(txt, str):
                    sys_tok += int(len(txt) / 2.8 * 1.1)
            elif isinstance(block, str):
                sys_tok += int(len(block) / 2.8 * 1.1)
    tool_tok = 0
    if tool_schemas:
        try:
            tool_tok = int(len(json.dumps(tool_schemas)) / 2.8 * 1.1)
        except Exception:
            tool_tok = 0
    input_est = msg_tok + sys_tok + tool_tok
    headroom = ctx_window - input_est - safety_margin
    if headroom < 256:
        return 256
    return min(configured, headroom)


def resolve_max_tokens(config: dict, provider: str, model: str,
                       base_url: str = "", api_key: str = "") -> int | None:
    """Return the effective max_tokens to use, auto-capping to the model's limit.

    Priority:
      1. Per-model hard limit from _MODEL_OUTPUT_LIMITS (known models)
      2. For 'custom' provider: query /v1/models for max_model_len
      3. Provider-level context_limit from PROVIDERS registry
      4. User's configured value unchanged (no cap available)

    Always respects the user's configured value as an upper bound — never
    increases it beyond what was requested.
    """
    requested = config.get("max_tokens")
    if not requested:
        return None  # let the caller use its own default

    # 1. Known per-model limit
    bare = bare_model(model)
    known = _MODEL_OUTPUT_LIMITS.get(bare)
    if known:
        return min(requested, known)

    # 2. Custom endpoint: query /v1/models
    if provider == "custom" and base_url:
        ctx_limit = _fetch_custom_model_limit(base_url, model, api_key)
        if ctx_limit:
            # Reserve 256 tokens so max_tokens never equals max_model_len exactly
            # (vLLM rejects max_tokens == max_model_len in some versions)
            safe = max(256, ctx_limit - 256)
            return min(requested, safe)

    # 3. Provider-level context limit (conservative: cap output to 1/2 context)
    prov_ctx = PROVIDERS.get(provider, {}).get("context_limit")
    if prov_ctx:
        cap = prov_ctx // 2
        return min(requested, cap)

    return requested


def get_api_key(provider_name: str, config: dict) -> str:
    prov = PROVIDERS.get(provider_name, {})
    # 1. Check config dict (e.g. config["kimi_api_key"])
    cfg_key = config.get(f"{provider_name}_api_key", "")
    if cfg_key:
        return cfg_key
    # 2. Check env var
    env_var = prov.get("api_key_env")
    if env_var:
        import os
        return os.environ.get(env_var, "")
    # 3. Hardcoded (for local providers)
    return prov.get("api_key", "")


def calc_cost(model: str, in_tok: int, out_tok: int) -> float:
    ic, oc = COSTS.get(bare_model(model), (0.0, 0.0))
    return (in_tok * ic + out_tok * oc) / 1_000_000


# ── Native tool-call format interceptors ──────────────────────────────────
#
# Some models (Gemma 3/4, Mistral, …) emit their NATIVE tool-call format
# even when the OpenAI-compatible API specifies tools via JSON schemas.
# vLLM can parse some of these via `--tool-call-parser <name>`, but only
# `hermes`, `mistral`, `llama3_json`, `granite`, `pythonic`, `phi4_mini_json`,
# and `deepseek_v3` are supported. Gemma has no parser.
#
# When the parser doesn't recognise the format, raw markers like
# `<|tool_call>call:Foo{"x":1}<tool_call|>` end up in `delta.content` and
# stream straight to the user as garbage text — and the model's intended
# tool call never fires.
#
# The interceptor below detects native tool-call markers in the streamed
# text, switches into "buffer" mode (stops yielding TextChunks), and at
# end-of-stream parses the buffered content into proper tool_calls. The
# user sees a brief pause instead of malformed XML-ish gibberish.
#
# Add a new format: extend `_NATIVE_TOOL_OPENERS` and write a parser
# branch in `_extract_native_tool_calls`.

_NATIVE_TOOL_OPENERS = (
    "<|tool_call|>",   # Gemma official
    "<|tool_call>",    # Gemma 4 variant seen in the wild (asymmetric)
    "<tool_call>",     # Hermes/Qwen (parsed by vLLM, but covered as fallback)
    "[TOOL_CALLS]",    # Mistral
)

_GEMMA_QUOTE_TOKEN_FIXES = (
    ("<|\"|>", '"'),
    ("<|'|>", "'"),
)

import re as _re_native
_NATIVE_FORMAT_V1 = _re_native.compile(
    r"<\|?tool_call\|?>\s*(\{.*?\})\s*<\|?(?:end_)?(?:/)?tool_call\|?>",
    _re_native.DOTALL,
)
# Format 2: <|tool_call>call:NAME{json}<tool_call|>
_NATIVE_FORMAT_V2 = _re_native.compile(
    r"<\|?tool_call\|?>\s*call:\s*(\w+)\s*(\{.*?\})\s*<\|?(?:end_)?(?:/)?tool_call\|?>",
    _re_native.DOTALL,
)
_NATIVE_FORMAT_MISTRAL = _re_native.compile(
    r"\[TOOL_CALLS\]\s*(\[.*?\])",
    _re_native.DOTALL,
)


def _find_native_tool_marker(text: str) -> int | None:
    """Return earliest index of a native tool-call opener, or None."""
    earliest = None
    for opener in _NATIVE_TOOL_OPENERS:
        idx = text.find(opener)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    return earliest


def _extract_native_tool_calls(buf: str) -> list[dict]:
    """Parse buffered text into a list of {id, name, input} tool-call dicts.

    Tries multiple formats in order. Returns [] if none matched.
    """
    if not buf:
        return []

    # Normalise quote-escapes Gemma sometimes emits inside its native format
    for tok, repl in _GEMMA_QUOTE_TOKEN_FIXES:
        buf = buf.replace(tok, repl)

    out: list[dict] = []

    # Format 2 first (more specific — has explicit name)
    for i, m in enumerate(_NATIVE_FORMAT_V2.finditer(buf)):
        name, body = m.group(1), m.group(2)
        try:
            args = json.loads(body)
            if not isinstance(args, dict):
                args = {"_raw": body}
        except json.JSONDecodeError:
            args = {"_raw": body}
        out.append({"id": f"native_call_{len(out)}", "name": name, "input": args})

    # Format 1: JSON envelope with `name` + `arguments`
    if not out:
        for m in _NATIVE_FORMAT_V1.finditer(buf):
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, dict):
                    name = parsed.get("name") or parsed.get("function") or ""
                    args = parsed.get("arguments") or parsed.get("args") or {}
                    if name:
                        if not isinstance(args, dict):
                            args = {"_raw": str(args)}
                        out.append({
                            "id": f"native_call_{len(out)}",
                            "name": name, "input": args,
                        })
            except json.JSONDecodeError:
                continue

    # Mistral [TOOL_CALLS] format
    if not out:
        for m in _NATIVE_FORMAT_MISTRAL.finditer(buf):
            try:
                arr = json.loads(m.group(1))
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, dict):
                            name = item.get("name") or ""
                            args = item.get("arguments") or {}
                            if name:
                                if not isinstance(args, dict):
                                    args = {"_raw": str(args)}
                                out.append({
                                    "id": f"native_call_{len(out)}",
                                    "name": name, "input": args,
                                })
            except json.JSONDecodeError:
                continue

    return out


# Format 3: Gemma 4 channel-tagged tool intent — surfaced in streamed
# text when vLLM's hermes parser ate the `<|tool_call|>` opener but
# left fragments behind. Patterns we've seen in the wild:
#   `<|channel|>commentary to=WebSearch <|message|>{"query":"x"}<|im_end|>`
#   `<|channel|>commentary tool=WebSearch <|message|>{...}<|im_end|>`
#   `<|channel|>thought<|channel|> ... call:WebSearch{"query":"x"}`
_GEMMA_CHANNEL_RE = _re_native.compile(
    r"<\|?channel\|?>\s*\w+\s+(?:to|tool|name)\s*=\s*(\w+)\s*"
    r"<\|?message\|?>\s*(\{.*?\})",
    _re_native.DOTALL,
)
_GEMMA_LOOSE_CALL_RE = _re_native.compile(
    r"\bcall\s*:\s*(\w+)\s*(\{.*?\})",
    _re_native.DOTALL,
)
# Asymmetric Gemma 4 form without `call:` prefix:
#   `<|tool_call>WebSearch{"query":"x"}<tool_call|>`
_GEMMA_INLINE_NAME_RE = _re_native.compile(
    r"<\|?tool_call\|?>\s*(\w+)\s*(\{.*?\})\s*<\|?(?:end_)?(?:/)?tool_call\|?>",
    _re_native.DOTALL,
)


def _recover_args_from_text(text: str, tool_name: str) -> dict | None:
    """Last-ditch recovery: when vLLM emits a tool_call with name but
    empty arguments, scan the surrounding streamed text for the
    matching Gemma-style channel/message or `call:NAME{json}` pair
    and return parsed args. Returns None if nothing recoverable.

    Used only when the primary parse path produced empty `input`.
    """
    if not text or not tool_name:
        return None

    for tok, repl in _GEMMA_QUOTE_TOKEN_FIXES:
        text = text.replace(tok, repl)

    # Try channel-tagged form first.
    for m in _GEMMA_CHANNEL_RE.finditer(text):
        if m.group(1) == tool_name:
            try:
                args = json.loads(m.group(2))
                if isinstance(args, dict) and args:
                    return args
            except json.JSONDecodeError:
                continue

    # Try loose `call:NAME{json}` form.
    for m in _GEMMA_LOOSE_CALL_RE.finditer(text):
        if m.group(1) == tool_name:
            try:
                args = json.loads(m.group(2))
                if isinstance(args, dict) and args:
                    return args
            except json.JSONDecodeError:
                continue

    # Try asymmetric inline-name form: `<|tool_call>NAME{json}<tool_call|>`.
    for m in _GEMMA_INLINE_NAME_RE.finditer(text):
        if m.group(1) == tool_name:
            try:
                args = json.loads(m.group(2))
                if isinstance(args, dict) and args:
                    return args
            except json.JSONDecodeError:
                continue

    # Try the native-format extractors (handles `<|tool_call|>...`).
    native = _extract_native_tool_calls(text)
    for nc in native:
        if nc.get("name") == tool_name and nc.get("input"):
            inp = nc["input"]
            if isinstance(inp, dict) and inp and "_raw" not in inp:
                return inp
    return None


# ── Tool schema conversion ─────────────────────────────────────────────────

def tools_to_openai(tool_schemas: list) -> list:
    """Convert Anthropic-style tool schemas to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"],
            },
        }
        for t in tool_schemas
    ]


# ── Message format conversion ──────────────────────────────────────────────
#
# Internal "neutral" message format:
#   {"role": "user",      "content": "text"}
#   {"role": "assistant", "content": "text", "tool_calls": [
#       {"id": "...", "name": "...", "input": {...}}
#   ]}
#   {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}

def messages_to_anthropic(messages: list) -> list:
    """Convert neutral messages → Anthropic API format."""
    result = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m["role"]

        if role == "user":
            result.append({"role": "user", "content": m["content"]})
            i += 1

        elif role == "assistant":
            blocks = []
            text = m.get("content", "")
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls", []):
                blocks.append({
                    "type":  "tool_use",
                    "id":    tc["id"],
                    "name":  tc["name"],
                    "input": tc["input"],
                })
            result.append({"role": "assistant", "content": blocks})
            i += 1

        elif role == "tool":
            # Collect consecutive tool results into one user message
            tool_blocks = []
            while i < len(messages) and messages[i]["role"] == "tool":
                t = messages[i]
                tool_blocks.append({
                    "type":        "tool_result",
                    "tool_use_id": t["tool_call_id"],
                    "content":     t["content"],
                })
                i += 1
            result.append({"role": "user", "content": tool_blocks})

        else:
            i += 1

    return result


def messages_to_openai(messages: list, ollama_native_images: bool = False) -> list:
    """Convert neutral messages → OpenAI API format.

    Args:
        ollama_native_images: if True, forward the 'images' list in user messages
                              using Ollama's /api/chat native format (a bare base64
                              list on the message object).  Set this only when
                              targeting the Ollama backend.
                              If False (default), images are converted to the
                              OpenAI/Gemini multipart ``image_url`` format so they
                              reach vision-capable cloud models correctly.
    """
    result = []
    for m in messages:
        role = m["role"]

        if role == "user":
            content = m["content"]
            if ollama_native_images and m.get("images"):
                # Ollama /api/chat native: bare base64 list on the message
                msg_out = {"role": "user", "content": content, "images": m["images"]}
            elif not ollama_native_images and m.get("images"):
                # OpenAI / Gemini multipart vision format
                parts = [{"type": "text", "text": content}]
                for img_b64 in m["images"]:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
                msg_out = {"role": "user", "content": parts}
            else:
                msg_out = {"role": "user", "content": content}
            result.append(msg_out)

        elif role == "assistant":
            # Use "" rather than None for the all-tool-calls case: Ollama's
            # OpenAI-compat endpoint rejects content: null with
            # `invalid message content type: <nil>` (issue #71). Empty string
            # is accepted by every OpenAI-compat backend we target.
            msg: dict = {"role": "assistant", "content": m.get("content") or ""}
            tcs = m.get("tool_calls", [])
            if tcs:
                msg["tool_calls"] = []
                for tc in tcs:
                    tc_msg = {
                        "id":   tc["id"],
                        "type": "function",
                        "function": {
                            "name":      tc["name"],
                            "arguments": json.dumps(tc["input"], ensure_ascii=False),
                        },
                    }
                    # Pass through provider-specific fields (e.g. Gemini thought_signature)
                    if tc.get("extra_content"):
                        tc_msg["extra_content"] = tc["extra_content"]
                    msg["tool_calls"].append(tc_msg)
                # DeepSeek v4 spec: when an assistant turn carries tool_calls,
                # its `reasoning_content` must be echoed back on subsequent
                # requests.  Benign for other OpenAI-compat providers — they
                # ignore unknown fields.
                rc = m.get("reasoning_content")
                if rc:
                    msg["reasoning_content"] = rc
            result.append(msg)

        elif role == "tool":
            result.append({
                "role":         "tool",
                "tool_call_id": m["tool_call_id"],
                "content":      m["content"],
            })

    return result


# ── Streaming adapters ─────────────────────────────────────────────────────

class TextChunk:
    def __init__(self, text): self.text = text

class ThinkingChunk:
    def __init__(self, text): self.text = text

class AssistantTurn:
    """Completed assistant turn with text + tool_calls.

    ``reasoning_content`` carries model-emitted chain-of-thought surfaced via an
    OpenAI-compat ``delta.reasoning_content`` field (DeepSeek v4, Kimi K2
    Thinking, GLM-4.6, etc.).  DeepSeek v4 requires it to be echoed back when
    the assistant turn contains tool_calls; see ``messages_to_openai``.
    """
    def __init__(self, text, tool_calls, in_tokens, out_tokens,
                 cache_read_tokens=0, cache_write_tokens=0,
                 reasoning_content=""):
        self.text                 = text
        self.tool_calls           = tool_calls   # list of {id, name, input}
        self.in_tokens            = in_tokens
        self.out_tokens           = out_tokens
        self.cache_read_tokens    = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens
        self.reasoning_content    = reasoning_content


def stream_anthropic(
    api_key: str,
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    config: dict,
) -> Generator:
    """Stream from Anthropic API. Yields TextChunk/ThinkingChunk, then AssistantTurn."""
    import anthropic as _ant
    base_url = config.get("anthropic_endpoint") or "https://api.anthropic.com"
    client = _ant.Anthropic(api_key=api_key, base_url=base_url)

    _mt = resolve_max_tokens(config, "anthropic", model) or 8192
    # Per-call dynamic cap: shrink max_tokens when the current prompt is already
    # large, so input + output never exceeds the model's context window.
    _ctx_window = get_model_context_window("anthropic", model)
    _mt = dynamic_cap_max_tokens(messages, system, tool_schemas, _ctx_window, _mt)
    kwargs = {
        "model":      model,
        "max_tokens": _mt,
        "system":     system,
        "messages":   messages_to_anthropic(messages),
        "tools":      tool_schemas,
    }
    if config.get("thinking"):
        kwargs["thinking"] = {
            "type":          "enabled",
            "budget_tokens": config.get("thinking_budget", 10000),
        }

    tool_calls = []
    text       = ""

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "content_block_delta":
                delta = event.delta
                dtype = getattr(delta, "type", None)
                if dtype == "text_delta":
                    text += delta.text
                    yield TextChunk(delta.text)
                elif dtype == "thinking_delta":
                    yield ThinkingChunk(delta.thinking)

        final = stream.get_final_message()
        for block in final.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id":    block.id,
                    "name":  block.name,
                    "input": block.input,
                })

        cache_r, cache_w = _anthropic_cache_tokens(final.usage)
        yield AssistantTurn(
            text, tool_calls,
            final.usage.input_tokens,
            final.usage.output_tokens,
            cache_read_tokens=cache_r,
            cache_write_tokens=cache_w,
        )


def _anthropic_cache_tokens(usage) -> tuple[int, int]:
    """Extract (cache_read, cache_write) token counts from an Anthropic usage object.

    Returns (0, 0) if the fields are missing -- older Anthropic SDKs, non-cached
    calls and most downstream wrappers (e.g. Bedrock over litellm) all fall
    through to this default rather than raising AttributeError.
    """
    read  = getattr(usage, "cache_read_input_tokens", 0) or 0
    write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return int(read), int(write)


def _openai_cached_read_tokens(usage) -> int:
    """Extract the OpenAI-compatible cached read-token count.

    OpenAI-compatible providers surface cache hits as
    `usage.prompt_tokens_details.cached_tokens`; there is no separate
    "cache creation" counter in the OpenAI schema (caching is implicit on
    their side), so the write-side is always 0 for this family of providers.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return int(getattr(details, "cached_tokens", 0) or 0)


def stream_openai_compat(
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    config: dict,
) -> Generator:
    """Stream from any OpenAI-compatible API. Yields TextChunk, then AssistantTurn."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

    oai_messages = [{"role": "system", "content": system}] + messages_to_openai(messages)

    kwargs: dict = {
        "model":    model,
        "messages": oai_messages,
        "stream":   True,
    }

    # Pass num_ctx for known Ollama/LM Studio ports only — avoids matching other local servers (e.g. vLLM on :8000)
    _is_local_ollama = "11434" in base_url
    _is_lmstudio     = "1234" in base_url and ("lmstudio" in base_url or "localhost" in base_url or "127.0.0.1" in base_url)
    if _is_local_ollama or _is_lmstudio:
        prov = detect_provider(model)
        ctx_limit = PROVIDERS.get(prov if prov in ("ollama", "lmstudio") else "ollama", {}).get("context_limit", 128000)
        kwargs["extra_body"] = {"options": {"num_ctx": ctx_limit}}

    if tool_schemas and not config.get("no_tools"):
        kwargs["tools"] = tools_to_openai(tool_schemas)
        # "auto" requires vLLM --enable-auto-tool-choice; omit if server doesn't support it
        if not config.get("disable_tool_choice"):
            kwargs["tool_choice"] = "auto"
    _prov = detect_provider(model)

    # DeepSeek v4: thinking is ON by default and controlled via extra_body.
    # `thinking` is tri-state in DEFAULTS (cc_config.py): None = unset (let
    # provider default stand → ON for v4), True = explicit ON (also default),
    # False = explicit OFF (user toggled via /thinking).  Only the explicit-OFF
    # case injects the disable toggle.  `is False` is intentional: distinguishes
    # explicit False from None.
    if _prov == "deepseek":
        if config.get("thinking") is False:
            kwargs.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
        eff = config.get("reasoning_effort")
        if eff:
            kwargs["reasoning_effort"] = eff
    _effective_mt = resolve_max_tokens(config, _prov, model, base_url, api_key)
    if _effective_mt:
        # Further cap by provider-level max_completion_tokens if present
        prov_cap = PROVIDERS.get(_prov, {}).get("max_completion_tokens")
        val = min(_effective_mt, prov_cap) if prov_cap else _effective_mt
        # Per-call dynamic cap: shrink based on current prompt size so input +
        # output never overflows the real context window. Critical for 32k
        # local models (qwen2.5, mistral) where the static cap alone is not
        # enough — input grows turn-by-turn.
        _ctx_window = get_model_context_window(_prov, model, base_url, api_key)
        # Pass the system prompt as a single-element list so dynamic_cap counts it.
        val = dynamic_cap_max_tokens(messages, system, kwargs.get("tools"), _ctx_window, val)
        # Newer OpenAI models (o1/o3/o4/gpt-5 family) dropped max_tokens in favour of
        # max_completion_tokens.  Use max_completion_tokens for the openai provider so
        # all current and future OpenAI models work without per-model special-casing.
        # All other OpenAI-compatible providers (Ollama, vLLM, Gemini, etc.) still
        # accept max_tokens, so we keep the old key for them.
        if _prov == "openai":
            kwargs["max_completion_tokens"] = val
        else:
            kwargs["max_tokens"] = val

    text            = ""
    reasoning_text  = ""
    tool_buf: dict = {}   # index → {id, name, args_str}
    in_tok = out_tok = 0
    cache_read_tok = cache_write_tok = 0

    # Native tool-call interceptor state — see comments around
    # `_extract_native_tool_calls`. Gemma 4 + vLLM hermes parser is the
    # primary trigger but this catches Mistral [TOOL_CALLS] etc. too.
    native_tool_buffering = False
    native_tool_buffer    = ""

    # Diagnostic: when CC_DEBUG_TOOL_CALLS=1, dump every streamed
    # delta to /tmp/cc_tool_call_debug.log so the user can see what
    # the upstream model server is really sending. Crucial for
    # diagnosing vLLM parser ↔ model-format mismatches.
    _debug_tc = os.environ.get("CC_DEBUG_TOOL_CALLS") == "1"
    _debug_path = "/tmp/cc_tool_call_debug.log"
    if _debug_tc:
        try:
            with open(_debug_path, "a", encoding="utf-8") as _df:
                _df.write(f"\n=== STREAM START model={kwargs.get('model','?')} ts={time.time():.0f} ===\n")
        except Exception:
            pass

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if not chunk.choices:
            # usage-only chunk (some providers send this last)
            if hasattr(chunk, "usage") and chunk.usage:
                in_tok  = chunk.usage.prompt_tokens
                out_tok = chunk.usage.completion_tokens
                cache_read_tok = _openai_cached_read_tokens(chunk.usage) or cache_read_tok
            continue

        choice = chunk.choices[0]
        delta  = choice.delta

        # Some providers (DeepSeek v4, Kimi K2 Thinking, GLM-4.6) stream
        # chain-of-thought on a sibling `reasoning_content` field before any
        # visible content.  Surface it as ThinkingChunk so the UI renders it
        # consistently with Anthropic extended-thinking / Ollama thinking.
        reasoning_delta = getattr(delta, "reasoning_content", None)
        if reasoning_delta:
            reasoning_text += reasoning_delta
            yield ThinkingChunk(reasoning_delta)

        if delta.content:
            new = delta.content
            if not native_tool_buffering:
                # Detect native tool-call markers split across chunks by
                # checking the joined accumulated text.
                joined = text + new
                marker_idx = _find_native_tool_marker(joined)
                if marker_idx is not None and marker_idx >= len(text):
                    split = marker_idx - len(text)
                    if split > 0:
                        text += new[:split]
                        yield TextChunk(new[:split])
                    native_tool_buffering = True
                    native_tool_buffer = new[split:]
                else:
                    text += new
                    yield TextChunk(new)
            else:
                native_tool_buffer += new

        # Diagnostic: dump raw delta when in debug mode.
        if _debug_tc:
            try:
                _dump = {
                    "content": getattr(delta, "content", None),
                    "tool_calls": [
                        {
                            "index": getattr(t, "index", None),
                            "id": getattr(t, "id", None),
                            "name": getattr(getattr(t, "function", None),
                                              "name", None),
                            "arguments": getattr(getattr(t, "function", None),
                                                   "arguments", None),
                        }
                        for t in (getattr(delta, "tool_calls", None) or [])
                    ] or None,
                    "reasoning": getattr(delta, "reasoning_content", None),
                }
                if any(_dump.values()):
                    with open(_debug_path, "a", encoding="utf-8") as _df:
                        _df.write(json.dumps(_dump, ensure_ascii=False,
                                              default=str) + "\n")
            except Exception:
                pass

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_buf:
                    tool_buf[idx] = {"id": "", "name": "", "args": "", "extra_content": None}
                if tc.id:
                    tool_buf[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_buf[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_buf[idx]["args"] += tc.function.arguments
                # Capture extra_content (e.g. Gemini thought_signature)
                extra = getattr(tc, "extra_content", None)
                if extra:
                    tool_buf[idx]["extra_content"] = extra

        # Some providers include usage in the last chunk
        if hasattr(chunk, "usage") and chunk.usage:
            in_tok  = chunk.usage.prompt_tokens  or in_tok
            out_tok = chunk.usage.completion_tokens or out_tok
            cache_read_tok = _openai_cached_read_tokens(chunk.usage) or cache_read_tok

    tool_calls = []
    for idx in sorted(tool_buf):
        v = tool_buf[idx]
        try:
            inp = json.loads(v["args"]) if v["args"] else {}
        except json.JSONDecodeError:
            inp = {"_raw": v["args"]}
        # Recovery: Gemma 4 + vLLM hermes parser sometimes emits a
        # tool_call with the right name but empty arguments because
        # the parser ate the `<|tool_call|>` opener but couldn't
        # locate the JSON body. The args might still be sitting in
        # the streamed text as `<|channel|>commentary tool=NAME
        # <|message|>{...}` or `call:NAME{json}` fragments. Try to
        # recover before we hand an empty dict to a tool that
        # requires args.
        if (not inp or inp == {} or (isinstance(inp, dict)
                                       and "_raw" in inp
                                       and len(inp) == 1)) \
                and v["name"] \
                and (text or native_tool_buffer):
            recovered = _recover_args_from_text(
                text + native_tool_buffer, v["name"],
            )
            if recovered:
                inp = recovered
        tc_entry = {"id": v["id"] or f"call_{idx}", "name": v["name"], "input": inp}
        if v.get("extra_content"):
            tc_entry["extra_content"] = v["extra_content"]
        tool_calls.append(tc_entry)

    # Native tool-call extraction (Gemma 4 etc.) — only kicks in when the
    # vLLM parser failed to extract the call and we buffered the markers
    # client-side. See `_extract_native_tool_calls` for format details.
    if native_tool_buffering:
        native_calls = _extract_native_tool_calls(native_tool_buffer)
        if native_calls:
            tool_calls.extend(native_calls)
        else:
            # Couldn't parse — fall back to yielding the buffer as text so the
            # user sees *something* rather than a silent stall.
            text += native_tool_buffer
            yield TextChunk(native_tool_buffer)

    yield AssistantTurn(
        text, tool_calls, in_tok, out_tok, cache_read_tok, cache_write_tok,
        reasoning_content=reasoning_text,
    )


def stream_ollama(
    base_url: str,
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    config: dict,
) -> Generator:
    # pass_images=True: Ollama /api/chat accepts base64 images natively in the message
    oai_messages = [{"role": "system", "content": system}] + messages_to_openai(messages, ollama_native_images=True)
    
    # Ollama requires tool arguments as dict objects, not strings. OpenAI uses strings.
    for m in oai_messages:
        if m.get("content") is None:
            m["content"] = ""
        if "tool_calls" in m and m["tool_calls"]:
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        import sys
                        print(f"[warn] Failed to parse tool arguments as JSON, leaving as string: {fn['arguments']!r}", file=sys.stderr)
    
    payload = {
        "model": model,
        "messages": oai_messages,
        "stream": True,
        "options": {
            "num_ctx": config.get("context_limit", 128000)
        }
    }
    
    if tool_schemas and not config.get("no_tools"):
        payload["tools"] = tools_to_openai(tool_schemas)

    def _make_request(p):
        return urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=json.dumps(p).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

    req = _make_request(payload)

    text = ""
    tool_buf: dict = {}

    try:
        resp_cm = urllib.request.urlopen(req)
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url}. "
            f"Is it running? Start with: ollama serve\n  ({e})"
        ) from e
    except urllib.error.HTTPError as e:
        if e.code == 500 and "tools" in payload:
            # Model doesn't support tool calling — retry without tools.
            # Close the error response before retrying.
            e.close()
            print(
                f"\n\033[33m[warn] {model} does not support tool calling."
                " Retrying in chat-only mode (no file editing, search, etc.).\033[0m"
            )
            payload.pop("tools", None)
            req = _make_request(payload)
            resp_cm = urllib.request.urlopen(req)
        elif e.code == 404:
            raise ValueError(
                f"Ollama model '{model}' not found. Pull it with: ollama pull {model}\n"
                f"  Or pick from local models: /model ollama"
            ) from e
        else:
            raise

    with resp_cm as resp:
        for line in resp:
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            msg = data.get("message", {})
            
            # Ollama native reasoning models stream thoughts here
            if "thinking" in msg and msg["thinking"]:
                yield ThinkingChunk(msg["thinking"])
                
            if "content" in msg and msg["content"]:
                text += msg["content"]
                yield TextChunk(msg["content"])
            
            # Handle native ollama tools format which mirrors OpenAI
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                idx = len(tool_buf) # Ollama sends complete tool calls, not delta
                tool_buf[idx] = {
                    "id": "call_ollama" + str(idx),
                    "name": fn.get("name", ""),
                    "args": json.dumps(fn.get("arguments", {})),
                    "input": fn.get("arguments", {})
                }

    tool_calls = []
    for idx in sorted(tool_buf):
        v = tool_buf[idx]
        tool_calls.append({"id": v["id"], "name": v["name"], "input": v["input"]})

    # Ollama doesn't return exact token counts via livestream easily until "done",
    # but we can do a rough estimate or 0, cheetahclaws handles zero gracefully
    yield AssistantTurn(text, tool_calls, 0, 0, 0, 0)


def stream(
    model: str,
    system: str,
    messages: list,
    tool_schemas: list,
    config: dict,
) -> Generator:
    """
    Unified streaming entry point.
    Auto-detects provider from model string.
    Yields: TextChunk | ThinkingChunk | AssistantTurn

    Wraps every provider with:
      - Circuit breaker: fails fast when a provider has repeated errors.
      - Structured logging: logs api_call_start / api_call_done / api_call_error.
    """
    import logging_utils as _log
    import circuit_breaker as _cb

    provider_name = detect_provider(model)
    model_name    = bare_model(model)
    prov          = PROVIDERS.get(provider_name, PROVIDERS["openai"])
    api_key       = get_api_key(provider_name, config)
    session_id    = config.get("_session_id", "default")

    # ── Circuit breaker gate ───────────────────────────────────────────────
    breaker = _cb.get_breaker(provider_name, config)
    if not breaker.allow_request():
        raise _cb.CircuitOpenError(
            f"Circuit breaker OPEN for provider '{provider_name}'. "
            f"Cooldown: {breaker.cooldown:.0f}s. Use /circuit reset {provider_name} to force-close."
        )

    _log.debug("api_call_start", session_id=session_id,
               provider=provider_name, model=model_name)

    # ── Build inner generator ──────────────────────────────────────────────
    if prov["type"] == "anthropic":
        inner = stream_anthropic(api_key, model_name, system, messages, tool_schemas, config)
    elif prov["type"] == "ollama":
        import os as _os
        base_url = (
            _os.environ.get("OLLAMA_BASE_URL")
            or config.get("ollama_base_url")
            or prov.get("base_url", "http://localhost:11434")
        )
        inner = stream_ollama(base_url, model_name, system, messages, tool_schemas, config)
    else:
        import os as _os
        if provider_name == "custom":
            base_url = (config.get("custom_base_url")
                        or _os.environ.get("CUSTOM_BASE_URL", ""))
            if not base_url:
                raise ValueError(
                    "custom provider requires a base_url. "
                    "Set CUSTOM_BASE_URL env var or run: /config custom_base_url=http://..."
                )
        else:
            base_url = prov.get("base_url", "https://api.openai.com/v1")
        inner = stream_openai_compat(
            api_key, base_url, model_name, system, messages, tool_schemas, config
        )

    # ── Yield with failure tracking ────────────────────────────────────────
    try:
        for event in inner:
            if isinstance(event, AssistantTurn):
                breaker.record_success()
                _log.info("api_call_done", session_id=session_id,
                          provider=provider_name, model=model_name,
                          in_tokens=event.in_tokens, out_tokens=event.out_tokens,
                          cache_read_tokens=getattr(event, 'cache_read_tokens', 0),
                          cache_write_tokens=getattr(event, 'cache_write_tokens', 0))
            yield event
    except Exception as exc:
        breaker.record_failure()
        _log.error("api_call_error", session_id=session_id,
                   provider=provider_name, model=model_name,
                   error_type=type(exc).__name__, error=str(exc)[:200])
        raise


def list_ollama_models(base_url: str) -> list[str]:
    """Fetch locally available model tags from Ollama server."""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Ollama returns {"models": [{"name": "llama3:latest", ...}, ...]}
            return [m["name"] for m in data.get("models", [])]
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return []
