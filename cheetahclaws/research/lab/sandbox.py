"""research/lab/sandbox.py — minimal subprocess sandbox for experiment code.

⚠️  Security caveat — v0 protections only:

  * Per-experiment workspace directory (isolated from $HOME).
  * Hard wall-clock timeout (kill via SIGKILL on expiry).
  * RLIMIT_CPU + RLIMIT_AS soft caps (Linux/macOS only).
  * subprocess.run with no shell=True; argv passed as list.
  * Working directory pinned to the workspace; the python process inherits
    the user's PATH but cannot escape the cwd via relative paths.

  This is **NOT** a security boundary against deliberately malicious code —
  the LLM-generated code can still:
    * import dangerous stdlib modules (os, ctypes, etc.)
    * make network calls (no firewall)
    * read user-readable files outside the workspace
    * use compute time within RLIMIT_CPU
    * persist files to the workspace

  For a real product (Phase 2.5+), wrap this with Docker + nsjail +
  network-egress restriction. For a single-user-on-trusted-machine v0,
  the protections above keep an honest LLM from accidentally hosing the
  user's environment, which is the realistic threat model here.

Usage::

    result = run_python_in_sandbox(
        code="import math; print(math.pi)",
        workspace_dir=Path("/tmp/run/workspace"),
        timeout_s=120,
    )
    print(result.stdout, result.exit_code)
"""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Soft resource caps (per-process); tweak via constructor args.
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024     # 256 KB stdout / stderr cap
DEFAULT_RLIMIT_CPU_S = 240                 # 4 minutes of CPU time
DEFAULT_RLIMIT_AS_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB virtual memory


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    workspace: Path
    artifacts: list[Path] = field(default_factory=list)
    """Files in workspace_dir produced by the run (PNGs, CSVs, etc.).

    Limited to extensions matching ``ARTIFACT_EXTS`` so we don't pick up
    junk; ordered by mtime ascending.
    """


ARTIFACT_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".svg",
                  ".csv", ".tsv", ".json", ".log", ".txt"}


# ── Code-block extraction ─────────────────────────────────────────────────


def extract_python_block(text: str) -> Optional[str]:
    """Pull the first ```python ... ``` block from a model response.

    Falls back to the first ``` ... ``` block if no language tag.
    Returns None if no fenced block found.
    """
    import re
    m = re.search(r"```python\n(.+?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\n(.+?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return None


def extract_bash_block(text: str) -> Optional[str]:
    """Pull a ```bash ... ``` block (used by Engineer for setup steps)."""
    import re
    m = re.search(r"```(?:bash|sh|shell)\n(.+?)```", text, re.DOTALL)
    return m.group(1) if m else None


# ── Workspace management ──────────────────────────────────────────────────


def make_workspace(run_id: str, *, root: Optional[Path] = None) -> Path:
    """Create (or reuse) a workspace directory for this run."""
    base = root or (Path.home() / ".cheetahclaws" / "research_papers")
    ws = base / run_id / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def collect_artifacts(workspace: Path) -> list[Path]:
    out = []
    for p in sorted(workspace.iterdir(), key=lambda p: p.stat().st_mtime):
        if p.is_file() and p.suffix.lower() in ARTIFACT_EXTS:
            out.append(p)
    return out


# ── Sandbox runner ────────────────────────────────────────────────────────


def run_python_in_sandbox(
    code: str,
    *,
    workspace_dir: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    rlimit_cpu_s: int = DEFAULT_RLIMIT_CPU_S,
    rlimit_as_bytes: int = DEFAULT_RLIMIT_AS_BYTES,
    extra_env: Optional[dict] = None,
) -> SandboxResult:
    """Run Python source code inside ``workspace_dir`` with timeout +
    resource limits. Returns a SandboxResult with stdout, stderr, exit code.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)
    script = workspace_dir / "experiment.py"
    script.write_text(code, encoding="utf-8")

    # Snapshot files before the run so we can show *new* artifacts only.
    pre_files = {p.name for p in workspace_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in ARTIFACT_EXTS}

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"  # no display server needed for matplotlib
    if extra_env:
        env.update(extra_env)

    # Linux/macOS only: apply rlimits via preexec_fn.
    preexec = _make_preexec(rlimit_cpu_s, rlimit_as_bytes) \
        if os.name != "nt" else None

    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(workspace_dir),
            env=env,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            preexec_fn=preexec,
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -1
        stdout = exc.stdout or b""
        stderr = (exc.stderr or b"") + (
            f"\n[sandbox] timed out after {timeout_s:.1f}s; killed."
        ).encode()
    except Exception as exc:
        return SandboxResult(
            exit_code=-2, stdout="",
            stderr=f"[sandbox] launch error: {exc}",
            duration_s=time.monotonic() - t0,
            timed_out=False, workspace=workspace_dir,
        )

    duration = time.monotonic() - t0
    stdout_s = _truncate(stdout, max_output_bytes).decode("utf-8", errors="replace")
    stderr_s = _truncate(stderr, max_output_bytes).decode("utf-8", errors="replace")

    # Persist outputs as files for the report
    (workspace_dir / "stdout.txt").write_text(stdout_s, encoding="utf-8")
    (workspace_dir / "stderr.txt").write_text(stderr_s, encoding="utf-8")
    (workspace_dir / "exit_code.txt").write_text(str(exit_code), encoding="utf-8")

    artifacts = [
        p for p in collect_artifacts(workspace_dir)
        if p.name not in pre_files
        and p.name not in ("stdout.txt", "stderr.txt", "exit_code.txt")
    ]

    return SandboxResult(
        exit_code=exit_code,
        stdout=stdout_s,
        stderr=stderr_s,
        duration_s=duration,
        timed_out=timed_out,
        workspace=workspace_dir,
        artifacts=artifacts,
    )


def _truncate(buf: bytes, limit: int) -> bytes:
    if len(buf) <= limit:
        return buf
    suffix = f"\n[sandbox] output truncated at {limit} bytes; {len(buf)} total]\n"
    return buf[:limit] + suffix.encode()


def _make_preexec(cpu_s: int, mem_bytes: int):
    """Return a preexec_fn that applies rlimits in the child process."""
    def _apply():
        try:
            import resource
            if cpu_s > 0:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            if mem_bytes > 0:
                # AS = address space; the closest portable proxy for memory.
                # macOS doesn't honor this fully — best-effort.
                try:
                    resource.setrlimit(resource.RLIMIT_AS,
                                        (mem_bytes, mem_bytes))
                except (ValueError, OSError):
                    pass
            # New session so a stuck child can't grab the parent's tty.
            os.setsid()
        except Exception:
            # Don't hard-fail in preexec — let the script run with whatever
            # limits the OS provides.
            pass
    return _apply


# ── Convenience: render result as compact text for prompts ───────────────


def format_result_for_prompt(result: SandboxResult, *,
                              max_lines: int = 60) -> str:
    """Compact stdout/stderr summary for feeding back to the Engineer."""
    head = f"exit_code: {result.exit_code}  ·  duration: {result.duration_s:.2f}s"
    if result.timed_out:
        head += "  ·  TIMED OUT"
    out_lines = result.stdout.splitlines()
    err_lines = result.stderr.splitlines()
    if len(out_lines) > max_lines:
        out_lines = out_lines[:max_lines] + [f"... [+{len(result.stdout.splitlines()) - max_lines} more]"]
    if len(err_lines) > max_lines:
        err_lines = err_lines[:max_lines] + [f"... [+{len(result.stderr.splitlines()) - max_lines} more]"]
    parts = [head, "", "stdout:", "\n".join(out_lines) or "(empty)"]
    if err_lines:
        parts += ["", "stderr:", "\n".join(err_lines)]
    if result.artifacts:
        parts += ["", "artifacts:"]
        for p in result.artifacts:
            parts.append(f"  - {p.name} ({p.stat().st_size} bytes)")
    return "\n".join(parts)
