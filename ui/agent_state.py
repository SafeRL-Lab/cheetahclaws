"""ui/agent_state.py — Shared concurrency primitives for the interactive REPL.

Three module-level singletons coordinate the main (input) thread and the
agent (run_query) thread:

  _cancel_event   — ESC sets this; agent loop checks it to stop early.
  _input_queue    — typed-ahead messages buffered while agent is running.
  _agent_running  — set while a run_query() turn is in progress.

Question routing (AskUserQuestion):
  _pending_question — non-empty string while agent is waiting for an answer.
  _answer_event     — agent thread blocks on this; Enter binding sets it.
  _answer_value     — the user's answer, written by Enter binding before set().
"""
from __future__ import annotations

import collections
import threading

_cancel_event: threading.Event = threading.Event()
_input_queue: collections.deque = collections.deque()
_agent_running: threading.Event = threading.Event()

_pending_question: str = ""
_answer_event: threading.Event = threading.Event()
_answer_value: str = ""
