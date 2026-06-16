"""daemon — reference scaffolding for the cheetahclaws daemon foundation.

Validates the contract proposed in docs/RFC/0001-daemon-design-note.md.
This is a SPIKE: it covers transport, auth, originator routing, and event
streaming, but does NOT integrate with agent.run, bridges, or the session
store. The foundation PR rebuilds on this or replaces it.

Linux-only for peer-cred. macOS support is a deliberate TODO.
"""
from __future__ import annotations

API_VERSION = "0"
API_VERSION_HEADER = "Cheetahclaws-Api-Version"

__all__ = ["API_VERSION", "API_VERSION_HEADER"]
