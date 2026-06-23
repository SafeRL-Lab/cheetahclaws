#!/usr/bin/env python3
"""Frozen entry point for the bundled CheetahClaws web server.

PyInstaller freezes this into a standalone binary that the desktop app spawns
as its sidecar. It just hands off to the normal CLI ``main()`` — the app passes
``--web --no-auth --host 127.0.0.1``, so this serves the exact same web UI as
``cheetahclaws --web``, with **no Python install required** on the user's
machine.
"""
import multiprocessing
import sys


def main() -> int:
    # No-op unless something forks via multiprocessing; required for frozen
    # apps that might, and harmless otherwise.
    multiprocessing.freeze_support()
    from cheetahclaws.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
