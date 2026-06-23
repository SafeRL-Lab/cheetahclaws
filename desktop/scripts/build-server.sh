#!/usr/bin/env bash
# Freeze `cheetahclaws --web` into a standalone server binary with PyInstaller.
#
# Builds in a CLEAN virtualenv that has only the core + [web] runtime deps —
# NOT your dev environment's kitchen sink — so the bundle stays lean (the
# trading/voice/research stacks and their heavy deps simply aren't present;
# the modular loaders degrade gracefully at runtime when they're absent).
#
# PyInstaller does not cross-compile: run this ON each target OS (macOS for the
# .app, Windows for the .exe). Output: desktop/server/dist/cheetahclaws-server/
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SERVER_DIR="$(cd "$HERE/../server" && pwd)"
VENV="${CC_BUILD_VENV:-$HERE/../.build-venv}"
PY="${PYTHON:-python3}"

echo "→ creating clean build venv: $VENV"
rm -rf "$VENV"
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip

echo "→ installing cheetahclaws[web] (core + web deps only)"
"$VENV/bin/pip" install -q -e "$REPO_ROOT[web]"
"$VENV/bin/pip" install -q pyinstaller

echo "→ freezing server (PyInstaller)"
cd "$SERVER_DIR"
rm -rf build dist
"$VENV/bin/pyinstaller" cheetahclaws-server.spec --noconfirm --log-level WARN

echo ""
echo "✓ built: $SERVER_DIR/dist/cheetahclaws-server/"
du -sh "$SERVER_DIR/dist/cheetahclaws-server"
