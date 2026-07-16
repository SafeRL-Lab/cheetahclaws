#!/usr/bin/env bash
# Build a self-contained desktop installer: freeze the Python server, then
# package the Electron shell around it with electron-builder.
#
# Produces, in desktop/out/:
#   macOS    → .dmg / .app        Windows → .exe (NSIS)
#   Linux    → .AppImage          (per the build.* targets in package.json)
#
# PyInstaller does not cross-compile — run this ON the OS you want to ship for.
# End users of the result need neither Node nor Python.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP="$(cd "$HERE/.." && pwd)"

echo "==> [1/3] Freeze the Python server (PyInstaller, clean venv)"
bash "$HERE/build-server.sh"

echo ""
echo "==> [2/3] Install Electron build deps"
cd "$DESKTOP"
npm install

echo ""
echo "==> [3/3] Package the installer (electron-builder)"
npm run dist

echo ""
echo "✓ Done. Installers in: $DESKTOP/out/"
ls -1 "$DESKTOP/out" 2>/dev/null | grep -iE '\.(dmg|exe|appimage|deb|zip)$' || true
