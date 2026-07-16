# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — freeze `cheetahclaws --web` into a standalone server
# binary that the desktop app ships and spawns (no Python on the user's box).
#
# Build:  pyinstaller desktop/server/cheetahclaws-server.spec --noconfirm
# Output: dist/cheetahclaws-server/  (onedir; the exe + _internal/)

import os
from PyInstaller.utils.hooks import collect_all

# The repo root holds the real cheetahclaws/ package dir. Put it on the
# analysis path so PyInstaller resolves submodules by FILE rather than via the
# editable-install meta-path finder (which it can't follow — that yields a
# binary missing all of cheetahclaws). SPECPATH = this spec's dir.
REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, '..', '..'))

# Pull in ALL of the cheetahclaws package: every submodule (it's full of
# dynamic imports — tool registry, plugin/modular loaders, bridges) and its
# data files (web/ html+js+css, prompts/*.md, agent_templates/*.md).
datas, binaries, hiddenimports = collect_all('cheetahclaws')

# Heavy, optional stacks. The web server + agent never need them, and the
# modular loaders already degrade gracefully when they're absent ("could not
# load modular.X"). Excluding keeps the bundle from ballooning into the GBs
# (torch alone is ~2 GB).
EXCLUDES = [
    'torch', 'torchvision', 'torchaudio', 'transformers',
    'sklearn', 'scipy', 'lightgbm', 'xgboost', 'pandas',
    'matplotlib', 'seaborn',
    'faster_whisper', 'whisper', 'sounddevice', 'soundfile',
    'PIL', 'moviepy', 'imageio', 'imageio_ffmpeg', 'cv2',
    'IPython', 'notebook', 'jupyter', 'jupyter_core',
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
]

a = Analysis(
    ['launch_server.py'],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cheetahclaws-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,            # it's a server — stdout carries the readiness line
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,        # build host arch; build on each target OS
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='cheetahclaws-server',
)
