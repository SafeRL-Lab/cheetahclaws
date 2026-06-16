"""
Backward-compatibility shim.
Real implementation: modular/video/

'from video import X' and 'from video.pipeline import Y' still work.
"""
import importlib as _il
import sys as _sys

# Import the real package — triggers its own __init__
_real = _il.import_module("cheetahclaws.modular.video")

# Re-export everything from the real package
from cheetahclaws.modular.video import *  # noqa: F401, F403

# Register submodules so 'from video.X import Y' works
for _sub in ["pipeline", "story", "tts", "images", "subtitles",
             "assembly", "source", "niches", "cmd"]:
    try:
        _m = _il.import_module(f"cheetahclaws.modular.video.{_sub}")
        _sys.modules.setdefault(f"{__name__}.{_sub}", _m)
    except ImportError:
        pass
