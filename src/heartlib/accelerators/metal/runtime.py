"""Backend availability detection for Metal/MPS.

This is a tiny, dependency-light helper used by the optional Metal fast path.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import TYPE_CHECKING

import torch

__all__ = [
    "metal_supported",
    "metal_build_tools_available",
]


def metal_supported() -> bool:
    """Whether the current runtime *can* execute custom Metal (MPS) ops."""
    if TYPE_CHECKING:
        return False
    if platform.system() != "Darwin":
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def metal_build_tools_available() -> bool:
    """Whether the host can compile Metal shaders via Xcode toolchain."""
    if TYPE_CHECKING:
        return False
    if not metal_supported():
        return False
    if shutil.which("xcrun") is None:
        return False
    try:
        subprocess.check_output(
            ["xcrun", "-sdk", "macosx", "--find", "metal"], stderr=subprocess.STDOUT
        )
        subprocess.check_output(
            ["xcrun", "-sdk", "macosx", "--find", "metallib"], stderr=subprocess.STDOUT
        )
    except Exception:
        return False
    return True

