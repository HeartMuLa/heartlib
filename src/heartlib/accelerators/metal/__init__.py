"""Optional Metal (MPS) fused kernels for Apple Silicon.

This is intentionally self-contained and opt-in:
- No import-time dependency on Xcode toolchains.
- The extension is built on-demand via `torch.utils.cpp_extension` when enabled.
"""

from __future__ import annotations

from .runtime import metal_supported, metal_build_tools_available
from .jit import load_heartlib_metal_ops
from .rmsnorm import metal_rmsnorm_available, rmsnorm_fp16
from .rope import metal_rope_available, rope_fp16

__all__ = [
    "metal_supported",
    "metal_build_tools_available",
    "load_heartlib_metal_ops",
    "metal_rmsnorm_available",
    "rmsnorm_fp16",
    "metal_rope_available",
    "rope_fp16",
]

