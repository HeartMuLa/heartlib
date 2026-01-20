"""JIT build + load the Metal extension.

Built only when explicitly enabled. Requires Xcode command line tools.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from .runtime import metal_build_tools_available, metal_supported


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


_CACHED_MOD: Any | None = None
_CACHED_ERR: Exception | None = None


def _xcrun_find(tool: str) -> str:
    out = subprocess.check_output(
        ["xcrun", "-sdk", "macosx", "--find", str(tool)], stderr=subprocess.STDOUT
    )
    p = out.decode("utf-8", errors="replace").strip()
    if not p:
        raise RuntimeError(f"xcrun returned empty path for tool {tool!r}")
    return p


def _compile_metallib(*, out_dir: Path, verbose: bool) -> Path:
    """Compile minimal Metal shaders -> `heartlib_ops.metallib` in `out_dir`."""
    sources = [
        _this_dir() / "rmsnorm.metal",
        _this_dir() / "rope.metal",
    ]
    airs = [out_dir / f"{src.stem}.air" for src in sources]
    metallib = out_dir / "heartlib_ops.metallib"

    metal = _xcrun_find("metal")
    metallib_tool = _xcrun_find("metallib")

    if metallib.exists():
        mt = metallib.stat().st_mtime
        if all(mt >= src.stat().st_mtime for src in sources):
            return metallib

    out_dir.mkdir(parents=True, exist_ok=True)

    for src, air in zip(sources, airs, strict=True):
        cmd = [metal, "-c", str(src), "-o", str(air)]
        if verbose:
            print("[heartlib] compiling Metal shader:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to compile Metal shaders.\n\n"
                f"Command:\n  {' '.join(cmd)}\n\n"
                f"stdout:\n{proc.stdout}\n\n"
                f"stderr:\n{proc.stderr}\n"
            )

    cmd2 = [metallib_tool, *[str(air) for air in airs], "-o", str(metallib)]
    if verbose:
        print("[heartlib] linking Metal metallib:", " ".join(cmd2))
    proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    if proc2.returncode != 0:
        raise RuntimeError(
            "Failed to link Metal metallib (`metallib`).\n\n"
            f"Command:\n  {' '.join(cmd2)}\n\n"
            f"stdout:\n{proc2.stdout}\n\n"
            f"stderr:\n{proc2.stderr}\n"
        )
    return metallib


def load_heartlib_metal_ops(*, verbose: bool = False) -> Any:
    """Build (if needed) and import the `heartlib_metal_ops` extension."""
    global _CACHED_MOD, _CACHED_ERR
    if _CACHED_MOD is not None:
        return _CACHED_MOD
    if _CACHED_ERR is not None:
        raise _CACHED_ERR

    if not metal_supported():
        err = RuntimeError("Metal/MPS is not supported on this runtime")
        _CACHED_ERR = err
        raise err
    if not metal_build_tools_available():
        err = RuntimeError(
            "Metal build tools unavailable.\n\n"
            "heartlib's fused Metal kernels require Xcode's Metal toolchain (`metal`, `metallib`).\n"
            "Install/select it:\n"
            "  - `xcode-select --install`\n"
            "  - or install Xcode.app then:\n"
            "      `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`\n"
            "      `sudo xcodebuild -license accept`\n\n"
            "Verify:\n"
            "  `xcrun -sdk macosx --find metal`\n"
            "  `xcrun -sdk macosx --find metallib`\n"
        )
        _CACHED_ERR = err
        raise err

    import torch.utils.cpp_extension as ce

    try:
        name = "heartlib_metal_ops"
        build_dir = Path(ce._get_build_directory(name, verbose=verbose))

        _compile_metallib(out_dir=build_dir, verbose=verbose)

        src_ops = str(_this_dir() / "ops.mm")
        extra_cflags = [
            "-O3",
            "-std=c++17",
            "-fobjc-arc",
        ]
        extra_ldflags = [
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
        ]
        mod = ce.load(
            name=name,
            sources=[src_ops],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            with_cuda=False,
            is_python_module=True,
            build_directory=str(build_dir),
            verbose=verbose,
        )
    except Exception as e:
        _CACHED_ERR = e
        raise

    _CACHED_MOD = mod
    return mod

