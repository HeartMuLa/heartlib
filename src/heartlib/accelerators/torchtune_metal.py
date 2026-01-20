from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MetalPatchReport:
    rmsnorm_replaced: int = 0
    rope_wrapped: int = 0
    enabled: bool = False
    reason: str = ""


class _MetalRMSNorm(nn.Module):
    """Drop-in replacement for torchtune.modules.rms_norm.RMSNorm (inference-safe)."""

    def __init__(self, *, scale: nn.Parameter, eps: float):
        super().__init__()
        self.eps = float(eps)
        # Keep parameter name stable for state_dict compatibility.
        self.scale = scale

        # Lazy import: optimizer is an optional local package.
        self._metal_impl = None
        try:
            from .metal.rmsnorm import rmsnorm_fp16 as _metal_rmsnorm

            self._metal_impl = _metal_rmsnorm
        except Exception:
            self._metal_impl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._metal_impl is not None
            and x.device.type == "mps"
            and x.dtype in (torch.float16, torch.float32)
        ):
            return self._metal_impl(x=x, weight=self.scale, eps=self.eps)

        # Fallback matches torchtune RMSNorm implementation (compute in fp32).
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale


class _MetalLlama3ScaledRoPE(nn.Module):
    """Wrapper for torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE.

    Uses Metal RoPE when the input positions are compatible with the kernel:
    - Training/unpacked: input_pos is None (positions 0..S-1)
    - Prompt paths that pass input_pos but are still 0..S-1 (same for all batches)
    - Decode step: input_pos is a scalar position (same for all batches) with seq_len==1
    Otherwise, falls back to the original implementation.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        # Avoid accidental wrapper-of-wrapper chains.
        while isinstance(inner, _MetalLlama3ScaledRoPE):
            inner = inner.inner
        self.inner = inner

        self._metal_impl = None
        try:
            from .metal.rope import rope_fp16 as _metal_rope

            self._metal_impl = _metal_rope
        except Exception:
            self._metal_impl = None

    def _maybe_expand_cache(self, need_len: int) -> None:
        cache = getattr(self.inner, "cache", None)
        if cache is None:
            return
        if int(cache.shape[0]) >= int(need_len):
            return
        # Rebuild cache to the required length.
        build = getattr(self.inner, "build_rope_cache", None)
        if callable(build):
            build(int(need_len))

    def _inner_rope(self) -> nn.Module:
        # Unwrap nested wrappers (defensive) and guard against cycles.
        inner: nn.Module = self.inner
        seen: set[int] = set()
        while isinstance(inner, _MetalLlama3ScaledRoPE):
            if id(inner) in seen:
                break
            seen.add(id(inner))
            inner = inner.inner
        return inner

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        inner = self._inner_rope()
        if (
            self._metal_impl is None
            or x.device.type != "mps"
            or x.dtype not in (torch.float16, torch.float32)
        ):
            return inner(x, input_pos=input_pos)

        # torchtune shape: [b, s, n_h, h_d]
        if x.ndim != 4:
            return inner(x, input_pos=input_pos)

        b, s, nh, hd = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
        rot_dim = int(getattr(inner, "dim", hd))
        if rot_dim <= 0 or (rot_dim % 2) != 0 or rot_dim > hd:
            return inner(x, input_pos=input_pos)

        cache = getattr(inner, "cache", None)
        if cache is None:
            return inner(x, input_pos=input_pos)

        # Fast-path selection for cos/sin.
        cos: Optional[torch.Tensor] = None
        sin: Optional[torch.Tensor] = None

        if input_pos is None:
            self._maybe_expand_cache(s)
            cos = cache[:s, :, 0]
            sin = cache[:s, :, 1]
        else:
            ip = input_pos
            # Common prompt path: ip is [B,S] and equals arange(S) (same for all batches).
            if ip.ndim == 2 and int(ip.shape[1]) == s:
                # Only accept when identical across batch and sequential.
                ip0 = ip[0]
                ar = torch.arange(s, device=ip.device, dtype=ip.dtype)
                if torch.equal(ip0, ar) and torch.all(ip == ip0):
                    self._maybe_expand_cache(s)
                    cos = cache[:s, :, 0]
                    sin = cache[:s, :, 1]
            # Decode path: ip is [B,1] and identical across batch; use a single-row cos/sin.
            if cos is None and ip.numel() == b and ip.ndim == 2 and int(ip.shape[1]) == 1 and s == 1:
                v0 = ip.view(-1)[0]
                if torch.all(ip == v0):
                    pos = int(v0.item())
                    self._maybe_expand_cache(pos + 1)
                    cos = cache[pos : pos + 1, :, 0]
                    sin = cache[pos : pos + 1, :, 1]

        if cos is None or sin is None:
            return inner(x, input_pos=input_pos)

        # Convert torchtune layout [B,S,H,D] -> kernel layout [B,H,S,D]
        x2 = x.permute(0, 2, 1, 3).contiguous()
        y2 = self._metal_impl(x=x2, cos=cos, sin=sin, rot_dim=rot_dim)
        return y2.permute(0, 2, 1, 3).contiguous()


def try_enable_torchtune_metal(
    model: nn.Module,
    *,
    enabled: Optional[bool] = None,
    verbose: bool = False,
) -> MetalPatchReport:
    """Best-effort: patch torchtune Llama3.* modules to use Metal RMSNorm/RoPE on MPS.

    This is intentionally opt-in and safe:
    - If optimizer/metal is missing, does nothing
    - If torchtune internals differ, does nothing
    - Falls back to original ops when the kernel can't represent input_pos layouts
    """
    if enabled is None:
        enabled = os.getenv("HEARTLIB_ENABLE_MPS_METAL", "0") == "1"
    if not enabled:
        return MetalPatchReport(enabled=False, reason="disabled")

    try:
        tt_rms_mod = __import__("torchtune.modules.rms_norm", fromlist=["RMSNorm"])
        TT_RMSNorm = getattr(tt_rms_mod, "RMSNorm")
        tt_rope_mod = __import__(
            "torchtune.models.llama3_1._position_embeddings",
            fromlist=["Llama3ScaledRoPE"],
        )
        TT_RoPE = getattr(tt_rope_mod, "Llama3ScaledRoPE")
    except Exception as e:
        return MetalPatchReport(enabled=False, reason=f"torchtune import failed: {e}")

    rms_count = 0
    rope_count = 0
    rope_wrappers: Dict[int, nn.Module] = {}

    for parent in model.modules():
        # Never patch inside our own wrappers; that can create wrapper chains/cycles.
        if isinstance(parent, _MetalLlama3ScaledRoPE):
            continue
        for name, child in list(parent.named_children()):
            # Replace RMSNorm.
            if isinstance(child, TT_RMSNorm) and not isinstance(child, _MetalRMSNorm):
                scale = getattr(child, "scale", None)
                eps = float(getattr(child, "eps", 1e-6))
                if isinstance(scale, nn.Parameter):
                    setattr(parent, name, _MetalRMSNorm(scale=scale, eps=eps))
                    rms_count += 1
                    continue

            # Wrap RoPE.
            if isinstance(child, TT_RoPE) and not isinstance(child, _MetalLlama3ScaledRoPE):
                key = id(child)
                wrapped = rope_wrappers.get(key)
                if wrapped is None:
                    wrapped = _MetalLlama3ScaledRoPE(child)
                    rope_wrappers[key] = wrapped
                setattr(parent, name, wrapped)
                rope_count += 1

    if verbose:
        print(
            f"[heartlib] torchtune metal patch: rmsnorm_replaced={rms_count}, rope_wrapped={rope_count}"
        )
    return MetalPatchReport(
        rmsnorm_replaced=rms_count,
        rope_wrapped=rope_count,
        enabled=True,
        reason="ok",
    )

