"""Fused RMSNorm wrapper for the Metal extension."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from .runtime import metal_supported
from .jit import load_heartlib_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


class _AutogradCtx(Protocol):
    saved_tensors: tuple["Tensor", ...]

    def save_for_backward(self, *tensors: "Tensor") -> None: ...


def metal_rmsnorm_available() -> bool:
    return metal_supported()


class _MetalRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: _AutogradCtx,
        x: "Tensor",
        weight: "Tensor | None",
        eps: float,
        verbose_build: bool,
    ) -> "Tensor":
        if x.device.type != "mps":
            raise RuntimeError("Metal RMSNorm requires device.type == 'mps'")
        if x.dtype not in (torch.float16, torch.float32):
            raise RuntimeError("Metal RMSNorm supports fp16/fp32 only")

        x2 = x.contiguous()
        ops = load_heartlib_metal_ops(verbose=bool(verbose_build))

        if weight is None:
            out, inv = ops.rmsnorm_noweight_forward_with_inv(x2, float(eps))
            ctx.save_for_backward(x2, inv)
            return out

        w2 = weight.to(device=x.device, dtype=x.dtype).contiguous()
        out, inv = ops.rmsnorm_forward_with_inv(x2, w2, float(eps))
        ctx.save_for_backward(x2, w2, inv)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: _AutogradCtx,
        grad_out: "Tensor",
    ) -> tuple["Tensor | None", ...]:
        if grad_out is None:
            raise RuntimeError("Metal RMSNorm backward requires grad_out")
        if grad_out.device.type != "mps":
            raise RuntimeError("Metal RMSNorm backward requires grad_out on MPS")

        saved = ctx.saved_tensors
        target_dtype = saved[0].dtype
        if grad_out.dtype != target_dtype:
            grad_out = grad_out.to(dtype=target_dtype)
        g = grad_out.contiguous()

        ops = load_heartlib_metal_ops(verbose=False)
        if len(saved) == 2:
            x, inv = saved
            grad_x = ops.rmsnorm_backward_x_noweight(g, x, inv)
            return (grad_x, None, None, None)
        if len(saved) == 3:
            x, w, inv = saved
            grad_x = ops.rmsnorm_backward_x(g, x, w, inv)
            grad_w = ops.rmsnorm_backward_w(g, x, inv)
            return (grad_x, grad_w, None, None)
        raise RuntimeError("Metal RMSNorm backward: invalid saved tensor state")


def rmsnorm_fp16(
    *,
    x: "Tensor",
    weight: "Tensor | None",
    eps: float = 1e-6,
    verbose_build: bool = False,
) -> "Tensor":
    """Fused RMSNorm (MPS/Metal) for fp16/fp32 tensors."""
    if x.device.type != "mps":
        raise RuntimeError("Metal RMSNorm requires device.type == 'mps'")
    if x.dtype not in (torch.float16, torch.float32):
        raise RuntimeError("Metal RMSNorm supports fp16/fp32 only")

    needs_grad = bool(x.requires_grad) or (
        weight is not None and bool(weight.requires_grad)
    )
    if not needs_grad:
        x2 = x.contiguous()
        ops = load_heartlib_metal_ops(verbose=bool(verbose_build))
        if weight is None:
            return ops.rmsnorm_noweight(x2, float(eps))
        w2 = weight.to(device=x.device, dtype=x.dtype).contiguous()
        return ops.rmsnorm(x2, w2, float(eps))

    y = _MetalRMSNormFn.apply(x, weight, float(eps), bool(verbose_build))
    if not isinstance(y, torch.Tensor):
        raise TypeError("Metal RMSNorm returned a non-tensor output")
    return y

