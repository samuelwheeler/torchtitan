"""Algebraically collapse independent shared SwiGLU experts into one dense MLP."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _check(x: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, down: torch.Tensor) -> None:
    if x.ndim < 2 or up.ndim != 3 or gate.shape != up.shape or down.ndim != 3:
        raise ValueError("expected x=[...,D], up/gate=[S,D,H], down=[S,H,D]")
    if (
        x.size(-1) != up.size(1)
        or up.size(0) != down.size(0)
        or up.size(2) != down.size(1)
        or down.size(2) != x.size(-1)
    ):
        raise ValueError("incompatible shared-expert dimensions")
    if not all(tensor.is_contiguous() for tensor in (x, up, gate, down)):
        raise ValueError("collapsed shared experts require contiguous inputs")


def collapse_shared_up_gate(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return ``[D, 2*S*H]`` packed as ``[all up | all gate]``."""

    if up.ndim != 3 or gate.shape != up.shape:
        raise ValueError("up and gate must have matching [S,D,H] shapes")
    model_dim = up.size(1)
    width = up.size(0) * up.size(2)
    return torch.cat(
        (
            up.permute(1, 0, 2).reshape(model_dim, width),
            gate.permute(1, 0, 2).reshape(model_dim, width),
        ),
        dim=-1,
    ).contiguous()


def collapsed_shared_expert_swiglu(
    x: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor,
    down: torch.Tensor,
    *,
    fused_pointwise: bool = False,
) -> torch.Tensor:
    """Return the shared-expert sum through one widened dense SwiGLU MLP.

    The packed projection is formed at runtime, so gradients map directly back
    to the existing separate expert parameters.  A production packed-parameter
    layout can instead call :func:`collapsed_packed_shared_expert_swiglu`.
    """

    _check(x, up, gate, down)
    if up.size(0) == 0:
        return torch.zeros_like(x)
    flat = x.reshape(-1, x.size(-1))
    width = up.size(0) * up.size(2)
    projection = flat.mm(collapse_shared_up_gate(up, gate))
    hidden = _swiglu(projection, width, fused_pointwise)
    return hidden.mm(down.reshape(width, x.size(-1))).reshape_as(x)


def collapsed_packed_shared_expert_swiglu(
    x: torch.Tensor,
    packed_up_gate: torch.Tensor,
    down: torch.Tensor,
    *,
    fused_pointwise: bool = False,
) -> torch.Tensor:
    """Run a widened shared SwiGLU from persistent ``[D,2*S*H]`` weights."""

    if x.ndim < 2 or packed_up_gate.ndim != 2 or down.ndim != 3:
        raise ValueError("expected x=[...,D], packed=[D,2*S*H], down=[S,H,D]")
    if not all(tensor.is_contiguous() for tensor in (x, packed_up_gate, down)):
        raise ValueError("collapsed packed shared experts require contiguous inputs")
    experts, hidden_dim, model_dim = down.shape
    width = experts * hidden_dim
    if x.size(-1) != model_dim or packed_up_gate.shape != (model_dim, 2 * width):
        raise ValueError("incompatible packed shared-expert dimensions")
    if experts == 0:
        return torch.zeros_like(x)
    flat = x.reshape(-1, model_dim)
    projection = flat.mm(packed_up_gate)
    hidden = _swiglu(projection, width, fused_pointwise)
    return hidden.mm(down.reshape(width, model_dim)).reshape_as(x)


def _swiglu(projection: torch.Tensor, width: int, fused_pointwise: bool) -> torch.Tensor:
    if not fused_pointwise:
        return F.silu(projection[:, width:]) * projection[:, :width]
    if projection.device.type != "xpu" or projection.dtype != torch.bfloat16:
        raise ValueError("fused collapsed SwiGLU requires a BF16 XPU projection")
    from aurora_moe._kernels.packed_swiglu_ops import packed_swiglu_bf16

    return packed_swiglu_bf16(projection)
