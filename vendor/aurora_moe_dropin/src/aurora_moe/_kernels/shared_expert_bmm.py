"""Exact shared-expert SwiGLU layouts without per-step parameter copies."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _check(x: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, down: torch.Tensor) -> None:
    if x.ndim != 2 or up.ndim != 3 or gate.ndim != 3 or down.ndim != 3:
        raise ValueError("expected x=[M,D], up/gate=[S,D,H], down=[S,H,D]")
    if up.shape != gate.shape or up.size(0) != down.size(0):
        raise ValueError("shared-expert weight group dimensions must agree")
    if x.size(1) != up.size(1) or up.size(2) != down.size(1) or down.size(2) != x.size(1):
        raise ValueError("incompatible shared-expert matrix dimensions")


def shared_expert_loop(
    x: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor,
    down: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the exact per-shared-expert loop result.

    ``initial`` is an already accumulated output with the same shape as ``x``.
    It permits a stream-separated continuation while retaining the ordinary
    left-to-right BF16 accumulation order.
    """

    _check(x, up, gate, down)
    if initial is None:
        out = torch.zeros_like(x)
    else:
        if initial.shape != x.shape or initial.device != x.device or initial.dtype != x.dtype:
            raise ValueError("initial shared-expert output must match x")
        out = initial
    for expert in range(up.size(0)):
        out = out + (F.silu(x.mm(gate[expert])) * x.mm(up[expert])).mm(down[expert])
    return out


def shared_expert_expand_bmm(
    x: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    """Use a zero-stride expanded activation view with three batched GEMMs."""

    _check(x, up, gate, down)
    batched_x = x.unsqueeze(0).expand(up.size(0), -1, -1)
    hidden = F.silu(torch.bmm(batched_x, gate)) * torch.bmm(batched_x, up)
    return torch.bmm(hidden, down).sum(dim=0)


def shared_expert_broadcast_matmul(
    x: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    """Use matmul broadcasting, which exposes the same zero-stride activation layout."""

    _check(x, up, gate, down)
    hidden = F.silu(torch.matmul(x, gate)) * torch.matmul(x, up)
    return torch.bmm(hidden, down).sum(dim=0)


def shared_expert_onemkl_batched(
    x: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    """Use oneMKL's shared-A GEMMs and strided batched down projection."""

    from aurora_moe._kernels.one_mkl_ops import shared_a_gemm_bf16, strided_batched_gemm_bf16

    _check(x, up, gate, down)
    hidden = F.silu(shared_a_gemm_bf16(x, gate)) * shared_a_gemm_bf16(x, up)
    return strided_batched_gemm_bf16(hidden, down).sum(dim=0)


def shared_expert_packed_bmm(
    x: torch.Tensor, packed_up_gate: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    """Use persistent [S,D,2H] parameters, eliminating a runtime up/gate concat."""

    if packed_up_gate.ndim != 3 or down.ndim != 3 or x.ndim != 2:
        raise ValueError("expected x=[M,D], packed=[S,D,2H], down=[S,H,D]")
    if (packed_up_gate.size(0), packed_up_gate.size(1), packed_up_gate.size(2) // 2) != (
        down.size(0), x.size(1), down.size(1)
    ) or packed_up_gate.size(2) % 2 or down.size(2) != x.size(1):
        raise ValueError("incompatible packed shared-expert matrix dimensions")
    projection = torch.matmul(x, packed_up_gate)
    hidden = F.silu(projection[..., down.size(1):]) * projection[..., :down.size(1)]
    return torch.bmm(hidden, down).sum(dim=0)


def flatten_shared_expert_up_gate(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Pack [S, D, H] up/gate weights as a [D, 2*S*H] projection matrix."""

    if up.ndim != 3 or gate.shape != up.shape:
        raise ValueError("up and gate must have matching [S, D, H] shapes")
    return torch.cat((up, gate), dim=-1).permute(1, 0, 2).reshape(up.size(1), -1).contiguous()


def shared_expert_flattened_projection(
    x: torch.Tensor, flattened_up_gate: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    """Evaluate shared experts from one [D, 2*S*H] up/gate projection."""

    if x.ndim != 2 or flattened_up_gate.ndim != 2 or down.ndim != 3:
        raise ValueError("expected x=[M,D], flattened_up_gate=[D,2*S*H], down=[S,H,D]")
    experts, hidden_dim, model_dim = down.shape
    if x.size(1) != model_dim or flattened_up_gate.shape != (
        model_dim,
        2 * experts * hidden_dim,
    ):
        raise ValueError("incompatible flattened shared-expert matrix dimensions")
    projection = x.mm(flattened_up_gate)
    up_values, gate_values = projection.reshape(
        x.size(0), experts, 2, hidden_dim
    ).transpose(0, 1).unbind(dim=2)
    return torch.bmm(F.silu(gate_values) * up_values, down).sum(dim=0)
