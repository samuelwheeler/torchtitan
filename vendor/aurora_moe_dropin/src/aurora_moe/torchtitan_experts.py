"""TorchTitan-compatible exact expert MLPs backed by Aurora SYCL kernels.

This module deliberately implements only the local, already-dispatched expert
calculation.  TorchTitan remains responsible for routing, expert-parallel
all-to-all communication, route-score application, shared experts, parameter
ownership, and distributed gradient synchronization.

TorchTitan stores all three routed-expert matrices as
``[expert, output_features, input_features]``.  The oneMKL extension consumes
that allocation through its transpose flag, so this adapter does not repack or
transpose expert weights in the training hot path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.autograd.function import once_differentiable


def _expert_rows_tuple(
    expert_rows: torch.Tensor | Sequence[int],
    *,
    num_experts: int,
    total_rows: int,
) -> tuple[int, ...]:
    """Return validated host counts for the compact expert-major tensor."""

    if isinstance(expert_rows, torch.Tensor):
        if expert_rows.ndim != 1:
            raise ValueError("expert_rows must be rank 1")
        raw_rows = expert_rows.detach().cpu().tolist()
    else:
        raw_rows = list(expert_rows)

    if len(raw_rows) != num_experts:
        raise ValueError(
            f"expert_rows has {len(raw_rows)} entries, expected {num_experts}"
        )

    rows: list[int] = []
    for value in raw_rows:
        row = int(value)
        if row < 0 or float(value) != row:
            raise ValueError("expert_rows must contain non-negative integers")
        rows.append(row)
    if sum(rows) != total_rows:
        raise ValueError(
            f"expert_rows sums to {sum(rows)}, expected {total_rows} compact rows"
        )
    return tuple(rows)


def _check_inputs(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> None:
    operands = {"x": x, "w1": w1, "w2": w2, "w3": w3}
    for name, tensor in operands.items():
        if tensor.device.type != "xpu":
            raise ValueError(f"{name} must be an XPU tensor")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or w3.ndim != 3:
        raise ValueError("x must be rank 2 and expert weights must be rank 3")
    experts, hidden_dim, model_dim = w1.shape
    expected_w2 = (experts, model_dim, hidden_dim)
    if tuple(w3.shape) != tuple(w1.shape):
        raise ValueError(f"w3 shape {tuple(w3.shape)} must match w1 {tuple(w1.shape)}")
    if tuple(w2.shape) != expected_w2:
        raise ValueError(f"w2 shape {tuple(w2.shape)} must be {expected_w2}")
    if x.shape[1] != model_dim:
        raise ValueError(f"x width {x.shape[1]} must match model dim {model_dim}")
    if not (x.device == w1.device == w2.device == w3.device):
        raise ValueError("all expert operands must share one XPU device")


def _load_ops():
    from aurora_moe._kernels.one_mkl_exact_expert_gemm import (
        load_one_mkl_exact_expert_ops,
    )
    from aurora_moe._kernels.swiglu_ops import load_swiglu_ops

    return load_one_mkl_exact_expert_ops(), load_swiglu_ops()


class _TorchTitanExactExperts(torch.autograd.Function):
    """Manual first-order autograd for TorchTitan-layout routed experts."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        expert_rows: tuple[int, ...],
    ) -> torch.Tensor:
        one_mkl, swiglu = _load_ops()

        # TorchTitan: silu(x @ w1.T) * (x @ w3.T), then hidden @ w2.T.
        gate = one_mkl.exact_expert_gemm_transposed_weight_bf16(x, w1, expert_rows)
        up = one_mkl.exact_expert_gemm_transposed_weight_bf16(x, w3, expert_rows)
        hidden = swiglu.swiglu_forward_bf16(up, gate)
        out = one_mkl.exact_expert_gemm_transposed_weight_bf16(
            hidden, w2, expert_rows
        )

        ctx.expert_rows = expert_rows
        # Recompute the cheap pointwise hidden value in backward instead of
        # retaining a third [routes, hidden_dim] activation for every layer.
        ctx.save_for_backward(x, w1, w2, w3, up, gate)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        x, w1, w2, w3, up, gate = ctx.saved_tensors
        expert_rows = ctx.expert_rows
        if grad_out.dtype != torch.bfloat16 or grad_out.device.type != "xpu":
            raise RuntimeError("SYCL expert output gradients must be BF16 XPU tensors")
        grad_out = grad_out.contiguous()

        one_mkl, swiglu = _load_ops()
        hidden = swiglu.swiglu_forward_bf16(up, gate)
        # With W stored [out, in], dX = dY @ W.  The regular exact GEMM
        # consumes that same allocation as [forward_out, forward_in].
        grad_hidden = one_mkl.exact_expert_gemm_bf16(
            grad_out, w2, expert_rows
        )
        grad_up, grad_gate = swiglu.swiglu_backward_bf16(
            grad_hidden, up, gate
        )

        grad_x_gate = one_mkl.exact_expert_gemm_bf16(
            grad_gate, w1, expert_rows
        )
        grad_x_up = one_mkl.exact_expert_gemm_bf16(
            grad_up, w3, expert_rows
        )
        grad_x = grad_x_gate + grad_x_up

        # exact_expert_weight_grad computes A.T @ B.  Reversing the usual
        # activation/gradient argument order returns [out, in], exactly the
        # TorchTitan parameter layout, with defined zeros for empty experts.
        grad_w1 = one_mkl.exact_expert_weight_grad_bf16(
            grad_gate, x, expert_rows
        )
        grad_w2 = one_mkl.exact_expert_weight_grad_bf16(
            grad_out, hidden, expert_rows
        )
        grad_w3 = one_mkl.exact_expert_weight_grad_bf16(
            grad_up, x, expert_rows
        )
        return grad_x, grad_w1, grad_w2, grad_w3, None


def torchtitan_exact_experts(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    expert_rows: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Run compact expert-major TorchTitan SwiGLU experts on Aurora XPU.

    Args:
        w1: Gate weights ``[experts, hidden_dim, model_dim]``.
        w2: Down weights ``[experts, model_dim, hidden_dim]``.
        w3: Up weights ``[experts, hidden_dim, model_dim]``.
        x: Already expert-major routed rows ``[routes, model_dim]``.
        expert_rows: True row count for every local expert.  Counts must sum
            exactly to ``x.shape[0]``; ragged and empty experts are supported.
    """

    _check_inputs(x, w1, w2, w3)
    rows = _expert_rows_tuple(
        expert_rows,
        num_experts=w1.shape[0],
        total_rows=x.shape[0],
    )
    return _TorchTitanExactExperts.apply(x, w1, w2, w3, rows)


__all__ = ["torchtitan_exact_experts"]
