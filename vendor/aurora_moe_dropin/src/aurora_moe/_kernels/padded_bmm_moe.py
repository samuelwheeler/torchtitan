"""Exact routed-expert SwiGLU/Squared-ReLU path using native XPU BMMs.

This is the first production-shaped control for the native Aurora effort.  It
uses custom SYCL kernels for route gather, exact ragged packing/unpacking, and
top-k reduction; the matrix products are PyTorch XPU ``bmm`` calls.  Every
shape is a runtime value.  The only rectangularization is zero padding to the
maximum *observed* per-expert row count for that batch, and all valid rows are
trimmed back before combining.  No capacity factor is imposed and no route is
dropped.

The BMM path gives a correct, generic backend while the PVC XMX grouped-GEMM
kernel is developed.  It also cleanly separates route metadata construction
from the steady-state expert forward/backward timing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aurora_moe._kernels.ops import load_route_ops
from aurora_moe._kernels.swiglu_ops import load_swiglu_ops


@dataclass(frozen=True)
class ExactRoutePlan:
    """Device-resident metadata for one exact local expert dispatch.

    ``route_ids[sorted_route]`` is the original flattened ``[token, choice]``
    position.  ``route_positions[original_route]`` is its inverse.  Expert
    rows are contiguous in sorted-route order, while ``group_offsets`` maps
    each expert to its half-open interval in that order.
    """

    route_ids: torch.Tensor
    route_positions: torch.Tensor
    sorted_experts: torch.Tensor
    group_rows: torch.Tensor
    group_offsets: torch.Tensor
    max_rows: int
    is_uniform: bool
    tokens: int
    top_k: int


def make_exact_route_plan(topk_indices: torch.Tensor, num_experts: int) -> ExactRoutePlan:
    """Build an exact, ragged local route plan from arbitrary top-k choices.

    The scalar ``max_rows`` is intentionally obtained once here because a
    rectangular BMM needs an extent known to its host API.  It is not part of
    the expert steady-state body when a plan is reused.  A later native ragged
    grouped kernel will consume ``group_rows`` directly and avoid that
    control-plane synchronization.
    """

    if topk_indices.device.type != "xpu":
        raise ValueError("topk_indices must be an XPU tensor")
    if topk_indices.dtype != torch.int64 or topk_indices.ndim != 2:
        raise ValueError("topk_indices must be contiguous int64 [tokens, top_k]")
    if not topk_indices.is_contiguous():
        topk_indices = topk_indices.contiguous()
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    flat = topk_indices.reshape(-1)
    sorted_experts, route_ids = torch.sort(flat)
    group_rows = torch.bincount(sorted_experts, minlength=num_experts).to(torch.int32)
    group_offsets = torch.cat(
        (
            group_rows.new_zeros(1),
            torch.cumsum(group_rows, dim=0, dtype=torch.int32),
        )
    )
    route_positions = load_route_ops().inverse_permutation_i64(route_ids)
    max_rows = int(group_rows.max().item())
    return ExactRoutePlan(
        route_ids=route_ids,
        route_positions=route_positions,
        sorted_experts=sorted_experts,
        group_rows=group_rows,
        group_offsets=group_offsets,
        max_rows=max_rows,
        # This CPU-side equality is exact: every count is <= max_rows, so the
        # sum can equal experts * max_rows only when all expert loads are equal.
        is_uniform=max_rows * num_experts == flat.numel(),
        tokens=topk_indices.shape[0],
        top_k=topk_indices.shape[1],
    )


def _check_inputs(
    x: torch.Tensor,
    topk_scores: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    plan: ExactRoutePlan,
    activation: str,
) -> None:
    if x.device.type != "xpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("x must be a BF16 XPU [tokens, model_dim] tensor")
    if topk_scores.device != x.device or topk_scores.ndim != 2:
        raise ValueError("topk_scores must be an XPU [tokens, top_k] tensor")
    if topk_scores.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("topk_scores must be BF16 or FP32")
    if topk_scores.shape != (plan.tokens, plan.top_k):
        raise ValueError("topk_scores shape does not match the route plan")
    if up.device != x.device or down.device != x.device:
        raise ValueError("expert weights must be on x's XPU device")
    if up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
        raise ValueError("expert weights must use BF16")
    if up.ndim != 3 or down.ndim != 3:
        raise ValueError("expert weights must be rank-3 tensors")
    experts, model_dim, hidden_dim = up.shape
    if x.shape != (plan.tokens, model_dim):
        raise ValueError("x shape does not match route plan and up weights")
    if down.shape != (experts, hidden_dim, model_dim):
        raise ValueError("down must have shape [experts, hidden_dim, model_dim]")
    if activation not in ("swiglu", "squared-relu"):
        raise ValueError("activation must be swiglu or squared-relu")
    if activation == "swiglu":
        if gate is None or gate.shape != up.shape:
            raise ValueError("SwiGLU requires gate shaped like up")
        if gate.device != x.device or gate.dtype != torch.bfloat16:
            raise ValueError("gate must be BF16 on x's XPU device")
    elif gate is not None:
        raise ValueError("squared-relu does not take a gate weight")
    if plan.group_rows.numel() != experts:
        raise ValueError("route plan expert count does not match expert weights")


class _ExactPaddedBmmMoE(torch.autograd.Function):
    """Manual autograd around route SYCL primitives and generic XPU BMMs."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        topk_scores: torch.Tensor,
        up: torch.Tensor,
        gate: torch.Tensor | None,
        down: torch.Tensor,
        plan: ExactRoutePlan,
        activation: str,
    ) -> torch.Tensor:
        _check_inputs(x, topk_scores, up, gate, down, plan, activation)
        ops = load_route_ops()
        packed = ops.route_gather_bf16(x.contiguous(), plan.route_ids, plan.top_k)
        if plan.is_uniform:
            padded = packed.reshape(up.size(0), plan.max_rows, x.size(1))
        else:
            padded = ops.sorted_to_padded_bf16(
                packed, plan.sorted_experts, plan.group_offsets, plan.max_rows
            )

        up_values = torch.bmm(padded, up)
        if activation == "swiglu":
            assert gate is not None
            gate_values = torch.bmm(padded, gate)
            hidden = load_swiglu_ops().swiglu_forward_bf16(up_values, gate_values)
        else:
            gate_values = torch.empty(0, dtype=x.dtype, device=x.device)
            hidden = F.relu(up_values).square()
        padded_values = torch.bmm(hidden, down)
        if plan.is_uniform:
            sorted_values = padded_values.reshape(-1, x.size(1))
        else:
            sorted_values = ops.padded_to_sorted_bf16(
                padded_values, plan.sorted_experts, plan.group_offsets
            )
        output = ops.weighted_route_reduce_bf16(
            sorted_values,
            plan.route_positions.reshape(plan.tokens, plan.top_k),
            topk_scores.contiguous(),
        )

        # The backward uses the exact same packed rows and route plan.  The
        # router scores are saved in sorted order, avoiding a second gather of
        # them in the performance-critical expert-gradient path.
        sorted_scores = topk_scores.reshape(-1).index_select(0, plan.route_ids)
        ctx.plan = plan
        ctx.activation = activation
        ctx.gate_present = gate is not None
        ctx.save_for_backward(
            padded,
            up,
            gate if gate is not None else torch.empty(0, dtype=x.dtype, device=x.device),
            down,
            up_values,
            gate_values,
            hidden,
            sorted_values,
            sorted_scores,
        )
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        (
            padded,
            up,
            gate,
            down,
            up_values,
            gate_values,
            hidden,
            sorted_values,
            sorted_scores,
        ) = ctx.saved_tensors
        plan: ExactRoutePlan = ctx.plan
        ops = load_route_ops()
        if grad_output.dtype != torch.bfloat16 or grad_output.ndim != 2:
            raise RuntimeError("the exact BF16 MoE path requires BF16 rank-2 grad_output")
        grad_output = grad_output.contiguous()

        # Recreate the per-route output gradient in expert-sorted order.  The
        # router multiplier is applied before packing, exactly as in the
        # reference expert loop.
        grad_sorted = ops.route_gather_bf16(grad_output, plan.route_ids, plan.top_k)
        grad_scores = None
        if ctx.needs_input_grad[1]:
            grad_scores_sorted = (
                grad_sorted.float() * sorted_values.float()
            ).sum(dim=-1)
            grad_scores_flat = torch.empty_like(sorted_scores)
            grad_scores_flat.index_copy_(
                0, plan.route_ids, grad_scores_sorted.to(sorted_scores.dtype)
            )
            grad_scores = grad_scores_flat.reshape(plan.tokens, plan.top_k)
        grad_sorted_values = (
            grad_sorted * sorted_scores.unsqueeze(-1)
        ).to(torch.bfloat16)
        if plan.is_uniform:
            grad_padded_values = grad_sorted_values.reshape(
                up.size(0), plan.max_rows, grad_sorted.size(1)
            )
        else:
            grad_padded_values = ops.sorted_to_padded_bf16(
                grad_sorted_values,
                plan.sorted_experts,
                plan.group_offsets,
                plan.max_rows,
            )

        # dW_down, d(hidden), and the two up/gate projections.  Every BMM has
        # the same dynamic [experts, max_rows, ...] layout used in forward.
        grad_down = torch.bmm(hidden.transpose(-1, -2), grad_padded_values)
        grad_hidden = torch.bmm(grad_padded_values, down.transpose(-1, -2))
        if ctx.activation == "swiglu":
            grad_up_values, grad_gate_values = load_swiglu_ops().swiglu_backward_bf16(
                grad_hidden, up_values, gate_values
            )
            grad_gate = torch.bmm(padded.transpose(-1, -2), grad_gate_values)
            grad_padded = torch.bmm(grad_gate_values, gate.transpose(-1, -2))
        else:
            grad_up_values = grad_hidden * (2.0 * F.relu(up_values))
            grad_gate = None
            grad_padded = torch.zeros_like(padded)
        grad_up = torch.bmm(padded.transpose(-1, -2), grad_up_values)
        grad_padded = grad_padded + torch.bmm(grad_up_values, up.transpose(-1, -2))
        if plan.is_uniform:
            grad_sorted_input = grad_padded.reshape(-1, grad_padded.size(-1))
        else:
            grad_sorted_input = ops.padded_to_sorted_bf16(
                grad_padded, plan.sorted_experts, plan.group_offsets
            )
        grad_x = ops.route_sum_bf16(
            grad_sorted_input,
            plan.route_positions.reshape(plan.tokens, plan.top_k),
        )

        # Inputs: x, topk_scores, up, gate, down, plan, activation.
        return grad_x, grad_scores, grad_up, grad_gate, grad_down, None, None


def exact_routed_expert_bmm(
    x: torch.Tensor,
    topk_scores: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    plan: ExactRoutePlan,
    *,
    activation: str = "swiglu",
) -> torch.Tensor:
    """Run exact local routed-expert forward+backward via XPU BMMs.

    Build a plan with :func:`make_exact_route_plan` for each new routing
    result, or reuse a plan when benchmarking a fixed route distribution.
    Shared experts intentionally remain an independent dense path.
    """

    return _ExactPaddedBmmMoE.apply(x, topk_scores, up, gate, down, plan, activation)
