"""Exact local routed-expert path for the Aurora MoE prototype."""

from __future__ import annotations

import torch

from aurora_moe._kernels.padded_bmm_moe import (
    ExactRoutePlan,
    exact_routed_expert_bmm,
    make_exact_route_plan,
)


def make_local_route_plan(
    topk_indices: torch.Tensor, num_experts: int
) -> ExactRoutePlan:
    """Create exact device-side route metadata for locally owned experts."""

    return make_exact_route_plan(topk_indices, num_experts)


def local_routed_expert_forward(
    x: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    activation: str = "swiglu",
) -> torch.Tensor:
    """Run exact local expert computation with custom SYCL route movement.

    This covers local experts after EP dispatch.  Shared experts stay on a
    separate dense path.  All route selection, score weighting, and expert
    extents are dynamic; no capacity clipping is applied.
    """

    if x.ndim < 2:
        raise ValueError("x must have token and hidden dimensions")
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    if topk_indices.shape != topk_scores.shape or topk_indices.ndim != 2:
        raise ValueError("topk_indices and topk_scores must be [tokens, top_k]")
    if topk_indices.shape[0] != flat.shape[0]:
        raise ValueError("routing token dimension must match x")
    plan = make_exact_route_plan(topk_indices, up.size(0))
    output = exact_routed_expert_bmm(
        flat, topk_scores, up, gate, down, plan, activation=activation
    )
    return output.reshape_as(x)
