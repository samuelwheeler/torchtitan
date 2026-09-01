"""Exact local-expert-sharded layouts for pipelined padded EP transport.

Each local expert owns a separate ``[destination, cap]`` all-to-all slice.
The cap is an observed maximum over destination/expert route counts, not a
capacity factor; every route remains represented by its slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aurora_moe._kernels.direct_layout_bmm_moe import direct_padded_expert_bmm
from aurora_moe._kernels.ep_route_ops import (
    make_route_slots,
    pack_routes_payload,
    reduce_route_rows,
)


@dataclass(frozen=True)
class ExpertShardedRoutes:
    """Chunk-free source layout for one independent all-to-all per local expert."""

    payload: torch.Tensor
    route_slots: torch.Tensor
    counts: torch.Tensor
    cap: int


def destination_expert_counts(
    destinations: torch.Tensor, local_ids: torch.Tensor, ep_size: int, local_count: int
) -> torch.Tensor:
    """Count routes by ``[local_expert, destination]`` on the device."""

    if (
        destinations.dtype != torch.int64
        or local_ids.dtype != torch.int64
        or destinations.shape != local_ids.shape
        or destinations.ndim != 2
        or not destinations.is_contiguous()
        or not local_ids.is_contiguous()
        or destinations.device.type != "xpu"
        or local_ids.device != destinations.device
    ):
        raise ValueError("destinations and local_ids must be matching contiguous int64 XPU matrices")
    if ep_size <= 0 or local_count <= 0:
        raise ValueError("EP size and local expert count must be positive")
    combined = local_ids * ep_size + destinations
    return torch.bincount(combined.reshape(-1), minlength=ep_size * local_count).reshape(
        local_count, ep_size
    ).to(torch.int64)


def make_expert_sharded_routes(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    destinations: torch.Tensor,
    local_ids: torch.Tensor,
    ep_size: int,
    local_count: int,
    cap: int,
) -> ExpertShardedRoutes:
    """Pack exact routes as ``[local_expert, destination, cap, token..., score]``."""

    if cap < 0:
        raise ValueError("expert-sharded cap must be nonnegative")
    counts = destination_expert_counts(destinations, local_ids, ep_size, local_count)
    if counts.numel() and int(counts.max().item()) > cap:
        raise ValueError("expert-sharded cap does not cover observed routes")
    combined = (local_ids * ep_size + destinations).contiguous()
    slots = make_route_slots(combined, ep_size * local_count, cap)
    packed = pack_routes_payload(
        tokens,
        scores,
        local_ids,
        slots,
        local_count * ep_size * cap,
    )
    return ExpertShardedRoutes(
        packed.payload.reshape(local_count, ep_size, cap, tokens.size(1) + 1),
        slots,
        counts,
        cap,
    )


def run_expert_sharded_local_experts(
    received_payload: torch.Tensor,
    source_expert_counts: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    activation: str = "swiglu",
) -> torch.Tensor:
    """Run exact one-expert BMMs on received expert-sharded EP slices.

    ``received_payload`` is ``[local_experts, sources, cap, model_dim + 1]``;
    ``source_expert_counts`` is ``[sources, local_experts]``.  The return has
    the same leading layout and is ready for one return all-to-all per expert.
    """

    if (
        received_payload.device.type != "xpu"
        or received_payload.dtype != torch.bfloat16
        or received_payload.ndim != 4
        or not received_payload.is_contiguous()
        or source_expert_counts.device != received_payload.device
        or source_expert_counts.dtype != torch.int64
        or source_expert_counts.ndim != 2
        or not source_expert_counts.is_contiguous()
    ):
        raise ValueError("expert-sharded payload and counts must be contiguous XPU tensors")
    local_count, sources, cap, columns = received_payload.shape
    if source_expert_counts.shape != (sources, local_count):
        raise ValueError("source counts must be [sources, local_experts]")
    if cap < 0 or columns != up.size(1) + 1:
        raise ValueError("expert-sharded payload columns must be token plus score")
    if up.shape != (local_count, up.size(1), down.size(1)) or down.shape != (
        local_count,
        up.size(2),
        up.size(1),
    ):
        raise ValueError("expert weights do not match the received local expert count")
    if gate is not None and gate.shape != up.shape:
        raise ValueError("SwiGLU gate weights must match up weights")

    zero_ids = torch.zeros(sources * cap, dtype=torch.long, device=received_payload.device)
    outputs = []
    for expert in range(local_count):
        output = direct_padded_expert_bmm(
            received_payload[expert].reshape(sources * cap, columns),
            source_expert_counts[:, expert].contiguous(),
            cap,
            1,
            up[expert : expert + 1],
            None if gate is None else gate[expert : expert + 1],
            down[expert : expert + 1],
            local_ids=zero_ids,
            activation=activation,
        )
        outputs.append(output.reshape(sources, cap, up.size(1)))
    return torch.stack(outputs).contiguous()


def reduce_expert_sharded_rows(
    returned_payload: torch.Tensor, route_slots: torch.Tensor
) -> torch.Tensor:
    """Reduce returned expert-sharded rows back to canonical token order."""

    if returned_payload.ndim != 4 or not returned_payload.is_contiguous():
        raise ValueError("returned expert-sharded values must be contiguous rank-4")
    return reduce_route_rows(returned_payload.reshape(-1, returned_payload.size(-1)), route_slots)
