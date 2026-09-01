"""Exact chunked local-expert BMMs for source-padded EP transport.

The transport layout is ``[chunk, source, rows_per_chunk, columns]``.  It
contains every observed route; the final chunk is transport padding only and
is guarded by the corresponding exact receive counts.
"""

from __future__ import annotations

import math

import torch

from aurora_moe._kernels.direct_layout_bmm_moe import direct_padded_expert_bmm


def chunked_counts(recv_counts: torch.Tensor, start: int, rows: int) -> torch.Tensor:
    """Return exact valid rows from each source in one source-cap chunk."""

    if start < 0 or rows <= 0:
        raise ValueError("chunk start must be nonnegative and rows must be positive")
    return (recv_counts - start).clamp(min=0, max=rows).contiguous()


def source_major_to_chunked(payload: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    """Copy ``[source, cap, columns]`` payload storage into chunk-major order."""

    if payload.ndim != 3 or not payload.is_contiguous() or chunk_rows <= 0:
        raise ValueError("payload must be contiguous rank-3 and chunk_rows must be positive")
    sources, cap, columns = payload.shape
    chunks = math.ceil(cap / chunk_rows)
    result = payload.new_zeros((chunks, sources, chunk_rows, columns))
    for chunk in range(chunks):
        start = chunk * chunk_rows
        width = min(chunk_rows, cap - start)
        result[chunk, :, :width].copy_(payload[:, start : start + width])
    return result


def chunked_to_source_major(values: torch.Tensor, cap: int) -> torch.Tensor:
    """Copy chunk-major rows back to the original ``[source, cap, columns]`` order."""

    if values.ndim != 4 or not values.is_contiguous() or cap < 0:
        raise ValueError("values must be contiguous rank-4 and cap must be nonnegative")
    chunks, sources, chunk_rows, columns = values.shape
    if chunks != math.ceil(cap / chunk_rows):
        raise ValueError("chunk count and cap do not agree")
    result = values.new_zeros((sources, cap, columns))
    for chunk in range(chunks):
        start = chunk * chunk_rows
        width = min(chunk_rows, cap - start)
        result[:, start : start + width].copy_(values[chunk, :, :width])
    return result


def chunked_direct_padded_expert_bmm(
    payload_chunks: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    num_experts: int,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    local_id_chunks: torch.Tensor | None = None,
    activation: str = "swiglu",
) -> torch.Tensor:
    """Run exact local experts independently on every source-cap chunk.

    The returned tensor remains chunk-major.  Reassembling it with
    :func:`chunked_to_source_major` gives the same source-padded rows as one
    unchunked direct-layout call.  Reusing the same parameter tensors in each
    call makes ordinary autograd accumulate their exact gradients.
    """

    if (
        payload_chunks.ndim != 4
        or not payload_chunks.is_contiguous()
        or recv_counts.ndim != 1
        or recv_counts.device != payload_chunks.device
        or recv_counts.dtype != torch.int64
        or payload_chunks.size(1) != recv_counts.numel()
    ):
        raise ValueError("chunked payload and receive counts have incompatible layouts")
    chunks, sources, chunk_rows, columns = payload_chunks.shape
    if chunk_rows <= 0 or chunks != math.ceil(cap / chunk_rows):
        raise ValueError("chunked payload shape does not describe cap")
    if local_id_chunks is not None and (
        local_id_chunks.shape != (chunks, sources, chunk_rows)
        or local_id_chunks.device != payload_chunks.device
        or local_id_chunks.dtype != torch.int64
        or not local_id_chunks.is_contiguous()
    ):
        raise ValueError("local_id_chunks must be contiguous [chunk, source, rows]")

    if chunks == 0:
        return payload_chunks[..., : up.size(1)] * 0

    outputs = []
    for chunk in range(chunks):
        counts = chunked_counts(recv_counts, chunk * chunk_rows, chunk_rows)
        payload = payload_chunks[chunk].reshape(sources * chunk_rows, columns)
        local_ids = (
            None
            if local_id_chunks is None
            else local_id_chunks[chunk].reshape(sources * chunk_rows)
        )
        output = direct_padded_expert_bmm(
            payload,
            counts,
            chunk_rows,
            num_experts,
            up,
            gate,
            down,
            local_ids=local_ids,
            activation=activation,
        )
        outputs.append(output.reshape(sources, chunk_rows, up.size(1)))
    return torch.stack(outputs).contiguous()
