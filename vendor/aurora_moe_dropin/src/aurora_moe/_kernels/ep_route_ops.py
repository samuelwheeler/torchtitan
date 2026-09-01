"""Exact device-side source routing for padded expert-parallel transport."""

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


@dataclass(frozen=True)
class DispatchBuffers:
    """Padded send buffers produced from canonical ``[token, choice]`` routes."""

    tokens: torch.Tensor
    scores: torch.Tensor
    expert_ids: torch.Tensor


@dataclass(frozen=True)
class FusedDispatchBuffers:
    """BF16 token/score payload and int64 IDs for a two-collective dispatch."""

    payload: torch.Tensor
    expert_ids: torch.Tensor


@dataclass(frozen=True)
class IdFusedDispatchBuffers:
    """BF16 token, score, and exact-small-ID payload for one EP collective."""

    payload: torch.Tensor


@dataclass(frozen=True)
class CompactDispatchBuffers:
    """Exact alltoallv-ready BF16 payload plus generic int64 expert IDs."""

    payload: torch.Tensor
    expert_ids: torch.Tensor


@dataclass(frozen=True)
class CompactIdDispatchBuffers:
    """Exact alltoallv-ready BF16 token, score, and small-ID payload."""

    payload: torch.Tensor


@dataclass(frozen=True)
class CompactSegmentedDispatchBuffers:
    """Exact compact token and score buffers with implicit local expert IDs."""

    tokens: torch.Tensor
    scores: torch.Tensor


def load_ep_route_ops(verbose: bool = False) -> ModuleType:
    """Load the SYCL extension on the current PyTorch XPU stream."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use EP route kernels")

    # A multi-rank job must never race a just-in-time extension build.  The
    # explicit escape hatch lets a worker use a known-compatible, precompiled
    # module while a different candidate is being compiled or validated.  It
    # is intentionally opt-in, and production/source builds retain the normal
    # ``torch.utils.cpp_extension.load`` path below.
    prebuilt = os.environ.get("AURORA_MOE_EP_ROUTE_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_ep_route_ops", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt EP route ops: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
        return _MODULE

    source = Path(__file__).with_name("csrc") / "ep_route_ops.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "ep_route_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_ep_route_ops",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _check_route_matrix(routes: torch.Tensor, name: str) -> None:
    if routes.device.type != "xpu" or routes.dtype != torch.int64 or routes.ndim != 2:
        raise ValueError(f"{name} must be an int64 XPU [tokens, top_k] tensor")
    if not routes.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def destination_counts(destinations: torch.Tensor, num_destinations: int) -> torch.Tensor:
    """Return exact per-destination route counts for canonical route IDs."""

    _check_route_matrix(destinations, "destinations")
    if num_destinations <= 0:
        raise ValueError("num_destinations must be positive")
    return torch.bincount(destinations.reshape(-1), minlength=num_destinations).to(torch.int64)


def destination_expert_counts(
    destinations: torch.Tensor,
    local_ids: torch.Tensor,
    num_destinations: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Return exact ``[destination, local_expert]`` route counts on XPU.

    This is the control-plane counterpart to the direct expert layout.  Each
    source already owns both a destination rank and that rank's local expert
    ID, so a single device ``bincount`` can provide the exact receive-side
    expert-row bounds before the payload all-to-all.  No route is dropped or
    capacity-limited: callers still derive the padded transport width from
    the observed maximum destination count.
    """

    _check_route_matrix(destinations, "destinations")
    _check_route_matrix(local_ids, "local_ids")
    if destinations.shape != local_ids.shape or destinations.device != local_ids.device:
        raise ValueError("destinations and local_ids must have matching XPU shapes")
    if num_destinations <= 0 or num_local_experts <= 0:
        raise ValueError("destination and local-expert counts must be positive")
    # ``topk_indices`` has already been range-checked by the model.  Avoid a
    # scalar readback here: this control path exists specifically to remove a
    # post-dispatch synchronization.  The composed ID keeps the histogram
    # entirely device-resident and works for arbitrary dynamic
    # destination/expert counts.
    combined = destinations.reshape(-1) * num_local_experts + local_ids.reshape(-1)
    return torch.bincount(
        combined, minlength=num_destinations * num_local_experts
    ).reshape(num_destinations, num_local_experts).to(torch.int64)


def make_route_slots(
    destinations: torch.Tensor, num_destinations: int, cap: int
) -> torch.Tensor:
    """Map every ``[token, choice]`` route to a unique flattened ``[ep, cap]`` slot.

    The map is constructed on device with per-destination atomic counters.  It
    has no capacity factor: ``cap`` must be the exact maximum supplied by the
    distributed count exchange, so every input route owns one transport slot.
    """

    _check_route_matrix(destinations, "destinations")
    if num_destinations <= 0 or cap < 0:
        raise ValueError("num_destinations must be positive and cap nonnegative")
    return load_ep_route_ops().route_slots_atomic_i64(
        destinations.reshape(-1), num_destinations, cap
    ).reshape_as(destinations)


def make_compact_route_slots(
    destinations: torch.Tensor, destination_offsets: torch.Tensor
) -> torch.Tensor:
    """Build exact destination-major alltoallv slots without padded capacity.

    ``destination_offsets`` is a contiguous int64 XPU vector of exclusive
    destination offsets, usually built as ``cumsum(send_counts) -
    send_counts``.  The result maps each canonical ``[token, choice]`` route
    to a unique compact send-buffer row.  Atomic ordering inside a destination
    is intentionally unspecified, but the route metadata travels with the
    payload, so it cannot change the mathematical result.

    The offsets must describe the exact destination-major partition of all
    routes: slots are expected to cover ``[0, destinations.numel())`` with no
    gaps.  This is the alltoallv counterpart to :func:`make_route_slots` and
    never allocates capacity padding.
    """

    _check_route_matrix(destinations, "destinations")
    if (
        destination_offsets.device.type != "xpu"
        or destination_offsets.dtype != torch.int64
        or destination_offsets.ndim != 1
        or not destination_offsets.is_contiguous()
        or destination_offsets.device != destinations.device
        or destination_offsets.numel() == 0
    ):
        raise ValueError("destination_offsets must be a contiguous nonempty int64 XPU vector")
    return load_ep_route_ops().compact_route_slots_atomic_i64(
        destinations.reshape(-1), destination_offsets
    ).reshape_as(destinations)


def make_destination_expert_offsets(
    destination_expert_counts: torch.Tensor, destination_offsets: torch.Tensor
) -> torch.Tensor:
    """Return compact global starts for each exact ``[destination, expert]`` segment."""

    if (
        destination_expert_counts.device.type != "xpu"
        or destination_expert_counts.dtype != torch.int64
        or destination_expert_counts.ndim != 2
        or not destination_expert_counts.is_contiguous()
        or destination_expert_counts.size(0) == 0
        or destination_expert_counts.size(1) == 0
    ):
        raise ValueError("destination_expert_counts must be contiguous int64 [destinations, experts]")
    if (
        destination_offsets.device != destination_expert_counts.device
        or destination_offsets.dtype != torch.int64
        or destination_offsets.ndim != 1
        or destination_offsets.numel() != destination_expert_counts.size(0)
        or not destination_offsets.is_contiguous()
    ):
        raise ValueError("destination_offsets must be contiguous int64 with one entry per destination")
    prefix = torch.cat(
        (
            destination_expert_counts.new_zeros((destination_expert_counts.size(0), 1)),
            destination_expert_counts.cumsum(dim=1),
        ),
        dim=1,
    )
    return (prefix + destination_offsets.unsqueeze(1)).contiguous()


def make_compact_destination_expert_slots(
    destinations: torch.Tensor,
    local_ids: torch.Tensor,
    destination_expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Map every route to an exact compact destination/expert segment slot."""

    _check_route_matrix(destinations, "destinations")
    _check_route_matrix(local_ids, "local_ids")
    if destinations.shape != local_ids.shape or destinations.device != local_ids.device:
        raise ValueError("destinations and local_ids must have matching XPU route shapes")
    if (
        destination_expert_offsets.device != destinations.device
        or destination_expert_offsets.dtype != torch.int64
        or destination_expert_offsets.ndim != 2
        or not destination_expert_offsets.is_contiguous()
        or destination_expert_offsets.shape[0] == 0
        or destination_expert_offsets.shape[1] < 2
    ):
        raise ValueError("destination_expert_offsets must be contiguous int64 [destinations, experts + 1]")
    return load_ep_route_ops().compact_destination_expert_slots_atomic_i64(
        destinations.reshape(-1), local_ids.reshape(-1), destination_expert_offsets
    ).reshape_as(destinations)


def pack_routes(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    expert_ids: torch.Tensor,
    route_slots: torch.Tensor,
    output_rows: int,
) -> DispatchBuffers:
    """Pack tokens, scores, and int64 expert IDs into exact padded EP buffers."""

    _check_route_matrix(route_slots, "route_slots")
    _check_route_matrix(expert_ids, "expert_ids")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if not tokens.is_contiguous():
        raise ValueError("tokens must be contiguous")
    if scores.device != tokens.device or scores.shape != route_slots.shape:
        raise ValueError("scores must share route_slots shape and device")
    if scores.dtype not in (torch.bfloat16, torch.float32) or not scores.is_contiguous():
        raise ValueError("scores must be contiguous BF16 or FP32")
    if expert_ids.shape != route_slots.shape or tokens.size(0) != route_slots.size(0):
        raise ValueError("route shapes do not match tokens")
    if output_rows < 0:
        raise ValueError("output_rows must be nonnegative")
    packed = load_ep_route_ops().pack_routes_bf16(
        tokens, scores, expert_ids, route_slots, output_rows
    )
    return DispatchBuffers(*packed)


def pack_token_rows(
    tokens: torch.Tensor, route_slots: torch.Tensor, output_rows: int
) -> torch.Tensor:
    """Copy each token row to every route's padded slot without host loops."""

    _check_route_matrix(route_slots, "route_slots")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if not tokens.is_contiguous() or tokens.size(0) != route_slots.size(0):
        raise ValueError("tokens must be contiguous and match route_slots")
    if output_rows < 0:
        raise ValueError("output_rows must be nonnegative")
    return load_ep_route_ops().pack_token_rows_bf16(tokens, route_slots, output_rows)


def pack_routes_payload(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    expert_ids: torch.Tensor,
    route_slots: torch.Tensor,
    output_rows: int,
) -> FusedDispatchBuffers:
    """Pack BF16 token rows and scores into ``[ep * cap, model_dim + 1]``.

    The last column carries the route score.  It is an optional transport
    layout for reducing forward EP all-to-all calls from three to two; views
    of its token and score columns remain device-resident.
    """

    _check_route_matrix(route_slots, "route_slots")
    _check_route_matrix(expert_ids, "expert_ids")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if scores.device != tokens.device or scores.dtype != torch.bfloat16 or scores.shape != route_slots.shape:
        raise ValueError("fused route payload requires BF16 scores matching route_slots")
    if not tokens.is_contiguous() or not scores.is_contiguous():
        raise ValueError("tokens and scores must be contiguous")
    if expert_ids.shape != route_slots.shape or tokens.size(0) != route_slots.size(0):
        raise ValueError("route shapes do not match tokens")
    if output_rows < 0:
        raise ValueError("output_rows must be nonnegative")
    packed = load_ep_route_ops().pack_routes_payload_bf16(
        tokens, scores, expert_ids, route_slots, output_rows
    )
    return FusedDispatchBuffers(*packed)


def pack_routes_chunked_payload(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    expert_ids: torch.Tensor,
    route_slots: torch.Tensor,
    num_destinations: int,
    cap: int,
    chunk_rows: int,
) -> FusedDispatchBuffers:
    """Pack routes into chunk-major BF16 payload plus exact int64 IDs.

    The output payload is ``[ceil(cap / chunk_rows), destinations,
    chunk_rows, model_dim + 1]`` and IDs share its leading dimensions.  This
    generic path has no small-ID limit and preserves all exact route slots.
    """

    _check_route_matrix(route_slots, "route_slots")
    _check_route_matrix(expert_ids, "expert_ids")
    if num_destinations <= 0 or cap < 0 or chunk_rows <= 0:
        raise ValueError("destination count and chunk rows must be positive and cap nonnegative")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if scores.device != tokens.device or scores.dtype != torch.bfloat16 or scores.shape != route_slots.shape:
        raise ValueError("chunked route payload requires BF16 scores matching route_slots")
    if not tokens.is_contiguous() or not scores.is_contiguous():
        raise ValueError("tokens and scores must be contiguous")
    if expert_ids.shape != route_slots.shape or tokens.size(0) != route_slots.size(0):
        raise ValueError("route shapes do not match tokens")
    if route_slots.numel() and cap == 0:
        raise ValueError("nonempty route slots require a positive cap")
    payload, ids = load_ep_route_ops().pack_routes_chunked_payload_bf16(
        tokens,
        scores,
        expert_ids,
        route_slots,
        num_destinations,
        cap,
        chunk_rows,
    )
    chunks = (cap + chunk_rows - 1) // chunk_rows
    return FusedDispatchBuffers(
        payload.reshape(chunks, num_destinations, chunk_rows, tokens.size(1) + 1),
        ids.reshape(chunks, num_destinations, chunk_rows),
    )


def pack_routes_payload_ids(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    local_ids: torch.Tensor,
    route_slots: torch.Tensor,
    output_rows: int,
    local_count: int,
) -> IdFusedDispatchBuffers:
    """Pack BF16 tokens, scores, and exact BF16 local IDs into one payload.

    BF16 represents every nonnegative integer through 256 exactly.  This
    transport layout is therefore valid only for ``1 <= local_count <= 256``;
    callers with more local experts must retain the int64-ID collective.
    """

    _check_route_matrix(route_slots, "route_slots")
    _check_route_matrix(local_ids, "local_ids")
    if not 1 <= local_count <= 256:
        raise ValueError("BF16 ID payload requires local_count in [1, 256]")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if scores.device != tokens.device or scores.dtype != torch.bfloat16 or scores.shape != route_slots.shape:
        raise ValueError("BF16 ID payload requires BF16 scores matching route_slots")
    if not tokens.is_contiguous() or not scores.is_contiguous():
        raise ValueError("tokens and scores must be contiguous")
    if local_ids.shape != route_slots.shape or tokens.size(0) != route_slots.size(0):
        raise ValueError("route shapes do not match tokens")
    if output_rows < 0:
        raise ValueError("output_rows must be nonnegative")
    payload = load_ep_route_ops().pack_routes_payload_ids_bf16(
        tokens, scores, local_ids, route_slots, output_rows, local_count
    )
    return IdFusedDispatchBuffers(payload)


def pack_routes_chunked_payload_ids(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    local_ids: torch.Tensor,
    route_slots: torch.Tensor,
    num_destinations: int,
    cap: int,
    chunk_rows: int,
    local_count: int,
) -> IdFusedDispatchBuffers:
    """Pack small-ID routes directly into chunk-major padded EP storage.

    The result has shape ``[ceil(cap / chunk_rows), num_destinations,
    chunk_rows, model_dim + 2]``.  It preserves every exact route slot while
    making each chunk independently contiguous for equal-size all-to-all.
    """

    _check_route_matrix(route_slots, "route_slots")
    _check_route_matrix(local_ids, "local_ids")
    if not 1 <= local_count <= 256:
        raise ValueError("chunked BF16 ID payload requires local_count in [1, 256]")
    if num_destinations <= 0 or cap < 0 or chunk_rows <= 0:
        raise ValueError("destination count and chunk rows must be positive and cap nonnegative")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if scores.device != tokens.device or scores.dtype != torch.bfloat16 or scores.shape != route_slots.shape:
        raise ValueError("chunked BF16 ID payload requires BF16 scores matching route_slots")
    if not tokens.is_contiguous() or not scores.is_contiguous():
        raise ValueError("tokens and scores must be contiguous")
    if local_ids.shape != route_slots.shape or tokens.size(0) != route_slots.size(0):
        raise ValueError("route shapes do not match tokens")
    if route_slots.numel() and cap == 0:
        raise ValueError("nonempty route slots require a positive cap")
    payload = load_ep_route_ops().pack_routes_chunked_payload_ids_bf16(
        tokens,
        scores,
        local_ids,
        route_slots,
        num_destinations,
        cap,
        chunk_rows,
        local_count,
    )
    chunks = (cap + chunk_rows - 1) // chunk_rows
    return IdFusedDispatchBuffers(
        payload.reshape(chunks, num_destinations, chunk_rows, tokens.size(1) + 2)
    )


def pack_compact_routes_payload(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    expert_ids: torch.Tensor,
    route_slots: torch.Tensor,
) -> CompactDispatchBuffers:
    """Pack an exact non-padded alltoallv payload and opaque int64 IDs.

    The returned payload has shape ``[routes, model_dim + 1]`` with BF16
    token values and the BF16 score in its final column.  ``expert_ids`` is a
    separate exact int64 transport buffer for arbitrary local expert counts;
    it is the generic fallback when BF16 small-ID fusion is not valid.
    ``route_slots`` must be an exact compact permutation from
    :func:`make_compact_route_slots`.
    """

    _check_route_matrix(route_slots, "route_slots")
    _check_route_matrix(expert_ids, "expert_ids")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if scores.device != tokens.device or scores.dtype != torch.bfloat16 or scores.shape != route_slots.shape:
        raise ValueError("compact route payload requires BF16 scores matching route_slots")
    if not tokens.is_contiguous() or not scores.is_contiguous():
        raise ValueError("tokens and scores must be contiguous")
    if expert_ids.shape != route_slots.shape or tokens.size(0) != route_slots.size(0):
        raise ValueError("route shapes do not match tokens")
    packed = load_ep_route_ops().pack_compact_routes_payload_bf16(
        tokens, scores, expert_ids, route_slots
    )
    return CompactDispatchBuffers(*packed)


def pack_compact_routes_payload_ids(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    local_ids: torch.Tensor,
    route_slots: torch.Tensor,
    local_count: int,
) -> CompactIdDispatchBuffers:
    """Pack a one-collective exact alltoallv payload for up to 256 local IDs.

    The final two BF16 columns are score and local expert ID.  IDs are exact
    only in the small-ID range; callers with ``local_count > 256`` must use
    :func:`pack_compact_routes_payload` and retain its separate int64 IDs.
    """

    _check_route_matrix(route_slots, "route_slots")
    _check_route_matrix(local_ids, "local_ids")
    if not 1 <= local_count <= 256:
        raise ValueError("compact BF16 ID payload requires local_count in [1, 256]")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if scores.device != tokens.device or scores.dtype != torch.bfloat16 or scores.shape != route_slots.shape:
        raise ValueError("compact BF16 ID payload requires BF16 scores matching route_slots")
    if not tokens.is_contiguous() or not scores.is_contiguous():
        raise ValueError("tokens and scores must be contiguous")
    if local_ids.shape != route_slots.shape or tokens.size(0) != route_slots.size(0):
        raise ValueError("route shapes do not match tokens")
    payload = load_ep_route_ops().pack_compact_routes_payload_ids_bf16(
        tokens, scores, local_ids, route_slots, local_count
    )
    return CompactIdDispatchBuffers(payload)


def pack_compact_tokens_scores(
    tokens: torch.Tensor, scores: torch.Tensor, route_slots: torch.Tensor
) -> CompactSegmentedDispatchBuffers:
    """Pack contiguous exact token and score buffers for segmented all-to-all-v."""

    _check_route_matrix(route_slots, "route_slots")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if (
        scores.device != tokens.device
        or scores.dtype != torch.bfloat16
        or scores.shape != route_slots.shape
        or not scores.is_contiguous()
        or not tokens.is_contiguous()
        or tokens.size(0) != route_slots.size(0)
    ):
        raise ValueError("scores and route_slots must match contiguous BF16 tokens")
    packed_tokens, packed_scores = load_ep_route_ops().pack_compact_tokens_scores_bf16(
        tokens, scores, route_slots
    )
    return CompactSegmentedDispatchBuffers(packed_tokens, packed_scores)


def pack_compact_tokens_score_payload(
    tokens: torch.Tensor, scores: torch.Tensor, route_slots: torch.Tensor
) -> torch.Tensor:
    """Pack one exact ``[routes, model_dim + 1]`` BF16 dispatch payload.

    This is the source-side counterpart of the expert-major fused payload
    reorder.  Fragment counts carry the local-expert identity, so unlike the
    generic compact payload it transports neither IDs nor a capacity-sized
    sentinel.  The final column is the routing score.
    """

    _check_route_matrix(route_slots, "route_slots")
    if tokens.device.type != "xpu" or tokens.dtype != torch.bfloat16 or tokens.ndim != 2:
        raise ValueError("tokens must be a BF16 XPU [tokens, model_dim] tensor")
    if (
        scores.device != tokens.device
        or scores.dtype != torch.bfloat16
        or scores.shape != route_slots.shape
        or not scores.is_contiguous()
        or not tokens.is_contiguous()
        or tokens.size(0) != route_slots.size(0)
    ):
        raise ValueError("scores and route_slots must match contiguous BF16 tokens")
    return load_ep_route_ops().pack_compact_tokens_score_payload_bf16(
        tokens, scores, route_slots
    )


def reduce_route_rows(route_values: torch.Tensor, route_slots: torch.Tensor) -> torch.Tensor:
    """Sum padded return rows into canonical tokens without atomic ``index_add_``."""

    _check_route_matrix(route_slots, "route_slots")
    if (
        route_values.device.type != "xpu"
        or route_values.dtype != torch.bfloat16
        or route_values.ndim != 2
        or not route_values.is_contiguous()
    ):
        raise ValueError("route_values must be a contiguous BF16 XPU [rows, model_dim] tensor")
    return load_ep_route_ops().reduce_route_rows_bf16(route_values, route_slots)


def unpack_route_scalars(padded_values: torch.Tensor, route_slots: torch.Tensor) -> torch.Tensor:
    """Gather one returned scalar per route into canonical ``[token, choice]`` order."""

    _check_route_matrix(route_slots, "route_slots")
    if (
        padded_values.device.type != "xpu"
        or padded_values.dtype not in (torch.bfloat16, torch.float32)
        or padded_values.ndim != 1
        or not padded_values.is_contiguous()
    ):
        raise ValueError("padded_values must be contiguous BF16 or FP32 XPU rank-1")
    return load_ep_route_ops().unpack_route_scalars(padded_values, route_slots)


def fuse_route_rows(route_values: torch.Tensor, route_scalars: torch.Tensor) -> torch.Tensor:
    """Fuse BF16 return vectors and one scalar into a transport payload."""

    if (
        route_values.device.type != "xpu"
        or route_values.dtype != torch.bfloat16
        or route_values.ndim != 2
        or not route_values.is_contiguous()
    ):
        raise ValueError("route_values must be contiguous BF16 XPU rank-2")
    if (
        route_scalars.device != route_values.device
        or route_scalars.dtype != torch.bfloat16
        or route_scalars.ndim != 1
        or route_scalars.numel() != route_values.size(0)
        or not route_scalars.is_contiguous()
    ):
        raise ValueError("route_scalars must be contiguous BF16 with one scalar per row")
    return load_ep_route_ops().fuse_route_rows_bf16(route_values, route_scalars)


def reduce_route_payload(
    payload: torch.Tensor, route_slots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return no-atomic token sums and canonical route scalars from fused payload."""

    _check_route_matrix(route_slots, "route_slots")
    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or not payload.is_contiguous()
    ):
        raise ValueError("payload must be contiguous BF16 XPU rank-2")
    if payload.size(1) < 2:
        raise ValueError("payload must contain at least one vector column and one scalar")
    return tuple(load_ep_route_ops().reduce_route_payload_bf16(payload, route_slots))
