"""Exact compact MoE local node with long expert-major PVC GEMMs.

The exact EP transport naturally delivers source-major ``[source, expert]``
fragments.  Running every fragment as a separate GEMM leaves only about one
twelfth of an expert's rows in each group at DP=2/EP=12.  This candidate makes
one exact physical-to-expert-major permutation, runs one true-length group per
local expert, and restores physical order for the reverse all-to-all-v.  It
never constructs ``[experts, max_rows, ...]`` storage or clips routes.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from aurora_moe._kernels.ep_local_ops import split_compact_payload_rows
from aurora_moe._kernels.segment_expert_reorder import (
    ExpertMajorLayout,
    expert_major_to_payload_zero_score_parallel,
    expert_major_to_payload_zero_score_row_parallel,
    expert_major_to_payload_parallel,
    expert_major_to_segments,
    expert_major_to_segments_parallel,
    expert_major_to_segments_scaled_parallel,
    expert_major_to_segments_scaled_row_parallel,
    make_expert_major_layout,
    pair_segments_to_expert_major,
    pair_segments_to_expert_major_parallel,
    payload_and_grad_to_expert_major_scaled_parallel,
    payload_and_grad_to_expert_major_scaled_row_parallel,
    payload_to_expert_major_parallel,
    payload_to_expert_major_row_parallel,
    segments_to_expert_major,
    segments_to_expert_major_parallel,
)
from aurora_moe._kernels.segmented_grouped_gemm import (
    segmented_grouped_gemm_bf16,
    segmented_weight_grad_bf16,
)
from aurora_moe._kernels.segmented_sonic import (
    PeerExpertSegments,
    make_peer_expert_segments,
)


def _record(name: str):
    return (
        torch.profiler.record_function(name)
        if os.environ.get("AURORA_MOE_PROFILE") == "1"
        else nullcontext()
    )


def _gemm_backend() -> str:
    backend = os.environ.get("AURORA_MOE_EXPERT_MAJOR_GEMM", "specialized")
    if backend not in ("generic", "specialized", "torch", "onemkl"):
        raise ValueError(
            "AURORA_MOE_EXPERT_MAJOR_GEMM must be 'generic', 'specialized', 'torch', or 'onemkl'"
        )
    return backend


def _weight_grad_backend() -> str:
    backend = os.environ.get("AURORA_MOE_EXPERT_MAJOR_DW", "segmented")
    if backend not in ("segmented", "xetla", "torch", "onemkl"):
        raise ValueError(
            "AURORA_MOE_EXPERT_MAJOR_DW must be 'segmented', 'xetla', 'torch', or 'onemkl'"
        )
    return backend


def _reorder_backend() -> str:
    backend = os.environ.get("AURORA_MOE_EXPERT_MAJOR_REORDER", "parallel")
    if backend not in ("serial", "parallel", "row_parallel"):
        raise ValueError(
            "AURORA_MOE_EXPERT_MAJOR_REORDER must be 'serial', 'parallel', or 'row_parallel'"
        )
    return backend


def _pointwise_backend() -> str:
    backend = os.environ.get("AURORA_MOE_SEGMENTED_POINTWISE", "torch")
    if backend not in ("torch", "sycl"):
        raise ValueError("AURORA_MOE_SEGMENTED_POINTWISE must be 'torch' or 'sycl'")
    return backend


def _down_backward_strategy() -> str:
    strategy = os.environ.get("AURORA_MOE_EXPERT_MAJOR_DOWN_BACKWARD", "reordered")
    if strategy not in ("reference", "reordered"):
        raise ValueError(
            "AURORA_MOE_EXPERT_MAJOR_DOWN_BACKWARD must be 'reference' or 'reordered'"
        )
    return strategy


def _packed_up_gate_requested(activation: str) -> bool:
    """Whether to fuse the two SwiGLU input projections exactly.

    The packed form is ``[up | gate]`` along the GEMM N dimension.  It keeps
    one true-length expert-major interval per expert and performs no capacity
    padding or route selection.  Keep it opt-in until its distributed
    numerical and throughput gates have run on PVC.
    """

    value = os.environ.get("AURORA_MOE_EXPERT_MAJOR_PACKED_UP_GATE", "0")
    if value not in ("0", "1"):
        raise ValueError("AURORA_MOE_EXPERT_MAJOR_PACKED_UP_GATE must be '0' or '1'")
    return activation == "swiglu" and value == "1"


def _ignore_router_grad_requested() -> bool:
    """Whether the caller explicitly excludes router-score gradients.

    The transport payload contains token columns and one route-score column.
    When router training is outside the experiment's scope, keeping a score
    gradient forces both a dot product and a second payload-wide inverse pack.
    This opt-in mode preserves forward, dX, and expert-weight mathematics,
    but deliberately returns an exact zero for the score-gradient column.
    It is only meaningful for the fused payload API; the regular two-input
    API continues to differentiate scores normally.
    """

    value = os.environ.get("AURORA_MOE_IGNORE_ROUTER_GRAD", "0")
    if value not in ("0", "1"):
        raise ValueError("AURORA_MOE_IGNORE_ROUTER_GRAD must be '0' or '1'")
    return value == "1"


def _onemkl_fuse_up_gate_dx_requested() -> bool:
    """Whether exact oneMKL may accumulate the two SwiGLU dX GEMMs in place."""

    value = os.environ.get("AURORA_MOE_ONEMKL_FUSE_UP_GATE_DX", "0")
    if value not in ("0", "1"):
        raise ValueError("AURORA_MOE_ONEMKL_FUSE_UP_GATE_DX must be '0' or '1'")
    return value == "1"


def make_expert_group_segments(physical_segments: PeerExpertSegments) -> PeerExpertSegments:
    """Return one runtime-size group per logical local expert.

    The leading singleton is only metadata: it describes one packed physical
    source whose contiguous segments are expert-major.  It does not create a
    source/capacity dimension in activation storage.
    """

    expert_rows = physical_segments.counts.sum(dim=0, dtype=torch.int32)
    return make_peer_expert_segments(expert_rows.reshape(1, -1).contiguous())


def _check_inputs(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    physical_segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    activation: str,
) -> None:
    if (
        tokens.device.type != "xpu"
        or tokens.dtype != torch.bfloat16
        or tokens.ndim != 2
        or not tokens.is_contiguous()
    ):
        raise ValueError("tokens must be a contiguous BF16 XPU [routes, model_dim] tensor")
    if (
        scores.device != tokens.device
        or scores.dtype != tokens.dtype
        or scores.ndim != 1
        or scores.numel() != tokens.size(0)
        or not scores.is_contiguous()
    ):
        raise ValueError("scores must be contiguous BF16 with one entry per route")
    _check_expert_inputs(
        tokens, tokens.size(1), physical_segments, up, gate, down, activation
    )


def _check_payload_inputs(
    payload: torch.Tensor,
    physical_segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    activation: str,
) -> None:
    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or payload.size(1) < 2
        or not payload.is_contiguous()
    ):
        raise ValueError(
            "payload must be contiguous BF16 XPU [routes, model_dim + 1]"
        )
    _check_expert_inputs(
        payload, payload.size(1) - 1, physical_segments, up, gate, down, activation
    )


def _split_expert_major_payload(payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous token and score columns from direct expert-major IPC."""

    rows = split_compact_payload_rows(payload)
    return rows.tokens, rows.scores


def _fuse_expert_major_payload_grad(
    tokens: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """Append exact direct-layout dscore values to contiguous dX rows."""

    from aurora_moe._kernels.ep_route_ops import fuse_route_rows

    return fuse_route_rows(tokens, scores)


def _check_expert_inputs(
    reference: torch.Tensor,
    model_dim: int,
    physical_segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    activation: str,
) -> None:
    if (
        physical_segments.counts.device != reference.device
        or physical_segments.counts.dtype != torch.int32
        or physical_segments.source_expert_offsets.device != reference.device
        or physical_segments.source_expert_offsets.dtype != torch.int64
        or physical_segments.source_offsets.device != reference.device
        or physical_segments.source_offsets.dtype != torch.int64
    ):
        raise ValueError("physical segment metadata must be int32/int64 XPU tensors")
    if (
        up.device != reference.device
        or down.device != reference.device
        or up.dtype != reference.dtype
        or down.dtype != reference.dtype
        or up.ndim != 3
        or down.ndim != 3
        or not up.is_contiguous()
        or not down.is_contiguous()
        or up.size(0) != physical_segments.experts
        or up.size(1) != model_dim
        or down.shape != (up.size(0), up.size(2), up.size(1))
    ):
        raise ValueError("expert weights do not match exact compact segments")
    if activation not in ("swiglu", "squared-relu"):
        raise ValueError("activation must be 'swiglu' or 'squared-relu'")
    if activation == "swiglu":
        if (
            gate is None
            or gate.device != reference.device
            or gate.dtype != reference.dtype
            or gate.shape != up.shape
            or not gate.is_contiguous()
        ):
            raise ValueError("SwiGLU gate weights must match contiguous up weights")
    elif gate is not None:
        raise ValueError("squared-relu does not use gate weights")


def _expert_rows_host(expert_segments: PeerExpertSegments) -> tuple[int, ...]:
    """Return exact logical expert row counts for the dense-GEMM diagnostic.

    This is intentionally only used by the explicit ``torch`` backend.  The
    production SYCL paths retain fully device-resident descriptors.  At the
    distributed boundary the all-to-all-v split control already requires a
    host-visible count synchronization, so this experiment tells us whether
    dense-class per-expert GEMMs can close the compute gap before investing in
    another persistent scheduler variant.
    """

    return tuple(int(row) for row in expert_segments.counts.reshape(-1).cpu().tolist())


def _resolve_expert_rows(
    expert_segments: PeerExpertSegments,
    expert_rows_hint: tuple[int, ...] | None,
    *,
    total_rows: int,
    required: bool,
) -> tuple[int, ...] | None:
    """Return one true row count per local expert for host-descriptor GEMMs.

    The distributed compact transport already transfers the complete
    ``[source, destination, local-expert]`` count control tensor to the host
    in order to form its all-to-all-v split descriptors.  Supplying the
    corresponding local-expert totals from that same transfer avoids a second
    device-to-host read after A1 has completed.  The hint remains optional so
    standalone callers retain the simple device-metadata API.
    """

    if not required:
        return None
    if expert_rows_hint is None:
        return _expert_rows_host(expert_segments)
    rows = tuple(int(value) for value in expert_rows_hint)
    if (
        len(rows) != expert_segments.experts
        or any(value < 0 for value in rows)
        or sum(rows) != total_rows
    ):
        raise ValueError(
            "precomputed expert_rows must contain nonnegative true row counts "
            "for every local expert"
        )
    return rows


def _torch_dense_expert_gemm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    expert_rows: tuple[int, ...],
) -> torch.Tensor:
    """Exact no-padding per-expert dense GEMMs used as a performance control.

    Each logical expert owns one contiguous true-length interval after the
    expert-major reorder.  This uses PyTorch's dense XPU GEMM for those
    intervals without BMM, capacity storage, route clipping, or activation
    gathers.  It is deliberately a diagnostic backend rather than a claim
    that host-side descriptor loops are the final SYCL kernel solution.
    """

    if weights.size(0) != len(expert_rows) or sum(expert_rows) != activations.size(0):
        raise ValueError("dense expert rows do not describe the compact activation extent")
    output = activations.new_empty((activations.size(0), weights.size(2)))
    begin = 0
    for expert, rows in enumerate(expert_rows):
        if rows:
            torch.mm(
                activations.narrow(0, begin, rows),
                weights[expert],
                out=output.narrow(0, begin, rows),
            )
        begin += rows
    return output


def _torch_dense_expert_weight_grad(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    expert_rows: tuple[int, ...],
) -> torch.Tensor:
    """Exact no-padding dW control using dense XPU GEMMs per logical expert."""

    if (
        len(expert_rows) == 0
        or sum(expert_rows) != activations.size(0)
        or gradients.size(0) != activations.size(0)
    ):
        raise ValueError("dense expert rows do not describe dW compact inputs")
    output = activations.new_empty(
        (len(expert_rows), activations.size(1), gradients.size(1))
    )
    begin = 0
    for expert, rows in enumerate(expert_rows):
        if rows:
            torch.mm(
                activations.narrow(0, begin, rows).transpose(0, 1),
                gradients.narrow(0, begin, rows),
                out=output[expert],
            )
        else:
            output[expert].zero_()
        begin += rows
    return output


def _grouped_gemm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    expert_segments: PeerExpertSegments,
    backend: str,
    expert_rows: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if backend == "generic":
        return segmented_grouped_gemm_bf16(
            activations,
            weights,
            expert_segments.counts,
            expert_segments.source_expert_offsets,
            expert_segments.source_offsets,
        )
    if backend == "specialized":
        from aurora_moe._kernels.segmented_moe_tla import segmented_moe_grouped_gemm_bf16

        return segmented_moe_grouped_gemm_bf16(
            activations, weights, expert_segments.counts
        )
    if backend == "torch":
        if expert_rows is None:
            expert_rows = _expert_rows_host(expert_segments)
        return _torch_dense_expert_gemm(activations, weights, expert_rows)
    if backend == "onemkl":
        if expert_rows is None:
            expert_rows = _expert_rows_host(expert_segments)
        from aurora_moe._kernels.one_mkl_exact_expert_gemm import exact_expert_gemm_bf16

        return exact_expert_gemm_bf16(activations, weights, expert_rows)
    raise AssertionError(f"unexpected expert-major GEMM backend: {backend}")


def _grouped_gemm_with_forward_weight_transpose(
    activations: torch.Tensor,
    forward_weights: torch.Tensor,
    expert_segments: PeerExpertSegments,
    backend: str,
    expert_rows: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Apply ``activations @ forward_weights.T`` without needless dX copies.

    The direct oneMKL implementation passes a transpose flag against the
    normal forward `[E, output, input]` weights.  Other diagnostic backends
    retain the established materialized-transpose fallback until they acquire
    an equivalent validated contract.
    """

    if backend == "onemkl":
        if expert_rows is None:
            expert_rows = _expert_rows_host(expert_segments)
        from aurora_moe._kernels.one_mkl_exact_expert_gemm import (
            exact_expert_gemm_transposed_weight_bf16,
        )

        return exact_expert_gemm_transposed_weight_bf16(
            activations, forward_weights, expert_rows
        )
    return _grouped_gemm(
        activations,
        forward_weights.transpose(-1, -2).contiguous(),
        expert_segments,
        backend,
        expert_rows,
    )


def _grouped_sum_with_forward_weight_transposes(
    left_activations: torch.Tensor,
    left_weights: torch.Tensor,
    right_activations: torch.Tensor,
    right_weights: torch.Tensor,
    expert_segments: PeerExpertSegments,
    expert_rows: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Accumulate two exact oneMKL dX GEMMs into one compact expert-major output."""

    if expert_rows is None:
        expert_rows = _expert_rows_host(expert_segments)
    from aurora_moe._kernels.one_mkl_exact_expert_gemm import (
        exact_expert_sum_transposed_weight_bf16,
    )

    return exact_expert_sum_transposed_weight_bf16(
        left_activations,
        left_weights,
        right_activations,
        right_weights,
        expert_rows,
    )


def _weight_grad(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    expert_segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
    backend: str,
    expert_rows: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if backend == "segmented":
        return segmented_weight_grad_bf16(
            activations,
            gradients,
            expert_segments.counts,
            expert_segments.source_expert_offsets,
            expert_segments.source_offsets,
        )
    if backend == "xetla":
        from aurora_moe._kernels.xetla_grouped_gemm import grouped_weight_grad_bf16

        return grouped_weight_grad_bf16(
            activations, gradients, layout.expert_offsets_i32
        )
    if backend == "torch":
        if expert_rows is None:
            expert_rows = _expert_rows_host(expert_segments)
        return _torch_dense_expert_weight_grad(activations, gradients, expert_rows)
    if backend == "onemkl":
        if expert_rows is None:
            expert_rows = _expert_rows_host(expert_segments)
        from aurora_moe._kernels.one_mkl_exact_expert_gemm import exact_expert_weight_grad_bf16

        return exact_expert_weight_grad_bf16(activations, gradients, expert_rows)
    raise AssertionError(f"unexpected expert-major dW backend: {backend}")


def _swiglu(up: torch.Tensor, gate: torch.Tensor, backend: str) -> torch.Tensor:
    if backend == "sycl":
        from aurora_moe._kernels.swiglu_ops import load_swiglu_ops

        return load_swiglu_ops().swiglu_forward_bf16(up, gate)
    return up * F.silu(gate)


def _swiglu_backward(
    grad_hidden: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, backend: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "sycl":
        from aurora_moe._kernels.swiglu_ops import load_swiglu_ops

        return load_swiglu_ops().swiglu_backward_bf16(grad_hidden, up, gate)
    sigmoid = torch.sigmoid(gate)
    return (
        grad_hidden * gate * sigmoid,
        grad_hidden * up * sigmoid * (1.0 + gate * (1.0 - sigmoid)),
    )


def _packed_swiglu(projection: torch.Tensor, backend: str) -> torch.Tensor:
    """Apply SwiGLU directly to a contiguous ``[up | gate]`` projection."""

    if projection.size(-1) <= 0 or projection.size(-1) % 2:
        raise ValueError("packed SwiGLU projection must have a positive even trailing dimension")
    if backend == "sycl":
        from aurora_moe._kernels.packed_swiglu_ops import packed_swiglu_bf16

        return packed_swiglu_bf16(projection)
    hidden = projection.size(-1) // 2
    return projection[..., :hidden] * F.silu(projection[..., hidden:])


def _packed_swiglu_backward(
    grad_hidden: torch.Tensor, projection: torch.Tensor, backend: str
) -> torch.Tensor:
    """Return contiguous gradients in the matching ``[up | gate]`` layout."""

    if backend == "sycl":
        from aurora_moe._kernels.packed_swiglu_ops import load_packed_swiglu_ops

        return load_packed_swiglu_ops().packed_swiglu_backward_bf16(
            grad_hidden.contiguous(), projection
        )
    hidden = projection.size(-1) // 2
    grad_up, grad_gate = _swiglu_backward(
        grad_hidden, projection[..., :hidden], projection[..., hidden:], backend
    )
    return torch.cat((grad_up, grad_gate), dim=-1)


def _pack(
    values: torch.Tensor,
    physical_segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
    reorder_backend: str,
) -> torch.Tensor:
    if reorder_backend in ("parallel", "row_parallel"):
        return segments_to_expert_major_parallel(values, physical_segments, layout)
    return segments_to_expert_major(values, physical_segments, layout)


def _pack_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    physical_segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
    reorder_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reorder_backend in ("parallel", "row_parallel"):
        return pair_segments_to_expert_major_parallel(
            first, second, physical_segments, layout
        )
    return pair_segments_to_expert_major(first, second, physical_segments, layout)


def _unpack(
    values: torch.Tensor,
    physical_segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
    reorder_backend: str,
) -> torch.Tensor:
    if reorder_backend in ("parallel", "row_parallel"):
        return expert_major_to_segments_parallel(values, physical_segments, layout)
    return expert_major_to_segments(values, physical_segments, layout)


def _unpack_scaled(
    values: torch.Tensor,
    scores: torch.Tensor,
    physical_segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
    reorder_backend: str,
) -> torch.Tensor:
    """Return physical rows after exact route-score weighting.

    The parallel path folds its score multiply into the required inverse
    source/expert permutation.  Retain the serial fallback for complete
    runtime-config support while avoiding a second implementation of its
    intentionally diagnostic schedule.
    """

    if reorder_backend == "row_parallel":
        return expert_major_to_segments_scaled_row_parallel(
            values, scores, physical_segments, layout
        )
    if reorder_backend == "parallel":
        return expert_major_to_segments_scaled_parallel(
            values, scores, physical_segments, layout
        )
    return _unpack(
        values * scores.unsqueeze(-1), physical_segments, layout, reorder_backend
    )


def _pack_payload(
    payload: torch.Tensor,
    physical_segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
    reorder_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused, exact payload unpack used by the one-collective route path."""

    if reorder_backend == "row_parallel":
        return payload_to_expert_major_row_parallel(payload, physical_segments, layout)
    return payload_to_expert_major_parallel(payload, physical_segments, layout)


def _unpack_payload(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    physical_segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
    reorder_backend: str,
) -> torch.Tensor:
    """Fused, exact payload pack for backward's reverse all-to-all-v."""

    # The normal score-gradient path is outside the router-free experiment.
    # Retain the validated element-tiled inverse here when `row_parallel` is
    # selected, rather than silently changing its numerical implementation.
    return expert_major_to_payload_parallel(tokens, scores, physical_segments, layout)


class _ExpertMajorSegmentedLocalMoE(torch.autograd.Function):
    """Custom autograd node with physical input and exact expert-major compute."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        physical_counts: torch.Tensor,
        physical_source_expert_offsets: torch.Tensor,
        physical_source_offsets: torch.Tensor,
        expert_counts: torch.Tensor,
        up: torch.Tensor,
        gate: torch.Tensor,
        down: torch.Tensor,
        activation: str,
        pointwise_backend: str,
        down_strategy: str,
        gemm_backend: str,
        dw_backend: str,
        reorder_backend: str,
        payload_input: bool,
        expert_rows_hint: tuple[int, ...] | None,
        already_expert_major: bool,
    ) -> torch.Tensor:
        physical_segments = PeerExpertSegments(
            physical_counts, physical_source_expert_offsets, physical_source_offsets
        )
        gate_arg = gate if activation == "swiglu" else None
        if payload_input:
            _check_payload_inputs(tokens, physical_segments, up, gate_arg, down, activation)
        else:
            _check_inputs(tokens, scores, physical_segments, up, gate_arg, down, activation)
        expert_segments = make_peer_expert_segments(expert_counts)
        expert_rows = _resolve_expert_rows(
            expert_segments,
            expert_rows_hint,
            total_rows=tokens.size(0),
            required=gemm_backend in ("torch", "onemkl")
            or dw_backend in ("torch", "onemkl"),
        )
        layout = make_expert_major_layout(
            physical_segments, include_row_schedule=reorder_backend == "row_parallel"
        )
        if already_expert_major and not payload_input:
            raise ValueError("only fused token/score payloads may already be expert-major")
        with _record(f"moe.expert_major.reorder.{reorder_backend}.forward"):
            if payload_input and already_expert_major:
                # Direct IPC has already performed the exact runtime
                # [source, expert] -> [expert, source] placement.  Retain
                # the normal compact payload columns but do not materialize
                # the inverse/forward reorder pair again.
                grouped_tokens, grouped_scores = _split_expert_major_payload(tokens)
            elif payload_input:
                grouped_tokens, grouped_scores = _pack_payload(
                    tokens, physical_segments, layout, reorder_backend
                )
            else:
                grouped_tokens, grouped_scores_matrix = _pack_pair(
                    tokens,
                    scores.unsqueeze(-1).contiguous(),
                    physical_segments,
                    layout,
                    reorder_backend,
                )
                grouped_scores = grouped_scores_matrix.reshape(-1)
        packed_up_gate = _packed_up_gate_requested(activation)
        if packed_up_gate:
            # One exact [D, 2H] grouped projection replaces the up and gate
            # [D, H] launches.  The rows remain compact and expert-major; the
            # temporary concatenation only changes the output-column layout
            # of each expert weight.
            with _record(f"moe.expert_major.{gemm_backend}.packed_up_gate"):
                preact_up = _grouped_gemm(
                    grouped_tokens,
                    torch.cat((up, gate), dim=-1).contiguous(),
                    expert_segments,
                    gemm_backend,
                    expert_rows,
                )
            preact_gate = tokens.new_empty(0)
        else:
            with _record(f"moe.expert_major.{gemm_backend}.up"):
                preact_up = _grouped_gemm(
                    grouped_tokens, up, expert_segments, gemm_backend, expert_rows
                )
        if activation == "swiglu":
            if not packed_up_gate:
                with _record(f"moe.expert_major.{gemm_backend}.gate"):
                    preact_gate = _grouped_gemm(
                        grouped_tokens, gate, expert_segments, gemm_backend, expert_rows
                    )
            with _record(f"moe.expert_major.pointwise.{pointwise_backend}.forward"):
                hidden = (
                    _packed_swiglu(preact_up, pointwise_backend)
                    if packed_up_gate
                    else _swiglu(preact_up, preact_gate, pointwise_backend)
                )
        else:
            preact_gate = tokens.new_empty(0)
            with _record("moe.expert_major.pointwise.squared_relu.forward"):
                hidden = F.relu(preact_up).square()
        with _record(f"moe.expert_major.{gemm_backend}.down"):
            values = _grouped_gemm(
                hidden, down, expert_segments, gemm_backend, expert_rows
            )
        with _record(f"moe.expert_major.reorder.{reorder_backend}.output"):
            output = (
                values * grouped_scores.unsqueeze(-1)
                if already_expert_major
                else _unpack_scaled(
                    values,
                    grouped_scores,
                    physical_segments,
                    layout,
                    reorder_backend,
                )
            )
        ctx.activation = activation
        ctx.pointwise_backend = pointwise_backend
        ctx.down_strategy = down_strategy
        ctx.gemm_backend = gemm_backend
        ctx.dw_backend = dw_backend
        ctx.reorder_backend = reorder_backend
        ctx.payload_input = payload_input
        ctx.already_expert_major = already_expert_major
        ctx.ignore_router_grad = payload_input and _ignore_router_grad_requested()
        ctx.packed_up_gate = packed_up_gate
        ctx.expert_rows = expert_rows
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            tokens,
            scores,
            physical_counts,
            physical_source_expert_offsets,
            physical_source_offsets,
            expert_counts,
            up,
            gate,
            down,
            layout.expert_offsets_i64,
            layout.expert_offsets_i32,
            layout.expert_source_offsets,
            layout.fragment_row_block_offsets,
            preact_up,
            preact_gate,
        )
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_output is None:
            return (None,) * 18
        (
            tokens,
            scores,
            physical_counts,
            physical_source_expert_offsets,
            physical_source_offsets,
            expert_counts,
            up,
            gate,
            down,
            expert_offsets_i64,
            expert_offsets_i32,
            expert_source_offsets,
            fragment_row_block_offsets,
            preact_up,
            preact_gate,
        ) = ctx.saved_tensors
        physical_segments = PeerExpertSegments(
            physical_counts, physical_source_expert_offsets, physical_source_offsets
        )
        expert_segments = make_peer_expert_segments(expert_counts)
        layout = ExpertMajorLayout(
            expert_offsets_i64,
            expert_offsets_i32,
            expert_source_offsets,
            fragment_row_block_offsets,
        )
        if grad_output.dtype != tokens.dtype or not grad_output.is_contiguous():
            grad_output = grad_output.contiguous().to(dtype=tokens.dtype)
        router_free_payload = ctx.payload_input and ctx.ignore_router_grad
        grouped_payload_tokens = None
        grouped_grad_output = None
        pre_scaled_grad_values = None
        with _record(f"moe.expert_major.reorder.{ctx.reorder_backend}.grad_output"):
            if router_free_payload and ctx.down_strategy == "reordered":
                # One physical-to-expert-major traversal produces exactly the
                # two operands required by the router-free backward: compact
                # tokens for dW-up/gate and dY*score for down dX/dW.
                if ctx.already_expert_major:
                    grouped_payload_tokens, grouped_scores = _split_expert_major_payload(
                        tokens
                    )
                    pre_scaled_grad_values = (
                        grad_output * grouped_scores.unsqueeze(-1)
                    )
                else:
                    grouped_payload_tokens, pre_scaled_grad_values = (
                        payload_and_grad_to_expert_major_scaled_row_parallel(
                            tokens, grad_output, physical_segments, layout
                        )
                        if ctx.reorder_backend == "row_parallel"
                        else payload_and_grad_to_expert_major_scaled_parallel(
                            tokens, grad_output, physical_segments, layout
                        )
                    )
            elif ctx.already_expert_major:
                grouped_grad_output = grad_output
            else:
                grouped_grad_output = _pack(
                    grad_output, physical_segments, layout, ctx.reorder_backend
                )
            if ctx.payload_input and grouped_payload_tokens is None:
                if ctx.already_expert_major:
                    grouped_payload_tokens, grouped_scores = _split_expert_major_payload(
                        tokens
                    )
                else:
                    grouped_payload_tokens, grouped_scores = _pack_payload(
                        tokens, physical_segments, layout, ctx.reorder_backend
                    )
            elif not ctx.payload_input:
                grouped_scores = _pack(
                    scores.unsqueeze(-1).contiguous(),
                    physical_segments,
                    layout,
                    ctx.reorder_backend,
                ).reshape(-1)
        if ctx.activation == "swiglu":
            with _record(f"moe.expert_major.pointwise.{ctx.pointwise_backend}.recompute"):
                hidden = (
                    _packed_swiglu(preact_up, ctx.pointwise_backend)
                    if ctx.packed_up_gate
                    else _swiglu(preact_up, preact_gate, ctx.pointwise_backend)
                )
        else:
            with _record("moe.expert_major.pointwise.squared_relu.recompute"):
                hidden = F.relu(preact_up).square()

        (
            need_tokens,
            need_scores,
            _,
            _,
            _,
            _,
            need_up,
            need_gate,
            need_down,
        ) = ctx.needs_input_grad[:9]
        if ctx.payload_input:
            # The one payload input owns both the token and score columns.
            # Its backward must therefore form both logical gradients even
            # though the dormant second tensor input is not differentiable.
            need_scores = need_tokens and not ctx.ignore_router_grad
        need_activation_backward = need_tokens or need_up or need_gate
        grad_values = None
        grad_hidden = None
        grouped_grad_scores = None
        if ctx.down_strategy == "reordered":
            if router_free_payload:
                assert pre_scaled_grad_values is not None
                if need_down or need_activation_backward:
                    grad_values = pre_scaled_grad_values
                if need_activation_backward:
                    assert grad_values is not None
                    with _record(f"moe.expert_major.{ctx.gemm_backend}.down_dx"):
                        grad_hidden = _grouped_gemm_with_forward_weight_transpose(
                            grad_values,
                            down,
                            expert_segments,
                            ctx.gemm_backend,
                            ctx.expert_rows,
                        )
            elif need_scores or need_activation_backward:
                assert grouped_grad_output is not None
                with _record(f"moe.expert_major.{ctx.gemm_backend}.down_dx_unscaled"):
                    down_input_unscaled = _grouped_gemm_with_forward_weight_transpose(
                        grouped_grad_output,
                        down,
                        expert_segments,
                        ctx.gemm_backend,
                        ctx.expert_rows,
                    )
                with _record("moe.expert_major.down_backward_pointwise"):
                    grad_hidden = down_input_unscaled * grouped_scores.unsqueeze(-1)
                    grouped_grad_scores = (
                        down_input_unscaled.float() * hidden.float()
                    ).sum(dim=-1).to(scores.dtype)
            if need_down and not router_free_payload:
                assert grouped_grad_output is not None
                grad_values = grouped_grad_output * grouped_scores.unsqueeze(-1)
        else:
            with _record(f"moe.expert_major.{ctx.gemm_backend}.down_recompute"):
                values = _grouped_gemm(
                    hidden, down, expert_segments, ctx.gemm_backend, ctx.expert_rows
                )
            if need_scores:
                grouped_grad_scores = (
                    grouped_grad_output.float() * values.float()
                ).sum(dim=-1).to(scores.dtype)
            if need_down or need_activation_backward:
                grad_values = grouped_grad_output * grouped_scores.unsqueeze(-1)
            if need_activation_backward:
                assert grad_values is not None
                with _record(f"moe.expert_major.{ctx.gemm_backend}.down_dx"):
                    grad_hidden = _grouped_gemm_with_forward_weight_transpose(
                        grad_values,
                        down,
                        expert_segments,
                        ctx.gemm_backend,
                        ctx.expert_rows,
                    )

        grad_down = None
        if need_down:
            assert grad_values is not None
            with _record(f"moe.expert_major.dw.{ctx.dw_backend}.down"):
                grad_down = _weight_grad(
                    hidden,
                    grad_values,
                    expert_segments,
                    layout,
                    ctx.dw_backend,
                    ctx.expert_rows,
                )
        grad_up = None
        grad_gate = None
        grouped_grad_tokens = None
        fused_up_gate_dx = (
            ctx.activation == "swiglu"
            and not ctx.packed_up_gate
            and need_tokens
            and ctx.gemm_backend == "onemkl"
            and _onemkl_fuse_up_gate_dx_requested()
        )
        if need_activation_backward:
            assert grad_hidden is not None
            grouped_tokens = None
            if need_up or need_gate:
                with _record(f"moe.expert_major.reorder.{ctx.reorder_backend}.tokens"):
                    if ctx.payload_input:
                        assert grouped_payload_tokens is not None
                        grouped_tokens = grouped_payload_tokens
                    else:
                        grouped_tokens = _pack(
                            tokens, physical_segments, layout, ctx.reorder_backend
                        )
            if ctx.activation == "swiglu":
                if ctx.packed_up_gate:
                    # The packed pointwise backward preserves the [up | gate]
                    # column layout.  One grouped dW and one grouped dX then
                    # replace the two independent SwiGLU-projection paths.
                    with _record(
                        f"moe.expert_major.pointwise.{ctx.pointwise_backend}.packed_backward"
                    ):
                        grad_up_gate_values = _packed_swiglu_backward(
                            grad_hidden, preact_up, ctx.pointwise_backend
                        )
                    hidden_dim = up.size(-1)
                    if need_up or need_gate:
                        assert grouped_tokens is not None
                        with _record(f"moe.expert_major.dw.{ctx.dw_backend}.packed_up_gate"):
                            grad_up_gate = _weight_grad(
                                grouped_tokens,
                                grad_up_gate_values,
                                expert_segments,
                                layout,
                                ctx.dw_backend,
                                ctx.expert_rows,
                            )
                        if need_up:
                            grad_up = grad_up_gate[..., :hidden_dim].contiguous()
                        if need_gate:
                            grad_gate = grad_up_gate[..., hidden_dim:].contiguous()
                    if need_tokens:
                        with _record(
                            f"moe.expert_major.{ctx.gemm_backend}.packed_up_gate_dx"
                        ):
                            grouped_grad_tokens = _grouped_gemm_with_forward_weight_transpose(
                                grad_up_gate_values,
                                torch.cat((up, gate), dim=-1).contiguous(),
                                expert_segments,
                                ctx.gemm_backend,
                                ctx.expert_rows,
                            )
                else:
                    with _record(f"moe.expert_major.pointwise.{ctx.pointwise_backend}.backward"):
                        grad_up_values, grad_gate_values = _swiglu_backward(
                            grad_hidden, preact_up, preact_gate, ctx.pointwise_backend
                        )
                    if need_gate:
                        assert grouped_tokens is not None
                        with _record(f"moe.expert_major.dw.{ctx.dw_backend}.gate"):
                            grad_gate = _weight_grad(
                                grouped_tokens,
                                grad_gate_values,
                                expert_segments,
                                layout,
                                ctx.dw_backend,
                                ctx.expert_rows,
                            )
                    if need_tokens and not fused_up_gate_dx:
                        with _record(f"moe.expert_major.{ctx.gemm_backend}.gate_dx"):
                            grouped_grad_tokens = _grouped_gemm_with_forward_weight_transpose(
                                grad_gate_values,
                                gate,
                                expert_segments,
                                ctx.gemm_backend,
                                ctx.expert_rows,
                            )
            else:
                with _record("moe.expert_major.pointwise.squared_relu.backward"):
                    grad_up_values = grad_hidden * (2.0 * F.relu(preact_up))
                grouped_grad_tokens = (
                    torch.zeros_like(grad_values) if need_tokens else None
                )
            if need_up and not ctx.packed_up_gate:
                assert grouped_tokens is not None
                with _record(f"moe.expert_major.dw.{ctx.dw_backend}.up"):
                    grad_up = _weight_grad(
                        grouped_tokens,
                        grad_up_values,
                        expert_segments,
                        layout,
                        ctx.dw_backend,
                        ctx.expert_rows,
                    )
            if need_tokens and not ctx.packed_up_gate:
                if fused_up_gate_dx:
                    with _record(f"moe.expert_major.{ctx.gemm_backend}.fused_up_gate_dx"):
                        grouped_grad_tokens = _grouped_sum_with_forward_weight_transposes(
                            grad_gate_values,
                            gate,
                            grad_up_values,
                            up,
                            expert_segments,
                            ctx.expert_rows,
                        )
                else:
                    with _record(f"moe.expert_major.{ctx.gemm_backend}.up_dx"):
                        up_tokens = _grouped_gemm_with_forward_weight_transpose(
                            grad_up_values,
                            up,
                            expert_segments,
                            ctx.gemm_backend,
                            ctx.expert_rows,
                        )
                    assert grouped_grad_tokens is not None
                    grouped_grad_tokens = grouped_grad_tokens + up_tokens

        grad_tokens = None
        if need_tokens:
            assert grouped_grad_tokens is not None
            if not ctx.payload_input:
                with _record(f"moe.expert_major.reorder.{ctx.reorder_backend}.input_grad"):
                    grad_tokens = _unpack(
                        grouped_grad_tokens,
                        physical_segments,
                        layout,
                        ctx.reorder_backend,
                    )
        grad_scores = None
        if ctx.payload_input and ctx.ignore_router_grad and need_tokens:
            assert grouped_grad_tokens is not None
            with _record(
                f"moe.expert_major.reorder.{ctx.reorder_backend}.payload_grad_zero_score"
            ):
                grad_tokens = (
                    _fuse_expert_major_payload_grad(
                        grouped_grad_tokens, torch.zeros_like(grouped_grad_tokens[:, 0])
                    )
                    if ctx.already_expert_major
                    else expert_major_to_payload_zero_score_row_parallel(
                        grouped_grad_tokens, physical_segments, layout
                    )
                    if ctx.reorder_backend == "row_parallel"
                    else expert_major_to_payload_zero_score_parallel(
                        grouped_grad_tokens, physical_segments, layout
                    )
                )
        if need_scores:
            assert grouped_grad_scores is not None
            if ctx.payload_input:
                assert grouped_grad_tokens is not None
                with _record("moe.expert_major.reorder.parallel.payload_grad"):
                    grad_tokens = (
                        _fuse_expert_major_payload_grad(
                            grouped_grad_tokens, grouped_grad_scores
                        )
                        if ctx.already_expert_major
                        else _unpack_payload(
                            grouped_grad_tokens,
                            grouped_grad_scores,
                            physical_segments,
                            layout,
                            ctx.reorder_backend,
                        )
                    )
            else:
                with _record(f"moe.expert_major.reorder.{ctx.reorder_backend}.score_grad"):
                    grad_scores = _unpack(
                        grouped_grad_scores.unsqueeze(-1),
                        physical_segments,
                        layout,
                        ctx.reorder_backend,
                    ).reshape(-1)
        return (
            grad_tokens,
            grad_scores,
            None,
            None,
            None,
            None,
            grad_up,
            grad_gate,
            grad_down,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def expert_major_segmented_local_moe(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    physical_segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    activation: str = "swiglu",
    expert_rows: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Run exact physical segments through long true-length expert GEMMs."""

    gate_tensor = gate if gate is not None else tokens.new_empty(0)
    expert_segments = make_expert_group_segments(physical_segments)
    return _ExpertMajorSegmentedLocalMoE.apply(
        tokens,
        scores,
        physical_segments.counts,
        physical_segments.source_expert_offsets,
        physical_segments.source_offsets,
        expert_segments.counts,
        up,
        gate_tensor,
        down,
        activation,
        _pointwise_backend(),
        _down_backward_strategy(),
        _gemm_backend(),
        _weight_grad_backend(),
        _reorder_backend(),
        False,
        expert_rows,
        False,
    )


def expert_major_segmented_local_moe_payload(
    payload: torch.Tensor,
    physical_segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    activation: str = "swiglu",
    expert_rows: tuple[int, ...] | None = None,
    already_expert_major: bool = False,
) -> torch.Tensor:
    """Run the exact expert-major local MoE from one token/score payload.

    ``payload`` has the normal contiguous transport layout ``[routes, D + 1]``
    rather than a strided GEMM input.  By default the custom node uses fused
    exact source/expert permutations to obtain ``[routes, D]`` expert-major
    GEMM rows and to return the matching fused input gradient.  When
    ``already_expert_major`` is true, a direct IPC transport has already made
    that runtime permutation; forward and backward retain the expert-major
    layout so its inverse IPC phase can return it without a local remap.
    """

    gate_tensor = gate if gate is not None else payload.new_empty(0)
    expert_segments = make_expert_group_segments(physical_segments)
    return _ExpertMajorSegmentedLocalMoE.apply(
        payload,
        payload.new_empty(0),
        physical_segments.counts,
        physical_segments.source_expert_offsets,
        physical_segments.source_offsets,
        expert_segments.counts,
        up,
        gate_tensor,
        down,
        activation,
        _pointwise_backend(),
        _down_backward_strategy(),
        _gemm_backend(),
        _weight_grad_backend(),
        _reorder_backend(),
        True,
        expert_rows,
        already_expert_major,
    )


def _reference_pack(
    values: torch.Tensor,
    physical_segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Portable exact permutation used only by CPU/reference tests."""

    output = torch.empty_like(values)
    for source in range(physical_segments.sources):
        source_begin = int(physical_segments.source_offsets[source])
        for expert in range(physical_segments.experts):
            rows = int(physical_segments.counts[source, expert])
            if not rows:
                continue
            physical_begin = source_begin + int(
                physical_segments.source_expert_offsets[source, expert]
            )
            expert_begin = int(layout.expert_offsets_i64[expert]) + int(
                layout.expert_source_offsets[source, expert]
            )
            output[expert_begin : expert_begin + rows] = values[
                physical_begin : physical_begin + rows
            ]
    return output


def _reference_unpack(
    values: torch.Tensor,
    physical_segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Portable inverse exact permutation used only by CPU/reference tests."""

    output = torch.empty_like(values)
    for source in range(physical_segments.sources):
        source_begin = int(physical_segments.source_offsets[source])
        for expert in range(physical_segments.experts):
            rows = int(physical_segments.counts[source, expert])
            if not rows:
                continue
            physical_begin = source_begin + int(
                physical_segments.source_expert_offsets[source, expert]
            )
            expert_begin = int(layout.expert_offsets_i64[expert]) + int(
                layout.expert_source_offsets[source, expert]
            )
            output[physical_begin : physical_begin + rows] = values[
                expert_begin : expert_begin + rows
            ]
    return output


def expert_major_segmented_reference_local_moe(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    physical_segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    activation: str = "swiglu",
) -> torch.Tensor:
    """Portable exact oracle with the same compact/expert-major layout contract.

    It intentionally uses ordinary per-expert matmuls and is never a
    performance path.  Its differing data order makes it useful for checking
    the physical-to-expert-major pipeline independently of the source-fragment
    loop reference.
    """

    if activation not in ("swiglu", "squared-relu"):
        raise ValueError("activation must be 'swiglu' or 'squared-relu'")
    if activation == "swiglu" and gate is None:
        raise ValueError("SwiGLU requires gate weights")
    if activation == "squared-relu" and gate is not None:
        raise ValueError("squared-relu does not use gate weights")
    layout = make_expert_major_layout(physical_segments)
    grouped_tokens = _reference_pack(tokens, physical_segments, layout)
    grouped_scores = _reference_pack(
        scores.unsqueeze(-1), physical_segments, layout
    ).reshape(-1)
    grouped_values = torch.empty_like(grouped_tokens)
    for expert in range(physical_segments.experts):
        begin = int(layout.expert_offsets_i64[expert])
        end = int(layout.expert_offsets_i64[expert + 1])
        if begin == end:
            continue
        preact = grouped_tokens[begin:end].matmul(up[expert])
        if activation == "swiglu":
            assert gate is not None
            hidden = preact * F.silu(grouped_tokens[begin:end].matmul(gate[expert]))
        else:
            hidden = F.relu(preact).square()
        grouped_values[begin:end] = hidden.matmul(down[expert]) * grouped_scores[
            begin:end, None
        ]
    return _reference_unpack(grouped_values, physical_segments, layout)
