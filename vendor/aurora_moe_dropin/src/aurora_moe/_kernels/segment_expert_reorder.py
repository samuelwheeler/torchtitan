"""Exact source/expert-to-expert-major row packing for segmented MoE dW."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch

from aurora_moe._kernels.segmented_sonic import PeerExpertSegments
from aurora_moe._kernels.xetla_grouped_gemm import _load_without_sycl2020


_MODULE: ModuleType | None = None
_PARALLEL_ELEMENTS_PER_WORKGROUP = 64 * 1024
# The row-tiled schedule is independent of the matrix width, unlike the
# older element-tiled schedule.  It is therefore built once with the layout
# and reused by every D-wide payload/reorder operation in a MoE invocation.
# This is a kernel scheduling constant, not a capacity or a model-shape
# restriction: short fragments simply receive fewer workgroups.
_ROW_PARALLEL_ROWS_PER_WORKGROUP = 32


@dataclass(frozen=True)
class ExpertMajorLayout:
    """Runtime offsets that map physical source/expert rows to expert-major rows."""

    expert_offsets_i64: torch.Tensor
    expert_offsets_i32: torch.Tensor
    expert_source_offsets: torch.Tensor
    fragment_row_block_offsets: torch.Tensor


def load_segment_expert_reorder_ops(verbose: bool = False) -> ModuleType:
    """Load the current-stream exact row-reorder kernels."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use source/expert reorder kernels")
    prebuilt = os.environ.get("AURORA_MOE_SEGMENT_EXPERT_REORDER_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_segment_expert_reorder", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt segment reorder ops: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        # A prebuilt may legitimately predate the optional row-tiled A/B
        # kernels while still contain the established serial/element-parallel
        # exact reorder path.  Do not reject that usable production backend
        # wholesale.  Require the row symbols only when the caller actually
        # selects the row-tiled schedule; its source build remains available
        # for that experiment.
        required = (
            "segments_to_expert_major_bf16",
            "pair_segments_to_expert_major_bf16",
            "expert_major_to_segments_bf16",
            "segments_to_expert_major_parallel_bf16",
            "pair_segments_to_expert_major_parallel_bf16",
            "expert_major_to_segments_parallel_bf16",
            "expert_major_to_segments_scaled_parallel_bf16",
            "payload_to_expert_major_parallel_bf16",
            "payload_and_grad_to_expert_major_scaled_parallel_bf16",
            "expert_major_to_payload_parallel_bf16",
            "expert_major_to_payload_zero_score_parallel_bf16",
        )
        if os.environ.get("AURORA_MOE_EXPERT_MAJOR_REORDER", "parallel") == "row_parallel":
            required += (
                "expert_major_to_segments_scaled_row_parallel_bf16",
                "payload_to_expert_major_row_parallel_bf16",
                "payload_and_grad_to_expert_major_scaled_row_parallel_bf16",
                "expert_major_to_payload_zero_score_row_parallel_bf16",
            )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(
                "prebuilt segment reorder ops are stale or incomplete; missing "
                f"{missing}"
            )
        _MODULE = module
        return _MODULE
    source = Path(__file__).with_name("csrc") / "segment_expert_reorder.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "segment_expert_reorder")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    # A preceding TLA extension adds PVC SPIR-V device-link flags to
    # PyTorch's process-global extension builder.  The framework default
    # emits both ``spir64_gen`` and generic ``spir64`` for a plain SYCL
    # extension, which makes that linker unable to attach the PVC-specific
    # SPIR-V flags to an unambiguous target.  This reorder kernel is XPU-only
    # already, so build one PVC AOT image just like the surrounding exact
    # grouped kernels.  It changes only build targeting, never its dynamic
    # rows/experts/layout contract.
    _MODULE = _load_without_sycl2020(
        name="aurora_moe_segment_expert_reorder",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=[
            "-fno-sycl-instrument-device-code",
            "-fsycl-targets=spir64_gen",
        ],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def make_expert_major_layout(
    segments: PeerExpertSegments, *, include_row_schedule: bool = False
) -> ExpertMajorLayout:
    """Build device-resident logical-expert offsets from exact fragment counts.

    ``include_row_schedule`` is enabled only by the row-tiled experimental
    backend.  The established element-tiled path should not pay for metadata
    that it never reads.
    """

    counts = segments.counts
    if counts.dtype != torch.int32 or counts.ndim != 2 or not counts.is_contiguous():
        raise ValueError("segments.counts must be contiguous int32 [sources, experts]")
    counts64 = counts.to(torch.int64)
    expert_rows = counts64.sum(dim=0)
    expert_offsets_i64 = torch.cat(
        (expert_rows.new_zeros(1), expert_rows.cumsum(dim=0))
    ).contiguous()
    # For physical [source, expert] fragment (s, e), this is the number of
    # earlier-source rows already assigned to logical expert e.
    expert_source_offsets = (
        counts64.cumsum(dim=0) - counts64
    ).contiguous()
    if include_row_schedule:
        fragment_blocks = torch.div(
            counts64.reshape(-1) + _ROW_PARALLEL_ROWS_PER_WORKGROUP - 1,
            _ROW_PARALLEL_ROWS_PER_WORKGROUP,
            rounding_mode="floor",
        )
        fragment_row_block_offsets = torch.cat(
            (fragment_blocks.new_zeros(1), fragment_blocks.cumsum(dim=0))
        ).contiguous()
    else:
        fragment_row_block_offsets = counts64.new_empty(0)
    return ExpertMajorLayout(
        expert_offsets_i64=expert_offsets_i64,
        expert_offsets_i32=expert_offsets_i64.to(torch.int32).contiguous(),
        expert_source_offsets=expert_source_offsets,
        fragment_row_block_offsets=fragment_row_block_offsets,
    )


def _check_values(values: torch.Tensor, segments: PeerExpertSegments) -> None:
    if (
        values.device.type != "xpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 2
        or not values.is_contiguous()
        or values.device != segments.counts.device
    ):
        raise ValueError("values must be a contiguous BF16 XPU matrix on the segment device")


def segments_to_expert_major(
    values: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Pack one exact physical fragment matrix into logical expert-major rows."""

    _check_values(values, segments)
    return load_segment_expert_reorder_ops().segments_to_expert_major_bf16(
        values,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
    )


def pair_segments_to_expert_major(
    first: torch.Tensor,
    second: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack two exact physical matrices with one per-fragment traversal."""

    _check_values(first, segments)
    _check_values(second, segments)
    if first.size(0) != second.size(0):
        raise ValueError("paired matrices must have the same route count")
    return tuple(load_segment_expert_reorder_ops().pair_segments_to_expert_major_bf16(
        first,
        second,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
    ))  # type: ignore[return-value]


def expert_major_to_segments(
    values: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Restore source/expert physical order from exact expert-major rows.

    This is the inverse of :func:`segments_to_expert_major` for every true
    source/expert fragment.  It does not allocate a capacity-sized buffer or
    encode a routing sentinel; its leading extent remains exactly ``routes``.
    """

    _check_values(values, segments)
    return load_segment_expert_reorder_ops().expert_major_to_segments_bf16(
        values,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
    )


def make_parallel_reorder_block_offsets(
    segments: PeerExpertSegments, columns: int
) -> torch.Tensor:
    """Return exact device block ranges for a bandwidth-parallel reorder.

    Each physical source/expert fragment receives
    ``ceil(rows * columns / 65536)`` workgroups.  The returned vector is a
    schedule only; the route matrices remain compact ``[routes, columns]``
    tensors, with no per-expert capacity extent.
    """

    if columns <= 0:
        raise ValueError("columns must be positive")
    counts = segments.counts
    fragment_elements = counts.to(torch.int64).reshape(-1) * columns
    fragment_blocks = torch.div(
        fragment_elements + _PARALLEL_ELEMENTS_PER_WORKGROUP - 1,
        _PARALLEL_ELEMENTS_PER_WORKGROUP,
        rounding_mode="floor",
    )
    return torch.cat(
        (fragment_blocks.new_zeros(1), fragment_blocks.cumsum(dim=0))
    ).contiguous()


def segments_to_expert_major_parallel(
    values: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Bandwidth-parallel exact pack into logical expert-major rows."""

    _check_values(values, segments)
    schedule = make_parallel_reorder_block_offsets(segments, values.size(1))
    return load_segment_expert_reorder_ops().segments_to_expert_major_parallel_bf16(
        values,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        schedule,
    )


def pair_segments_to_expert_major_parallel(
    first: torch.Tensor,
    second: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bandwidth-parallel fused exact pack of two physical matrices."""

    _check_values(first, segments)
    _check_values(second, segments)
    if first.size(0) != second.size(0):
        raise ValueError("paired matrices must have the same route count")
    schedule = make_parallel_reorder_block_offsets(
        segments, max(first.size(1), second.size(1))
    )
    return tuple(load_segment_expert_reorder_ops().pair_segments_to_expert_major_parallel_bf16(
        first,
        second,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        schedule,
    ))  # type: ignore[return-value]


def expert_major_to_segments_parallel(
    values: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Bandwidth-parallel inverse pack back to source/expert physical rows."""

    _check_values(values, segments)
    schedule = make_parallel_reorder_block_offsets(segments, values.size(1))
    return load_segment_expert_reorder_ops().expert_major_to_segments_parallel_bf16(
        values,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        schedule,
    )


def expert_major_to_segments_scaled_parallel(
    values: torch.Tensor,
    scores: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Inverse-pack expert-major rows while applying their BF16 route score.

    This is the exact compact forward-return operation.  It performs the
    otherwise separate ``values * scores[:, None]`` inside the D-wide inverse
    permutation, so no weighted expert-major activation tensor is allocated.
    """

    _check_values(values, segments)
    if (
        scores.device != values.device
        or scores.dtype != values.dtype
        or scores.ndim != 1
        or scores.numel() != values.size(0)
        or not scores.is_contiguous()
    ):
        raise ValueError("scores must be contiguous BF16 with one value per expert-major row")
    schedule = make_parallel_reorder_block_offsets(segments, values.size(1))
    return load_segment_expert_reorder_ops().expert_major_to_segments_scaled_parallel_bf16(
        values,
        scores,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        schedule,
    )


def expert_major_to_segments_scaled_row_parallel(
    values: torch.Tensor,
    scores: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Row-tiled exact inverse pack with fused BF16 route-score scaling.

    The schedule follows true source/expert row fragments and is width
    independent, so it remains valid for arbitrary runtime model dimensions.
    It is a distinct implementation for A/B measurement; no route is padded
    or dropped.
    """

    _check_values(values, segments)
    if (
        scores.device != values.device
        or scores.dtype != values.dtype
        or scores.ndim != 1
        or scores.numel() != values.size(0)
        or not scores.is_contiguous()
    ):
        raise ValueError("scores must be contiguous BF16 with one value per expert-major row")
    return load_segment_expert_reorder_ops().expert_major_to_segments_scaled_row_parallel_bf16(
        values,
        scores,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        layout.fragment_row_block_offsets,
    )


def payload_to_expert_major_parallel(
    payload: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused exact payload-to-expert-major token/score permutation.

    ``payload`` is compact ``[routes, model_dim + 1]`` BF16, with the score
    in its last column.  The result is a dense expert-major ``[routes, D]``
    GEMM operand and its corresponding contiguous score vector.  This does
    not create a capacity extent or a strided GEMM operand.
    """

    _check_values(payload, segments)
    if payload.size(1) < 2:
        raise ValueError("payload must contain a token column and a score column")
    schedule = make_parallel_reorder_block_offsets(segments, payload.size(1) - 1)
    return tuple(load_segment_expert_reorder_ops().payload_to_expert_major_parallel_bf16(
        payload,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        schedule,
    ))  # type: ignore[return-value]


def payload_to_expert_major_row_parallel(
    payload: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-tiled fused exact payload-to-expert-major token/score permutation."""

    _check_values(payload, segments)
    if payload.size(1) < 2:
        raise ValueError("payload must contain a token column and a score column")
    return tuple(load_segment_expert_reorder_ops().payload_to_expert_major_row_parallel_bf16(
        payload,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        layout.fragment_row_block_offsets,
    ))  # type: ignore[return-value]


def payload_and_grad_to_expert_major_scaled_parallel(
    payload: torch.Tensor,
    grad_output: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack payload tokens and score-scale an incoming gradient in one pass.

    This is the exact expert-training backward primitive for the explicitly
    router-gradient-free path.  It produces compact expert-major tokens for
    expert dW and compact ``dY * score`` rows for down dX/dW, without a
    capacity layout or a separate D-wide score-multiply temporary.
    """

    _check_values(payload, segments)
    _check_values(grad_output, segments)
    if payload.size(1) != grad_output.size(1) + 1:
        raise ValueError("payload must have exactly one score column after grad_output columns")
    schedule = make_parallel_reorder_block_offsets(segments, grad_output.size(1))
    return tuple(
        load_segment_expert_reorder_ops().payload_and_grad_to_expert_major_scaled_parallel_bf16(
            payload,
            grad_output,
            segments.counts,
            segments.source_expert_offsets,
            segments.source_offsets,
            layout.expert_offsets_i64,
            layout.expert_source_offsets,
            schedule,
        )
    )  # type: ignore[return-value]


def payload_and_grad_to_expert_major_scaled_row_parallel(
    payload: torch.Tensor,
    grad_output: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-tiled fused payload-token and BF16 score-scaled-gradient pack."""

    _check_values(payload, segments)
    _check_values(grad_output, segments)
    if payload.size(1) != grad_output.size(1) + 1:
        raise ValueError("payload must have exactly one score column after grad_output columns")
    return tuple(
        load_segment_expert_reorder_ops().payload_and_grad_to_expert_major_scaled_row_parallel_bf16(
            payload,
            grad_output,
            segments.counts,
            segments.source_expert_offsets,
            segments.source_offsets,
            layout.expert_offsets_i64,
            layout.expert_source_offsets,
            layout.fragment_row_block_offsets,
        )
    )  # type: ignore[return-value]


def expert_major_to_payload_parallel(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Fused exact expert-major token/score inverse permutation to payload."""

    _check_values(tokens, segments)
    if (
        scores.device != tokens.device
        or scores.dtype != tokens.dtype
        or scores.ndim != 1
        or scores.numel() != tokens.size(0)
        or not scores.is_contiguous()
    ):
        raise ValueError("scores must be contiguous BF16 with one value per expert-major row")
    schedule = make_parallel_reorder_block_offsets(segments, tokens.size(1))
    return load_segment_expert_reorder_ops().expert_major_to_payload_parallel_bf16(
        tokens,
        scores,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        schedule,
    )


def expert_major_to_payload_zero_score_parallel(
    tokens: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Inverse-pack token gradients and write an exact zero score-gradient column."""

    _check_values(tokens, segments)
    schedule = make_parallel_reorder_block_offsets(segments, tokens.size(1))
    return load_segment_expert_reorder_ops().expert_major_to_payload_zero_score_parallel_bf16(
        tokens,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        schedule,
    )


def expert_major_to_payload_zero_score_row_parallel(
    tokens: torch.Tensor,
    segments: PeerExpertSegments,
    layout: ExpertMajorLayout,
) -> torch.Tensor:
    """Row-tiled exact dX inverse payload map with a zero score-gradient column."""

    _check_values(tokens, segments)
    return load_segment_expert_reorder_ops().expert_major_to_payload_zero_score_row_parallel_bf16(
        tokens,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        layout.expert_offsets_i64,
        layout.expert_source_offsets,
        layout.fragment_row_block_offsets,
    )
