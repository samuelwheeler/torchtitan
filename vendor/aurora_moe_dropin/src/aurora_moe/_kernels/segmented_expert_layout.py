"""Exact source-block expert segmentation for the isolated MoE kernel probe."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


@dataclass(frozen=True)
class SegmentedExpertLayout:
    """Exact source-block expert partitions derived from a route count matrix."""

    source_expert_counts: torch.Tensor
    source_expert_offsets: torch.Tensor
    source_rows: torch.Tensor
    expert_rows: torch.Tensor
    cap: int

    @property
    def sources(self) -> int:
        return self.source_expert_counts.size(0)

    @property
    def num_experts(self) -> int:
        return self.source_expert_counts.size(1)

    @property
    def padded_rows(self) -> int:
        return self.sources * self.cap


@dataclass(frozen=True)
class SegmentedRoutePayload:
    """Reference payload and exact source/segment inverse maps."""

    payload: torch.Tensor
    source_to_segment: torch.Tensor
    segment_to_source: torch.Tensor


def load_segmented_expert_layout_ops(verbose: bool = False) -> ModuleType:
    """Load the current-stream XPU source-segmentation extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use segmented expert layout kernels")
    source = Path(__file__).with_name("csrc") / "segmented_expert_layout.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "segmented_expert_layout")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_segmented_expert_layout",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _counts(counts: torch.Tensor) -> None:
    if (
        counts.device.type != "xpu"
        or counts.dtype != torch.int32
        or counts.ndim != 2
        or not counts.is_contiguous()
    ):
        raise ValueError("source_expert_counts must be a contiguous int32 XPU matrix")
    if counts.size(1) == 0:
        raise ValueError("source_expert_counts must have at least one expert")


def _i64_vector(values: torch.Tensor, name: str) -> None:
    if (
        values.device.type != "xpu"
        or values.dtype != torch.int64
        or values.ndim != 1
        or not values.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous int64 XPU vector")


def _bf16_matrix(values: torch.Tensor, name: str) -> None:
    if (
        values.device.type != "xpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 2
        or not values.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous BF16 XPU matrix")


def make_segmented_expert_layout(
    source_expert_counts: torch.Tensor, cap: int
) -> SegmentedExpertLayout:
    """Build exact source-block expert offsets without host scalar reads.

    ``source_expert_counts[s, e]`` must be the exact route count for source
    peer ``s`` and this rank's local expert ``e``.  Every source-row sum must
    be at most ``cap``; callers deriving both values from the pre-dispatch
    route histogram preserve all routes without capacity clipping.
    """

    _counts(source_expert_counts)
    if cap < 0:
        raise ValueError("cap must be nonnegative")
    offsets, source_rows, expert_rows = load_segmented_expert_layout_ops().build_source_expert_offsets_i32(
        source_expert_counts, cap
    )
    return SegmentedExpertLayout(
        source_expert_counts=source_expert_counts,
        source_expert_offsets=offsets,
        source_rows=source_rows,
        expert_rows=expert_rows,
        cap=cap,
    )


def validate_segmented_expert_layout(layout: SegmentedExpertLayout) -> None:
    """Synchronously validate exact count invariants for tests and diagnostics."""

    _counts(layout.source_expert_counts)
    if layout.cap < 0:
        raise ValueError("cap must be nonnegative")
    counts = layout.source_expert_counts.cpu()
    offsets = layout.source_expert_offsets.cpu()
    source_rows = layout.source_rows.cpu()
    expert_rows = layout.expert_rows.cpu()
    if (counts < 0).any():
        raise ValueError("source_expert_counts must be nonnegative")
    expected_offsets = torch.cat(
        (
            torch.zeros((layout.sources, 1), dtype=torch.int64),
            counts.to(torch.int64).cumsum(dim=1),
        ),
        dim=1,
    )
    expected_source_rows = counts.to(torch.int64).sum(dim=1).to(torch.int32)
    expected_expert_rows = counts.to(torch.int64).sum(dim=0).to(torch.int32)
    if not torch.equal(offsets, expected_offsets):
        raise AssertionError("source_expert_offsets do not match source_expert_counts")
    if not torch.equal(source_rows, expected_source_rows):
        raise AssertionError("source_rows do not match source_expert_counts")
    if not torch.equal(expert_rows, expected_expert_rows):
        raise AssertionError("expert_rows do not match source_expert_counts")
    if (source_rows > layout.cap).any():
        raise ValueError("cap does not cover an exact source route count")


def pack_padded_payload_to_source_expert_segments(
    payload: torch.Tensor,
    local_ids: torch.Tensor,
    recv_counts: torch.Tensor,
    layout: SegmentedExpertLayout,
) -> SegmentedRoutePayload:
    """Reference-pack arbitrary source rows into exact source-expert segments.

    This validates the data layout independently of the eventual segmented
    GEMM.  The production route pack will write directly to the same slots,
    avoiding this bridge copy.  IDs must be local IDs in
    ``[0, layout.num_experts)`` for every valid row, and ``recv_counts`` must
    equal the source-row totals represented by ``layout``.
    """

    _bf16_matrix(payload, "payload")
    _i64_vector(local_ids, "local_ids")
    _i64_vector(recv_counts, "recv_counts")
    _counts(layout.source_expert_counts)
    if payload.device != local_ids.device or payload.device != recv_counts.device:
        raise ValueError("payload, local_ids, and recv_counts must share an XPU device")
    if payload.device != layout.source_expert_counts.device:
        raise ValueError("payload and layout must share an XPU device")
    if recv_counts.numel() != layout.sources:
        raise ValueError("recv_counts must have one entry per source")
    if payload.size(0) != layout.padded_rows or local_ids.numel() != layout.padded_rows:
        raise ValueError("payload and local_ids must have one row per source padded slot")
    segmented, source_to_segment, segment_to_source = load_segmented_expert_layout_ops().pack_padded_payload_to_source_expert_segments_bf16(
        payload,
        local_ids,
        recv_counts,
        layout.source_expert_counts,
        layout.source_expert_offsets,
        layout.cap,
    )
    return SegmentedRoutePayload(segmented, source_to_segment, segment_to_source)
