"""Exact compact expert-row layout and route-transform fusion prototype."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


@dataclass(frozen=True)
class CompactExpertLayout:
    """Exact concatenated expert rows and inverse source-padded maps."""

    tokens: torch.Tensor
    scores: torch.Tensor
    source_to_compact: torch.Tensor
    compact_to_source: torch.Tensor
    group_rows: torch.Tensor
    expert_offsets: torch.Tensor
    total_rows: int
    num_experts: int
    cap: int

    @property
    def padded_source_rows(self) -> int:
        """Number of source-major padded transport rows."""

        return self.source_to_compact.numel()


def load_compact_expert_layout_fusion_ops(verbose: bool = False) -> ModuleType:
    """Load the current-stream compact-layout SYCL extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use compact layout fusion kernels")
    source = Path(__file__).with_name("csrc") / "compact_expert_layout_fusion.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "compact_expert_layout_fusion")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_compact_expert_layout_fusion",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _bf16_matrix(tensor: torch.Tensor, name: str) -> None:
    if (
        tensor.device.type != "xpu"
        or tensor.dtype != torch.bfloat16
        or tensor.ndim != 2
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous BF16 XPU matrix")


def _i64_vector(tensor: torch.Tensor, name: str) -> None:
    if (
        tensor.device.type != "xpu"
        or tensor.dtype != torch.int64
        or tensor.ndim != 1
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous int64 XPU vector")


def _bf16_vector(tensor: torch.Tensor, name: str) -> None:
    if (
        tensor.device.type != "xpu"
        or tensor.dtype != torch.bfloat16
        or tensor.ndim != 1
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous BF16 XPU vector")


def _check_counts(recv_counts: torch.Tensor, cap: int) -> None:
    _i64_vector(recv_counts, "recv_counts")
    if cap < 0:
        raise ValueError("cap must be nonnegative")


def _check_layout(layout: CompactExpertLayout, recv_counts: torch.Tensor) -> None:
    _check_counts(recv_counts, layout.cap)
    if recv_counts.device != layout.tokens.device:
        raise ValueError("recv_counts and layout must share an XPU device")
    if layout.padded_source_rows != recv_counts.numel() * layout.cap:
        raise ValueError("recv_counts and cap do not describe the compact layout")
    if layout.tokens.shape[0] != layout.total_rows:
        raise ValueError("compact layout token rows do not match total_rows")
    if layout.scores.shape != (layout.total_rows,):
        raise ValueError("compact layout scores do not match total_rows")
    if layout.compact_to_source.shape != (layout.total_rows,):
        raise ValueError("compact inverse map does not match total_rows")
    if layout.source_to_compact.shape != (layout.padded_source_rows,):
        raise ValueError("compact source map does not match padded rows")


def make_compact_expert_layout(
    payload: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    num_experts: int,
    *,
    local_ids: torch.Tensor | None = None,
    total_rows: int | None = None,
) -> CompactExpertLayout:
    """Pack every valid padded route into exact concatenated expert rows.

    ``local_ids`` selects the arbitrary-expert-count int64 transport path.
    Omitting it selects the embedded BF16-ID path, which is exact for at most
    256 local experts.  ``total_rows`` must equal ``sum(recv_counts)`` when
    supplied; callers that already own the dispatch count exchange can pass
    that scalar and avoid another host read.  The default obtains this exact
    scalar from ``recv_counts`` and is intended for standalone validation.

    The returned ``expert_offsets`` is device-resident and defines each
    expert's exact half-open range in ``tokens``.  A future grouped BMM can
    consume those ranges directly, so no capacity factor, route clipping, or
    ragged ``[experts, max_rows]`` tail is required.
    """

    _bf16_matrix(payload, "payload")
    _check_counts(recv_counts, cap)
    if num_experts <= 0 or payload.device != recv_counts.device:
        raise ValueError("num_experts must be positive and payload/counts must share a device")
    padded_rows = recv_counts.numel() * cap
    if payload.size(0) != padded_rows:
        raise ValueError("payload rows must equal recv_counts.numel() * cap")
    embedded_ids = local_ids is None
    if embedded_ids:
        if num_experts > 256 or payload.size(1) < 3:
            raise ValueError("embedded BF16 IDs require 1..256 experts and token, score, ID columns")
        ids = torch.empty(0, device=payload.device, dtype=torch.int64)
    else:
        _i64_vector(local_ids, "local_ids")
        if local_ids.device != payload.device or local_ids.numel() != padded_rows:
            raise ValueError("local_ids must share payload's device and padded row count")
        if payload.size(1) < 2:
            raise ValueError("payload must contain token columns and one score column")
        ids = local_ids
    if total_rows is None:
        total_rows = int(recv_counts.sum().item())
    if not 0 <= total_rows <= padded_rows:
        raise ValueError("total_rows must be in [0, recv_counts.numel() * cap]")
    outputs = load_compact_expert_layout_fusion_ops().build_compact_layout_bf16(
        payload,
        ids,
        recv_counts,
        cap,
        num_experts,
        total_rows,
        embedded_ids,
    )
    tokens, scores, source_to_compact, compact_to_source, group_rows, offsets = outputs
    return CompactExpertLayout(
        tokens=tokens,
        scores=scores,
        source_to_compact=source_to_compact,
        compact_to_source=compact_to_source,
        group_rows=group_rows,
        expert_offsets=offsets,
        total_rows=total_rows,
        num_experts=num_experts,
        cap=cap,
    )


def weighted_compact_rows_to_padded(
    expert_values: torch.Tensor,
    recv_counts: torch.Tensor,
    layout: CompactExpertLayout,
) -> torch.Tensor:
    """Score and scatter compact expert output directly to padded source rows."""

    _bf16_matrix(expert_values, "expert_values")
    _check_layout(layout, recv_counts)
    if (
        expert_values.device != layout.tokens.device
        or expert_values.shape != (layout.total_rows, layout.tokens.size(1))
    ):
        raise ValueError("expert_values must match compact token rows and model dimension")
    return load_compact_expert_layout_fusion_ops().weighted_compact_rows_to_padded_bf16(
        expert_values,
        layout.scores,
        layout.source_to_compact,
        recv_counts,
        layout.cap,
    )


def compact_route_grad_scale_score(
    grad_output: torch.Tensor,
    expert_values: torch.Tensor,
    layout: CompactExpertLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare exact compact ``[dY, dscore]`` inputs for expert backward BMMs."""

    _bf16_matrix(grad_output, "grad_output")
    _bf16_matrix(expert_values, "expert_values")
    if (
        grad_output.device != layout.tokens.device
        or grad_output.shape != (layout.padded_source_rows, layout.tokens.size(1))
        or expert_values.device != layout.tokens.device
        or expert_values.shape != (layout.total_rows, layout.tokens.size(1))
    ):
        raise ValueError("route-gradient tensors do not match the compact layout")
    values, scores = load_compact_expert_layout_fusion_ops().compact_route_grad_scale_score_bf16(
        grad_output,
        expert_values,
        layout.scores,
        layout.compact_to_source,
    )
    return values, scores


def compact_token_score_to_padded(
    expert_tokens: torch.Tensor,
    expert_scores: torch.Tensor,
    recv_counts: torch.Tensor,
    layout: CompactExpertLayout,
    *,
    payload_columns: int,
) -> torch.Tensor:
    """Scatter one compact token gradient and score gradient into A4 payload rows."""

    _bf16_matrix(expert_tokens, "expert_tokens")
    _bf16_vector(expert_scores, "expert_scores")
    _check_layout(layout, recv_counts)
    if (
        expert_tokens.device != layout.tokens.device
        or expert_tokens.shape != layout.tokens.shape
        or expert_scores.device != layout.tokens.device
        or expert_scores.shape != (layout.total_rows,)
    ):
        raise ValueError("expert token/score gradients do not match compact layout")
    if payload_columns not in (expert_tokens.size(1) + 1, expert_tokens.size(1) + 2):
        raise ValueError("payload_columns must hold tokens, score, and optional ID gradient")
    return load_compact_expert_layout_fusion_ops().compact_token_score_to_padded_bf16(
        expert_tokens,
        expert_scores,
        layout.source_to_compact,
        recv_counts,
        layout.cap,
        payload_columns,
    )


def fused_compact_token_add_score_to_padded(
    up_tokens: torch.Tensor,
    gate_tokens: torch.Tensor,
    expert_scores: torch.Tensor,
    recv_counts: torch.Tensor,
    layout: CompactExpertLayout,
    *,
    payload_columns: int,
) -> torch.Tensor:
    """Fuse compact SwiGLU dX add with exact reverse-A4 payload scatter."""

    _bf16_matrix(up_tokens, "up_tokens")
    _bf16_matrix(gate_tokens, "gate_tokens")
    _bf16_vector(expert_scores, "expert_scores")
    _check_layout(layout, recv_counts)
    if (
        up_tokens.shape != layout.tokens.shape
        or gate_tokens.shape != up_tokens.shape
        or up_tokens.device != layout.tokens.device
        or gate_tokens.device != layout.tokens.device
        or expert_scores.device != layout.tokens.device
        or expert_scores.shape != (layout.total_rows,)
    ):
        raise ValueError("fused compact token/score gradients do not match compact layout")
    if payload_columns not in (up_tokens.size(1) + 1, up_tokens.size(1) + 2):
        raise ValueError("payload_columns must hold tokens, score, and optional ID gradient")
    return load_compact_expert_layout_fusion_ops().fused_compact_token_add_score_to_padded_bf16(
        up_tokens,
        gate_tokens,
        expert_scores,
        layout.source_to_compact,
        recv_counts,
        layout.cap,
        payload_columns,
    )
