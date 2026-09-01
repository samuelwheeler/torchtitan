"""Experimental fused route-gradient preparation for direct padded EP MoE."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

from aurora_moe._kernels.expert_layout_ops import PaddedExpertLayout


_MODULE: ModuleType | None = None


def load_route_grad_ops(verbose: bool = False) -> ModuleType:
    """Load the raw BF16 route-gradient SYCL extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use route-gradient kernels")
    source = Path(__file__).with_name("csrc") / "route_grad_ops.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "route_grad_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_route_grad_ops",
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


def route_grad_scale_score(
    grad_output: torch.Tensor,
    expert_values: torch.Tensor,
    expert_scores: torch.Tensor,
    source_slots: torch.Tensor,
    recv_counts: torch.Tensor,
    group_rows: torch.Tensor,
    *,
    cap: int,
    max_tail: int,
    zero_initialize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact BF16 ``[dY, dscore]`` expert-major route gradients.

    ``source_slots`` maps each valid padded source row to one unique flattened
    ``[expert, row]`` slot.  With ``zero_initialize=False``, ``max_tail`` must
    cover every dynamic expert tail described by ``group_rows``.
    """

    _bf16_matrix(grad_output, "grad_output")
    _bf16_matrix(expert_scores, "expert_scores")
    _i64_vector(source_slots, "source_slots")
    _i64_vector(recv_counts, "recv_counts")
    if (
        expert_values.device.type != "xpu"
        or expert_values.dtype != torch.bfloat16
        or expert_values.ndim != 3
        or not expert_values.is_contiguous()
    ):
        raise ValueError("expert_values must be contiguous BF16 [experts, rows, model_dim]")
    if (
        group_rows.device != grad_output.device
        or group_rows.dtype != torch.int32
        or group_rows.ndim != 1
        or not group_rows.is_contiguous()
    ):
        raise ValueError("group_rows must be a contiguous int32 XPU vector")
    if cap < 0 or max_tail < 0 or max_tail > expert_values.size(1):
        raise ValueError("cap and max_tail must describe a nonnegative valid layout")
    if (
        expert_values.device != grad_output.device
        or expert_scores.device != grad_output.device
        or source_slots.device != grad_output.device
        or recv_counts.device != grad_output.device
    ):
        raise ValueError("all route-gradient inputs must share an XPU device")
    if grad_output.shape != (recv_counts.numel() * cap, expert_values.size(2)):
        raise ValueError("grad_output shape must equal [recv_counts.numel() * cap, model_dim]")
    if expert_scores.shape != expert_values.shape[:2]:
        raise ValueError("expert_scores must have shape [experts, rows]")
    if source_slots.numel() != grad_output.size(0):
        raise ValueError("source_slots must provide one entry per padded source row")
    if group_rows.numel() != expert_values.size(0):
        raise ValueError("group_rows must provide one count per expert")
    return tuple(
        load_route_grad_ops().route_grad_scale_score_bf16(
            grad_output,
            expert_values,
            expert_scores,
            source_slots,
            recv_counts,
            group_rows,
            cap,
            max_tail,
            zero_initialize,
        )
    )


def route_grad_scale_score_from_layout(
    grad_output: torch.Tensor,
    expert_values: torch.Tensor,
    recv_counts: torch.Tensor,
    layout: PaddedExpertLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply :func:`route_grad_scale_score` using a saved direct EP layout."""

    return route_grad_scale_score(
        grad_output,
        expert_values,
        layout.scores,
        layout.source_slots,
        recv_counts,
        layout.group_rows,
        cap=layout.cap,
        max_tail=layout.max_tail,
        zero_initialize=not layout.use_tail_zero,
    )
