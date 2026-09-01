"""Experimental fused dX-add and reverse-A4 payload scatter."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

from aurora_moe._kernels.expert_layout_ops import PaddedExpertLayout


_MODULE: ModuleType | None = None


def load_direct_layout_payload_fusion_ops(verbose: bool = False) -> ModuleType:
    """Load the current-stream BF16 reverse-A4 fusion extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use direct-layout fusion kernels")
    source = Path(__file__).with_name("csrc") / "direct_layout_payload_fusion.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "direct_layout_payload_fusion")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_direct_layout_payload_fusion",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def fused_expert_token_add_score_to_padded(
    up_tokens: torch.Tensor,
    gate_tokens: torch.Tensor,
    expert_scores: torch.Tensor,
    recv_counts: torch.Tensor,
    layout: PaddedExpertLayout,
    *,
    payload_columns: int,
) -> torch.Tensor:
    """Add two BF16 expert-token gradients while producing reverse-A4 payload rows.

    This is equivalent to ``expert_token_score_to_padded(up_tokens +
    gate_tokens, expert_scores, ...)`` for the ordinary separate-BMM dX path.
    It deliberately does not change the baddbmm accumulation variant.
    """

    if (
        up_tokens.device.type != "xpu"
        or up_tokens.dtype != torch.bfloat16
        or up_tokens.ndim != 3
        or not up_tokens.is_contiguous()
    ):
        raise ValueError("up_tokens must be contiguous BF16 [experts, rows, model_dim]")
    if (
        gate_tokens.device != up_tokens.device
        or gate_tokens.dtype != torch.bfloat16
        or gate_tokens.shape != up_tokens.shape
        or not gate_tokens.is_contiguous()
    ):
        raise ValueError("gate_tokens must match up_tokens as contiguous BF16")
    if (
        expert_scores.device != up_tokens.device
        or expert_scores.dtype != torch.bfloat16
        or expert_scores.ndim != 2
        or expert_scores.shape != up_tokens.shape[:2]
        or not expert_scores.is_contiguous()
    ):
        raise ValueError("expert_scores must be contiguous BF16 [experts, rows]")
    if (
        recv_counts.device != up_tokens.device
        or recv_counts.dtype != torch.int64
        or recv_counts.ndim != 1
        or not recv_counts.is_contiguous()
    ):
        raise ValueError("recv_counts must be a contiguous int64 XPU vector")
    if up_tokens.shape[:2] != (layout.num_experts, layout.max_rows):
        raise ValueError("expert token rows must match the saved layout")
    if up_tokens.device != layout.tokens.device:
        raise ValueError("all direct-layout inputs must share an XPU device")
    if recv_counts.numel() * layout.cap != layout.padded_source_rows:
        raise ValueError("recv_counts and cap do not describe layout")
    if payload_columns not in (up_tokens.size(2) + 1, up_tokens.size(2) + 2):
        raise ValueError("payload_columns must hold token, score, and optional ID columns")
    return load_direct_layout_payload_fusion_ops().fused_expert_token_add_score_to_padded_bf16(
        up_tokens,
        gate_tokens,
        expert_scores,
        layout.source_slots,
        recv_counts,
        layout.cap,
        payload_columns,
        layout.use_tail_zero,
    )
