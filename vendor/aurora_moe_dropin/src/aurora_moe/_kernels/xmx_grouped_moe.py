"""Experimental exact MoE pack/XMX-GEMM/scatter primitives for PVC."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None
_XMX_K_TILE = 16
_TILE_SHAPES = {
    "2x4": (2, 4),
    "4x2": (4, 2),
    "4x4": (4, 4),
    "4x8": (4, 8),
}


@dataclass(frozen=True)
class XmxGroupedLayout:
    """Aligned expert rows plus the exact inverse source-row map."""

    tokens: torch.Tensor
    scores: torch.Tensor
    source_slots: torch.Tensor
    group_rows: torch.Tensor
    padded_rows: int
    padded_model_dim: int
    num_experts: int
    cap: int


def _tile_shape() -> tuple[int, int]:
    value = os.environ.get("AURORA_MOE_XMX_GROUPED_TILE", "4x4")
    try:
        return _TILE_SHAPES[value]
    except KeyError as error:
        raise ValueError(
            "AURORA_MOE_XMX_GROUPED_TILE must be one of "
            f"{', '.join(_TILE_SHAPES)}, got {value!r}"
        ) from error


def _macro_tiles() -> tuple[int, int]:
    tiles_m, tiles_n = _tile_shape()
    return 8 * tiles_m, 16 * tiles_n


def load_xmx_grouped_moe_ops(verbose: bool = False) -> ModuleType:
    """Build/load the isolated PVC grouped-XMX experiment."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to build/use grouped XMX MoE")
    source = Path(__file__).with_name("csrc") / "xmx_grouped_moe.sycl"
    tiles_m, tiles_n = _tile_shape()
    variant = f"m{tiles_m}_n{tiles_n}"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / f"xmx_grouped_moe_{variant}")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    defines = [
        f"-DAURORA_MOE_XMX_GROUPED_TILE_M={tiles_m}",
        f"-DAURORA_MOE_XMX_GROUPED_TILE_N={tiles_n}",
    ]
    _MODULE = load(
        name=f"aurora_moe_xmx_grouped_moe_{variant}",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG", *defines],
        extra_sycl_cflags=[
            "-fno-sycl-instrument-device-code",
            "-fsycl-targets=spir64_gen",
            *defines,
        ],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _round_up(value: int, tile: int) -> int:
    return ((value + tile - 1) // tile) * tile


def _check_payload_inputs(
    payload: torch.Tensor,
    local_ids: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    num_experts: int,
) -> None:
    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or payload.size(1) < 2
        or not payload.is_contiguous()
    ):
        raise ValueError("payload must be contiguous BF16 [padded_rows, token..., score]")
    if (
        local_ids.device != payload.device
        or local_ids.dtype != torch.int64
        or local_ids.ndim != 1
        or not local_ids.is_contiguous()
    ):
        raise ValueError("local_ids must be contiguous int64 on payload.device")
    if (
        recv_counts.device != payload.device
        or recv_counts.dtype != torch.int64
        or recv_counts.ndim != 1
        or not recv_counts.is_contiguous()
    ):
        raise ValueError("recv_counts must be contiguous int64 on payload.device")
    if cap < 0 or num_experts <= 0:
        raise ValueError("cap must be nonnegative and num_experts must be positive")
    if payload.size(0) != recv_counts.numel() * cap or local_ids.numel() != payload.size(0):
        raise ValueError("payload/local_ids rows must equal recv_counts.numel() * cap")


def make_xmx_grouped_layout(
    payload: torch.Tensor,
    local_ids: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    num_experts: int,
) -> XmxGroupedLayout:
    """Pack exact source-padded rows into XMX-aligned expert rows.

    This is an experimental boundary primitive.  It synchronizes once to size
    the aligned expert-row allocation; a production implementation should
    replace that host shape decision with a device-resident plan allocator.
    """

    _check_payload_inputs(payload, local_ids, recv_counts, cap, num_experts)
    ops = load_xmx_grouped_moe_ops()
    group_rows = ops.xmx_count_padded_expert_ids_i64(
        local_ids, recv_counts, cap, num_experts
    )
    max_rows = int(group_rows.max().item())
    macro_m, _ = _macro_tiles()
    padded_rows = _round_up(max_rows, macro_m)
    padded_model_dim = _round_up(payload.size(1) - 1, _XMX_K_TILE)
    tokens, scores, source_slots = ops.xmx_pack_padded_payload_ids_i64_bf16(
        payload,
        local_ids,
        recv_counts,
        cap,
        num_experts,
        padded_rows,
        padded_model_dim,
    )
    return XmxGroupedLayout(
        tokens=tokens,
        scores=scores,
        source_slots=source_slots,
        group_rows=group_rows,
        padded_rows=padded_rows,
        padded_model_dim=padded_model_dim,
        num_experts=num_experts,
        cap=cap,
    )


def pad_xmx_grouped_weights(weights: torch.Tensor, padded_k: int) -> torch.Tensor:
    """Return zero-padded runtime-shape [E, padded_k, padded_n] weights."""

    if (
        weights.device.type != "xpu"
        or weights.dtype != torch.bfloat16
        or weights.ndim != 3
        or not weights.is_contiguous()
    ):
        raise ValueError("weights must be contiguous BF16 [experts, K, N]")
    if padded_k < weights.size(1) or padded_k % _XMX_K_TILE:
        raise ValueError("padded_k must cover K and be divisible by 16")
    _, macro_n = _macro_tiles()
    padded_n = _round_up(weights.size(2), macro_n)
    padded = torch.zeros(
        weights.size(0), padded_k, padded_n, dtype=weights.dtype, device=weights.device
    )
    padded[:, : weights.size(1), : weights.size(2)].copy_(weights)
    return padded


def xmx_grouped_gemm_bf16(
    layout: XmxGroupedLayout, padded_weights: torch.Tensor
) -> torch.Tensor:
    """Run raw grouped XMX GEMM over a layout's exact valid expert rows."""

    if (
        padded_weights.device != layout.tokens.device
        or padded_weights.dtype != torch.bfloat16
        or padded_weights.ndim != 3
        or not padded_weights.is_contiguous()
        or padded_weights.shape[:2] != (layout.num_experts, layout.padded_model_dim)
        or padded_weights.size(2) % _macro_tiles()[1]
    ):
        raise ValueError("padded_weights must match XMX layout [experts, padded_K, padded_N]")
    return load_xmx_grouped_moe_ops().xmx_grouped_gemm_bf16(
        layout.tokens, padded_weights, layout.group_rows
    )


def scatter_xmx_grouped_rows(
    expert_values: torch.Tensor,
    layout: XmxGroupedLayout,
    recv_counts: torch.Tensor,
    output_columns: int,
) -> torch.Tensor:
    """Restore score-weighted valid XMX rows to source-padded order."""

    if recv_counts.device != layout.tokens.device or recv_counts.dtype != torch.int64:
        raise ValueError("recv_counts must be int64 on the layout device")
    return load_xmx_grouped_moe_ops().xmx_scatter_rows_to_padded_bf16(
        expert_values,
        layout.scores,
        layout.source_slots,
        recv_counts.contiguous(),
        layout.cap,
        output_columns,
    )
