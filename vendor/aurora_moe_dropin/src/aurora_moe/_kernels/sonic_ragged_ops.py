"""SYCL route-layout primitives for the exact Sonic-style MoE path.

This module intentionally has no padded-capacity interface.  A route appears
once in source order and once in its local-expert segment.  The resulting
``group_rows`` and inverse maps are device-resident and can be passed directly
to the grouped GEMM and route-gradient stages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


@dataclass(frozen=True)
class SonicRaggedLayout:
    """Exact expert-grouped routes and source/grouped inverse maps."""

    grouped_tokens: torch.Tensor
    grouped_scores: torch.Tensor
    grouped_to_source: torch.Tensor
    source_to_grouped: torch.Tensor
    group_rows: torch.Tensor
    expert_offsets: torch.Tensor

    @property
    def routes(self) -> int:
        return self.grouped_tokens.size(0)


def load_sonic_ragged_ops(verbose: bool = False) -> ModuleType:
    """Load the current-stream exact-ragged SYCL extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use Sonic ragged SYCL kernels")
    # All ranks in a DP×EP launch must use one already-built image rather than
    # race a JIT compile in their first routed step.  Keep source build as the
    # default for development; launch scripts can point this at the validated
    # shared object after a one-rank warm build.
    prebuilt = os.environ.get("AURORA_MOE_SONIC_RAGGED_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_sonic_ragged_ops", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt Sonic ragged ops: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
        return _MODULE
    source = Path(__file__).with_name("csrc") / "sonic_ragged_ops.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "sonic_ragged_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_sonic_ragged_ops",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _bf16_matrix(values: torch.Tensor, name: str) -> None:
    if (
        values.device.type != "xpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 2
        or not values.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous BF16 XPU matrix")


def _bf16_vector(values: torch.Tensor, name: str) -> None:
    if (
        values.device.type != "xpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 1
        or not values.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous BF16 XPU vector")


def _i64_vector(values: torch.Tensor, name: str) -> None:
    if (
        values.device.type != "xpu"
        or values.dtype != torch.int64
        or values.ndim != 1
        or not values.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous int64 XPU vector")


def build_sonic_ragged_layout(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    local_ids: torch.Tensor,
    num_experts: int,
) -> SonicRaggedLayout:
    """Group exact source-route rows by local expert entirely on the device.

    The caller is responsible for the normal router invariant that each local
    ID lies in ``[0, num_experts)``.  Intra-expert route order is intentionally
    unspecified: it does not affect forward results or exact route ownership,
    and avoids a global sort in the hot path.
    """

    _bf16_matrix(tokens, "tokens")
    _bf16_vector(scores, "scores")
    _i64_vector(local_ids, "local_ids")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if scores.device != tokens.device or local_ids.device != tokens.device:
        raise ValueError("tokens, scores, and local_ids must share an XPU device")
    if scores.numel() != tokens.size(0) or local_ids.numel() != tokens.size(0):
        raise ValueError("scores and local_ids must provide one entry per token route")
    outputs = load_sonic_ragged_ops().build_ragged_layout_bf16(
        tokens, scores, local_ids, num_experts
    )
    return SonicRaggedLayout(*outputs)


def gather_source_rows(values: torch.Tensor, grouped_to_source: torch.Tensor) -> torch.Tensor:
    """Gather exact source-order BF16 rows into current grouped-route order."""

    _bf16_matrix(values, "values")
    _i64_vector(grouped_to_source, "grouped_to_source")
    if values.device != grouped_to_source.device:
        raise ValueError("values and grouped_to_source must share an XPU device")
    return load_sonic_ragged_ops().gather_source_rows_bf16(values, grouped_to_source)


def scatter_grouped_rows(
    values: torch.Tensor, grouped_to_source: torch.Tensor, source_rows: int
) -> torch.Tensor:
    """Scatter one exact grouped route row to each source-route position."""

    _bf16_matrix(values, "values")
    _i64_vector(grouped_to_source, "grouped_to_source")
    if values.device != grouped_to_source.device or values.size(0) != grouped_to_source.numel():
        raise ValueError("values and grouped_to_source must describe the same XPU routes")
    if source_rows != grouped_to_source.numel():
        raise ValueError("exact ragged scatter requires one destination row per route")
    return load_sonic_ragged_ops().scatter_grouped_rows_bf16(
        values, grouped_to_source, source_rows
    )


def weighted_scatter_grouped_rows(
    values: torch.Tensor,
    grouped_scores: torch.Tensor,
    grouped_to_source: torch.Tensor,
    source_rows: int,
) -> torch.Tensor:
    """Apply the route score and scatter exact grouped rows to source order."""

    _bf16_matrix(values, "values")
    _bf16_vector(grouped_scores, "grouped_scores")
    _i64_vector(grouped_to_source, "grouped_to_source")
    if (
        values.device != grouped_scores.device
        or values.device != grouped_to_source.device
        or values.size(0) != grouped_scores.numel()
        or values.size(0) != grouped_to_source.numel()
    ):
        raise ValueError("ragged value, score, and map tensors must describe the same XPU routes")
    if source_rows != grouped_to_source.numel():
        raise ValueError("exact ragged scatter requires one destination row per route")
    return load_sonic_ragged_ops().weighted_scatter_grouped_rows_bf16(
        values, grouped_scores, grouped_to_source, source_rows
    )


def scatter_grouped_scalars(
    values: torch.Tensor, grouped_to_source: torch.Tensor, source_rows: int
) -> torch.Tensor:
    """Scatter exact grouped BF16 scalars to source-route order."""

    _bf16_vector(values, "values")
    _i64_vector(grouped_to_source, "grouped_to_source")
    if values.device != grouped_to_source.device or values.numel() != grouped_to_source.numel():
        raise ValueError("values and grouped_to_source must describe the same XPU routes")
    if source_rows != grouped_to_source.numel():
        raise ValueError("exact ragged scatter requires one destination row per route")
    return load_sonic_ragged_ops().scatter_grouped_scalars_bf16(
        values, grouped_to_source, source_rows
    )


def ragged_route_grad(
    grad_source: torch.Tensor,
    expert_values: torch.Tensor,
    grouped_scores: torch.Tensor,
    grouped_to_source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse source-gradient gather, score scaling, and route-score reduction."""

    _bf16_matrix(grad_source, "grad_source")
    _bf16_matrix(expert_values, "expert_values")
    _bf16_vector(grouped_scores, "grouped_scores")
    _i64_vector(grouped_to_source, "grouped_to_source")
    if (
        grad_source.device != expert_values.device
        or grad_source.device != grouped_scores.device
        or grad_source.device != grouped_to_source.device
        or grad_source.shape != expert_values.shape
        or expert_values.size(0) != grouped_scores.numel()
        or expert_values.size(0) != grouped_to_source.numel()
        or grad_source.size(0) != grouped_to_source.numel()
    ):
        raise ValueError("ragged gradient tensors must share exact source/grouped route shapes")
    return load_sonic_ragged_ops().ragged_route_grad_bf16(
        grad_source, expert_values, grouped_scores, grouped_to_source
    )


def ragged_down_backward(
    grouped_grad_output: torch.Tensor,
    down_input_grad_unscaled: torch.Tensor,
    hidden: torch.Tensor,
    grouped_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Sonic's reordered down-projection backward pointwise work.

    ``down_input_grad_unscaled`` is the exact grouped result
    ``grouped_grad_output @ down.T``.  The kernel produces the scaled down
    weight-gradient input, the scaled hidden gradient, and
    ``dscore = row_dot(down_input_grad_unscaled, hidden)``.  The latter is
    algebraically equal to ``row_dot(dO, hidden @ down)`` but does not require
    regenerating/saving the unweighted expert output.
    """

    _bf16_matrix(grouped_grad_output, "grouped_grad_output")
    _bf16_matrix(down_input_grad_unscaled, "down_input_grad_unscaled")
    _bf16_matrix(hidden, "hidden")
    _bf16_vector(grouped_scores, "grouped_scores")
    if (
        grouped_grad_output.device != down_input_grad_unscaled.device
        or grouped_grad_output.device != hidden.device
        or grouped_grad_output.device != grouped_scores.device
        or grouped_grad_output.size(0) != down_input_grad_unscaled.size(0)
        or down_input_grad_unscaled.shape != hidden.shape
        or grouped_grad_output.size(0) != grouped_scores.numel()
    ):
        raise ValueError("Sonic down-backward tensors must share exact grouped route shapes")
    return load_sonic_ragged_ops().ragged_down_backward_bf16(
        grouped_grad_output,
        down_input_grad_unscaled,
        hidden,
        grouped_scores,
    )
