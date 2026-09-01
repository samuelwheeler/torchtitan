"""Exact local EP row gather/scatter kernels for received expert routes."""

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
class PayloadRows:
    """Compact BF16 token rows and one BF16 score gathered from EP payload."""

    tokens: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class PayloadRowsWithIds:
    """Compact BF16 token/score rows plus decoded exact-small local IDs."""

    tokens: torch.Tensor
    scores: torch.Tensor
    local_ids: torch.Tensor


def load_ep_local_ops(verbose: bool = False) -> ModuleType:
    """Load local receive-side row movement kernels on the current XPU stream."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use local EP route kernels")
    prebuilt = os.environ.get("AURORA_MOE_EP_LOCAL_OPS_SO")
    if prebuilt:
        spec = spec_from_file_location("aurora_moe_ep_local_ops", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load prebuilt local EP ops: {prebuilt}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        required = (
            "gather_rows_bf16",
            "gather_payload_rows_bf16",
            "split_compact_payload_bf16",
            "compact_payload_rows_from_counts_bf16",
            "scatter_rows_bf16",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(f"prebuilt local EP ops is missing symbols: {missing}")
        _MODULE = module
        return _MODULE
    source = Path(__file__).with_name("csrc") / "ep_local_ops.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "ep_local_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_ep_local_ops",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _positions(positions: torch.Tensor) -> None:
    if (
        positions.device.type != "xpu"
        or positions.dtype != torch.int64
        or positions.ndim != 1
        or not positions.is_contiguous()
    ):
        raise ValueError("positions must be a contiguous int64 XPU rank-1 tensor")


def _counts(counts: torch.Tensor) -> None:
    if (
        counts.device.type != "xpu"
        or counts.dtype != torch.int64
        or counts.ndim != 1
        or not counts.is_contiguous()
    ):
        raise ValueError("counts must be a contiguous int64 XPU rank-1 tensor")


def _valid_rows(counts: torch.Tensor, cap: int, known_rows: int | None = None) -> int:
    if cap < 0:
        raise ValueError("cap must be nonnegative")
    if known_rows is not None:
        if known_rows < 0 or known_rows > counts.numel() * cap:
            raise ValueError("known_rows must be in [0, ep_size * cap]")
        return known_rows
    # Dynamic compact output requires its exact extent.  The existing
    # count-exchange path already needs a scalar synchronization for cap;
    # this replaces dynamic-shape nonzero with a single scalar extent.
    rows = int(counts.sum().item())
    if rows < 0 or rows > counts.numel() * cap:
        raise ValueError("counts must be in [0, cap]")
    return rows


def gather_rows(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather BF16 matrix rows at received-route positions into compact order."""

    _positions(positions)
    if (
        values.device.type != "xpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 2
        or not values.is_contiguous()
    ):
        raise ValueError("values must be a contiguous BF16 XPU rank-2 tensor")
    if values.device != positions.device:
        raise ValueError("values and positions must share an XPU device")
    return load_ep_local_ops().gather_rows_bf16(values, positions)


def gather_payload_rows(payload: torch.Tensor, positions: torch.Tensor) -> PayloadRows:
    """Gather compact token rows and scores from fused ``[rows, model_dim + 1]``."""

    _positions(positions)
    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or payload.size(1) < 2
        or not payload.is_contiguous()
        or payload.device != positions.device
    ):
        raise ValueError("payload must be contiguous BF16 XPU [rows, model_dim + 1]")
    return PayloadRows(*load_ep_local_ops().gather_payload_rows_bf16(payload, positions))


def gather_payload_rows_with_ids(
    payload: torch.Tensor, positions: torch.Tensor
) -> PayloadRowsWithIds:
    """Gather ``[token..., score, BF16-local-ID]`` payload rows exactly.

    The final BF16 column is decoded to int64.  Callers must use this only
    with the small-ID transport layout, whose local IDs are in ``[0, 255]``;
    the valid-row positions must still come from the exact count exchange,
    rather than from this ID column (padding is zero-filled).
    """

    _positions(positions)
    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or payload.size(1) < 3
        or not payload.is_contiguous()
        or payload.device != positions.device
    ):
        raise ValueError("payload must be contiguous BF16 XPU [rows, model_dim + 2]")
    return PayloadRowsWithIds(
        *load_ep_local_ops().gather_payload_rows_with_ids_bf16(payload, positions)
    )


def split_compact_payload_rows(payload: torch.Tensor) -> PayloadRows:
    """Split a non-padded alltoallv ``[token..., score]`` payload contiguously.

    Unlike :func:`gather_payload_rows`, every received row is valid, so this
    needs no position vector or padding mask.  The returned tensors are fresh
    contiguous buffers suitable for local expert BMMs.
    """

    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or payload.size(1) < 2
        or not payload.is_contiguous()
    ):
        raise ValueError("payload must be contiguous BF16 XPU [routes, model_dim + 1]")
    return PayloadRows(*load_ep_local_ops().split_compact_payload_bf16(payload))


def split_compact_payload_rows_with_ids(payload: torch.Tensor) -> PayloadRowsWithIds:
    """Split a non-padded ``[token..., score, BF16-local-ID]`` payload.

    The final BF16 ID column is decoded to int64.  This must only be used for
    the exact-small-ID transport path (local IDs in ``[0, 255]``); arbitrary
    local expert counts retain the separate int64-ID alltoallv buffer.
    """

    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or payload.size(1) < 3
        or not payload.is_contiguous()
    ):
        raise ValueError("payload must be contiguous BF16 XPU [routes, model_dim + 2]")
    return PayloadRowsWithIds(
        *load_ep_local_ops().split_compact_payload_ids_bf16(payload)
    )


def compact_payload_rows_from_counts(
    payload: torch.Tensor, recv_counts: torch.Tensor, cap: int
) -> PayloadRows:
    """Compact source-major valid rows directly from exact received counts.

    ``payload`` is a flattened ``[ep_size, cap, model_dim + 1]`` transport
    buffer.  This is equivalent to gathering at
    ``(arange(cap) < recv_counts[:, None]).nonzero()`` in source-major order,
    but avoids materializing the mask, nonzero result, and a second position
    gather.  Counts must be the exact destination column of the EP exchange.
    """

    _counts(recv_counts)
    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or payload.size(1) < 2
        or not payload.is_contiguous()
        or payload.device != recv_counts.device
        or payload.size(0) != recv_counts.numel() * cap
    ):
        raise ValueError("payload must be contiguous BF16 [ep_size * cap, model_dim + 1]")
    rows = _valid_rows(recv_counts, cap)
    return PayloadRows(
        *load_ep_local_ops().compact_payload_rows_from_counts_bf16(
            payload, recv_counts, cap, rows
        )
    )


def compact_payload_rows_with_ids_from_counts(
    payload: torch.Tensor, recv_counts: torch.Tensor, cap: int
) -> PayloadRowsWithIds:
    """Count-driven compact of ``[token..., score, BF16-local-ID]`` rows.

    This is the direct companion to :func:`pack_routes_payload_ids` for the
    ``local_count <= 256`` fast path.  As with its positional counterpart,
    validity is defined solely by ``recv_counts`` rather than zero-filled
    payload contents.
    """

    _counts(recv_counts)
    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or payload.size(1) < 3
        or not payload.is_contiguous()
        or payload.device != recv_counts.device
        or payload.size(0) != recv_counts.numel() * cap
    ):
        raise ValueError("payload must be contiguous BF16 [ep_size * cap, model_dim + 2]")
    rows = _valid_rows(recv_counts, cap)
    return PayloadRowsWithIds(
        *load_ep_local_ops().compact_payload_rows_with_ids_from_counts_bf16(
            payload, recv_counts, cap, rows
        )
    )


def compact_rows_from_counts(
    values: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    *,
    known_rows: int | None = None,
) -> torch.Tensor:
    """Compact BF16 source-major padded rows directly from exact counts.

    The output row order is identical to selecting each source's first
    ``recv_counts[source]`` entries from a flattened ``[ep_size, cap, cols]``
    buffer.  It is useful for the backward output-gradient receive leg.  When
    a compact extent saved from the matching forward path is available,
    passing it as ``known_rows`` avoids another count-sum scalar read.
    """

    _counts(recv_counts)
    if (
        values.device.type != "xpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 2
        or not values.is_contiguous()
        or values.device != recv_counts.device
        or values.size(0) != recv_counts.numel() * cap
    ):
        raise ValueError("values must be contiguous BF16 [ep_size * cap, columns]")
    rows = _valid_rows(recv_counts, cap, known_rows)
    return load_ep_local_ops().compact_rows_from_counts_bf16(
        values, recv_counts, cap, rows
    )


def expand_rows_to_counts(
    values: torch.Tensor, recv_counts: torch.Tensor, cap: int
) -> torch.Tensor:
    """Expand exact source-major compact BF16 rows into padded EP layout.

    This is the inverse of :func:`compact_rows_from_counts` for valid route
    rows.  It zero-fills all padding and therefore replaces position-based
    ``index_copy``/scatter without an intermediate position tensor.  The
    caller supplies compact rows produced from the same exact counts.
    """

    _counts(recv_counts)
    if (
        values.device.type != "xpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 2
        or not values.is_contiguous()
        or values.device != recv_counts.device
        or values.size(0) > recv_counts.numel() * cap
    ):
        raise ValueError("values must be contiguous BF16 compact rows for [ep_size, cap]")
    if cap < 0:
        raise ValueError("cap must be nonnegative")
    return load_ep_local_ops().expand_rows_to_counts_bf16(values, recv_counts, cap)


def scatter_rows(
    values: torch.Tensor, positions: torch.Tensor, output_rows: int
) -> torch.Tensor:
    """Scatter compact BF16 rows to unique received-route positions exactly."""

    _positions(positions)
    if (
        values.device.type != "xpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 2
        or not values.is_contiguous()
        or values.size(0) != positions.numel()
    ):
        raise ValueError("values must be contiguous BF16 rows matching positions")
    if values.device != positions.device or output_rows < positions.numel():
        raise ValueError("output shape or device is invalid")
    return load_ep_local_ops().scatter_rows_bf16(values, positions, output_rows)


def gather_scalars(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather BF16 or FP32 scalars at received-route positions."""

    _positions(positions)
    if (
        values.device.type != "xpu"
        or values.dtype not in (torch.bfloat16, torch.float32)
        or values.ndim != 1
        or not values.is_contiguous()
        or values.device != positions.device
    ):
        raise ValueError("values must be contiguous BF16/FP32 XPU rank-1")
    return load_ep_local_ops().gather_scalars(values, positions)


def scatter_scalars(
    values: torch.Tensor, positions: torch.Tensor, output_rows: int
) -> torch.Tensor:
    """Scatter compact BF16 or FP32 scalars to unique received-route positions."""

    _positions(positions)
    if (
        values.device.type != "xpu"
        or values.dtype not in (torch.bfloat16, torch.float32)
        or values.ndim != 1
        or not values.is_contiguous()
        or values.numel() != positions.numel()
        or values.device != positions.device
        or output_rows < positions.numel()
    ):
        raise ValueError("values or output shape is invalid")
    return load_ep_local_ops().scatter_scalars(values, positions, output_rows)
