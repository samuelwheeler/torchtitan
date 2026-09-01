"""Dynamic direct-peer remap candidate for the reverse MoE A4 payload.

The normal direct-layout backward first materializes a source-major
``[EP * cap, model_dim + 1]`` tensor, then hands that tensor to an equal-size
all-to-all.  This module is deliberately transport-agnostic: it writes that
same logical payload directly into caller-supplied peer receive buffers from
the expert-major token and score gradients.

It is not wired into ``MOE.py`` yet.  A future transport integration must own
the peer-address exchange and the existing producer/consumer readiness
barriers.  The primitive below only owns the device-side remap/push.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


def load_fused_a4_remap_ops(verbose: bool = False) -> ModuleType:
    """Load the standalone BF16 direct-peer reverse-A4 remap extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use fused A4 remap kernels")
    source = Path(__file__).with_name("csrc") / "fused_a4_remap.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "fused_a4_remap")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_fused_a4_remap",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def _bf16_xpu_contiguous(tensor: torch.Tensor, name: str, ndim: int) -> None:
    if (
        tensor.device.type != "xpu"
        or tensor.dtype != torch.bfloat16
        or tensor.ndim != ndim
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous BF16 XPU rank-{ndim} tensor")


def _i64_xpu_vector(tensor: torch.Tensor, name: str) -> None:
    if (
        tensor.device.type != "xpu"
        or tensor.dtype != torch.int64
        or tensor.ndim != 1
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous int64 XPU vector")


def _address_table(tensor: torch.Tensor, peers: int) -> None:
    _i64_xpu_vector(tensor, "remote_receive_addresses")
    if tensor.numel() != peers:
        raise ValueError("remote_receive_addresses must contain one address per peer")


def fused_a4_remap_peer_push(
    expert_tokens: torch.Tensor,
    expert_scores: torch.Tensor,
    source_slots: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    receive_staging: torch.Tensor,
    remote_receive_addresses: torch.Tensor,
    rank: int,
    *,
    asynchronous: bool = True,
) -> torch.Tensor:
    """Write the exact reverse-A4 payload directly into peer receive buffers.

    ``source_slots`` maps each valid source-major padded row to its flattened
    expert-major row.  For peer ``p`` and route row ``r``, the kernel writes
    the token gradient followed by the score gradient to peer ``p`` at its
    row for this local ``rank``.  Invalid padded rows are written as BF16
    zero.  ``receive_staging`` supplies the runtime receive stride and is
    usually the local member of the persistent peer-buffer set.

    The call does not synchronize in asynchronous mode.  Its current-stream
    bridge protects the source tensors, but callers still must place their
    normal collective readiness barriers around matching peer calls.  The
    address table is an XPU int64 tensor rather than a host-specialized peer
    array, so peer count is runtime data rather than a compile-time limit.
    """

    _bf16_xpu_contiguous(expert_tokens, "expert_tokens", 3)
    _bf16_xpu_contiguous(expert_scores, "expert_scores", 2)
    _i64_xpu_vector(source_slots, "source_slots")
    _i64_xpu_vector(recv_counts, "recv_counts")
    _address_table(remote_receive_addresses, recv_counts.numel())
    _bf16_xpu_contiguous(receive_staging, "receive_staging", 2)
    if expert_scores.shape != expert_tokens.shape[:2]:
        raise ValueError("expert_scores must match expert_tokens expert-row dimensions")
    if (
        expert_scores.device != expert_tokens.device
        or source_slots.device != expert_tokens.device
        or recv_counts.device != expert_tokens.device
        or remote_receive_addresses.device != expert_tokens.device
        or receive_staging.device != expert_tokens.device
    ):
        raise ValueError("all fused A4 remap tensors must share one XPU device")
    if cap < 0:
        raise ValueError("cap must be nonnegative")
    peers = recv_counts.numel()
    if peers <= 0:
        raise ValueError("fused A4 remap requires at least one peer")
    if receive_staging.size(0) != peers:
        raise ValueError("receive_staging must have one row per peer")
    if source_slots.numel() != peers * cap:
        raise ValueError("source_slots must contain peers * cap entries")
    if rank < 0 or rank >= peers:
        raise ValueError("rank is outside the peer group")
    payload_columns = expert_tokens.size(2) + 1
    elements_per_peer = cap * payload_columns
    if receive_staging.size(1) < elements_per_peer:
        raise ValueError("receive_staging capacity is smaller than the reverse-A4 payload")
    if elements_per_peer == 0:
        return receive_staging
    ops = load_fused_a4_remap_ops()
    launch = (
        ops.push_fused_a4_remap_bf16_async
        if asynchronous
        else ops.push_fused_a4_remap_bf16
    )
    launch(
        expert_tokens,
        expert_scores,
        source_slots,
        recv_counts,
        receive_staging,
        remote_receive_addresses,
        int(rank),
        int(cap),
    )
    return receive_staging


def fused_a4_remap_reference(
    expert_tokens: torch.Tensor,
    expert_scores: torch.Tensor,
    source_slots: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
) -> torch.Tensor:
    """Small exact reference used by the focused kernel test.

    This intentionally accepts CPU or XPU tensors and is not performance
    code.  The result has source-major ``[peers, cap, model_dim + 1]`` layout.
    """

    if expert_tokens.ndim != 3 or expert_scores.shape != expert_tokens.shape[:2]:
        raise ValueError("expert token/score reference shapes are invalid")
    if recv_counts.ndim != 1 or source_slots.ndim != 1 or cap < 0:
        raise ValueError("reference route metadata is invalid")
    peers = recv_counts.numel()
    if source_slots.numel() != peers * cap:
        raise ValueError("reference source_slots must contain peers * cap entries")
    columns = expert_tokens.size(2)
    result = torch.zeros(
        (peers, cap, columns + 1), dtype=expert_tokens.dtype, device=expert_tokens.device
    )
    for peer in range(peers):
        count = int(recv_counts[peer].item())
        if count < 0 or count > cap:
            raise ValueError("reference recv_counts must be in [0, cap]")
        for row in range(count):
            slot = int(source_slots[peer * cap + row].item())
            if slot < 0 or slot >= expert_tokens.size(0) * expert_tokens.size(1):
                raise ValueError("reference source_slots contains an invalid live route slot")
            expert = slot // expert_tokens.size(1)
            expert_row = slot % expert_tokens.size(1)
            result[peer, row, :columns] = expert_tokens[expert, expert_row]
            result[peer, row, columns] = expert_scores[expert, expert_row]
    return result
