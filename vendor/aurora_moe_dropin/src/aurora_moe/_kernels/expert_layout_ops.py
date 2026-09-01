"""Exact source-major EP to expert-major padded row layout kernels."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_MODULE: ModuleType | None = None


@dataclass(frozen=True)
class PaddedExpertLayout:
    """Dynamic expert-major rows and their inverse source-padded slot map."""

    tokens: torch.Tensor
    scores: torch.Tensor
    source_slots: torch.Tensor
    group_rows: torch.Tensor
    max_rows: int
    max_tail: int
    all_expert_rows_filled: bool
    use_tail_zero: bool
    num_experts: int
    cap: int

    @property
    def padded_source_rows(self) -> int:
        return self.source_slots.numel()


def load_expert_layout_ops(verbose: bool = False) -> ModuleType:
    """Load the current-stream XPU expert-layout extension."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use expert-layout kernels")
    source = Path(__file__).with_name("csrc") / "expert_layout_ops.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "expert_layout_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_expert_layout_ops",
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
        or counts.dtype != torch.int64
        or counts.ndim != 1
        or not counts.is_contiguous()
    ):
        raise ValueError("recv_counts must be a contiguous int64 XPU rank-1 tensor")


def _payload(payload: torch.Tensor, count_columns: int) -> None:
    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or payload.size(1) < count_columns
        or not payload.is_contiguous()
    ):
        raise ValueError("payload must be a contiguous BF16 XPU matrix with route metadata")


def _ids(local_ids: torch.Tensor) -> None:
    if (
        local_ids.device.type != "xpu"
        or local_ids.dtype != torch.int64
        or local_ids.ndim != 1
        or not local_ids.is_contiguous()
    ):
        raise ValueError("local_ids must be a contiguous int64 XPU rank-1 tensor")


def _shape(payload: torch.Tensor, recv_counts: torch.Tensor, cap: int, num_experts: int) -> None:
    if cap < 0 or num_experts <= 0:
        raise ValueError("cap must be nonnegative and num_experts must be positive")
    if payload.device != recv_counts.device:
        raise ValueError("payload and recv_counts must share an XPU device")
    if payload.size(0) != recv_counts.numel() * cap:
        raise ValueError("payload rows must equal recv_counts.numel() * cap")


def _row_shape(group_rows: torch.Tensor) -> tuple[int, int, bool]:
    group_rows_cpu = tuple(int(value) for value in group_rows.cpu().tolist())
    max_rows = max(group_rows_cpu, default=0)
    max_tail = max_rows - min(group_rows_cpu, default=max_rows)
    return max_rows, max_tail, max_rows > 0 and max_tail == 0


def _tail_zero_requested() -> bool:
    return os.environ.get("AURORA_MOE_DIRECT_LAYOUT_TAIL_ZERO") == "1"


def _zero_expert_row_tails(
    values: torch.Tensor, group_rows: torch.Tensor, max_tail: int
) -> None:
    if max_tail:
        load_expert_layout_ops().zero_expert_row_tails_bf16(
            values, group_rows, max_tail
        )


def _layout_row_shape(group_rows: torch.Tensor, use_tail_zero: bool) -> tuple[int, int, bool]:
    if use_tail_zero:
        return _row_shape(group_rows)
    return int(group_rows.max().item()), 0, False


def _known_layout_row_shape(
    group_rows: torch.Tensor,
    num_experts: int,
    max_rows: int,
    max_tail: int,
    use_tail_zero: bool,
) -> tuple[int, int, bool]:
    """Validate host-known exact row bounds without a device scalar readback."""

    if (
        group_rows.device.type != "xpu"
        or group_rows.dtype != torch.int32
        or group_rows.ndim != 1
        or not group_rows.is_contiguous()
        or group_rows.numel() != num_experts
    ):
        raise ValueError(
            "known_group_rows must be a contiguous int32 XPU vector with one entry per expert"
        )
    if max_rows < 0 or max_tail < 0 or max_tail > max_rows:
        raise ValueError("known expert-row bounds are invalid")
    # Tail-zero mode requires the exact maximum tail extent.  In the
    # zero-initialized fallback, it is intentionally ignored: the pack kernel
    # initializes all unused rows itself.
    return max_rows, max_tail if use_tail_zero else 0, max_rows > 0 and max_tail == 0


def make_padded_expert_layout(
    payload: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    num_experts: int,
    *,
    local_ids: torch.Tensor | None = None,
    known_group_rows: torch.Tensor | None = None,
    known_max_rows: int | None = None,
    known_max_tail: int | None = None,
) -> PaddedExpertLayout:
    """Map valid padded EP payload rows directly into exact expert BMM rows.

    ``payload`` is source-major ``[ep_size * cap, token..., score]`` when
    ``local_ids`` is supplied, or ``[ep_size * cap, token..., score, id]``
    for the BF16 small-ID transport.  The returned row count is the observed
    maximum local-expert load, never a configured capacity factor.  A caller
    that has exchanged exact expert counts before dispatch may provide the
    three ``known_*`` values.  That skips the receive-side count kernel and,
    crucially, avoids a post-dispatch device-to-host scalar synchronization.
    """

    _counts(recv_counts)
    known_values = (known_group_rows, known_max_rows, known_max_tail)
    if any(value is not None for value in known_values) and not all(
        value is not None for value in known_values
    ):
        raise ValueError(
            "known_group_rows, known_max_rows, and known_max_tail must be supplied together"
        )
    use_known_rows = known_group_rows is not None
    if use_known_rows:
        assert known_max_rows is not None and known_max_tail is not None
        use_tail_zero = _tail_zero_requested()
        max_rows, max_tail, all_expert_rows_filled = _known_layout_row_shape(
            known_group_rows,
            num_experts,
            int(known_max_rows),
            int(known_max_tail),
            use_tail_zero,
        )
    if local_ids is None:
        _payload(payload, 3)
        _shape(payload, recv_counts, cap, num_experts)
        if num_experts > 256:
            raise ValueError("BF16 payload IDs require num_experts in [1, 256]")
        ops = load_expert_layout_ops()
        if use_known_rows:
            assert known_group_rows is not None
            group_rows = known_group_rows
        else:
            use_tail_zero = _tail_zero_requested()
            group_rows = ops.count_padded_payload_expert_ids_bf16(
                payload, recv_counts, cap, num_experts
            )
            max_rows, max_tail, all_expert_rows_filled = _layout_row_shape(
                group_rows, use_tail_zero
            )
        tokens, scores, source_slots = ops.pack_padded_payload_ids_bf16_to_expert_bf16(
            payload,
            recv_counts,
            cap,
            num_experts,
            max_rows,
            not use_tail_zero,
        )
        if use_tail_zero:
            _zero_expert_row_tails(tokens, group_rows, max_tail)
            _zero_expert_row_tails(scores, group_rows, max_tail)
    else:
        _payload(payload, 2)
        _ids(local_ids)
        _shape(payload, recv_counts, cap, num_experts)
        if local_ids.device != payload.device or local_ids.numel() != payload.size(0):
            raise ValueError("local_ids must share payload's device and padded row count")
        ops = load_expert_layout_ops()
        if use_known_rows:
            assert known_group_rows is not None
            group_rows = known_group_rows
        else:
            use_tail_zero = _tail_zero_requested()
            group_rows = ops.count_padded_expert_ids_i64(
                local_ids, recv_counts, cap, num_experts
            )
            max_rows, max_tail, all_expert_rows_filled = _layout_row_shape(
                group_rows, use_tail_zero
            )
        tokens, scores, source_slots = ops.pack_padded_payload_ids_i64_to_expert_bf16(
            payload,
            local_ids,
            recv_counts,
            cap,
            num_experts,
            max_rows,
            not use_tail_zero,
        )
        if use_tail_zero:
            _zero_expert_row_tails(tokens, group_rows, max_tail)
            _zero_expert_row_tails(scores, group_rows, max_tail)
    return PaddedExpertLayout(
        tokens=tokens,
        scores=scores,
        source_slots=source_slots,
        group_rows=group_rows,
        max_rows=max_rows,
        max_tail=max_tail,
        all_expert_rows_filled=all_expert_rows_filled,
        use_tail_zero=use_tail_zero,
        num_experts=num_experts,
        cap=cap,
    )


def padded_rows_to_expert(
    values: torch.Tensor, recv_counts: torch.Tensor, layout: PaddedExpertLayout
) -> torch.Tensor:
    """Place valid source-padded BF16 vector rows in an existing expert layout."""

    _counts(recv_counts)
    _payload(values, 1)
    if values.device != recv_counts.device or values.device != layout.tokens.device:
        raise ValueError("values, recv_counts, and layout must share an XPU device")
    if values.size(0) != layout.padded_source_rows:
        raise ValueError("values must have one row per layout source slot")
    if recv_counts.numel() * layout.cap != values.size(0):
        raise ValueError("recv_counts and cap do not describe values")
    output = load_expert_layout_ops().padded_rows_to_expert_bf16(
        values,
        layout.source_slots,
        recv_counts,
        layout.cap,
        layout.num_experts,
        layout.max_rows,
        not layout.use_tail_zero,
    )
    if layout.use_tail_zero:
        _zero_expert_row_tails(output, layout.group_rows, layout.max_tail)
    return output


def expert_rows_to_padded(
    expert_values: torch.Tensor, recv_counts: torch.Tensor, layout: PaddedExpertLayout
) -> torch.Tensor:
    """Restore expert-major BF16 vector rows to zero-filled source-padded order."""

    _counts(recv_counts)
    if (
        expert_values.device.type != "xpu"
        or expert_values.dtype != torch.bfloat16
        or expert_values.ndim != 3
        or not expert_values.is_contiguous()
    ):
        raise ValueError("expert_values must be a contiguous BF16 XPU rank-3 tensor")
    if expert_values.device != recv_counts.device or expert_values.device != layout.tokens.device:
        raise ValueError("expert_values, recv_counts, and layout must share an XPU device")
    if expert_values.shape[:2] != (layout.num_experts, layout.max_rows):
        raise ValueError("expert_values expert and row dimensions must match layout")
    if recv_counts.numel() * layout.cap != layout.padded_source_rows:
        raise ValueError("recv_counts and cap do not describe layout")
    return load_expert_layout_ops().expert_rows_to_padded_bf16(
        expert_values,
        layout.source_slots,
        recv_counts,
        layout.cap,
        layout.use_tail_zero,
    )


def expert_token_score_to_padded(
    expert_tokens: torch.Tensor,
    expert_scores: torch.Tensor,
    recv_counts: torch.Tensor,
    layout: PaddedExpertLayout,
    *,
    payload_columns: int,
) -> torch.Tensor:
    """Restore token/score gradients directly to a fused padded payload.

    ``payload_columns`` is ``model_dim + 1`` for a separate-ID transport or
    ``model_dim + 2`` when the final BF16 local-ID column is embedded.  The
    latter column is returned as exact zeros, as required for a non-differentiable
    route ID, without materializing either of the intermediate concatenations
    used by the generic PyTorch expression.
    """

    _counts(recv_counts)
    if (
        expert_tokens.device.type != "xpu"
        or expert_tokens.dtype != torch.bfloat16
        or expert_tokens.ndim != 3
        or not expert_tokens.is_contiguous()
    ):
        raise ValueError("expert_tokens must be contiguous BF16 [experts, rows, model_dim]")
    if (
        expert_scores.device != expert_tokens.device
        or expert_scores.dtype != torch.bfloat16
        or expert_scores.ndim != 2
        or not expert_scores.is_contiguous()
        or expert_scores.shape != expert_tokens.shape[:2]
    ):
        raise ValueError("expert_scores must be contiguous BF16 [experts, rows]")
    if expert_tokens.device != recv_counts.device or expert_tokens.device != layout.tokens.device:
        raise ValueError("expert token/score layout inputs must share an XPU device")
    if expert_tokens.shape[:2] != (layout.num_experts, layout.max_rows):
        raise ValueError("expert token rows must match the saved layout")
    if payload_columns not in (expert_tokens.size(2) + 1, expert_tokens.size(2) + 2):
        raise ValueError("payload_columns must hold token, score, and optional ID columns")
    if recv_counts.numel() * layout.cap != layout.padded_source_rows:
        raise ValueError("recv_counts and cap do not describe layout")
    return load_expert_layout_ops().expert_token_score_to_padded_bf16(
        expert_tokens,
        expert_scores,
        layout.source_slots,
        recv_counts,
        layout.cap,
        payload_columns,
        layout.use_tail_zero,
    )


def weighted_expert_rows_to_padded(
    expert_values: torch.Tensor, recv_counts: torch.Tensor, layout: PaddedExpertLayout
) -> torch.Tensor:
    """Restore exact expert rows after multiplying their BF16 route scores."""

    if expert_values.shape[:2] != (layout.num_experts, layout.max_rows):
        raise ValueError("expert_values expert and row dimensions must match layout")
    if expert_values.dtype != torch.bfloat16 or not expert_values.is_contiguous():
        raise ValueError("expert_values must be contiguous BF16")
    _counts(recv_counts)
    if expert_values.device != layout.tokens.device or expert_values.device != recv_counts.device:
        raise ValueError("expert_values, recv_counts, and layout must share an XPU device")
    return load_expert_layout_ops().weighted_expert_rows_to_padded_bf16(
        expert_values,
        layout.scores,
        layout.source_slots,
        recv_counts,
        layout.cap,
        layout.use_tail_zero,
    )
