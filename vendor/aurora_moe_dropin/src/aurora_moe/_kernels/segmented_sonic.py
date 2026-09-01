"""Exact source-segmented local MoE with no expert-major route copy."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aurora_moe._kernels.segmented_grouped_gemm import (
    segmented_grouped_gemm_bf16,
    segmented_weight_grad_bf16,
)


def _record(name: str):
    return torch.profiler.record_function(name) if os.environ.get("AURORA_MOE_PROFILE") == "1" else nullcontext()


@dataclass(frozen=True)
class PeerExpertSegments:
    """Exact physical source/expert segments in a compact all-to-all-v receive buffer."""

    counts: torch.Tensor
    source_expert_offsets: torch.Tensor
    source_offsets: torch.Tensor

    @property
    def sources(self) -> int:
        return self.counts.size(0)

    @property
    def experts(self) -> int:
        return self.counts.size(1)


def make_peer_expert_segments(source_expert_counts: torch.Tensor) -> PeerExpertSegments:
    """Build exact device-resident source and expert prefix offsets."""

    if (
        source_expert_counts.ndim != 2
        or source_expert_counts.dtype != torch.int32
        or not source_expert_counts.is_contiguous()
        or source_expert_counts.size(0) == 0
        or source_expert_counts.size(1) == 0
    ):
        raise ValueError("source_expert_counts must be contiguous int32 [sources, experts]")
    counts64 = source_expert_counts.to(torch.int64)
    zero_column = counts64.new_zeros((counts64.size(0), 1))
    source_expert_offsets = torch.cat((zero_column, counts64.cumsum(dim=1)), dim=1).contiguous()
    source_offsets = torch.cat(
        (counts64.new_zeros(1), counts64.sum(dim=1).cumsum(dim=0))
    ).contiguous()
    return PeerExpertSegments(
        counts=source_expert_counts,
        source_expert_offsets=source_expert_offsets,
        source_offsets=source_offsets,
    )


def _pointwise_backend() -> str:
    backend = os.environ.get("AURORA_MOE_SEGMENTED_POINTWISE", "torch")
    if backend not in ("torch", "sycl"):
        raise ValueError("AURORA_MOE_SEGMENTED_POINTWISE must be 'torch' or 'sycl'")
    return backend


def _down_backward_strategy() -> str:
    strategy = os.environ.get("AURORA_MOE_SEGMENTED_DOWN_BACKWARD", "reference")
    if strategy not in ("reference", "reordered"):
        raise ValueError("AURORA_MOE_SEGMENTED_DOWN_BACKWARD must be 'reference' or 'reordered'")
    return strategy


def _gemm_backend() -> str:
    """Select the fragment GEMM scheduler without changing the MoE contract."""

    backend = os.environ.get("AURORA_MOE_SEGMENTED_GEMM", "generic")
    if backend not in ("generic", "specialized"):
        raise ValueError("AURORA_MOE_SEGMENTED_GEMM must be 'generic' or 'specialized'")
    return backend


def _weight_grad_backend() -> str:
    """Choose exact dW reduction layout independently of forward GEMM choice."""

    backend = os.environ.get("AURORA_MOE_SEGMENTED_DW", "fragment")
    if backend not in ("fragment", "expert_major"):
        raise ValueError("AURORA_MOE_SEGMENTED_DW must be 'fragment' or 'expert_major'")
    return backend


def _grouped_gemm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    segments: PeerExpertSegments,
    backend: str,
) -> torch.Tensor:
    if backend == "generic":
        return segmented_grouped_gemm_bf16(
            activations,
            weights,
            segments.counts,
            segments.source_expert_offsets,
            segments.source_offsets,
        )
    if backend == "specialized":
        # The PVC MoE scheduler works directly over the physical
        # source-major/expert-minor fragments.  It maps group % local_experts
        # to the reused weight, avoiding both a D-wide reorder and replicated
        # [source, expert, K, N] weights.
        from aurora_moe._kernels.segmented_moe_tla import segmented_moe_grouped_gemm_bf16

        return segmented_moe_grouped_gemm_bf16(
            activations, weights, segments.counts
        )
    raise AssertionError(f"unexpected segmented GEMM backend: {backend}")


def _swiglu(up: torch.Tensor, gate: torch.Tensor, backend: str) -> torch.Tensor:
    if backend == "sycl":
        from aurora_moe._kernels.swiglu_ops import load_swiglu_ops

        return load_swiglu_ops().swiglu_forward_bf16(up, gate)
    return up * F.silu(gate)


def _swiglu_backward(
    grad_hidden: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, backend: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "sycl":
        from aurora_moe._kernels.swiglu_ops import load_swiglu_ops

        return load_swiglu_ops().swiglu_backward_bf16(grad_hidden, up, gate)
    sigmoid = torch.sigmoid(gate)
    return grad_hidden * gate * sigmoid, grad_hidden * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))


def _check_inputs(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    activation: str,
) -> None:
    if (
        tokens.device.type != "xpu"
        or tokens.dtype != torch.bfloat16
        or tokens.ndim != 2
        or not tokens.is_contiguous()
    ):
        raise ValueError("tokens must be a contiguous BF16 XPU [routes, model_dim] tensor")
    if (
        scores.device != tokens.device
        or scores.dtype != tokens.dtype
        or scores.ndim != 1
        or scores.numel() != tokens.size(0)
        or not scores.is_contiguous()
    ):
        raise ValueError("scores must be contiguous BF16 with one entry per route")
    if (
        segments.counts.device != tokens.device
        or segments.source_expert_offsets.device != tokens.device
        or segments.source_offsets.device != tokens.device
        or segments.counts.dtype != torch.int32
        or segments.source_expert_offsets.dtype != torch.int64
        or segments.source_offsets.dtype != torch.int64
    ):
        raise ValueError("segment metadata must be int32/int64 XPU tensors on tokens.device")
    if (
        up.device != tokens.device
        or down.device != tokens.device
        or up.dtype != tokens.dtype
        or down.dtype != tokens.dtype
        or up.ndim != 3
        or down.ndim != 3
        or not up.is_contiguous()
        or not down.is_contiguous()
        or up.size(0) != segments.experts
        or up.size(1) != tokens.size(1)
        or down.shape != (up.size(0), up.size(2), up.size(1))
    ):
        raise ValueError("expert weights do not match segmented tokens")
    if activation not in ("swiglu", "squared-relu"):
        raise ValueError("activation must be 'swiglu' or 'squared-relu'")
    if activation == "swiglu":
        if gate is None or gate.shape != up.shape or gate.device != tokens.device or gate.dtype != tokens.dtype:
            raise ValueError("SwiGLU gate weights must match up weights")
        if not gate.is_contiguous():
            raise ValueError("gate must be contiguous")
    elif gate is not None:
        raise ValueError("squared-relu does not use gate weights")


class _SegmentedLocalMoE(torch.autograd.Function):
    """Autograd node for source-major, expert-segmented exact routed rows."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        counts: torch.Tensor,
        source_expert_offsets: torch.Tensor,
        source_offsets: torch.Tensor,
        up: torch.Tensor,
        gate: torch.Tensor,
        down: torch.Tensor,
        activation: str,
        pointwise_backend: str,
        down_strategy: str,
    ) -> torch.Tensor:
        segments = PeerExpertSegments(counts, source_expert_offsets, source_offsets)
        gate_arg = gate if activation == "swiglu" else None
        _check_inputs(tokens, scores, segments, up, gate_arg, down, activation)
        ctx.gemm_backend = _gemm_backend()
        ctx.weight_grad_backend = _weight_grad_backend()
        with _record(f"moe.segmented.{ctx.gemm_backend}.up"):
            preact_up = _grouped_gemm(tokens, up, segments, ctx.gemm_backend)
        if activation == "swiglu":
            with _record(f"moe.segmented.{ctx.gemm_backend}.gate"):
                preact_gate = _grouped_gemm(tokens, gate, segments, ctx.gemm_backend)
            with _record(f"moe.segmented.pointwise.{pointwise_backend}.forward"):
                hidden = _swiglu(preact_up, preact_gate, pointwise_backend)
        else:
            preact_gate = tokens.new_empty(0)
            with _record("moe.segmented.pointwise.squared_relu.forward"):
                hidden = F.relu(preact_up).square()
        with _record(f"moe.segmented.{ctx.gemm_backend}.down"):
            values = _grouped_gemm(hidden, down, segments, ctx.gemm_backend)
        with _record("moe.segmented.scale_output"):
            output = values * scores.unsqueeze(-1)
        ctx.activation = activation
        ctx.pointwise_backend = pointwise_backend
        ctx.down_strategy = down_strategy
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            tokens,
            scores,
            counts,
            source_expert_offsets,
            source_offsets,
            up,
            gate,
            down,
            preact_up,
            preact_gate,
        )
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_output is None:
            return (None,) * 11
        (
            tokens,
            scores,
            counts,
            source_expert_offsets,
            source_offsets,
            up,
            gate,
            down,
            preact_up,
            preact_gate,
        ) = ctx.saved_tensors
        segments = PeerExpertSegments(counts, source_expert_offsets, source_offsets)
        if grad_output.dtype != tokens.dtype or not grad_output.is_contiguous():
            grad_output = grad_output.contiguous().to(dtype=tokens.dtype)
        if ctx.activation == "swiglu":
            with _record(f"moe.segmented.pointwise.{ctx.pointwise_backend}.recompute"):
                hidden = _swiglu(preact_up, preact_gate, ctx.pointwise_backend)
        else:
            with _record("moe.segmented.pointwise.squared_relu.recompute"):
                hidden = F.relu(preact_up).square()
        with _record("moe.segmented.scale_grad_output"):
            score = scores.unsqueeze(-1)
            grad_values = grad_output * score
        if ctx.down_strategy == "reordered":
            with _record(f"moe.segmented.{ctx.gemm_backend}.down_dx_unscaled"):
                down_input_unscaled = _grouped_gemm(
                    grad_output,
                    down.transpose(-1, -2).contiguous(),
                    segments,
                    ctx.gemm_backend,
                )
            with _record("moe.segmented.down_grad_scale_score"):
                grad_hidden = down_input_unscaled * score
                grad_scores = (down_input_unscaled.float() * hidden.float()).sum(dim=-1).to(scores.dtype)
        else:
            with _record(f"moe.segmented.{ctx.gemm_backend}.down_recompute"):
                values = _grouped_gemm(hidden, down, segments, ctx.gemm_backend)
            with _record("moe.segmented.down_grad_score"):
                grad_scores = (grad_output.float() * values.float()).sum(dim=-1).to(scores.dtype)
            with _record(f"moe.segmented.{ctx.gemm_backend}.down_dx"):
                grad_hidden = _grouped_gemm(
                    grad_values,
                    down.transpose(-1, -2).contiguous(),
                    segments,
                    ctx.gemm_backend,
                )

        need_tokens, need_scores, _, _, _, need_up, need_gate, need_down = ctx.needs_input_grad[:8]
        expert_major_layout = None
        expert_major_tokens = None
        if ctx.weight_grad_backend == "expert_major" and (need_down or need_up or need_gate):
            from aurora_moe._kernels.segment_expert_reorder import make_expert_major_layout

            expert_major_layout = make_expert_major_layout(segments)

        def fragment_weight_grad(
            activation_values: torch.Tensor, gradient_values: torch.Tensor
        ) -> torch.Tensor:
            return segmented_weight_grad_bf16(
                activation_values,
                gradient_values,
                counts,
                source_expert_offsets,
                source_offsets,
            )

        def expert_major_weight_grad(
            activation_values: torch.Tensor, gradient_values: torch.Tensor,
            *, pair: bool = True,
        ) -> torch.Tensor:
            assert expert_major_layout is not None
            from aurora_moe._kernels.segment_expert_reorder import (
                pair_segments_to_expert_major,
                segments_to_expert_major,
            )
            from aurora_moe._kernels.xetla_grouped_gemm import grouped_weight_grad_bf16

            if pair:
                activation_major, gradient_major = pair_segments_to_expert_major(
                    activation_values, gradient_values, segments, expert_major_layout
                )
            else:
                activation_major = segments_to_expert_major(
                    activation_values, segments, expert_major_layout
                )
                gradient_major = segments_to_expert_major(
                    gradient_values, segments, expert_major_layout
                )
            return grouped_weight_grad_bf16(
                activation_major,
                gradient_major,
                expert_major_layout.expert_offsets_i32,
            )

        if need_down:
            with _record(f"moe.segmented.dw.{ctx.weight_grad_backend}.down"):
                grad_down = (
                    expert_major_weight_grad(hidden, grad_values)
                    if ctx.weight_grad_backend == "expert_major"
                    else fragment_weight_grad(hidden, grad_values)
                )
        else:
            grad_down = None
        grad_tokens = None
        grad_up = None
        grad_gate = None
        if need_tokens or need_up or need_gate:
            if ctx.activation == "swiglu":
                with _record(f"moe.segmented.pointwise.{ctx.pointwise_backend}.backward"):
                    grad_up_values, grad_gate_values = _swiglu_backward(
                        grad_hidden, preact_up, preact_gate, ctx.pointwise_backend
                    )
                if ctx.weight_grad_backend == "expert_major" and (need_up or need_gate):
                    assert expert_major_layout is not None
                    from aurora_moe._kernels.segment_expert_reorder import (
                        pair_segments_to_expert_major,
                        segments_to_expert_major,
                    )
                    from aurora_moe._kernels.xetla_grouped_gemm import grouped_weight_grad_bf16

                    expert_major_tokens = segments_to_expert_major(
                        tokens, segments, expert_major_layout
                    )
                    if need_up and need_gate:
                        gate_values_major, up_values_major = pair_segments_to_expert_major(
                            grad_gate_values,
                            grad_up_values,
                            segments,
                            expert_major_layout,
                        )
                    elif need_gate:
                        gate_values_major = segments_to_expert_major(
                            grad_gate_values, segments, expert_major_layout
                        )
                        up_values_major = None
                    else:
                        gate_values_major = None
                        up_values_major = segments_to_expert_major(
                            grad_up_values, segments, expert_major_layout
                        )
                    if need_gate:
                        assert gate_values_major is not None
                        with _record("moe.segmented.dw.expert_major.gate"):
                            grad_gate = grouped_weight_grad_bf16(
                                expert_major_tokens,
                                gate_values_major,
                                expert_major_layout.expert_offsets_i32,
                            )
                    if need_up:
                        assert up_values_major is not None
                        with _record("moe.segmented.dw.expert_major.up"):
                            grad_up = grouped_weight_grad_bf16(
                                expert_major_tokens,
                                up_values_major,
                                expert_major_layout.expert_offsets_i32,
                            )
                elif need_gate:
                    with _record("moe.segmented.dw.fragment.gate"):
                        grad_gate = fragment_weight_grad(tokens, grad_gate_values)
                if need_tokens:
                    with _record(f"moe.segmented.{ctx.gemm_backend}.gate_dx"):
                        grad_tokens = _grouped_gemm(
                            grad_gate_values,
                            gate.transpose(-1, -2).contiguous(),
                            segments,
                            ctx.gemm_backend,
                        )
            else:
                with _record("moe.segmented.pointwise.squared_relu.backward"):
                    grad_up_values = grad_hidden * (2.0 * F.relu(preact_up))
                if need_tokens:
                    grad_tokens = torch.zeros_like(tokens)
            if need_up and ctx.weight_grad_backend != "expert_major":
                with _record("moe.segmented.dw.fragment.up"):
                    grad_up = fragment_weight_grad(tokens, grad_up_values)
            elif need_up and ctx.activation != "swiglu":
                # Squared-ReLU has no paired gate dW path above.  Keep the
                # exact expert-major reduction available for that activation.
                with _record("moe.segmented.dw.expert_major.up"):
                    grad_up = expert_major_weight_grad(tokens, grad_up_values)
            if need_tokens:
                with _record(f"moe.segmented.{ctx.gemm_backend}.up_dx"):
                    up_tokens = _grouped_gemm(
                        grad_up_values,
                        up.transpose(-1, -2).contiguous(),
                        segments,
                        ctx.gemm_backend,
                    )
                with _record("moe.segmented.combine_input_grads"):
                    grad_tokens = up_tokens if grad_tokens is None else grad_tokens + up_tokens

        return (
            grad_tokens,
            grad_scores if need_scores else None,
            None,
            None,
            None,
            grad_up,
            grad_gate,
            grad_down,
            None,
            None,
            None,
        )


def segmented_local_moe(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    activation: str = "swiglu",
) -> torch.Tensor:
    """Run exact local experts directly over compact source/expert route segments."""

    pointwise_backend = _pointwise_backend()
    down_strategy = _down_backward_strategy()
    gate_tensor = gate if gate is not None else tokens.new_empty(0)
    return _SegmentedLocalMoE.apply(
        tokens,
        scores,
        segments.counts,
        segments.source_expert_offsets,
        segments.source_offsets,
        up,
        gate_tensor,
        down,
        activation,
        pointwise_backend,
        down_strategy,
    )


def segmented_reference_local_moe(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    activation: str = "swiglu",
) -> torch.Tensor:
    """Portable source-segmented reference used only by tests and diagnostics."""

    counts = segments.counts.cpu()
    source_offsets = segments.source_offsets.cpu()
    expert_offsets = segments.source_expert_offsets.cpu()
    result = torch.empty_like(tokens)
    for source in range(counts.size(0)):
        source_begin = int(source_offsets[source])
        for expert in range(counts.size(1)):
            rows = int(counts[source, expert])
            if not rows:
                continue
            begin = source_begin + int(expert_offsets[source, expert])
            end = begin + rows
            preact = tokens[begin:end].matmul(up[expert])
            if activation == "swiglu":
                assert gate is not None
                hidden = preact * F.silu(tokens[begin:end].matmul(gate[expert]))
            elif activation == "squared-relu":
                hidden = F.relu(preact).square()
            else:
                raise ValueError("activation must be 'swiglu' or 'squared-relu'")
            result[begin:end] = hidden.matmul(down[expert]) * scores[begin:end].unsqueeze(-1)
    return result
