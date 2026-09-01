"""Exact ragged local MoE primitives with SonicMoE-style saved state."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RaggedRouteLayout:
    """Expert-sorted route metadata with exact runtime extents."""

    grouped_experts: torch.Tensor | None
    grouped_to_source: torch.Tensor
    source_to_grouped: torch.Tensor
    group_rows: torch.Tensor
    expert_offsets: torch.Tensor


def make_ragged_route_layout(local_ids: torch.Tensor, num_experts: int) -> RaggedRouteLayout:
    """Sort exact local routes by expert and return device-resident offsets."""

    if local_ids.ndim != 1 or local_ids.dtype != torch.int64 or not local_ids.is_contiguous():
        raise ValueError("local_ids must be a contiguous int64 rank-1 tensor")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    grouped_experts, grouped_to_source = torch.sort(local_ids, stable=True)
    group_rows = torch.bincount(grouped_experts, minlength=num_experts)
    if group_rows.numel() != num_experts:
        raise ValueError("local_ids must be in [0, num_experts)")
    group_rows = group_rows.to(torch.int32)
    expert_offsets = torch.cat(
        (group_rows.new_zeros(1, dtype=torch.int64), group_rows.to(torch.int64).cumsum(0))
    )
    source_to_grouped = torch.empty_like(grouped_to_source)
    source_to_grouped.scatter_(0, grouped_to_source, torch.arange(local_ids.numel(), device=local_ids.device))
    return RaggedRouteLayout(
        grouped_experts=grouped_experts,
        grouped_to_source=grouped_to_source,
        source_to_grouped=source_to_grouped,
        group_rows=group_rows.contiguous(),
        expert_offsets=expert_offsets.contiguous(),
    )


def _make_layout_and_grouped_inputs(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    local_ids: torch.Tensor,
    num_experts: int,
    layout_backend: str,
) -> tuple[RaggedRouteLayout, torch.Tensor, torch.Tensor]:
    """Create exact expert segments and source/grouped route maps.

    The pure-Torch form is deliberately retained as the portable correctness
    oracle.  The SYCL form does its count, prefix scan, and pack directly on
    the current XPU stream, so it has neither a sort temporary nor host-side
    expert extents.
    """

    if layout_backend == "torch":
        layout = make_ragged_route_layout(local_ids, num_experts)
        return (
            layout,
            tokens.index_select(0, layout.grouped_to_source).contiguous(),
            scores.index_select(0, layout.grouped_to_source).contiguous(),
        )
    if layout_backend == "sycl":
        from aurora_moe._kernels.sonic_ragged_ops import build_sonic_ragged_layout

        packed = build_sonic_ragged_layout(tokens, scores, local_ids, num_experts)
        return (
            RaggedRouteLayout(
                grouped_experts=None,
                grouped_to_source=packed.grouped_to_source,
                source_to_grouped=packed.source_to_grouped,
                group_rows=packed.group_rows,
                expert_offsets=packed.expert_offsets,
            ),
            packed.grouped_tokens,
            packed.grouped_scores,
        )
    raise ValueError("layout_backend must be 'torch' or 'sycl'")


def _gather_source_rows(
    values: torch.Tensor, grouped_to_source: torch.Tensor, layout_backend: str
) -> torch.Tensor:
    if layout_backend == "sycl":
        from aurora_moe._kernels.sonic_ragged_ops import gather_source_rows

        return gather_source_rows(values, grouped_to_source)
    return values.index_select(0, grouped_to_source).contiguous()


def _scatter_grouped_rows(
    values: torch.Tensor, grouped_to_source: torch.Tensor, layout_backend: str
) -> torch.Tensor:
    if layout_backend == "sycl":
        from aurora_moe._kernels.sonic_ragged_ops import scatter_grouped_rows

        return scatter_grouped_rows(values, grouped_to_source, values.size(0))
    output = torch.empty_like(values)
    output.index_copy_(0, grouped_to_source, values)
    return output


def _weighted_scatter_grouped_rows(
    values: torch.Tensor,
    grouped_scores: torch.Tensor,
    grouped_to_source: torch.Tensor,
    layout_backend: str,
) -> torch.Tensor:
    if layout_backend == "sycl":
        from aurora_moe._kernels.sonic_ragged_ops import weighted_scatter_grouped_rows

        return weighted_scatter_grouped_rows(
            values, grouped_scores, grouped_to_source, values.size(0)
        )
    output = torch.empty_like(values)
    output.index_copy_(0, grouped_to_source, values * grouped_scores.unsqueeze(-1))
    return output


def _scatter_grouped_scalars(
    values: torch.Tensor, grouped_to_source: torch.Tensor, layout_backend: str
) -> torch.Tensor:
    if layout_backend == "sycl":
        from aurora_moe._kernels.sonic_ragged_ops import scatter_grouped_scalars

        return scatter_grouped_scalars(values, grouped_to_source, values.numel())
    output = torch.empty_like(values)
    output.index_copy_(0, grouped_to_source, values)
    return output


def _row_counts(group_rows: torch.Tensor) -> list[int]:
    return [int(value) for value in group_rows.cpu().tolist()]


def _reference_grouped_gemm(
    activations: torch.Tensor, weights: torch.Tensor, group_rows: torch.Tensor
) -> torch.Tensor:
    output = activations.new_empty((activations.size(0), weights.size(2)))
    offset = 0
    for expert, rows in enumerate(_row_counts(group_rows)):
        if rows:
            output[offset : offset + rows] = activations[offset : offset + rows].matmul(weights[expert])
        offset += rows
    return output


def _grouped_gemm(
    activations: torch.Tensor, weights: torch.Tensor, group_rows: torch.Tensor, backend: str
) -> torch.Tensor:
    if activations.size(0) == 0:
        return activations.new_empty((0, weights.size(2)))
    if backend == "reference":
        return _reference_grouped_gemm(activations, weights, group_rows)
    if backend == "tla":
        if activations.device.type != "xpu" or activations.dtype != torch.bfloat16:
            raise ValueError("the TLA backend requires BF16 XPU activations")
        from aurora_moe._kernels.tla_ops import load_tla_ops

        return load_tla_ops().grouped_gemm_bf16(activations, weights, group_rows)
    raise ValueError("backend must be 'reference' or 'tla'")


def _weight_grad(
    activations: torch.Tensor,
    grad_output: torch.Tensor,
    group_rows: torch.Tensor,
    weights: torch.Tensor,
    backend: str,
    expert_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute exact grouped dW without a max-expert-row reduction operand."""

    if backend == "tla":
        from aurora_moe._kernels.xetla_grouped_gemm import grouped_weight_grad_bf16

        if expert_offsets is None:
            expert_offsets = torch.cat(
                (
                    group_rows.new_zeros(1, dtype=torch.int32),
                    group_rows.cumsum(0, dtype=torch.int32),
                )
            )
        elif expert_offsets.dtype != torch.int32:
            # There are only E+1 offsets, so this conversion is control-plane
            # work.  The route matrices and true per-expert K extents stay on
            # device and are never padded.
            expert_offsets = expert_offsets.to(dtype=torch.int32)
        return grouped_weight_grad_bf16(activations, grad_output, expert_offsets)
    if backend != "reference":
        raise ValueError("backend must be 'reference' or 'tla'")
    output = torch.zeros_like(weights)
    offset = 0
    for expert, rows in enumerate(_row_counts(group_rows)):
        if rows:
            output[expert] = activations[offset : offset + rows].transpose(0, 1).matmul(
                grad_output[offset : offset + rows]
            ).to(dtype=weights.dtype)
        offset += rows
    return output


def _swiglu(up: torch.Tensor, gate: torch.Tensor, backend: str) -> torch.Tensor:
    if backend == "tla":
        from aurora_moe._kernels.swiglu_ops import load_swiglu_ops

        return load_swiglu_ops().swiglu_forward_bf16(up, gate)
    return up * F.silu(gate)


def _swiglu_backward(
    grad_hidden: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, backend: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "tla":
        from aurora_moe._kernels.swiglu_ops import load_swiglu_ops

        return load_swiglu_ops().swiglu_backward_bf16(grad_hidden, up, gate)
    sigmoid = torch.sigmoid(gate)
    return grad_hidden * (gate * sigmoid), grad_hidden * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))


def _check_inputs(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    local_ids: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    activation: str,
) -> None:
    if tokens.ndim != 2 or not tokens.is_contiguous() or not tokens.is_floating_point():
        raise ValueError("tokens must be a contiguous floating-point [routes, model_dim] tensor")
    if scores.ndim != 1 or not scores.is_contiguous() or not scores.is_floating_point():
        raise ValueError("scores must be a contiguous floating-point [routes] tensor")
    if scores.device != tokens.device or scores.numel() != tokens.size(0):
        raise ValueError("scores must share tokens.device and route count")
    if local_ids.device != tokens.device or local_ids.numel() != tokens.size(0):
        raise ValueError("local_ids must share tokens.device and route count")
    if up.ndim != 3 or not up.is_contiguous() or up.device != tokens.device:
        raise ValueError("up must be contiguous [experts, model_dim, hidden_dim] on tokens.device")
    if down.ndim != 3 or not down.is_contiguous() or down.device != tokens.device:
        raise ValueError("down must be contiguous [experts, hidden_dim, model_dim] on tokens.device")
    if up.dtype != tokens.dtype or down.dtype != tokens.dtype:
        raise ValueError("tokens and expert weights must have a common dtype")
    if up.size(1) != tokens.size(1) or down.shape != (up.size(0), up.size(2), up.size(1)):
        raise ValueError("expert weight shapes do not match tokens")
    if activation not in ("swiglu", "squared-relu"):
        raise ValueError("activation must be 'swiglu' or 'squared-relu'")
    if activation == "swiglu":
        if gate is None or gate.shape != up.shape or gate.dtype != tokens.dtype or not gate.is_contiguous():
            raise ValueError("SwiGLU requires contiguous gate weights shaped like up")
        if gate.device != tokens.device:
            raise ValueError("gate must share tokens.device")
    elif gate is not None:
        raise ValueError("squared-relu does not use gate weights")


def _down_backward_strategy() -> str:
    """Select the mathematically equivalent Sonic down-backward ordering.

    The reordered form eliminates a backward-only regeneration of the
    down-projection output.  It is a separate opt-in while BF16 XPU numerical
    error and timing are compared against the current route-loop oracle.
    """

    strategy = os.environ.get("AURORA_MOE_SONIC_DOWN_BACKWARD", "reference")
    if strategy not in ("reference", "reordered"):
        raise ValueError(
            "AURORA_MOE_SONIC_DOWN_BACKWARD must be 'reference' or 'reordered'"
        )
    return strategy


class _RaggedLocalMoE(torch.autograd.Function):
    """Exact ragged SwiGLU/squared-ReLU local-expert autograd node."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        local_ids: torch.Tensor,
        up: torch.Tensor,
        gate: torch.Tensor | None,
        down: torch.Tensor,
        activation: str,
        backend: str,
        layout_backend: str,
    ) -> torch.Tensor:
        _check_inputs(tokens, scores, local_ids, up, gate, down, activation)
        layout, grouped_tokens, grouped_scores = _make_layout_and_grouped_inputs(
            tokens, scores, local_ids, up.size(0), layout_backend
        )
        up_values = _grouped_gemm(grouped_tokens, up, layout.group_rows, backend)
        if activation == "swiglu":
            assert gate is not None
            gate_values = _grouped_gemm(grouped_tokens, gate, layout.group_rows, backend)
            # Keep the two logical halves as their native GEMM outputs.  The
            # old concatenation allocated and copied another [routes, 2H]
            # temporary even though backward consumes the halves separately.
            preact_up = up_values
            preact_gate = gate_values
            hidden = _swiglu(preact_up, preact_gate, backend)
        else:
            preact_up = up_values
            preact_gate = tokens.new_empty(0)
            hidden = F.relu(preact_up).square()
        values = _grouped_gemm(hidden, down, layout.group_rows, backend)
        output = _weighted_scatter_grouped_rows(
            values, grouped_scores.to(values.dtype), layout.grouped_to_source, layout_backend
        )

        ctx.activation = activation
        ctx.backend = backend
        ctx.layout_backend = layout_backend
        ctx.down_backward_strategy = _down_backward_strategy()
        ctx.gate_present = gate is not None
        # Like SonicMoE, retain an input reference and the activation
        # preactivation only.  Grouped input/output buffers are forward
        # temporaries and are reconstructed from the inverse route map in
        # backward rather than becoming saved activations.
        ctx.save_for_backward(
            tokens,
            scores,
            up,
            gate if gate is not None else tokens.new_empty(0),
            down,
            preact_up,
            preact_gate,
            layout.grouped_to_source,
            layout.group_rows,
            layout.expert_offsets,
        )
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_output is None:
            return (None,) * 9
        (
            tokens,
            scores,
            up,
            gate_saved,
            down,
            preact_up,
            preact_gate,
            grouped_to_source,
            group_rows,
            expert_offsets,
        ) = ctx.saved_tensors
        # Scores are only one BF16 scalar per route.  The matrix route-movement
        # kernel is the important bandwidth path; keep this tiny gather in
        # PyTorch until it can be folded into the following GEMM epilogue.
        grouped_scores = scores.index_select(0, grouped_to_source).contiguous()
        need_tokens, need_scores, _, need_up, need_gate, need_down = ctx.needs_input_grad[:6]
        if ctx.activation == "swiglu":
            hidden = _swiglu(preact_up, preact_gate, ctx.backend)
        else:
            hidden = F.relu(preact_up).square()

        grouped_tokens: torch.Tensor | None = None
        grouped_grad_output: torch.Tensor | None = None

        def get_grouped_tokens() -> torch.Tensor:
            nonlocal grouped_tokens
            if grouped_tokens is None:
                # Sonic's indirect-input contract: only dW1 needs a grouped
                # X buffer.  dX and dW2 operate from other exact grouped
                # values, so they do not force this materialization.
                grouped_tokens = _gather_source_rows(
                    tokens, grouped_to_source, ctx.layout_backend
                )
            return grouped_tokens

        def get_grouped_grad_output() -> torch.Tensor:
            nonlocal grouped_grad_output
            if grouped_grad_output is None:
                grouped_grad_output = _gather_source_rows(
                    grad_output.contiguous(), grouped_to_source, ctx.layout_backend
                )
            return grouped_grad_output

        grad_scores = None
        grad_values = None
        grad_hidden = None
        need_activation_backward = need_tokens or need_up or need_gate

        if ctx.down_backward_strategy == "reordered":
            # Algebraic Sonic ordering:
            #
            # dS = <dO, A @ W_down> = <dO @ W_down.T, A>.
            #
            # This replaces the reference path's backward-only ``A @ W_down``
            # regeneration.  The unscaled dA' GEMM is already needed for dH;
            # route scaling occurs afterwards, so neither Y nor dY is saved.
            if need_scores or need_activation_backward:
                down_input_grad_unscaled = _grouped_gemm(
                    get_grouped_grad_output(),
                    down.transpose(-1, -2).contiguous(),
                    group_rows,
                    ctx.backend,
                )
                if ctx.layout_backend == "sycl":
                    from aurora_moe._kernels.sonic_ragged_ops import ragged_down_backward

                    (
                        candidate_grad_values,
                        grad_hidden,
                        grouped_grad_scores,
                    ) = ragged_down_backward(
                        get_grouped_grad_output(),
                        down_input_grad_unscaled,
                        hidden,
                        grouped_scores,
                    )
                    if need_down:
                        grad_values = candidate_grad_values
                else:
                    grad_hidden = down_input_grad_unscaled * grouped_scores.to(
                        down_input_grad_unscaled.dtype
                    ).unsqueeze(-1)
                    if need_down:
                        grad_values = get_grouped_grad_output() * grouped_scores.to(
                            grouped_grad_output.dtype
                        ).unsqueeze(-1)
                    grouped_grad_scores = (
                        down_input_grad_unscaled.float() * hidden.float()
                    ).sum(dim=-1).to(grouped_scores.dtype)
                if need_scores:
                    grad_scores = _scatter_grouped_scalars(
                        grouped_grad_scores, grouped_to_source, ctx.layout_backend
                    )
            if need_down and grad_values is None:
                grad_values = get_grouped_grad_output() * grouped_scores.to(
                    grouped_grad_output.dtype
                ).unsqueeze(-1)
        else:
            # Numerically conservative reference ordering.  It is retained as
            # the default while the reordered BF16 path receives its Aurora
            # loop-reference gate.
            if need_scores:
                values = _grouped_gemm(hidden, down, group_rows, ctx.backend)
                if ctx.layout_backend == "sycl":
                    from aurora_moe._kernels.sonic_ragged_ops import ragged_route_grad

                    grad_values, grouped_grad_scores = ragged_route_grad(
                        grad_output.contiguous(), values, grouped_scores, grouped_to_source
                    )
                else:
                    grad_values = get_grouped_grad_output() * grouped_scores.to(
                        grouped_grad_output.dtype
                    ).unsqueeze(-1)
                    grouped_grad_scores = (
                        get_grouped_grad_output().float() * values.float()
                    ).sum(dim=-1).to(grouped_scores.dtype)
                grad_scores = _scatter_grouped_scalars(
                    grouped_grad_scores, grouped_to_source, ctx.layout_backend
                )
            if (need_down or need_activation_backward) and grad_values is None:
                grad_values = get_grouped_grad_output() * grouped_scores.to(
                    grouped_grad_output.dtype
                ).unsqueeze(-1)
            if need_activation_backward:
                grad_hidden = _grouped_gemm(
                    grad_values,
                    down.transpose(-1, -2).contiguous(),
                    group_rows,
                    ctx.backend,
                )

        grad_tokens = None
        grad_up = None
        grad_gate = None
        grad_down = None
        if need_down:
            assert grad_values is not None
            grad_down = _weight_grad(
                hidden, grad_values, group_rows, down, ctx.backend, expert_offsets
            )
        if need_activation_backward:
            assert grad_hidden is not None
            if ctx.activation == "swiglu":
                grad_up_values, grad_gate_values = _swiglu_backward(
                    grad_hidden, preact_up, preact_gate, ctx.backend
                )
                if need_gate:
                    grad_gate = _weight_grad(
                        get_grouped_tokens(),
                        grad_gate_values,
                        group_rows,
                        gate_saved,
                        ctx.backend,
                        expert_offsets,
                    )
                grouped_grad_tokens = _grouped_gemm(
                    grad_gate_values,
                    gate_saved.transpose(-1, -2).contiguous(),
                    group_rows,
                    ctx.backend,
                )
            else:
                grad_up_values = grad_hidden * (2.0 * F.relu(preact_up))
                grouped_grad_tokens = torch.zeros_like(tokens)
            if need_up:
                grad_up = _weight_grad(
                    get_grouped_tokens(),
                    grad_up_values,
                    group_rows,
                    up,
                    ctx.backend,
                    expert_offsets,
                )
            if need_tokens:
                grouped_grad_tokens = grouped_grad_tokens + _grouped_gemm(
                    grad_up_values, up.transpose(-1, -2).contiguous(), group_rows, ctx.backend
                )
                grad_tokens = _scatter_grouped_rows(
                    grouped_grad_tokens, grouped_to_source, ctx.layout_backend
                )

        return grad_tokens, grad_scores, None, grad_up, grad_gate, grad_down, None, None, None


def ragged_local_moe(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    local_ids: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    activation: str = "swiglu",
    backend: str = "reference",
    layout_backend: str | None = None,
) -> torch.Tensor:
    """Apply exact local experts without BMM or max-expert-row padding.

    ``tokens`` and the returned values are both in the caller's source-route
    order.  ``local_ids`` assigns each route to one local expert.  The
    ``reference`` backend is a correctness implementation; ``tla`` uses the
    SYCL*TLA grouped GEMM for forward and token-gradient projections plus the
    exact variable-reduction grouped dW primitive.  The TLA path remains
    opt-in until its Aurora correctness and performance gates complete.
    ``layout_backend='sycl'`` is the opt-in exact device route pack/scatter
    implementation.  It remains separately selectable until its XPU gate has
    completed; the portable default is ``torch``.

    Set ``AURORA_MOE_SONIC_DOWN_BACKWARD=reordered`` to use SonicMoE's
    algebraic down-projection backward ordering.  It eliminates the
    backward-only down-output GEMM used to form score gradients, but remains
    opt-in until its BF16 XPU loop-reference gate completes.
    """

    if scores.ndim == 2 and scores.shape == (tokens.size(0), 1):
        scores = scores.reshape(-1)
    if layout_backend is None:
        layout_backend = os.environ.get("AURORA_MOE_SONIC_RAGGED_LAYOUT", "torch")
    return _RaggedLocalMoE.apply(
        tokens, scores, local_ids, up, gate, down, activation, backend, layout_backend
    )


def sonic_saved_activation_bytes(routes: int, model_dim: int, hidden_dim: int, dtype: torch.dtype) -> int:
    """Return newly allocated BF16 activation state retained by the local node.

    The input is retained by reference, as in SonicMoE's indirect-input path;
    the only new bulk activation kept for SwiGLU backward is its preactivation.
    """

    if routes < 0 or model_dim < 0 or hidden_dim < 0:
        raise ValueError("routes and dimensions must be nonnegative")
    return routes * (2 * hidden_dim) * torch.empty((), dtype=dtype).element_size()
