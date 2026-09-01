"""Exact padded-EP expert BMM using direct expert-major layout kernels.

Integration point: replace the receive-side compact/sort/gather/pad block in
``_RoutedMOESyclEP`` after its dispatch all-to-all.  This module consumes the
source-major padded receive payload directly and returns the identically shaped
padded, score-weighted expert outputs.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from aurora_moe._kernels.expert_layout_ops import (
    PaddedExpertLayout,
    expert_token_score_to_padded,
    make_padded_expert_layout,
    padded_rows_to_expert,
    weighted_expert_rows_to_padded,
)
from aurora_moe._kernels.route_grad_ops import route_grad_scale_score_from_layout
from aurora_moe._kernels.swiglu_ops import load_swiglu_ops


def _fused_route_grad_requested() -> bool:
    return os.environ.get("AURORA_MOE_FUSED_ROUTE_GRAD") == "1"


def _fused_dx_baddbmm_requested() -> bool:
    return os.environ.get("AURORA_MOE_FUSE_DX_BADDBMM") == "1"


def _fused_dx_payload_requested() -> bool:
    """Whether to fuse separate SwiGLU dX add with reverse-A4 packing."""

    requested = os.environ.get("AURORA_MOE_FUSE_DX_PAYLOAD") == "1"
    if requested and _fused_dx_baddbmm_requested():
        raise RuntimeError(
            "AURORA_MOE_FUSE_DX_PAYLOAD and AURORA_MOE_FUSE_DX_BADDBMM "
            "are mutually exclusive"
        )
    return requested


def _check_inputs(
    payload: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    num_experts: int,
    local_ids: torch.Tensor | None,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    activation: str,
) -> None:
    if (
        payload.device.type != "xpu"
        or payload.dtype != torch.bfloat16
        or payload.ndim != 2
        or not payload.is_contiguous()
    ):
        raise ValueError("payload must be a contiguous BF16 XPU matrix")
    if (
        recv_counts.device != payload.device
        or recv_counts.dtype != torch.int64
        or recv_counts.ndim != 1
        or not recv_counts.is_contiguous()
    ):
        raise ValueError("recv_counts must be a contiguous int64 XPU vector")
    if cap < 0 or num_experts <= 0:
        raise ValueError("cap must be nonnegative and num_experts must be positive")
    if payload.size(0) != recv_counts.numel() * cap:
        raise ValueError("payload rows must equal recv_counts.numel() * cap")
    if (
        up.device != payload.device
        or up.dtype != torch.bfloat16
        or up.ndim != 3
        or not up.is_contiguous()
    ):
        raise ValueError("up must be a contiguous BF16 XPU [experts, model_dim, hidden_dim]")
    if up.size(0) != num_experts:
        raise ValueError("up expert dimension must equal num_experts")
    if (
        down.device != payload.device
        or down.dtype != torch.bfloat16
        or down.ndim != 3
        or not down.is_contiguous()
        or down.shape != (num_experts, up.size(2), up.size(1))
    ):
        raise ValueError("down must be contiguous BF16 [experts, hidden_dim, model_dim]")
    if activation not in ("swiglu", "squared-relu"):
        raise ValueError("activation must be 'swiglu' or 'squared-relu'")
    if activation == "swiglu":
        if (
            gate is None
            or gate.device != payload.device
            or gate.dtype != torch.bfloat16
            or not gate.is_contiguous()
            or gate.shape != up.shape
        ):
            raise ValueError("SwiGLU requires contiguous BF16 gate shaped like up")
    elif gate is not None:
        raise ValueError("squared-relu does not use a gate weight")
    expected_columns = up.size(1) + (1 if local_ids is not None else 2)
    if payload.size(1) != expected_columns:
        raise ValueError("payload columns must be [token..., score] or [token..., score, id]")
    if local_ids is not None and (
        local_ids.device != payload.device
        or local_ids.dtype != torch.int64
        or local_ids.ndim != 1
        or not local_ids.is_contiguous()
        or local_ids.numel() != payload.size(0)
    ):
        raise ValueError("local_ids must be contiguous int64 with one entry per padded payload row")


class _DirectPaddedLayoutBmmMoE(torch.autograd.Function):
    """Manual autograd for direct padded EP layout plus expert BMMs."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        payload: torch.Tensor,
        recv_counts: torch.Tensor,
        cap: int,
        num_experts: int,
        local_ids: torch.Tensor | None,
        up: torch.Tensor,
        gate: torch.Tensor | None,
        down: torch.Tensor,
        activation: str,
        known_group_rows: torch.Tensor | None,
        known_max_rows: int | None,
        known_max_tail: int | None,
    ) -> torch.Tensor:
        _check_inputs(
            payload, recv_counts, cap, num_experts, local_ids, up, gate, down, activation
        )
        layout = make_padded_expert_layout(
            payload,
            recv_counts,
            cap,
            num_experts,
            local_ids=local_ids,
            known_group_rows=known_group_rows,
            known_max_rows=known_max_rows,
            known_max_tail=known_max_tail,
        )
        ctx.layout = layout
        ctx.activation = activation
        ctx.gate_present = gate is not None
        ctx.payload_columns = payload.size(1)
        ctx.model_dim = up.size(1)
        ctx.has_embedded_ids = local_ids is None
        ctx.zero_routes = layout.max_rows == 0
        if ctx.zero_routes:
            ctx.save_for_backward(
                recv_counts,
                up,
                gate if gate is not None else torch.empty(0, dtype=payload.dtype, device=payload.device),
                down,
            )
            return payload.new_zeros((payload.size(0), up.size(1)))

        tokens = layout.tokens
        up_values = torch.bmm(tokens, up)
        if activation == "swiglu":
            assert gate is not None
            gate_values = torch.bmm(tokens, gate)
            hidden = load_swiglu_ops().swiglu_forward_bf16(up_values, gate_values)
        else:
            gate_values = torch.empty(0, dtype=payload.dtype, device=payload.device)
            hidden = F.relu(up_values).square()
        expert_values = torch.bmm(hidden, down)
        output = weighted_expert_rows_to_padded(
            expert_values.contiguous(), recv_counts, layout
        )
        ctx.save_for_backward(
            recv_counts,
            tokens,
            up,
            gate if gate is not None else torch.empty(0, dtype=payload.dtype, device=payload.device),
            down,
            up_values,
            gate_values,
            hidden,
            expert_values,
        )
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_output.dtype != torch.bfloat16 or grad_output.ndim != 2:
            raise RuntimeError("direct padded expert BMM requires BF16 rank-2 grad_output")
        if ctx.zero_routes:
            _, up, gate, down = ctx.saved_tensors
            grad_payload = (
                grad_output.new_zeros((grad_output.size(0), ctx.payload_columns))
                if ctx.needs_input_grad[0]
                else None
            )
            return (
                grad_payload,
                None,
                None,
                None,
                None,
                torch.zeros_like(up) if ctx.needs_input_grad[5] else None,
                torch.zeros_like(gate) if ctx.gate_present and ctx.needs_input_grad[6] else None,
                torch.zeros_like(down) if ctx.needs_input_grad[7] else None,
                None,
                None,
                None,
                None,
            )

        (
            recv_counts,
            tokens,
            up,
            gate,
            down,
            up_values,
            gate_values,
            hidden,
            expert_values,
        ) = ctx.saved_tensors
        layout: PaddedExpertLayout = ctx.layout
        grad_output = grad_output.contiguous()
        if grad_output.shape != (layout.padded_source_rows, ctx.model_dim):
            raise RuntimeError("grad_output shape does not match the direct padded expert layout")
        need_payload = ctx.needs_input_grad[0]
        need_up = ctx.needs_input_grad[5]
        need_gate = ctx.gate_present and ctx.needs_input_grad[6]
        need_down = ctx.needs_input_grad[7]

        grad_payload = None
        if _fused_route_grad_requested():
            grad_weighted_values, grad_scores = route_grad_scale_score_from_layout(
                grad_output, expert_values, recv_counts, layout
            )
        else:
            grad_expert_values = padded_rows_to_expert(
                grad_output, recv_counts, layout
            )
            if need_payload:
                grad_scores = (
                    grad_expert_values.float() * expert_values.float()
                ).sum(dim=-1).to(torch.bfloat16)
            grad_weighted_values = (
                grad_expert_values * layout.scores.unsqueeze(-1)
            ).to(torch.bfloat16)

        grad_down = (
            torch.bmm(hidden.transpose(-1, -2), grad_weighted_values)
            if need_down
            else None
        )
        grad_up = None
        grad_gate = None
        grad_tokens = None
        if need_payload or need_up or need_gate:
            grad_hidden = torch.bmm(grad_weighted_values, down.transpose(-1, -2))
            if ctx.activation == "swiglu":
                grad_up_values, grad_gate_values = load_swiglu_ops().swiglu_backward_bf16(
                    grad_hidden, up_values, gate_values
                )
                if need_up:
                    grad_up = torch.bmm(tokens.transpose(-1, -2), grad_up_values)
                if need_gate:
                    grad_gate = torch.bmm(tokens.transpose(-1, -2), grad_gate_values)
                if need_payload:
                    grad_tokens = torch.bmm(grad_up_values, up.transpose(-1, -2))
                    if _fused_dx_payload_requested():
                        from aurora_moe._kernels.direct_layout_payload_fusion import (
                            fused_expert_token_add_score_to_padded,
                        )

                        grad_gate_tokens = torch.bmm(
                            grad_gate_values, gate.transpose(-1, -2)
                        )
                        grad_payload = fused_expert_token_add_score_to_padded(
                            grad_tokens.contiguous(),
                            grad_gate_tokens.contiguous(),
                            grad_scores.contiguous(),
                            recv_counts,
                            layout,
                            payload_columns=ctx.payload_columns,
                        )
                    elif _fused_dx_baddbmm_requested():
                        grad_tokens.baddbmm_(
                            grad_gate_values, gate.transpose(-1, -2)
                        )
                    else:
                        grad_tokens = grad_tokens + torch.bmm(
                            grad_gate_values, gate.transpose(-1, -2)
                        )
            else:
                grad_up_values = grad_hidden * (2.0 * F.relu(up_values))
                if need_up:
                    grad_up = torch.bmm(tokens.transpose(-1, -2), grad_up_values)
                if need_payload:
                    grad_tokens = torch.bmm(grad_up_values, up.transpose(-1, -2))

        if need_payload:
            if grad_payload is None:
                assert grad_tokens is not None
                grad_payload = expert_token_score_to_padded(
                    grad_tokens.contiguous(),
                    grad_scores.contiguous(),
                    recv_counts,
                    layout,
                    payload_columns=ctx.payload_columns,
                )

        return (
            grad_payload,
            None,
            None,
            None,
            None,
            grad_up,
            grad_gate,
            grad_down,
            None,
            None,
            None,
            None,
        )


class DirectPaddedLayoutBmmTwoPhaseBackward:
    """Retained first-order backward state for direct-layout EP experts.

    This is an opt-in manual-autograd seam for the direct padded layout.  It
    splits the local expert backward into two ordered phases:

    1. :meth:`payload_gradient` produces the exact padded token/score
       gradient consumed by the reverse EP all-to-all (A4).
    2. :meth:`weight_gradients` runs the independent expert-weight BMMs.

    A caller can therefore submit A4 immediately after phase 1, enqueue phase
    2 on another XPU stream, and join that stream before returning the weight
    gradients to its enclosing autograd Function.  This object deliberately
    does *not* create events or synchronize streams: the caller owns that
    dependency.  Retain this context until both streams complete, or call
    :meth:`record_weight_tail_stream` after queueing phase 2 so its retained
    storage remains live if an enclosing autograd context releases it early.
    The returned A4 payload has the usual caller-owned tensor lifetime and
    must likewise be recorded on a nondefault collective stream if necessary.

    Instances are created only by
    :func:`prepare_direct_padded_expert_bmm_two_phase`.  The helper executes a
    manual, first-order forward under ``torch.no_grad()``; use the existing
    :func:`direct_padded_expert_bmm` for ordinary autograd or higher-order
    gradient use.  The exact dynamic ``recv_counts`` layout is retained, so
    this API never clips routes or assumes fixed expert/model/batch sizes.
    """

    def __init__(
        self,
        *,
        recv_counts: torch.Tensor,
        layout: PaddedExpertLayout,
        up: torch.Tensor,
        gate: torch.Tensor | None,
        down: torch.Tensor,
        activation: str,
        payload_columns: int,
        tokens: torch.Tensor | None,
        up_values: torch.Tensor | None,
        gate_values: torch.Tensor | None,
        hidden: torch.Tensor | None,
        expert_values: torch.Tensor | None,
    ) -> None:
        self._recv_counts = recv_counts
        self._layout = layout
        self._up = up
        self._gate = gate
        self._down = down
        self._activation = activation
        self._payload_columns = payload_columns
        self._model_dim = up.size(1)
        self._tokens = tokens
        self._up_values = up_values
        self._gate_values = gate_values
        self._hidden = hidden
        self._expert_values = expert_values
        self._zero_routes = layout.max_rows == 0
        self._payload_phase_started = False
        self._weight_phase_started = False
        self._grad_weighted_values: torch.Tensor | None = None
        self._grad_up_values: torch.Tensor | None = None
        self._grad_gate_values: torch.Tensor | None = None
        # Retain the returned payload too.  Besides documenting the ownership
        # boundary, this keeps its storage alive while a caller submits A4.
        self._grad_payload: torch.Tensor | None = None
        self._weight_outputs: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ] | None = None

    @property
    def payload_columns(self) -> int:
        """Columns in the exact reverse-A4 payload, including score/ID slots."""

        return self._payload_columns

    @property
    def zero_routes(self) -> bool:
        """Whether this receive side had no valid routes."""

        return self._zero_routes

    def payload_gradient(
        self, grad_output: torch.Tensor, *, payload_columns: int | None = None
    ) -> torch.Tensor:
        """Produce the exact source-padded A4 payload gradient once.

        ``grad_output`` must be the BF16 source-padded gradient of the forward
        result returned by :func:`prepare_direct_padded_expert_bmm_two_phase`.
        This phase also retains the BMM inputs needed by
        :meth:`weight_gradients`, but does not run any expert-weight-gradient
        BMM.  Call it on the stream that will make the returned tensor
        available to A4; the caller is responsible for establishing any
        cross-stream dependency before starting the tail phase.  By default
        the returned column count matches forward ``payload``.  An embedded
        BF16-ID payload may instead request ``model_dim + 1`` columns to omit
        its known-zero ID gradient before A4.
        """

        if self._payload_phase_started:
            raise RuntimeError("payload_gradient may be called only once per two-phase context")
        if (
            grad_output.device != self._up.device
            or grad_output.dtype != torch.bfloat16
            or grad_output.ndim != 2
        ):
            raise ValueError("grad_output must be a BF16 rank-2 XPU tensor on the expert device")
        if grad_output.shape != (self._layout.padded_source_rows, self._model_dim):
            raise ValueError("grad_output shape does not match the direct padded expert layout")
        output_columns = self._payload_columns if payload_columns is None else payload_columns
        if output_columns not in (self._model_dim + 1, self._payload_columns):
            raise ValueError(
                "payload_columns must be model_dim + 1 or the original payload column count"
            )

        self._payload_phase_started = True
        grad_output = grad_output.contiguous()
        with torch.no_grad():
            if self._zero_routes:
                self._grad_payload = grad_output.new_zeros(
                    (grad_output.size(0), output_columns)
                )
                return self._grad_payload

            tokens = self._tokens
            up_values = self._up_values
            hidden = self._hidden
            expert_values = self._expert_values
            if tokens is None or up_values is None or hidden is None or expert_values is None:
                raise RuntimeError("two-phase context is missing nonempty forward state")

            if _fused_route_grad_requested():
                grad_weighted_values, grad_scores = route_grad_scale_score_from_layout(
                    grad_output,
                    expert_values,
                    self._recv_counts,
                    self._layout,
                )
            else:
                grad_expert_values = padded_rows_to_expert(
                    grad_output, self._recv_counts, self._layout
                )
                grad_scores = (
                    grad_expert_values.float() * expert_values.float()
                ).sum(dim=-1).to(torch.bfloat16)
                grad_weighted_values = (
                    grad_expert_values * self._layout.scores.unsqueeze(-1)
                ).to(torch.bfloat16)

            # These intermediates are the only dependency of the later dW
            # BMMs.  Computing them here leaves phase 2 as BMM-only work that
            # can overlap the reverse EP collective.
            grad_hidden = torch.bmm(
                grad_weighted_values, self._down.transpose(-1, -2)
            )
            if self._activation == "swiglu":
                gate = self._gate
                gate_values = self._gate_values
                if gate is None or gate_values is None:
                    raise RuntimeError("SwiGLU two-phase context is missing gate state")
                grad_up_values, grad_gate_values = load_swiglu_ops().swiglu_backward_bf16(
                    grad_hidden, up_values, gate_values
                )
                grad_tokens = torch.bmm(
                    grad_up_values, self._up.transpose(-1, -2)
                )
                if _fused_dx_payload_requested():
                    from aurora_moe._kernels.direct_layout_payload_fusion import (
                        fused_expert_token_add_score_to_padded,
                    )

                    grad_gate_tokens = torch.bmm(
                        grad_gate_values, gate.transpose(-1, -2)
                    )
                    self._grad_payload = fused_expert_token_add_score_to_padded(
                        grad_tokens.contiguous(),
                        grad_gate_tokens.contiguous(),
                        grad_scores.contiguous(),
                        self._recv_counts,
                        self._layout,
                        payload_columns=output_columns,
                    )
                elif _fused_dx_baddbmm_requested():
                    grad_tokens.baddbmm_(
                        grad_gate_values, gate.transpose(-1, -2)
                    )
                else:
                    grad_tokens = grad_tokens + torch.bmm(
                        grad_gate_values, gate.transpose(-1, -2)
                    )
            else:
                grad_up_values = grad_hidden * (2.0 * F.relu(up_values))
                grad_gate_values = None
                grad_tokens = torch.bmm(
                    grad_up_values, self._up.transpose(-1, -2)
                )

            self._grad_weighted_values = grad_weighted_values
            self._grad_up_values = grad_up_values
            self._grad_gate_values = grad_gate_values
            if self._grad_payload is None:
                self._grad_payload = expert_token_score_to_padded(
                    grad_tokens.contiguous(),
                    grad_scores.contiguous(),
                    self._recv_counts,
                    self._layout,
                    payload_columns=output_columns,
                )
        return self._grad_payload

    def weight_gradients(
        self,
        *,
        need_up: bool | None = None,
        need_gate: bool | None = None,
        need_down: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Run the dW BMM tail after :meth:`payload_gradient`.

        By default, each gradient follows the corresponding weight's
        ``requires_grad`` flag; explicit arguments allow an enclosing custom
        autograd Function to omit unused gradients.  This method may be
        called exactly once and never synchronizes with A4.  If it runs on a
        separate stream, record/wait events outside this object.  If outer
        autograd can release this state before that stream finishes, call
        :meth:`record_weight_tail_stream` after queueing the BMMs.
        """

        if not self._payload_phase_started:
            raise RuntimeError("call payload_gradient before weight_gradients")
        if self._weight_phase_started:
            raise RuntimeError("weight_gradients may be called only once per two-phase context")
        self._weight_phase_started = True

        requested_up = self._up.requires_grad if need_up is None else bool(need_up)
        requested_gate = (
            self._gate is not None
            and (self._gate.requires_grad if need_gate is None else bool(need_gate))
        )
        requested_down = self._down.requires_grad if need_down is None else bool(need_down)

        with torch.no_grad():
            if self._zero_routes:
                self._weight_outputs = (
                    torch.zeros_like(self._up) if requested_up else None,
                    torch.zeros_like(self._gate) if requested_gate else None,
                    torch.zeros_like(self._down) if requested_down else None,
                )
                return self._weight_outputs

            tokens = self._tokens
            hidden = self._hidden
            grad_weighted_values = self._grad_weighted_values
            grad_up_values = self._grad_up_values
            if (
                tokens is None
                or hidden is None
                or grad_weighted_values is None
                or grad_up_values is None
            ):
                raise RuntimeError("payload_gradient did not retain the required dW state")

            grad_up = (
                torch.bmm(tokens.transpose(-1, -2), grad_up_values)
                if requested_up
                else None
            )
            grad_down = (
                torch.bmm(hidden.transpose(-1, -2), grad_weighted_values)
                if requested_down
                else None
            )
            grad_gate = None
            if requested_gate:
                grad_gate_values = self._grad_gate_values
                if grad_gate_values is None:
                    raise RuntimeError("SwiGLU dW gate gradient state is unavailable")
                grad_gate = torch.bmm(tokens.transpose(-1, -2), grad_gate_values)
        self._weight_outputs = grad_up, grad_gate, grad_down
        return self._weight_outputs

    def record_weight_tail_stream(self, stream: torch.xpu.Stream) -> None:
        """Record all phase-2 storage on ``stream`` before this state is freed.

        Call this after :meth:`weight_gradients` has enqueued its BMMs on a
        nondefault stream and before the enclosing autograd context can be
        released.  This is an allocator-lifetime record only; it does not add
        an execution dependency or host synchronization.  The caller must
        still join ``stream`` with the consumer stream before returning dW.
        """

        if self._weight_outputs is None:
            raise RuntimeError("call weight_gradients before recording its stream")
        tensors = (
            self._up,
            self._gate,
            self._down,
            self._tokens,
            self._hidden,
            self._grad_weighted_values,
            self._grad_up_values,
            self._grad_gate_values,
            *self._weight_outputs,
        )
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(stream)


def prepare_direct_padded_expert_bmm_two_phase(
    payload: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    num_experts: int,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    local_ids: torch.Tensor | None = None,
    activation: str = "swiglu",
    known_group_rows: torch.Tensor | None = None,
    known_max_rows: int | None = None,
    known_max_tail: int | None = None,
) -> tuple[torch.Tensor, DirectPaddedLayoutBmmTwoPhaseBackward]:
    """Run a manual direct-layout forward and retain an A4/dW split backward.

    This opt-in API has the same exact input/layout contract as
    :func:`direct_padded_expert_bmm`, including arbitrary dynamic
    ``recv_counts`` and no capacity-derived route clipping.  Unlike that
    ordinary autograd API, it executes forward under ``torch.no_grad()`` and
    returns a context whose two explicit backward phases are intended for an
    enclosing custom autograd Function:

    ``grad_payload = state.payload_gradient(grad_output)``
    ``grad_up, grad_gate, grad_down = state.weight_gradients(...)``

    To overlap A4 and dW, submit A4 immediately after the first line, make a
    tail stream wait for phase-1 completion before the second line, then join
    the tail stream before returning dW tensors.  Keep ``state`` alive until
    both operations complete.  The helper itself introduces no stream events,
    synchronization, shape specializations, or route-set changes.
    """

    _check_inputs(
        payload, recv_counts, cap, num_experts, local_ids, up, gate, down, activation
    )
    with torch.no_grad():
        layout = make_padded_expert_layout(
            payload,
            recv_counts,
            cap,
            num_experts,
            local_ids=local_ids,
            known_group_rows=known_group_rows,
            known_max_rows=known_max_rows,
            known_max_tail=known_max_tail,
        )
        if layout.max_rows == 0:
            output = payload.new_zeros((payload.size(0), up.size(1)))
            state = DirectPaddedLayoutBmmTwoPhaseBackward(
                recv_counts=recv_counts,
                layout=layout,
                up=up,
                gate=gate,
                down=down,
                activation=activation,
                payload_columns=payload.size(1),
                tokens=None,
                up_values=None,
                gate_values=None,
                hidden=None,
                expert_values=None,
            )
            return output, state

        tokens = layout.tokens
        up_values = torch.bmm(tokens, up)
        if activation == "swiglu":
            assert gate is not None
            gate_values = torch.bmm(tokens, gate)
            hidden = load_swiglu_ops().swiglu_forward_bf16(up_values, gate_values)
        else:
            gate_values = None
            hidden = F.relu(up_values).square()
        expert_values = torch.bmm(hidden, down)
        output = weighted_expert_rows_to_padded(
            expert_values.contiguous(), recv_counts, layout
        )
        state = DirectPaddedLayoutBmmTwoPhaseBackward(
            recv_counts=recv_counts,
            layout=layout,
            up=up,
            gate=gate,
            down=down,
            activation=activation,
            payload_columns=payload.size(1),
            tokens=tokens,
            up_values=up_values,
            gate_values=gate_values,
            hidden=hidden,
            expert_values=expert_values,
        )
    return output, state


def direct_padded_expert_bmm(
    payload: torch.Tensor,
    recv_counts: torch.Tensor,
    cap: int,
    num_experts: int,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    *,
    local_ids: torch.Tensor | None = None,
    activation: str = "swiglu",
    known_group_rows: torch.Tensor | None = None,
    known_max_rows: int | None = None,
    known_max_tail: int | None = None,
) -> torch.Tensor:
    """Run exact local experts from a padded EP receive payload.

    Valid rows are determined only by ``recv_counts``.  With ``local_ids``
    omitted, the final BF16 payload column carries local expert IDs in
    ``[0, 255]``; otherwise ``local_ids`` supplies arbitrary int64 IDs and
    payload columns are ``[token..., score]``.  Every valid ID must be in
    ``[0, num_experts)``; padding IDs are ignored.
    """

    return _DirectPaddedLayoutBmmMoE.apply(
        payload,
        recv_counts,
        cap,
        num_experts,
        local_ids,
        up,
        gate,
        down,
        activation,
        known_group_rows,
        known_max_rows,
        known_max_tail,
    )
