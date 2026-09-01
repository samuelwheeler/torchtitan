"""Exact compact expert-major MoE backward split for reverse-A4 overlap.

This module is deliberately separate from the ordinary custom-autograd node.
It exposes the same mathematical local expert computation as the router-free
fused-payload oneMKL path, but separates backward into an A4-payload phase and
an independent expert-weight-gradient tail.  The caller may submit reverse EP
communication after :meth:`payload_gradient` and run
:meth:`weight_gradients` concurrently on another stream.

There is no capacity layout, BMM, route clipping, or fixed model/expert/token
shape in this implementation.  It is restricted to the experiment's explicit
router-gradient-free SwiGLU + exact oneMKL configuration, since returning a
nonzero router gradient creates an additional dependency that is outside the
current scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aurora_moe._kernels.expert_major_segmented_sonic import (
    _check_payload_inputs,
    _grouped_gemm,
    _grouped_gemm_with_forward_weight_transpose,
    _grouped_sum_with_forward_weight_transposes,
    _onemkl_fuse_up_gate_dx_requested,
    _pack_payload,
    _resolve_expert_rows,
    _swiglu,
    _swiglu_backward,
    _unpack_scaled,
    _weight_grad,
    make_expert_group_segments,
)
from aurora_moe._kernels.segment_expert_reorder import (
    ExpertMajorLayout,
    expert_major_to_payload_zero_score_parallel,
    expert_major_to_payload_zero_score_row_parallel,
    make_expert_major_layout,
    payload_and_grad_to_expert_major_scaled_parallel,
    payload_and_grad_to_expert_major_scaled_row_parallel,
)
from aurora_moe._kernels.segmented_sonic import PeerExpertSegments


@dataclass
class ExactExpertMajorTwoPhaseBackward:
    """Retained exact forward state for a split compact-expert backward.

    Call order is intentionally strict:

    1. :meth:`payload_gradient` produces compact physical ``[dX, 0]`` rows;
    2. submit reverse A4 using that tensor; and
    3. :meth:`weight_gradients` runs dW on a stream independent of A4.

    All tensors are compact physical or expert-major matrices whose leading
    extent is the actual route count.  The class owns no streams or
    collectives, so its caller remains responsible for stream dependencies and
    keeping the state alive through the tail launch.
    """

    payload: torch.Tensor
    physical_segments: PeerExpertSegments
    expert_segments: PeerExpertSegments
    layout: ExpertMajorLayout
    up: torch.Tensor
    gate: torch.Tensor
    down: torch.Tensor
    preact_up: torch.Tensor
    preact_gate: torch.Tensor
    expert_rows: tuple[int, ...]
    reorder_backend: str
    pointwise_backend: str
    already_expert_major: bool = False
    _payload_started: bool = False
    _weight_started: bool = False
    _zero_routes: bool = False
    _grouped_tokens: torch.Tensor | None = None
    _grad_values: torch.Tensor | None = None
    _hidden: torch.Tensor | None = None
    _grad_up_values: torch.Tensor | None = None
    _grad_gate_values: torch.Tensor | None = None
    _grad_payload: torch.Tensor | None = None
    _weight_outputs: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None

    @property
    def zero_routes(self) -> bool:
        return self._zero_routes

    def payload_gradient(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Return exact physical ``[dX..., 0]`` rows without running dW.

        The BF16 route-score multiplication is deliberately performed in the
        fused source/expert permutation, exactly as in the router-free
        autograd node.  The score-gradient column is zero because router
        optimization is explicitly excluded by this experimental path.
        """

        if self._payload_started:
            raise RuntimeError("payload_gradient may be called only once")
        if (
            grad_output.device != self.payload.device
            or grad_output.dtype != self.payload.dtype
            or grad_output.ndim != 2
            or grad_output.shape != (self.payload.size(0), self.up.size(1))
            or not grad_output.is_contiguous()
        ):
            raise ValueError("grad_output must be contiguous BF16 [routes, model_dim]")
        self._payload_started = True
        with torch.no_grad():
            if self._zero_routes:
                self._grad_payload = self.payload.new_empty(
                    (0, self.payload.size(1))
                )
                return self._grad_payload

            if self.already_expert_major:
                grouped_tokens = self.payload[:, :-1].contiguous()
                grad_values = grad_output * self.payload[:, -1:].contiguous()
            elif self.reorder_backend == "row_parallel":
                grouped_tokens, grad_values = (
                    payload_and_grad_to_expert_major_scaled_row_parallel(
                        self.payload, grad_output, self.physical_segments, self.layout
                    )
                )
            else:
                grouped_tokens, grad_values = (
                    payload_and_grad_to_expert_major_scaled_parallel(
                        self.payload, grad_output, self.physical_segments, self.layout
                    )
                )
            hidden = _swiglu(self.preact_up, self.preact_gate, self.pointwise_backend)
            grad_hidden = _grouped_gemm_with_forward_weight_transpose(
                grad_values,
                self.down,
                self.expert_segments,
                "onemkl",
                self.expert_rows,
            )
            grad_up_values, grad_gate_values = _swiglu_backward(
                grad_hidden, self.preact_up, self.preact_gate, self.pointwise_backend
            )
            if _onemkl_fuse_up_gate_dx_requested():
                grouped_grad_tokens = _grouped_sum_with_forward_weight_transposes(
                    grad_gate_values,
                    self.gate,
                    grad_up_values,
                    self.up,
                    self.expert_segments,
                    self.expert_rows,
                )
            else:
                grouped_grad_tokens = _grouped_gemm_with_forward_weight_transpose(
                    grad_gate_values,
                    self.gate,
                    self.expert_segments,
                    "onemkl",
                    self.expert_rows,
                )
                grouped_grad_tokens.add_(
                    _grouped_gemm_with_forward_weight_transpose(
                        grad_up_values,
                        self.up,
                        self.expert_segments,
                        "onemkl",
                        self.expert_rows,
                    )
                )
            if self.already_expert_major:
                grad_payload = torch.cat(
                    (grouped_grad_tokens, torch.zeros_like(self.payload[:, -1:])), dim=-1
                )
            elif self.reorder_backend == "row_parallel":
                grad_payload = expert_major_to_payload_zero_score_row_parallel(
                    grouped_grad_tokens, self.physical_segments, self.layout
                )
            else:
                grad_payload = expert_major_to_payload_zero_score_parallel(
                    grouped_grad_tokens, self.physical_segments, self.layout
                )
            self._grouped_tokens = grouped_tokens
            self._grad_values = grad_values
            self._hidden = hidden
            self._grad_up_values = grad_up_values
            self._grad_gate_values = grad_gate_values
            self._grad_payload = grad_payload
        return grad_payload

    def weight_gradients(
        self,
        *,
        need_up: bool,
        need_gate: bool,
        need_down: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Run exact per-expert dW GEMMs after :meth:`payload_gradient`."""

        if not self._payload_started:
            raise RuntimeError("call payload_gradient before weight_gradients")
        if self._weight_started:
            raise RuntimeError("weight_gradients may be called only once")
        self._weight_started = True
        with torch.no_grad():
            if self._zero_routes:
                self._weight_outputs = (
                    torch.zeros_like(self.up) if need_up else None,
                    torch.zeros_like(self.gate) if need_gate else None,
                    torch.zeros_like(self.down) if need_down else None,
                )
                return self._weight_outputs
            if (
                self._grouped_tokens is None
                or self._grad_values is None
                or self._hidden is None
                or self._grad_up_values is None
                or self._grad_gate_values is None
            ):
                raise RuntimeError("payload_gradient did not retain dW operands")
            grad_up = (
                _weight_grad(
                    self._grouped_tokens,
                    self._grad_up_values,
                    self.expert_segments,
                    self.layout,
                    "onemkl",
                    self.expert_rows,
                )
                if need_up
                else None
            )
            grad_gate = (
                _weight_grad(
                    self._grouped_tokens,
                    self._grad_gate_values,
                    self.expert_segments,
                    self.layout,
                    "onemkl",
                    self.expert_rows,
                )
                if need_gate
                else None
            )
            grad_down = (
                _weight_grad(
                    self._hidden,
                    self._grad_values,
                    self.expert_segments,
                    self.layout,
                    "onemkl",
                    self.expert_rows,
                )
                if need_down
                else None
            )
        self._weight_outputs = grad_up, grad_gate, grad_down
        return self._weight_outputs

    def record_weight_tail_stream(self, stream: torch.xpu.Stream) -> None:
        """Record all retained tail storage on a caller-owned XPU stream."""

        if self._weight_outputs is None:
            raise RuntimeError("call weight_gradients before recording its stream")
        for tensor in (
            self.payload,
            self.physical_segments.counts,
            self.physical_segments.source_expert_offsets,
            self.physical_segments.source_offsets,
            self.layout.expert_offsets_i64,
            self.layout.expert_offsets_i32,
            self.layout.expert_source_offsets,
            self.layout.fragment_row_block_offsets,
            self.up,
            self.gate,
            self.down,
            self.preact_up,
            self.preact_gate,
            self._grouped_tokens,
            self._grad_values,
            self._hidden,
            self._grad_up_values,
            self._grad_gate_values,
            *self._weight_outputs,
        ):
            if tensor is not None:
                tensor.record_stream(stream)


def prepare_exact_expert_major_two_phase(
    payload: torch.Tensor,
    physical_segments: PeerExpertSegments,
    up: torch.Tensor,
    gate: torch.Tensor,
    down: torch.Tensor,
    *,
    expert_rows: tuple[int, ...] | None = None,
    reorder_backend: str = "parallel",
    pointwise_backend: str = "torch",
    already_expert_major: bool = False,
) -> tuple[torch.Tensor, ExactExpertMajorTwoPhaseBackward]:
    """Run exact forward and retain a router-free dX/dW split backward state.

    This is intentionally not a generic substitute for the ordinary autograd
    function.  It is an opt-in overlap primitive for the exact oneMKL SwiGLU
    path only, where all three weight gradients and the token gradient are
    required by normal training.
    """

    if reorder_backend not in ("parallel", "row_parallel"):
        raise ValueError("two-phase expert-major reorder must be 'parallel' or 'row_parallel'")
    if pointwise_backend not in ("torch", "sycl"):
        raise ValueError("two-phase expert-major pointwise backend must be 'torch' or 'sycl'")
    _check_payload_inputs(payload, physical_segments, up, gate, down, "swiglu")
    expert_segments = make_expert_group_segments(physical_segments)
    resolved_rows = _resolve_expert_rows(
        expert_segments,
        expert_rows,
        total_rows=payload.size(0),
        required=True,
    )
    assert resolved_rows is not None
    layout = make_expert_major_layout(
        physical_segments, include_row_schedule=reorder_backend == "row_parallel"
    )
    zero_routes = payload.size(0) == 0
    with torch.no_grad():
        if zero_routes:
            output = payload.new_empty((0, up.size(1)))
            preact_up = payload.new_empty((0, up.size(2)))
            preact_gate = payload.new_empty((0, gate.size(2)))
        else:
            if already_expert_major:
                grouped_tokens = payload[:, :-1].contiguous()
                grouped_scores = payload[:, -1].contiguous()
            else:
                grouped_tokens, grouped_scores = _pack_payload(
                    payload, physical_segments, layout, reorder_backend
                )
            preact_up = _grouped_gemm(
                grouped_tokens, up, expert_segments, "onemkl", resolved_rows
            )
            preact_gate = _grouped_gemm(
                grouped_tokens, gate, expert_segments, "onemkl", resolved_rows
            )
            hidden = _swiglu(preact_up, preact_gate, pointwise_backend)
            values = _grouped_gemm(
                hidden, down, expert_segments, "onemkl", resolved_rows
            )
            output = (
                values * grouped_scores.unsqueeze(-1)
                if already_expert_major
                else _unpack_scaled(
                    values, grouped_scores, physical_segments, layout, reorder_backend
                )
            )
    state = ExactExpertMajorTwoPhaseBackward(
        payload=payload,
        physical_segments=physical_segments,
        expert_segments=expert_segments,
        layout=layout,
        up=up,
        gate=gate,
        down=down,
        preact_up=preact_up,
        preact_gate=preact_gate,
        expert_rows=resolved_rows,
        reorder_backend=reorder_backend,
        pointwise_backend=pointwise_backend,
        already_expert_major=already_expert_major,
        _zero_routes=zero_routes,
    )
    return output, state
