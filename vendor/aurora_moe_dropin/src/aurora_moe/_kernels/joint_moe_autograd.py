"""DDP-safe, stream-ordered routed plus shared MoE autograd orchestration."""

from __future__ import annotations

from collections.abc import Callable

import torch

from aurora_moe._kernels.phase_shared_expert import PhaseSharedExpertController
from aurora_moe._kernels.shared_expert_bmm import shared_expert_loop

RoutedForward = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def _wait(consumer: torch.xpu.Stream, producer: torch.xpu.Stream) -> None:
    if consumer != producer:
        consumer.wait_stream(producer)


def _leaf(tensor: torch.Tensor, needed: bool) -> torch.Tensor:
    return tensor.detach().requires_grad_(needed)


def _gradients(
    output: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    indices: tuple[int, ...],
    grad_output: torch.Tensor,
    result: list[torch.Tensor | None],
) -> None:
    if not tensors:
        return
    for index, gradient in zip(
        indices, torch.autograd.grad(output, tensors, grad_output, allow_unused=True)
    ):
        result[index] = gradient


def _check(
    x: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
    routed_up: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_down: torch.Tensor,
    shared_up: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_down: torch.Tensor,
    shared_stream: torch.xpu.Stream,
) -> None:
    tensors = (
        scores,
        indices,
        routed_up,
        routed_gate,
        routed_down,
        shared_up,
        shared_gate,
        shared_down,
    )
    if x.device.type != "xpu" or any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all joint-MoE inputs must be on one XPU device")
    if x.ndim < 2 or scores.ndim != 2 or indices.shape != scores.shape:
        raise ValueError("x must end in model dim and scores/indices must be [tokens, top_k]")
    if x.reshape(-1, x.shape[-1]).size(0) != scores.size(0):
        raise ValueError("router scores must have one row per flattened x token")
    if routed_up.ndim != 3 or routed_gate.shape != routed_up.shape or routed_down.ndim != 3:
        raise ValueError("routed weights must be up/gate=[E,D,H], down=[E,H,D]")
    if shared_up.ndim != 3 or shared_gate.shape != shared_up.shape or shared_down.ndim != 3:
        raise ValueError("shared weights must be up/gate=[S,D,H], down=[S,H,D]")
    if (
        routed_up.size(1) != x.shape[-1]
        or routed_up.size(2) != routed_down.size(1)
        or routed_down.size(2) != x.shape[-1]
        or shared_up.size(1) != x.shape[-1]
        or shared_up.size(2) != shared_down.size(1)
        or shared_down.size(2) != x.shape[-1]
        or shared_up.size(0) != shared_down.size(0)
    ):
        raise ValueError("incompatible routed/shared MoE dimensions")
    if not isinstance(shared_stream, torch.xpu.Stream) or shared_stream.device != x.device:
        raise ValueError("shared_stream must be an XPU stream on x.device")


class _JointRoutedSharedMoE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
        routed_up: torch.Tensor,
        routed_gate: torch.Tensor,
        routed_down: torch.Tensor,
        shared_up: torch.Tensor,
        shared_gate: torch.Tensor,
        shared_down: torch.Tensor,
        routed_forward: RoutedForward,
        shared_stream: torch.xpu.Stream,
        phase_controller: PhaseSharedExpertController | None,
    ) -> torch.Tensor:
        _check(
            x,
            scores,
            indices,
            routed_up,
            routed_gate,
            routed_down,
            shared_up,
            shared_gate,
            shared_down,
            shared_stream,
        )
        ctx.needs = ctx.needs_input_grad[:9]
        forward_stream = torch.xpu.current_stream(x.device)
        ctx.forward_stream = forward_stream
        ctx.shared_stream = shared_stream
        ctx.shared_empty = shared_up.size(0) == 0
        if phase_controller is not None:
            if not isinstance(phase_controller, PhaseSharedExpertController):
                raise TypeError("phase_controller must be a PhaseSharedExpertController")
            if (
                phase_controller.x is not x
                or phase_controller.up is not shared_up
                or phase_controller.gate is not shared_gate
                or phase_controller.down is not shared_down
                or phase_controller.stream is not shared_stream
            ):
                raise ValueError(
                    "phase_controller must own this call's x, shared parameters, and shared stream"
                )
            if phase_controller.needs != (ctx.needs[0], *ctx.needs[6:9]):
                raise ValueError("phase_controller gradient requirements do not match joint MoE inputs")
        ctx.phase_controller = phase_controller

        with torch.enable_grad():
            if phase_controller is None:
                _wait(shared_stream, forward_stream)
                with torch.xpu.stream(shared_stream):
                    inner_shared = tuple(
                        _leaf(tensor, needed)
                        for tensor, needed in zip(
                            (x, shared_up, shared_gate, shared_down),
                            (ctx.needs[0], *ctx.needs[6:9]),
                        )
                    )
                    shared = shared_expert_loop(
                        inner_shared[0].reshape(-1, inner_shared[0].shape[-1]), *inner_shared[1:]
                    ).reshape_as(inner_shared[0])
                    for tensor in (x, shared_up, shared_gate, shared_down, *inner_shared, shared):
                        tensor.record_stream(shared_stream)

            with torch.xpu.stream(forward_stream):
                inner_routed = tuple(
                    _leaf(tensor, needed)
                    for tensor, needed in zip(
                        (x, scores, routed_up, routed_gate, routed_down),
                        (ctx.needs[0], ctx.needs[1], *ctx.needs[3:6]),
                    )
                )
                routed = routed_forward(*inner_routed)
                for tensor in (x, scores, routed_up, routed_gate, routed_down, *inner_routed, routed):
                    tensor.record_stream(forward_stream)

            if phase_controller is None:
                _wait(forward_stream, shared_stream)
                shared.record_stream(forward_stream)
            else:
                shared = phase_controller.finish_forward(forward_stream)
                inner_shared = phase_controller.saved_tensors()[:4]
            output = routed.detach() + shared.detach()
            output.record_stream(forward_stream)

        ctx.save_for_backward(*inner_routed, *inner_shared, routed, shared)
        return output.detach()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        (
            inner_routed_x,
            inner_scores,
            inner_routed_up,
            inner_routed_gate,
            inner_routed_down,
            inner_shared_x,
            inner_shared_up,
            inner_shared_gate,
            inner_shared_down,
            routed,
            shared,
        ) = ctx.saved_tensors
        consumer = torch.xpu.current_stream(grad_output.device)
        _wait(consumer, ctx.forward_stream)
        shared_result: list[torch.Tensor | None] = [None] * 9
        if ctx.phase_controller is None:
            _wait(ctx.shared_stream, consumer)
            with torch.xpu.stream(ctx.shared_stream), torch.enable_grad():
                for tensor in (
                    inner_shared_x,
                    inner_shared_up,
                    inner_shared_gate,
                    inner_shared_down,
                    shared,
                    grad_output,
                ):
                    tensor.record_stream(ctx.shared_stream)
                if ctx.shared_empty:
                    # ``shared_expert_loop`` returns a freshly allocated
                    # zero output for S=0, so the empty parameter tensors are
                    # genuinely unused.  Returning zero parameter gradients
                    # makes their `.grad` presence differ from the reference
                    # loop; leave them as None.  A zero x contribution is
                    # retained so it can combine with the routed x gradient.
                    if ctx.needs[0]:
                        shared_result[0] = torch.zeros_like(inner_shared_x)
                else:
                    pairs = tuple(
                        (index, tensor)
                        for index, tensor in zip(
                            (0, 6, 7, 8),
                            (
                                inner_shared_x,
                                inner_shared_up,
                                inner_shared_gate,
                                inner_shared_down,
                            ),
                        )
                        if ctx.needs[index]
                    )
                    _gradients(
                        shared,
                        tuple(tensor for _, tensor in pairs),
                        tuple(index for index, _ in pairs),
                        grad_output,
                        shared_result,
                    )
        else:
            ctx.phase_controller.arm_backward(grad_output, consumer)

        routed_result: list[torch.Tensor | None] = [None] * 9
        _wait(ctx.forward_stream, consumer)
        with torch.xpu.stream(ctx.forward_stream), torch.enable_grad():
            for tensor in (
                inner_routed_x,
                inner_scores,
                inner_routed_up,
                inner_routed_gate,
                inner_routed_down,
                routed,
                grad_output,
            ):
                tensor.record_stream(ctx.forward_stream)
            pairs = tuple(
                (index, tensor)
                for index, tensor in zip(
                    (0, 1, 3, 4, 5),
                    (
                        inner_routed_x,
                        inner_scores,
                        inner_routed_up,
                        inner_routed_gate,
                        inner_routed_down,
                    ),
                )
                if ctx.needs[index]
            )
            _gradients(
                routed,
                tuple(tensor for _, tensor in pairs),
                tuple(index for index, _ in pairs),
                grad_output,
                routed_result,
            )

        # A frozen routed branch has no nested-autograd edge, therefore no
        # routed backward invocation in which to submit reverse A4.  It is an
        # uncommon but valid configuration; preserve exact shared gradients by
        # running the controller after the stream handoff without assuming an
        # A4 window exists.  Normal training takes the overlap path above.
        if ctx.phase_controller is not None and not pairs:
            ctx.phase_controller.mark_payload_ready(ctx.forward_stream)
            if ctx.phase_controller.backward_after_a4_required:
                ctx.phase_controller.start_backward_after_a4()

        if ctx.phase_controller is None:
            _wait(consumer, ctx.shared_stream)
        else:
            phase_result = ctx.phase_controller.finish_backward(consumer)
            for index, gradient in zip((0, 6, 7, 8), phase_result):
                shared_result[index] = gradient
        _wait(consumer, ctx.forward_stream)
        result = routed_result
        with torch.xpu.stream(consumer):
            if ctx.needs[0]:
                routed_x = routed_result[0]
                shared_x = shared_result[0]
                result[0] = (
                    shared_x
                    if routed_x is None
                    else routed_x
                    if shared_x is None
                    else routed_x + shared_x
                )
        for index in (6, 7, 8):
            result[index] = shared_result[index]
        for gradient in result:
            if gradient is not None:
                gradient.record_stream(consumer)
        return (*result, None, None, None)


def joint_routed_shared_moe(
    x: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
    routed_up: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_down: torch.Tensor,
    shared_up: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_down: torch.Tensor,
    *,
    routed_forward: RoutedForward,
    shared_stream: torch.xpu.Stream,
    phase_controller: PhaseSharedExpertController | None = None,
) -> torch.Tensor:
    """Run exact routed/shared graphs, optionally phase-separating shared XMX work."""

    return _JointRoutedSharedMoE.apply(
        x,
        scores,
        indices,
        routed_up,
        routed_gate,
        routed_down,
        shared_up,
        shared_gate,
        shared_down,
        routed_forward,
        shared_stream,
        phase_controller,
    )
