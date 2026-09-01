"""Controlled first-order backward ordering for routed and shared SwiGLU branches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from aurora_moe._kernels.shared_expert_bmm import shared_expert_loop


def _routed_swiglu(
    x: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    return (F.silu(x.mm(gate)) * x.mm(up)).mm(down)


def _check_inputs(
    x: torch.Tensor,
    routed_up: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_down: torch.Tensor,
    shared_up: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_down: torch.Tensor,
    shared_stream: torch.xpu.Stream,
) -> None:
    tensors = (routed_up, routed_gate, routed_down, shared_up, shared_gate, shared_down)
    if x.device.type != "xpu" or any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all inputs must be on one XPU device")
    if x.ndim != 2 or routed_up.ndim != 2 or routed_gate.shape != routed_up.shape:
        raise ValueError("expected x=[M,D] and routed up/gate=[D,H]")
    if routed_down.ndim != 2 or routed_down.shape != (routed_up.size(1), x.size(1)):
        raise ValueError("routed_down must have shape [H,D]")
    if shared_up.ndim != 3 or shared_gate.shape != shared_up.shape or shared_down.ndim != 3:
        raise ValueError("expected shared up/gate=[S,D,H], down=[S,H,D]")
    if (
        shared_up.size(0) != shared_down.size(0)
        or shared_up.size(1) != x.size(1)
        or shared_up.size(2) != shared_down.size(1)
        or shared_down.size(2) != x.size(1)
    ):
        raise ValueError("incompatible shared-expert matrix dimensions")
    if not isinstance(shared_stream, torch.xpu.Stream) or shared_stream.device != x.device:
        raise ValueError("shared_stream must be an XPU stream on x.device")


def _wait(consumer: torch.xpu.Stream, producer: torch.xpu.Stream) -> None:
    if consumer != producer:
        consumer.wait_stream(producer)


def _leaf(tensor: torch.Tensor, needed: bool) -> torch.Tensor:
    return tensor.detach().requires_grad_(needed)


def _map_gradients(
    output: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    indices: tuple[int, ...],
    grad_output: torch.Tensor,
    result: list[torch.Tensor | None],
) -> None:
    if not tensors:
        return
    gradients = torch.autograd.grad(output, tensors, grad_output, allow_unused=True)
    for index, gradient in zip(indices, gradients):
        result[index] = gradient


class _JointStreamedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        routed_up: torch.Tensor,
        routed_gate: torch.Tensor,
        routed_down: torch.Tensor,
        shared_up: torch.Tensor,
        shared_gate: torch.Tensor,
        shared_down: torch.Tensor,
        shared_stream: torch.xpu.Stream,
        trace: list[str] | None,
    ) -> torch.Tensor:
        _check_inputs(
            x,
            routed_up,
            routed_gate,
            routed_down,
            shared_up,
            shared_gate,
            shared_down,
            shared_stream,
        )
        if trace is not None and not isinstance(trace, list):
            raise ValueError("trace must be a list or None")
        ctx.needs = ctx.needs_input_grad[:7]
        forward_stream = torch.xpu.current_stream(x.device)
        ctx.forward_stream = forward_stream
        ctx.shared_stream = shared_stream
        ctx.trace = trace
        with torch.enable_grad():
            # A device-autograd node normally inherits the stream of its
            # forward operation.  Build the nested shared graph on the
            # private stream, while the routed graph remains on the caller's
            # stream.  The x leaves must be distinct: one autograd leaf may
            # not participate in nested graphs recorded on two XPU streams.
            # Their independent forward work can overlap, and the split is
            # preserved when their nested backward graphs run.
            _wait(shared_stream, forward_stream)
            with torch.xpu.stream(shared_stream):
                inner_shared = tuple(
                    _leaf(tensor, needed)
                    for tensor, needed in zip(
                        (x, shared_up, shared_gate, shared_down),
                        (ctx.needs[0], *ctx.needs[4:]),
                    )
                )
                shared = shared_expert_loop(*inner_shared)
                for tensor in (
                    x,
                    shared_up,
                    shared_gate,
                    shared_down,
                    *inner_shared,
                    shared,
                ):
                    tensor.record_stream(shared_stream)
            with torch.xpu.stream(forward_stream):
                inner_routed = tuple(
                    _leaf(tensor, needed)
                    for tensor, needed in zip(
                        (x, routed_up, routed_gate, routed_down),
                        ctx.needs[:4],
                    )
                )
                routed = _routed_swiglu(*inner_routed)
                for tensor in (x, routed_up, routed_gate, routed_down, *inner_routed, routed):
                    tensor.record_stream(forward_stream)

            # The custom Function owns the backward, so avoid retaining an
            # otherwise unused graph for the output addition.  The caller
            # stream joins only when its visible output must be formed.
            _wait(forward_stream, shared_stream)
            shared.record_stream(forward_stream)
            output = routed.detach() + shared.detach()
            output.record_stream(forward_stream)
        ctx.shared_empty = shared_up.size(0) == 0
        ctx.save_for_backward(*inner_routed, *inner_shared, routed, shared)
        return output.detach()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        (
            inner_routed_x,
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
        _wait(ctx.shared_stream, consumer)
        shared_result: list[torch.Tensor | None] = [None] * 7
        with torch.xpu.stream(ctx.shared_stream), torch.enable_grad():
            # These saved tensors and the incoming gradient are consumed on a
            # non-owning stream.  Record that use so their allocations cannot
            # be recycled before the private-stream backward has completed.
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
                for index, tensor in zip(
                    (0, 4, 5, 6),
                    (
                        inner_shared_x,
                        inner_shared_up,
                        inner_shared_gate,
                        inner_shared_down,
                    ),
                ):
                    if ctx.needs[index]:
                        shared_result[index] = torch.zeros_like(tensor)
            else:
                pairs = tuple(
                    (index, tensor)
                    for index, tensor in zip(
                        (0, 4, 5, 6),
                        (
                            inner_shared_x,
                            inner_shared_up,
                            inner_shared_gate,
                            inner_shared_down,
                        ),
                    )
                    if ctx.needs[index]
                )
                _map_gradients(
                    shared,
                    tuple(tensor for _, tensor in pairs),
                    tuple(index for index, _ in pairs),
                    grad_output,
                    shared_result,
                )
        if ctx.trace is not None:
            ctx.trace.append("shared_grad_enqueued")

        routed_result: list[torch.Tensor | None] = [None] * 7
        # Retain the caller's routed stream rather than relying on the stream
        # that happened to call backward.  This is normally the default
        # stream, and it lets the two nested backward graphs execute apart.
        _wait(ctx.forward_stream, consumer)
        with torch.xpu.stream(ctx.forward_stream), torch.enable_grad():
            for tensor in (
                inner_routed_x,
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
                    (0, 1, 2, 3),
                    (
                        inner_routed_x,
                        inner_routed_up,
                        inner_routed_gate,
                        inner_routed_down,
                    ),
                )
                if ctx.needs[index]
            )
            _map_gradients(
                routed,
                tuple(tensor for _, tensor in pairs),
                tuple(index for index, _ in pairs),
                grad_output,
                routed_result,
            )
        if ctx.trace is not None:
            ctx.trace.append("routed_grad_enqueued")

        _wait(consumer, ctx.shared_stream)
        _wait(consumer, ctx.forward_stream)
        if ctx.trace is not None:
            ctx.trace.append("shared_join_enqueued")
        result = routed_result
        with torch.xpu.stream(consumer):
            if ctx.needs[0]:
                routed_x = routed_result[0]
                shared_x = shared_result[0]
                if routed_x is None:
                    result[0] = shared_x
                elif shared_x is None:
                    result[0] = routed_x
                else:
                    result[0] = routed_x + shared_x
        for index in (4, 5, 6):
            result[index] = shared_result[index]
        for gradient in result:
            if gradient is not None:
                gradient.record_stream(consumer)
        return (*result, None, None)


def joint_streamed_routed_shared_swiglu(
    x: torch.Tensor,
    routed_up: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_down: torch.Tensor,
    shared_up: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_down: torch.Tensor,
    *,
    shared_stream: torch.xpu.Stream,
    trace: list[str] | None = None,
) -> torch.Tensor:
    """Run synthetic routed plus shared SwiGLU with controlled backward ordering."""

    return _JointStreamedSwiGLU.apply(
        x,
        routed_up,
        routed_gate,
        routed_down,
        shared_up,
        shared_gate,
        shared_down,
        shared_stream,
        trace,
    )
