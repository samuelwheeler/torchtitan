"""Exact first-order shared-SwiGLU autograd on a caller-owned XPU stream."""

from __future__ import annotations

import torch

from aurora_moe._kernels.shared_expert_bmm import shared_expert_loop


def _check_inputs(
    x: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor,
    down: torch.Tensor,
    stream: torch.xpu.Stream,
) -> None:
    if x.device.type != "xpu" or any(
        tensor.device != x.device for tensor in (up, gate, down)
    ):
        raise ValueError("x, up, gate, and down must be on one XPU device")
    if up.ndim != 3 or gate.shape != up.shape or down.ndim != 3 or x.ndim != 2:
        raise ValueError("expected x=[M,D], up/gate=[S,D,H], down=[S,H,D]")
    if (
        up.size(0) != down.size(0)
        or x.size(1) != up.size(1)
        or up.size(2) != down.size(1)
        or down.size(2) != x.size(1)
    ):
        raise ValueError("incompatible shared-expert matrix dimensions")
    if not isinstance(stream, torch.xpu.Stream) or stream.device != x.device:
        raise ValueError("stream must be an XPU stream on x.device")


def _wait(consumer: torch.xpu.Stream, producer: torch.xpu.Stream) -> None:
    if consumer != producer:
        consumer.wait_stream(producer)


class _StreamedSharedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        up: torch.Tensor,
        gate: torch.Tensor,
        down: torch.Tensor,
        stream: torch.xpu.Stream,
        defer_forward_join: bool,
    ) -> torch.Tensor:
        _check_inputs(x, up, gate, down, stream)
        if not isinstance(defer_forward_join, bool):
            raise ValueError("defer_forward_join must be a bool")
        producer = torch.xpu.current_stream(x.device)
        ctx.stream = stream
        ctx.needs = ctx.needs_input_grad[:4]
        _wait(stream, producer)
        with torch.xpu.stream(stream):
            if up.size(0) == 0:
                output = x.new_zeros(x.shape)
                ctx.empty = True
                ctx.save_for_backward(x, up, gate, down)
            else:
                ctx.empty = False
                with torch.enable_grad():
                    inner_x = x.detach().requires_grad_(ctx.needs[0])
                    inner_up = up.detach().requires_grad_(ctx.needs[1])
                    inner_gate = gate.detach().requires_grad_(ctx.needs[2])
                    inner_down = down.detach().requires_grad_(ctx.needs[3])
                    output = shared_expert_loop(inner_x, inner_up, inner_gate, inner_down)
                ctx.save_for_backward(inner_x, inner_up, inner_gate, inner_down, output)
                for tensor in (x, up, gate, down, inner_x, inner_up, inner_gate, inner_down):
                    tensor.record_stream(stream)
            output.record_stream(stream)
        returned = output.detach()
        if not defer_forward_join:
            _wait(producer, stream)
            returned.record_stream(producer)
        return returned

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        if ctx.empty:
            x, up, gate, down = ctx.saved_tensors
            consumer = torch.xpu.current_stream(grad_output.device)
            with torch.xpu.stream(ctx.stream):
                result = [
                    torch.zeros_like(tensor) if needed else None
                    for tensor, needed in zip((x, up, gate, down), ctx.needs)
                ]
                for gradient in result:
                    if gradient is not None:
                        gradient.record_stream(ctx.stream)
            _wait(consumer, ctx.stream)
            for gradient in result:
                if gradient is not None:
                    gradient.record_stream(consumer)
            return (*result, None, None)

        inner_x, inner_up, inner_gate, inner_down, output = ctx.saved_tensors
        consumer = torch.xpu.current_stream(grad_output.device)
        _wait(ctx.stream, consumer)
        grad_output.record_stream(ctx.stream)
        inputs = tuple(
            tensor
            for tensor, needed in zip((inner_x, inner_up, inner_gate, inner_down), ctx.needs)
            if needed
        )
        with torch.xpu.stream(ctx.stream), torch.enable_grad():
            gradients = torch.autograd.grad(output, inputs, grad_output, allow_unused=True)
        gradient_iter = iter(gradients)
        result: list[torch.Tensor | None] = []
        for needed in ctx.needs:
            result.append(next(gradient_iter) if needed else None)
        for gradient in result:
            if gradient is not None:
                gradient.record_stream(ctx.stream)
        _wait(consumer, ctx.stream)
        for gradient in result:
            if gradient is not None:
                gradient.record_stream(consumer)
        return (*result, None, None)


def streamed_shared_expert_swiglu(
    x: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor,
    down: torch.Tensor,
    *,
    stream: torch.xpu.Stream,
    defer_forward_join: bool = False,
) -> torch.Tensor:
    """Run exact shared SwiGLU on ``stream`` with an optional deferred join.

    When ``defer_forward_join`` is true, call
    :func:`join_streamed_shared_expert_forward` on the consumer stream before
    reading the returned tensor.
    """

    return _StreamedSharedSwiGLU.apply(x, up, gate, down, stream, defer_forward_join)


def join_streamed_shared_expert_forward(
    output: torch.Tensor, *, stream: torch.xpu.Stream
) -> torch.Tensor:
    """Make a deferred shared-expert output safe on the current consumer stream."""

    if (
        output.device.type != "xpu"
        or not isinstance(stream, torch.xpu.Stream)
        or stream.device != output.device
    ):
        raise ValueError("output and stream must share one XPU device")
    consumer = torch.xpu.current_stream(output.device)
    _wait(consumer, stream)
    output.record_stream(consumer)
    return output
