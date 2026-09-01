"""Exact first-order shared-expert SwiGLU with explicit XPU backward GEMMs."""

from __future__ import annotations

import torch

from aurora_moe._kernels.swiglu_ops import load_swiglu_ops


def _check(
    x: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, down: torch.Tensor
) -> None:
    tensors = (x, up, gate, down)
    if x.device.type != "xpu" or any(tensor.device != x.device for tensor in tensors):
        raise ValueError("x and shared weights must be on one XPU device")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError("manual shared SwiGLU requires BF16 tensors")
    if x.ndim != 2 or up.ndim != 3 or gate.shape != up.shape or down.ndim != 3:
        raise ValueError("expected x=[M,D], up/gate=[S,D,H], down=[S,H,D]")
    if (
        up.size(0) != down.size(0)
        or x.size(1) != up.size(1)
        or up.size(2) != down.size(1)
        or down.size(2) != x.size(1)
    ):
        raise ValueError("incompatible shared-expert dimensions")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("manual shared SwiGLU requires contiguous tensors")


class _ManualSharedSwiGLU(torch.autograd.Function):
    """Manual first-order BF16 SwiGLU for loop or batched shared GEMMs."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        up: torch.Tensor,
        gate: torch.Tensor,
        down: torch.Tensor,
        batched: bool,
    ) -> torch.Tensor:
        _check(x, up, gate, down)
        if not isinstance(batched, bool):
            raise ValueError("batched must be a bool")
        ctx.needs = ctx.needs_input_grad[:4]
        ctx.batched = batched
        ctx.experts = up.size(0)
        if ctx.experts == 0:
            ctx.empty = True
            ctx.save_for_backward(x, up, gate, down)
            return x.new_zeros(x.shape)

        ctx.empty = False
        ops = load_swiglu_ops()
        if batched:
            tokens = x.unsqueeze(0).expand(ctx.experts, -1, -1)
            up_values = torch.bmm(tokens, up)
            gate_values = torch.bmm(tokens, gate)
            hidden = ops.swiglu_forward_bf16(up_values, gate_values)
            output = torch.bmm(hidden, down).sum(dim=0)
            ctx.save_for_backward(x, up, gate, down, up_values, gate_values, hidden)
            return output

        output = x.new_zeros(x.shape)
        up_values: list[torch.Tensor] = []
        gate_values: list[torch.Tensor] = []
        hidden_values: list[torch.Tensor] = []
        for expert in range(ctx.experts):
            up_value = x.mm(up[expert])
            gate_value = x.mm(gate[expert])
            hidden = ops.swiglu_forward_bf16(up_value, gate_value)
            output.addmm_(hidden, down[expert])
            up_values.append(up_value)
            gate_values.append(gate_value)
            hidden_values.append(hidden)
        ctx.save_for_backward(x, up, gate, down, *up_values, *gate_values, *hidden_values)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        saved = ctx.saved_tensors
        x, up, gate, down = saved[:4]
        need_x, need_up, need_gate, need_down = ctx.needs
        if grad_output.device != x.device or grad_output.dtype != torch.bfloat16:
            raise RuntimeError("manual shared SwiGLU requires a BF16 XPU grad_output")
        grad_output = grad_output.contiguous()
        if grad_output.shape != x.shape:
            raise RuntimeError("grad_output shape does not match x")
        if ctx.empty:
            return (
                torch.zeros_like(x) if need_x else None,
                torch.zeros_like(up) if need_up else None,
                torch.zeros_like(gate) if need_gate else None,
                torch.zeros_like(down) if need_down else None,
                None,
            )

        ops = load_swiglu_ops()
        if ctx.batched:
            _, _, _, _, up_values, gate_values, hidden = saved
            tokens = x.unsqueeze(0).expand(ctx.experts, -1, -1)
            grad_values = grad_output.unsqueeze(0).expand(ctx.experts, -1, -1)
            grad_down = (
                torch.bmm(hidden.transpose(-1, -2), grad_values) if need_down else None
            )
            grad_up = grad_gate = grad_x = None
            if need_x or need_up or need_gate:
                grad_hidden = torch.bmm(grad_values, down.transpose(-1, -2))
                grad_up_values, grad_gate_values = ops.swiglu_backward_bf16(
                    grad_hidden, up_values, gate_values
                )
                if need_up:
                    grad_up = torch.bmm(tokens.transpose(-1, -2), grad_up_values)
                if need_gate:
                    grad_gate = torch.bmm(tokens.transpose(-1, -2), grad_gate_values)
                if need_x:
                    grad_x = torch.bmm(grad_up_values, up.transpose(-1, -2))
                    grad_x.baddbmm_(grad_gate_values, gate.transpose(-1, -2))
                    grad_x = grad_x.sum(dim=0)
            return grad_x, grad_up, grad_gate, grad_down, None

        up_values = saved[4 : 4 + ctx.experts]
        gate_values = saved[4 + ctx.experts : 4 + 2 * ctx.experts]
        hidden_values = saved[4 + 2 * ctx.experts : 4 + 3 * ctx.experts]
        grad_x = torch.zeros_like(x) if need_x else None
        grad_up = torch.empty_like(up) if need_up else None
        grad_gate = torch.empty_like(gate) if need_gate else None
        grad_down = torch.empty_like(down) if need_down else None
        for expert, (up_value, gate_value, hidden) in enumerate(
            zip(up_values, gate_values, hidden_values, strict=True)
        ):
            if need_down:
                assert grad_down is not None
                grad_down[expert].copy_(hidden.transpose(-1, -2).mm(grad_output))
            if need_x or need_up or need_gate:
                grad_hidden = grad_output.mm(down[expert].transpose(-1, -2))
                grad_up_value, grad_gate_value = ops.swiglu_backward_bf16(
                    grad_hidden, up_value, gate_value
                )
                if need_up:
                    assert grad_up is not None
                    grad_up[expert].copy_(x.transpose(-1, -2).mm(grad_up_value))
                if need_gate:
                    assert grad_gate is not None
                    grad_gate[expert].copy_(x.transpose(-1, -2).mm(grad_gate_value))
                if need_x:
                    assert grad_x is not None
                    grad_x.addmm_(grad_up_value, up[expert].transpose(-1, -2))
                    grad_x.addmm_(grad_gate_value, gate[expert].transpose(-1, -2))
        return grad_x, grad_up, grad_gate, grad_down, None


def manual_shared_expert_swiglu(
    x: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor,
    down: torch.Tensor,
    *,
    batched: bool = False,
) -> torch.Tensor:
    """Return the exact sum of dynamic-shape shared SwiGLU experts.

    ``x`` may have arbitrary leading dimensions; the last is the model width.
    ``batched=False`` keeps one GEMM stream per shared expert, while
    ``batched=True`` uses zero-stride expanded activations and BMMs.
    """

    if x.ndim < 2:
        raise ValueError("x must have a model-dimension axis")
    flat = x.reshape(-1, x.size(-1))
    _check(flat, up, gate, down)
    output = _ManualSharedSwiGLU.apply(flat, up, gate, down, batched)
    return output.reshape_as(x)
