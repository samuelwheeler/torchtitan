"""Cheap DDP-forward anchors for custom-autograd parameter producers.

``DistributedDataParallel`` prepares its reducer from the graph returned by
its wrapped module's forward method.  A joint custom autograd function may
instead produce gradients for those parameters outside that module.  The
anchor below keeps the parameters visible to DDP's graph walk without adding
zero tensors to their gradients: its backward deliberately returns ``None``
for every parameter.
"""

from __future__ import annotations

import torch


class _DDPParameterAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, *parameters: torch.Tensor) -> torch.Tensor:
        if not parameters:
            raise ValueError("ddp_parameter_anchor requires at least one parameter")
        if any(parameter.device != parameters[0].device for parameter in parameters):
            raise ValueError("all anchored parameters must share a device")
        ctx.count = len(parameters)
        # This custom Function retains autograd edges to every input despite
        # not reading their values.  In particular it does not create a
        # full-sized zero gradient through a select/sum expression.
        return parameters[0].new_zeros(())

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[None, ...]:
        del grad_output
        return (None,) * ctx.count


def ddp_parameter_anchor(*parameters: torch.Tensor) -> torch.Tensor:
    """Return a scalar DDP graph anchor with no parameter-gradient contribution.

    Call this from the forward of each DDP-wrapped owner module and add the
    resulting scalar to the output of the joint custom Function.  The joint
    Function must be the only path returning non-``None`` gradients for these
    parameters in that iteration.
    """

    return _DDPParameterAnchor.apply(*parameters)
