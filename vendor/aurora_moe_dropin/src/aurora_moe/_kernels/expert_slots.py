"""Exact device-side local-expert packing plans for routed BF16 MoE rows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from aurora_moe._kernels.ep_local_ops import gather_rows, scatter_rows
from aurora_moe._kernels.swiglu_ops import load_swiglu_ops


_MODULE: ModuleType | None = None


@dataclass(frozen=True)
class ExactExpertSlotPlan:
    """One exact padded row slot per original local expert route."""

    route_slots: torch.Tensor
    group_rows: torch.Tensor
    max_rows: int
    num_experts: int
    routes: int

    @property
    def padded_rows(self) -> int:
        return self.num_experts * self.max_rows


def load_expert_slot_ops(verbose: bool = False) -> ModuleType:
    """Load device-side expert count and exact slot-assignment kernels."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.xpu.is_available():
        raise RuntimeError("Aurora XPU is required to use expert-slot kernels")
    source = Path(__file__).with_name("csrc") / "expert_slot_ops.sycl"
    build_dir = os.environ.get("AURORA_MOE_SYCL_BUILD_DIR")
    if build_dir:
        build_dir = str(Path(build_dir) / "expert_slot_ops")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "pvc")
    _MODULE = load(
        name="aurora_moe_expert_slot_ops",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        extra_sycl_cflags=["-fno-sycl-instrument-device-code"],
        with_cuda=False,
        with_sycl=True,
        verbose=verbose,
        build_directory=build_dir,
    )
    return _MODULE


def make_exact_expert_slot_plan(
    expert_ids: torch.Tensor, num_experts: int
) -> ExactExpertSlotPlan:
    """Create exact dynamic expert rows without sorting or capacity clipping.

    ``expert_ids`` is one local expert ID per already-expanded route.  Each
    route receives a unique ``expert * max_rows + row`` slot; ``max_rows`` is
    the observed maximum local load, not a configured capacity factor.
    """

    if (
        expert_ids.device.type != "xpu"
        or expert_ids.dtype != torch.int64
        or expert_ids.ndim != 1
        or not expert_ids.is_contiguous()
    ):
        raise ValueError("expert_ids must be a contiguous int64 XPU rank-1 tensor")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    ops = load_expert_slot_ops()
    group_rows = ops.expert_counts_i32(expert_ids, num_experts)
    routes = expert_ids.numel()
    max_rows = int(group_rows.max().item()) if routes else 0
    route_slots = ops.expert_route_slots_i64(expert_ids, num_experts, max_rows)
    return ExactExpertSlotPlan(
        route_slots=route_slots,
        group_rows=group_rows,
        max_rows=max_rows,
        num_experts=num_experts,
        routes=routes,
    )


def pack_expert_rows(route_values: torch.Tensor, plan: ExactExpertSlotPlan) -> torch.Tensor:
    """Pack route rows into exact ``[experts, max_rows, columns]`` BF16 rows."""

    if (
        route_values.device.type != "xpu"
        or route_values.dtype != torch.bfloat16
        or route_values.ndim != 2
        or not route_values.is_contiguous()
        or route_values.size(0) != plan.routes
    ):
        raise ValueError("route_values must be contiguous BF16 rows matching the plan")
    packed = scatter_rows(route_values, plan.route_slots, plan.padded_rows)
    return packed.reshape(plan.num_experts, plan.max_rows, route_values.size(1))


def unpack_expert_rows(padded_values: torch.Tensor, plan: ExactExpertSlotPlan) -> torch.Tensor:
    """Restore exact original route order from dynamic padded expert rows."""

    if (
        padded_values.device.type != "xpu"
        or padded_values.dtype != torch.bfloat16
        or padded_values.ndim != 3
        or not padded_values.is_contiguous()
        or padded_values.size(0) != plan.num_experts
        or padded_values.size(1) != plan.max_rows
    ):
        raise ValueError("padded_values must match the exact expert-slot plan")
    return gather_rows(padded_values.reshape(plan.padded_rows, padded_values.size(2)), plan.route_slots)


class _ExactExpertSlotBmm(torch.autograd.Function):
    """Manual autograd for exact route-row expert BMMs using device slot metadata."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        route_tokens: torch.Tensor,
        route_scores: torch.Tensor,
        up: torch.Tensor,
        gate: torch.Tensor | None,
        down: torch.Tensor,
        plan: ExactExpertSlotPlan,
        activation: str,
    ) -> torch.Tensor:
        if route_tokens.device.type != "xpu" or route_tokens.dtype != torch.bfloat16:
            raise ValueError("route_tokens must be BF16 XPU")
        if route_tokens.ndim != 2 or not route_tokens.is_contiguous():
            raise ValueError("route_tokens must be contiguous rank-2")
        if route_tokens.size(0) != plan.routes or plan.routes == 0:
            raise ValueError("expert-slot BMM requires a nonempty plan matching route_tokens")
        if (
            route_scores.device != route_tokens.device
            or route_scores.dtype != torch.bfloat16
            or route_scores.shape != (plan.routes, 1)
            or not route_scores.is_contiguous()
        ):
            raise ValueError("route_scores must be contiguous BF16 [routes, 1]")
        if up.ndim != 3 or up.shape[:2] != (plan.num_experts, route_tokens.size(1)):
            raise ValueError("up shape does not match route tokens and expert plan")
        if up.device != route_tokens.device or up.dtype != torch.bfloat16:
            raise ValueError("up must be BF16 on the route-token XPU")
        if down.ndim != 3 or down.shape != (plan.num_experts, up.size(2), route_tokens.size(1)):
            raise ValueError("down shape does not match up and route tokens")
        if down.device != route_tokens.device or down.dtype != torch.bfloat16:
            raise ValueError("down must be BF16 on the route-token XPU")
        if activation not in ("swiglu", "squared-relu"):
            raise ValueError("activation must be swiglu or squared-relu")
        if activation == "swiglu" and (
            gate is None
            or gate.shape != up.shape
            or gate.device != route_tokens.device
            or gate.dtype != torch.bfloat16
        ):
            raise ValueError("SwiGLU requires BF16 gate shaped like up on the route-token XPU")
        if activation == "squared-relu" and gate is not None:
            raise ValueError("squared-relu does not take a gate tensor")

        padded = pack_expert_rows(route_tokens, plan)
        up_values = torch.bmm(padded, up)
        if activation == "swiglu":
            assert gate is not None
            gate_values = torch.bmm(padded, gate)
            hidden = load_swiglu_ops().swiglu_forward_bf16(up_values, gate_values)
        else:
            gate_values = torch.empty(0, dtype=route_tokens.dtype, device=route_tokens.device)
            hidden = F.relu(up_values).square()
        padded_values = torch.bmm(hidden, down)
        route_values = unpack_expert_rows(padded_values, plan)
        output = (route_values * route_scores).to(torch.bfloat16)

        ctx.plan = plan
        ctx.activation = activation
        ctx.gate_present = gate is not None
        ctx.save_for_backward(
            padded,
            up,
            gate if gate is not None else torch.empty(0, dtype=route_tokens.dtype, device=route_tokens.device),
            down,
            up_values,
            gate_values,
            hidden,
            route_values,
            route_scores,
        )
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        (
            padded,
            up,
            gate,
            down,
            up_values,
            gate_values,
            hidden,
            route_values,
            route_scores,
        ) = ctx.saved_tensors
        if (
            grad_output.device != route_scores.device
            or grad_output.dtype != torch.bfloat16
            or grad_output.ndim != 2
            or grad_output.shape != route_values.shape
            or not grad_output.is_contiguous()
        ):
            raise RuntimeError("exact expert-slot BMM requires matching contiguous BF16 grad_output")
        plan: ExactExpertSlotPlan = ctx.plan

        grad_scores = None
        if ctx.needs_input_grad[1]:
            grad_scores = (grad_output.float() * route_values.float()).sum(-1, keepdim=True).to(torch.bfloat16)
        grad_route_values = (grad_output * route_scores).to(torch.bfloat16)
        grad_padded_values = pack_expert_rows(grad_route_values, plan)
        grad_down = torch.bmm(hidden.transpose(-1, -2), grad_padded_values)
        grad_hidden = torch.bmm(grad_padded_values, down.transpose(-1, -2))
        if ctx.activation == "swiglu":
            grad_up_values, grad_gate_values = load_swiglu_ops().swiglu_backward_bf16(
                grad_hidden, up_values, gate_values
            )
            grad_gate = torch.bmm(padded.transpose(-1, -2), grad_gate_values)
            grad_padded = torch.bmm(grad_gate_values, gate.transpose(-1, -2))
        else:
            grad_up_values = grad_hidden * (2.0 * F.relu(up_values))
            grad_gate = None
            grad_padded = torch.zeros_like(padded)
        grad_up = torch.bmm(padded.transpose(-1, -2), grad_up_values)
        grad_padded = grad_padded + torch.bmm(grad_up_values, up.transpose(-1, -2))
        grad_tokens = unpack_expert_rows(grad_padded, plan)
        return grad_tokens, grad_scores, grad_up, grad_gate, grad_down, None, None


def exact_expert_slot_bmm(
    route_tokens: torch.Tensor,
    route_scores: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor | None,
    down: torch.Tensor,
    plan: ExactExpertSlotPlan,
    *,
    activation: str = "swiglu",
) -> torch.Tensor:
    """Run exact local expert BF16 BMMs from device-built per-expert slots.

    The route dimension may be any flattened top-k route count.  The plan is
    independent of model dimension, hidden dimension, and expert count.
    """

    return _ExactExpertSlotBmm.apply(route_tokens, route_scores, up, gate, down, plan, activation)
