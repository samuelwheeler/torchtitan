"""TorchTitan adapter for Aurora's complete routed-MoE runtime."""

from __future__ import annotations

import os

import torch
from torch.distributed.tensor import DeviceMesh

from ._core import _phase_shared_experts_requested, _routed_moe
from .distributed import MoEProcessGroups, ParallelMesh
from ._kernels.joint_moe_autograd import joint_routed_shared_moe
from ._kernels.phase_shared_expert import PhaseSharedExpertController
from ._kernels.shared_expert_bmm import shared_expert_loop
_MESHES: dict[tuple[int, int], ParallelMesh] = {}
_SHARED_STREAMS: dict[int, torch.xpu.Stream] = {}


def _runtime_mesh(ep_mesh: DeviceMesh, device: torch.device) -> ParallelMesh:
    group = ep_mesh.get_group()
    key = (id(group), device.index if device.index is not None else -1)
    mesh = _MESHES.get(key)
    if mesh is None:
        mesh = ParallelMesh(
            MoEProcessGroups(ep_dispatch=group, dense_dp=None, sparse_dp=None),
            device,
        )
        _MESHES[key] = mesh
    return mesh


def torchtitan_full_moe(
    x: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor,
    down: torch.Tensor,
    shared_up: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_down: torch.Tensor,
    ep_mesh: DeviceMesh,
    *,
    backend: str,
) -> torch.Tensor:
    """Run exact Aurora routed and shared expert forward/backward."""

    if backend not in {"loop", "sycl_sonic"}:
        raise ValueError(f"unsupported Aurora routed backend {backend!r}")
    if x.device.type != "xpu" or x.dtype != torch.bfloat16:
        raise ValueError("Aurora routed MoE requires BF16 XPU activations")
    if not all(weight.is_contiguous() for weight in (up, gate, down)):
        raise ValueError("Aurora routed expert weights must be contiguous")
    if up.shape != gate.shape or down.shape != (up.shape[0], up.shape[2], up.shape[1]):
        raise ValueError("invalid Aurora expert weight layouts")

    mesh = _runtime_mesh(ep_mesh, x.device)
    if up.shape[0] * mesh.group_size["ep_dispatch"] <= 0:
        raise ValueError("Aurora routed MoE requires local experts")
    if shared_up.shape != shared_gate.shape or shared_down.shape != (
        shared_up.shape[0], shared_up.shape[2], shared_up.shape[1]
    ):
        raise ValueError("invalid Aurora shared-expert weight layouts")

    scores = scores.to(torch.bfloat16)
    local_ids = range(up.shape[0])

    def routed(inner_x, inner_scores, inner_up, inner_gate, inner_down, phase=None):
        return _routed_moe(
            inner_x, inner_scores, indices, local_ids, mesh,
            inner_up, inner_gate, inner_down, backend, phase,
        )

    if backend == "loop":
        shared = shared_expert_loop(
            x.reshape(-1, x.shape[-1]), shared_up, shared_gate, shared_down
        ).reshape_as(x)
        return routed(x, scores, up, gate, down) + shared

    # Keep a true ordinary-autograd Sonic path for distributed liveness
    # isolation.  The optimized production preset explicitly enables the
    # joint wrapper, so this opt-out does not alter its default execution.
    # Running the two branches in the outer graph is also the conservative
    # fallback if a framework/runtime combination cannot safely execute
    # nested autograd across the private shared-expert stream.
    joint_requested = (
        os.environ.get("AURORA_MOE_JOINT_ROUTED_SHARED_AUTOGRAD") == "1"
    )
    if not joint_requested:
        if os.environ.get("AURORA_MOE_PHASE_SHARED_EXPERTS") == "1":
            raise RuntimeError(
                "AURORA_MOE_PHASE_SHARED_EXPERTS=1 requires "
                "AURORA_MOE_JOINT_ROUTED_SHARED_AUTOGRAD=1"
            )
        routed_output = routed(x, scores, up, gate, down)
        shared = shared_expert_loop(
            x.reshape(-1, x.shape[-1]), shared_up, shared_gate, shared_down
        ).reshape_as(x)
        return routed_output + shared

    stream = _SHARED_STREAMS.get(x.device.index)
    if stream is None:
        stream = torch.xpu.Stream()
        _SHARED_STREAMS[x.device.index] = stream
    phase = None
    if _phase_shared_experts_requested(mesh):
        phase = PhaseSharedExpertController(
            x, shared_up, shared_gate, shared_down,
            (
                x.requires_grad, shared_up.requires_grad,
                shared_gate.requires_grad, shared_down.requires_grad,
            ),
            stream,
        )
    return joint_routed_shared_moe(
        x, scores, indices, up, gate, down, shared_up, shared_gate, shared_down,
        routed_forward=lambda *args: routed(*args, phase),
        shared_stream=stream,
        phase_controller=phase,
    )


__all__ = ["torchtitan_full_moe"]
