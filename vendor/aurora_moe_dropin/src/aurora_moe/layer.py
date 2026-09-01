"""PyTorch module facade for the exact Aurora expert-parallel MoE runtime."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

from ._core import MOELayer
from .distributed import MoEProcessGroups, ParallelMesh, default_device


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError("AuroraMoE supports torch.bfloat16 or torch.float32")


class AuroraMoE(nn.Module):
    """Exact top-k SwiGLU MoE with Aurora-native expert parallelism.

    The module owns its DP wrappers for router/shared and sharded expert
    parameters.  Do not wrap the full module in world-size DDP; construct the
    DP x EP groups with :func:`create_dp_ep_groups` and pass them here instead.
    """

    def __init__(
        self,
        model_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        top_k: int,
        *,
        shared_experts: int = 0,
        groups: MoEProcessGroups | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backend: str = "sycl_sonic",
        seed: int = 1234,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or expert_hidden_dim <= 0 or num_experts <= 0:
            raise ValueError("model_dim, expert_hidden_dim, and num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if shared_experts < 0:
            raise ValueError("shared_experts must be nonnegative")
        if groups is None:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() != 1:
                raise RuntimeError(
                    "pass DP x EP groups for a multi-rank run; use create_dp_ep_groups"
                )
            groups = MoEProcessGroups()
        target_device = torch.device(device) if device is not None else default_device()
        self.mesh = ParallelMesh(groups, target_device)
        if num_experts % self.mesh.group_size["ep_dispatch"]:
            raise ValueError("num_experts must divide evenly across the EP group")
        if backend not in {"loop", "sycl_ep", "sycl_sonic"}:
            raise ValueError("backend must be 'loop', 'sycl_ep', or 'sycl_sonic'")

        self.model_dim = model_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_experts = shared_experts
        config = SimpleNamespace(
            model_dim=model_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            shared_experts=shared_experts,
            dtype=_dtype_name(dtype),
            expert_backend=backend,
            seed=seed,
        )
        self._impl = MOELayer(config, self.mesh, layer_id=layer_id)

    @property
    def router_weight(self) -> torch.nn.Parameter:
        """The replicated ``[num_experts, model_dim]`` router projection."""

        return self._impl.router_weight

    @property
    def experts_up(self) -> torch.nn.Parameter:
        return self._impl.experts_up

    @property
    def experts_gate(self) -> torch.nn.Parameter:
        return self._impl.experts_gate

    @property
    def experts_down(self) -> torch.nn.Parameter:
        return self._impl.experts_down

    def finalize_gradients(self) -> None:
        """Run/scale the configured DP gradient reduction after ``backward``."""

        self._impl.scale_ddp_grads()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route and evaluate ``[..., model_dim]`` activations exactly."""

        if x.ndim < 2 or x.size(-1) != self.model_dim:
            raise ValueError("x must have shape [..., model_dim]")
        return self._impl(x)

    def extra_repr(self) -> str:
        return (
            f"model_dim={self.model_dim}, expert_hidden_dim={self.expert_hidden_dim}, "
            f"num_experts={self.num_experts}, top_k={self.top_k}, "
            f"shared_experts={self.shared_experts}, ep={self.mesh.group_size['ep_dispatch']}"
        )
