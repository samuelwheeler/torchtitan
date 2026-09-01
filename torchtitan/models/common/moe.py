# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear

from torchtitan.protocols.module import Module

from .token_dispatcher import LocalTokenDispatcher

try:
    from scattermoe.parallel_experts import flatten_sort_count, parallel_linear
except ImportError:
    flatten_sort_count = None
    parallel_linear = None

ExpertComputeBackend = Literal[
    "for_loop",
    "grouped_mm",
    "batched_mm_padded",
    "scattermoe",
    "aurora_sycl",
    "aurora_full_loop",
    "aurora_full_sonic",
]


# NOTE: keeping this for-loop implementation for comparison
#       and readability, may remove later
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    # NOTE: x is not sliced because padding was removed in #2774, so
    # sum(num_tokens_per_expert) == x.shape[0] always holds.
    x_splits = torch.split(
        x,
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    return out


def _run_experts_batched_mm_padded(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    counts = num_tokens_per_expert.to(device=x.device, dtype=torch.int64)
    if counts.numel() == 0:
        return x.new_empty((0, w2.shape[1]))

    max_tokens = int(counts.max().item())
    if max_tokens == 0:
        return x.new_empty((0, w2.shape[1]))

    num_experts = counts.numel()
    total_tokens = x.shape[0]
    device = x.device

    offsets = counts.cumsum(0) - counts
    expert_indices = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int64),
        counts,
    )
    token_indices_within_expert = torch.arange(
        total_tokens, device=device, dtype=torch.int64
    ) - torch.repeat_interleave(offsets, counts)

    padded_x = x.new_zeros((num_experts, max_tokens, x.shape[-1]))
    padded_x[expert_indices, token_indices_within_expert] = x

    h = F.silu(torch.bmm(padded_x, w1.transpose(-2, -1)))
    h = h * torch.bmm(padded_x, w3.transpose(-2, -1))
    out_padded = torch.bmm(h, w2.transpose(-2, -1))

    return out_padded[expert_indices, token_indices_within_expert]


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = F.silu(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


def _run_experts_aurora_sycl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run exact compact expert GEMMs through the optional Aurora package.

    Routing and score application remain in TorchTitan's token dispatcher.
    Import lazily so every other backend remains usable without aurora-moe.
    """

    try:
        from aurora_moe.torchtitan_experts import torchtitan_exact_experts
    except ImportError as error:
        raise ImportError(
            "aurora_sycl expert backend requires aurora_moe_dropin on PYTHONPATH"
        ) from error
    return torchtitan_exact_experts(w1, w2, w3, x, num_tokens_per_expert)


def _run_experts_scattermoe(
    w13: torch.Tensor,
    w2: torch.Tensor,
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
) -> torch.Tensor:
    if flatten_sort_count is None or parallel_linear is None:
        raise ImportError(
            "scattermoe backend requested, but the scattermoe package is not installed."
        )

    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
        selected_experts_indices, num_experts=w13.shape[0]
    )

    h_and_gates = parallel_linear(
        x,
        w13,
        top_scores.shape[1],
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        grouped_out=True,
    )
    h, gates = h_and_gates.chunk(2, dim=-1)
    h = F.silu(gates) * h

    return parallel_linear(
        h,
        w2,
        1,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gates=top_scores,
        grouped_in=True,
    )


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        use_grouped_mm: bool = True
        compute_backend: ExpertComputeBackend | None = None
        token_dispatcher: LocalTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.compute_backend: ExpertComputeBackend = (
            config.compute_backend
            if config.compute_backend is not None
            else ("grouped_mm" if config.use_grouped_mm else "for_loop")
        )
        if self.compute_backend == "scattermoe":
            # Store weights directly in scattermoe's kernel layout:
            #   w13: [num_experts, dim, 2 * hidden_dim]
            #       first half = logical w3^T, second half = logical w1^T
            #   w2:  [num_experts, hidden_dim, dim]
            #       equals logical w2^T
            self.scattermoe_w13 = nn.Parameter(
                torch.empty(config.num_experts, config.dim, 2 * config.hidden_dim)
            )
            self.scattermoe_w2 = nn.Parameter(
                torch.empty(config.num_experts, config.hidden_dim, config.dim)
            )
        elif self.compute_backend in {"aurora_full_loop", "aurora_full_sonic"}:
            # Aurora's routed runtime consumes GEMM-ready [expert, in, out]
            # matrices, avoiding per-step transposes or parameter copies.
            self.aurora_up = nn.Parameter(
                torch.empty(config.num_experts, config.dim, config.hidden_dim)
            )
            self.aurora_gate = nn.Parameter(
                torch.empty(config.num_experts, config.dim, config.hidden_dim)
            )
            self.aurora_down = nn.Parameter(
                torch.empty(config.num_experts, config.hidden_dim, config.dim)
            )
        else:
            self.w1 = nn.Parameter(
                torch.empty(config.num_experts, config.hidden_dim, config.dim)
            )
            self.w2 = nn.Parameter(
                torch.empty(config.num_experts, config.dim, config.hidden_dim)
            )
            self.w3 = nn.Parameter(
                torch.empty(config.num_experts, config.hidden_dim, config.dim)
            )
        self.use_grouped_mm = self.compute_backend == "grouped_mm"
        self.token_dispatcher = config.token_dispatcher.build()

    def _init_self_parameters(self) -> None:
        if self.compute_backend not in {
            "scattermoe",
            "aurora_full_loop",
            "aurora_full_sonic",
        }:
            super()._init_self_parameters()
            return

        if self._param_init is None:
            raise ValueError(
                f"No param_init found for {self.compute_backend} GroupedExperts. "
                "Set param_init on this module's Config or use skip_param_init."
            )
        missing = [name for name in ("w1", "w2", "w3") if name not in self._param_init]
        if missing:
            raise ValueError(
                f"Missing {self.compute_backend} initializers for {missing} "
                f"in {type(self).__name__}. "
                f"Available: {list(self._param_init.keys())}"
            )

        # Initialize in the logical TorchTitan layout via transposed views so
        # we preserve the existing w1/w2/w3 initialization rules without
        # repacking at every forward.
        if self.compute_backend == "scattermoe":
            self._param_init["w3"](
                self.scattermoe_w13[:, :, : self.hidden_dim].transpose(-2, -1)
            )
            self._param_init["w1"](
                self.scattermoe_w13[:, :, self.hidden_dim :].transpose(-2, -1)
            )
            self._param_init["w2"](self.scattermoe_w2.transpose(-2, -1))
            return

        self._param_init["w3"](self.aurora_up.transpose(-2, -1))
        self._param_init["w1"](self.aurora_gate.transpose(-2, -1))
        self._param_init["w2"](self.aurora_down.transpose(-2, -1))

    def _scattermoe_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        w13_param = self.scattermoe_w13
        w2_param = self.scattermoe_w2
        if isinstance(w13_param, DTensor):
            w13 = w13_param.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = w2_param.to_local()
        else:
            w13 = w13_param
            w2 = w2_param
        return w13, w2

    def _aurora_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(self.aurora_up, DTensor):
            return (
                self.aurora_up.to_local(),
                self.aurora_gate.to_local(),
                self.aurora_down.to_local(),
            )
        return self.aurora_up, self.aurora_gate, self.aurora_down

    def _aurora_shared_weights(
        self, shared: FeedForward
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = (shared.w3.weight, shared.w1.weight, shared.w2.weight)
        if isinstance(weights[0], DTensor):
            weights = tuple(weight.to_local() for weight in weights)
        up_weight, gate_weight, down_weight = weights
        shared_hidden = up_weight.shape[0]
        if shared_hidden % self.hidden_dim:
            raise ValueError(
                "Aurora shared hidden dimension must be divisible by expert hidden dimension"
            )
        count = shared_hidden // self.hidden_dim
        up = up_weight.view(count, self.hidden_dim, self.dim).transpose(-2, -1)
        gate = gate_weight.view(count, self.hidden_dim, self.dim).transpose(-2, -1)
        down = down_weight.view(self.dim, count, self.hidden_dim).permute(1, 2, 0)
        return up, gate, down

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine."""
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.compute_backend == "grouped_mm":
            return _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
        if self.compute_backend == "batched_mm_padded":
            return _run_experts_batched_mm_padded(
                w1, w2, w3, x, num_tokens_per_expert
            )
        if self.compute_backend == "aurora_sycl":
            return _run_experts_aurora_sycl(
                w1, w2, w3, x, num_tokens_per_expert
            )
        if self.compute_backend == "for_loop":
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
        raise ValueError(f"Unknown expert compute backend: {self.compute_backend}")

    def forward(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        shared_experts is passed to combine() where it overlaps with the async
        combine all-to-all (NCCL stream) or async DeepEP combine.
        """
        if self.compute_backend == "scattermoe":
            if not isinstance(self.token_dispatcher, LocalTokenDispatcher):
                raise NotImplementedError(
                    "scattermoe backend is currently only supported for EP=1 local dispatch."
                )
            if self.token_dispatcher.score_before_experts:
                raise NotImplementedError(
                    "scattermoe backend requires score_before_experts=False."
                )

            w13, w2 = self._scattermoe_weights()

            out = _run_experts_scattermoe(w13, w2, x, top_scores, selected_experts_indices)
            if shared_experts is not None:
                out = out + shared_experts(x)
            return out
        if self.compute_backend in {"aurora_full_loop", "aurora_full_sonic"}:
            if self.token_dispatcher.score_before_experts:
                raise ValueError("Aurora full runtime requires score_after_experts")
            ep_mesh = getattr(self.token_dispatcher, "ep_mesh", None)
            if ep_mesh is None:
                raise ValueError("Aurora full runtime requires expert parallelism")
            from aurora_moe.torchtitan_full import torchtitan_full_moe

            up, gate, down = self._aurora_weights()
            backend = (
                "loop"
                if self.compute_backend == "aurora_full_loop"
                else "sycl_sonic"
            )
            if not isinstance(shared_experts, FeedForward):
                raise ValueError("Aurora full runtime requires TorchTitan shared experts")
            shared_up, shared_gate, shared_down = self._aurora_shared_weights(
                shared_experts
            )
            return torchtitan_full_moe(
                x,
                top_scores,
                selected_experts_indices,
                up,
                gate,
                down,
                shared_up,
                shared_gate,
                shared_down,
                ep_mesh,
                backend=backend,
            )
        routed_input, num_tokens_local, metadata = self.token_dispatcher.dispatch(
            x, top_scores, selected_experts_indices
        )
        routed_output = self._experts_forward(routed_input, num_tokens_local)
        return self.token_dispatcher.combine(
            routed_output, metadata, x, shared_experts
        )


class TokenChoiceTopKRouter(Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: Linear.Config
        num_expert_groups: int | None = None  # must be a divisor of num_experts
        num_limited_groups: int | None = None
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.size(0)
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(
                n_tokens * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
            and set expert scores in non-selected groups as -inf

        Args:
            scores_for_choice: Router scores with expert_bias (if any), shape (bs*slen, num_experts)

        Returns:
            scores_for_choice: shape (bs*slen, num_experts)
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice.view(
            -1, self.num_expert_groups, experts_per_group
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, self.num_experts)

        return scores_for_choice

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        # Compute gate in float32 to help stability of expert load balancing.
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        # scored is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
        # XPU topk does not promise stable indices for equal values. Router
        # scores are BF16 on Aurora, so ties are common enough for activation
        # checkpoint recomputation to change one route and therefore a ragged
        # saved-tensor shape. Promote only the choice keys and add a sub-BF16-
        # ULP expert-id tiebreak; gathered scores and their gradients remain
        # unchanged.
        expert_ids = torch.arange(
            self.num_experts, device=scores_for_choice.device, dtype=torch.float32
        )
        choice_keys = scores_for_choice.float() - expert_ids * 1.0e-7
        _, selected_experts_indices = torch.topk(
            choice_keys, k=self.top_k, dim=-1, sorted=False
        )

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        # Expert assignments are integer indices, so retain integer counts.
        # This also avoids the non-deterministic/stale XPU histc path used by
        # older TorchTitan revisions.
        num_tokens_per_expert = torch.bincount(
            selected_experts_indices.view(-1),
            minlength=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments
    2. GroupedExperts.forward() handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
          With EP, also performs all-to-all communication to send tokens
          to expert-owning ranks.
       b. expert computation
       c. combine (TokenDispatcher) — reverse the dispatch reordering.
          With EP, starts async communication (NCCL all-to-all or DeepEP
          combine), runs shared_experts in parallel, then forces sync
          (scatter_add for NCCL AllToAll, sync_combine for DeepEP) and
          produces final output.
          Without EP (LocalTokenDispatcher), no communication is needed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        experts: GroupedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.experts = config.experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        # Convert DTensor to local tensor for MoE-internal computation.
        # grad_placements=(Partial(),) ensures x.grad is Partial on the tp_mesh
        # in backward, so gradient reduction (reduce-scatter from Partial to
        # Shard(1)) happens once at the MoE boundary rather than being
        # duplicated inside the MoE.
        #
        # Why grad(x) is Partial on the tp_mesh across all parallelism:
        # - TP only / TP+EP with ETP=TP: TP-sharded expert weights (Colwise on
        #   w1/w3, Rowwise on w2) produce Partial output gradients.
        # - TP+EP with ETP=1: each TP rank processes a disjoint token subset
        #   (via sequence-parallel token splitting in AllToAllTokenDispatcher),
        #   so grad(x) is non-zero only at each rank's token positions (Partial).
        #
        # This holds for all MoE components (router.gate, routed experts, shared
        # experts) and regardless of score_before_experts.
        if isinstance(x, DTensor):
            assert (
                x.device_mesh.ndim == 1
            ), f"Expected 1D mesh, got {x.device_mesh.ndim}D mesh"
            assert x.device_mesh.mesh_dim_names == (
                "tp",
            ), f"Expected TP mesh, got mesh_dim_names={x.device_mesh.mesh_dim_names}"
            x = x.to_local(grad_placements=(Partial(),))
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # top_scores and selected_experts_indices shape (bs*slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        out = self.experts(
            x,
            top_scores,
            selected_experts_indices,
            shared_experts=self.shared_experts,
        )

        return out.reshape(bs, slen, dim)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert isinstance(buffer_device, torch.device)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
